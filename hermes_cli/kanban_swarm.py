"""Kanban Swarm v1: thin swarm topology helpers on top of Kanban.

This module intentionally does not introduce a second scheduler. It writes a
small task graph into the existing Kanban kernel:

    planning root (completed immediately)
        ├─ parallel specialist workers (ready)
        └─ verifier (todo until all workers done)
             └─ synthesizer (todo until verifier done)

The shared blackboard is also deliberately low-tech: structured JSON comments on
the root task. That keeps all state in existing task_comments/task_events rows,
so the dashboard, notifier, slash command, and dispatcher keep working without a
new service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

BLACKBOARD_PREFIX = "[swarm:blackboard] "


@dataclass(frozen=True)
class SwarmWorkerSpec:
    """A single parallel worker card in a swarm."""

    profile: str
    title: str
    body: str
    skills: list[str] = field(default_factory=list)
    priority: int = 0
    max_runtime_seconds: Optional[int] = None


@dataclass(frozen=True)
class SwarmCreated:
    """IDs produced by :func:`create_swarm`."""

    root_id: str
    worker_ids: list[str]
    verifier_id: str
    synthesizer_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "worker_ids": list(self.worker_ids),
            "verifier_id": self.verifier_id,
            "synthesizer_id": self.synthesizer_id,
        }


def _require_text(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _swarm_context(root_id: str, goal: str) -> str:
    return (
        "\n\n## Swarm protocol\n"
        f"- Swarm root / shared blackboard: `{root_id}`.\n"
        "- Read sibling/parent handoffs from Kanban context before working.\n"
        "- Put machine-readable facts in completion metadata.\n"
        "- Put cross-worker notes on the root task using structured comments.\n"
        f"- Goal: {goal.strip()}\n"
    )


def create_swarm(
    conn: sqlite3.Connection,
    *,
    goal: str,
    workers: Iterable[SwarmWorkerSpec],
    verifier_assignee: str,
    synthesizer_assignee: str,
    root_title: Optional[str] = None,
    verifier_title: str = "Verify swarm outputs",
    synthesizer_title: str = "Synthesize swarm outputs",
    tenant: Optional[str] = None,
    created_by: str = "swarm-orchestrator",
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
) -> SwarmCreated:
    """Create a durable Kanban swarm graph.

    The returned graph is immediately dispatchable: the planning root is marked
    ``done`` with topology metadata, parallel workers are ``ready``, the verifier
    waits for every worker, and the synthesizer waits for the verifier.
    """

    goal = _require_text(goal, "goal")
    verifier_assignee = _require_text(verifier_assignee, "verifier_assignee")
    synthesizer_assignee = _require_text(synthesizer_assignee, "synthesizer_assignee")
    worker_specs = list(workers)
    if not worker_specs:
        raise ValueError("at least one worker is required")
    for i, spec in enumerate(worker_specs, start=1):
        _require_text(spec.profile, f"workers[{i}].profile")
        _require_text(spec.title, f"workers[{i}].title")

    root = kb.create_task(
        conn,
        title=root_title or f"Swarm: {goal.splitlines()[0][:80]}",
        body=(
            "Kanban Swarm v1 planning/root card. This card is completed "
            "immediately so parallel workers can start while it remains the "
            "shared blackboard and audit anchor.\n\n"
            f"Goal:\n{goal}"
        ),
        assignee=created_by,
        created_by=created_by,
        tenant=tenant,
        priority=priority,
        idempotency_key=idempotency_key,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["kanban-orchestrator"],
    )

    # If idempotency returned an existing non-archived root, do not duplicate the
    # swarm graph. Recover the topology from the root's latest blackboard, if it
    # was created by this helper previously.
    existing = latest_blackboard(conn, root).get("topology")
    if isinstance(existing, dict):
        worker_ids = [str(x) for x in existing.get("worker_ids", []) if x]
        verifier_id = existing.get("verifier_id")
        synthesizer_id = existing.get("synthesizer_id")
        if worker_ids and verifier_id and synthesizer_id:
            return SwarmCreated(
                root_id=root,
                worker_ids=worker_ids,
                verifier_id=str(verifier_id),
                synthesizer_id=str(synthesizer_id),
            )

    kb.complete_task(
        conn,
        root,
        summary="Swarm topology planned; root remains the shared blackboard.",
        metadata={
            "kind": "kanban_swarm_v1",
            "goal": goal,
            "worker_count": len(worker_specs),
        },
    )

    context_suffix = _swarm_context(root, goal)
    worker_ids: list[str] = []
    for spec in worker_specs:
        worker_id = kb.create_task(
            conn,
            title=spec.title,
            body=(spec.body or "") + context_suffix,
            assignee=spec.profile,
            created_by=created_by,
            parents=[root],
            tenant=tenant,
            priority=spec.priority or priority,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            skills=spec.skills or None,
            max_runtime_seconds=spec.max_runtime_seconds,
        )
        worker_ids.append(worker_id)

    verifier_body = (
        "Review every worker handoff and blackboard update. Gate the swarm: "
        "complete only with metadata {\"gate\": \"pass\"} when evidence is "
        "sufficient; otherwise block with exact missing work."
        + context_suffix
    )
    verifier = kb.create_task(
        conn,
        title=verifier_title,
        body=verifier_body,
        assignee=verifier_assignee,
        created_by=created_by,
        parents=worker_ids,
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["requesting-code-review"],
    )

    synthesizer_body = (
        "Synthesize the verified worker outputs into the final deliverable. "
        "Do not start until the verifier has passed the gate."
        + context_suffix
    )
    synthesizer = kb.create_task(
        conn,
        title=synthesizer_title,
        body=synthesizer_body,
        assignee=synthesizer_assignee,
        created_by=created_by,
        parents=[verifier],
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["humanizer"],
    )

    created = SwarmCreated(root, worker_ids, verifier, synthesizer)
    post_blackboard_update(
        conn,
        root,
        author=created_by,
        key="topology",
        value=created.as_dict() | {"goal": goal},
    )
    return created


def post_blackboard_update(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    author: str,
    key: str,
    value: Any,
) -> int:
    """Append one structured update to the swarm root blackboard."""

    _require_text(root_id, "root_id")
    author = _require_text(author, "author")
    key = _require_text(key, "key")
    payload = json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True)
    return kb.add_comment(conn, root_id, author=author, body=BLACKBOARD_PREFIX + payload)


def latest_blackboard(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    """Merge structured blackboard comments on a root card.

    Later comments replace earlier values for the same key. ``_authors`` records
    the author of the winning value for traceability.
    """

    merged: dict[str, Any] = {}
    authors: dict[str, str] = {}
    for comment in kb.list_comments(conn, root_id):
        body = comment.body or ""
        if not body.startswith(BLACKBOARD_PREFIX):
            continue
        try:
            payload = json.loads(body[len(BLACKBOARD_PREFIX):])
        except json.JSONDecodeError:
            continue
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            continue
        merged[key] = payload.get("value")
        authors[key] = comment.author
    if authors:
        merged["_authors"] = authors
    return merged


def parse_worker_arg(raw: str) -> SwarmWorkerSpec:
    """Parse CLI ``--worker profile:title[:skill,skill]`` values."""

    parts = [p.strip() for p in raw.split(":", 2)]
    if len(parts) < 2:
        raise ValueError("worker must be profile:title or profile:title:skill,skill")
    skills: list[str] = []
    if len(parts) == 3 and parts[2]:
        skills = [s.strip() for s in parts[2].split(",") if s.strip()]
    return SwarmWorkerSpec(profile=parts[0], title=parts[1], body=parts[1], skills=skills)


def _latest_worker_coverage_metadata(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? AND metadata IS NOT NULL ORDER BY ended_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["metadata"]:
        return {}
    try:
        loaded = json.loads(str(row["metadata"]))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _scope_reduction_approved(scope_reduction_evidence: Any) -> bool:
    if isinstance(scope_reduction_evidence, dict):
        approval = scope_reduction_evidence.get("scope_reduction_approval")
        if isinstance(approval, dict):
            return bool(str(approval.get("source") or approval.get("text") or "").strip())
        return bool(str(scope_reduction_evidence.get("approval") or "").strip())
    return bool(str(scope_reduction_evidence or "").strip())


def _summarize_parent_rollup_rows(
    expected_child_ids: Iterable[str],
    child_coverage: Iterable[dict[str, Any]] | None,
    *,
    scope_reduced_child_ids: Iterable[str] | None = None,
    scope_reduction_evidence: Any = None,
) -> dict[str, Any]:
    child_ids = [str(child_id) for child_id in expected_child_ids if str(child_id)]
    reduced_ids = [str(child_id) for child_id in (scope_reduced_child_ids or []) if str(child_id)]
    child_rows: dict[str, dict[str, Any]] = {}
    malformed: list[dict[str, Any]] = []
    for row in list(child_coverage or []):
        if not isinstance(row, dict):
            malformed.append({})
            continue
        child_id = str(row.get("child_id") or "").strip()
        if not child_id:
            malformed.append(row)
            continue
        child_rows[child_id] = row

    covered: list[str] = []
    partial: list[str] = []
    descoped: list[str] = []
    missing: list[str] = []
    reduction_unapproved = False
    for child_id in child_ids:
        row = child_rows.get(child_id)
        if row is None:
            missing.append(child_id)
            continue
        status = str(row.get("status") or "").strip().lower()
        evidence = str(row.get("evidence") or row.get("summary") or row.get("reason") or "").strip()
        approval = row if status == "descoped" else scope_reduction_evidence
        if child_id in reduced_ids or status == "descoped":
            if _scope_reduction_approved(approval):
                descoped.append(child_id)
            else:
                missing.append(child_id)
                reduction_unapproved = True
            continue
        if status == "complete" and evidence:
            covered.append(child_id)
        else:
            partial.append(child_id)

    blockers: list[str] = []
    if reduction_unapproved or (reduced_ids and not _scope_reduction_approved(scope_reduction_evidence)):
        blockers.append("parent_rollup_scope_reduction_requires_evidence")
        status = "needs_user_decision"
    elif malformed:
        blockers.append("parent_rollup_malformed_child_coverage")
        status = "partial"
    elif missing:
        blockers.append("parent_rollup_missing_child_coverage")
        status = "review_blocked" if not partial else "needs_user_decision"
    elif partial:
        blockers.append("parent_rollup_partial_child_coverage")
        status = "partial"
    else:
        status = "complete"
    return {
        "schema": "kanban_swarm_parent_rollup.v1",
        "status": status,
        "child_ids": child_ids,
        "required_child_ids": child_ids,
        "covered_child_ids": covered,
        "partial_child_ids": partial,
        "descoped_child_ids": descoped,
        "missing_child_ids": missing,
        "malformed_child_coverage": malformed,
        "ready_to_complete": status == "complete",
        "required_scope_reduction_evidence": missing if "parent_rollup_scope_reduction_requires_evidence" in blockers else [],
        "blockers": blockers,
    }


def summarize_parent_rollup(
    conn_or_expected_child_ids: sqlite3.Connection | Iterable[str] | None = None,
    parent_task_id: str | None = None,
    *,
    expected_child_ids: Iterable[str] | None = None,
    child_coverage: Iterable[dict[str, Any]] | None = None,
    scope_reduced_child_ids: Iterable[str] | None = None,
    scope_reduction_evidence: Any = None,
) -> dict[str, Any]:
    """Summarize parent swarm child coverage without silently shrinking scope."""
    if expected_child_ids is not None:
        return _summarize_parent_rollup_rows(
            expected_child_ids,
            child_coverage,
            scope_reduced_child_ids=scope_reduced_child_ids,
            scope_reduction_evidence=scope_reduction_evidence,
        )
    if not isinstance(conn_or_expected_child_ids, sqlite3.Connection) or parent_task_id is None:
        raise TypeError("summarize_parent_rollup requires either explicit child inputs or (conn, parent_task_id)")
    conn = conn_or_expected_child_ids
    required_child_ids = kb.child_ids(conn, parent_task_id)
    rows: list[dict[str, Any]] = []
    for child_id in required_child_ids:
        task = kb.get_task(conn, child_id)
        if task is None:
            continue
        metadata = _latest_worker_coverage_metadata(conn, child_id)
        coverage = metadata.get("coverage") if isinstance(metadata.get("coverage"), dict) else None
        if coverage is None:
            if task.status == "done":
                rows.append({"child_id": child_id, "status": "partial", "evidence": task.result or ""})
            continue
        rows.append(
            {
                "child_id": child_id,
                "status": str(coverage.get("status") or "").strip().lower(),
                "evidence": coverage.get("evidence") or coverage.get("summary") or task.result or "",
                "scope_reduction_approval": coverage.get("scope_reduction_approval"),
            }
        )
    return _summarize_parent_rollup_rows(required_child_ids, rows)
