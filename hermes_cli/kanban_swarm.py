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

# Re-export structured key helpers for callers that want namespaced keys.
# The blackboard itself stays string-keyed — these are naming convention
# helpers, not a schema change.
from hermes_cli.kv_helpers import (  # noqa: F401  — public re-export
    worker_key,
    worker_result_key,
    worker_status_key,
    worker_handoff_key,
    team_key,
    coordinator_key,
    parse_worker_key,
    parse_namespaced_key,
    worker_prefix,
    worker_fields,
    get_worker_field,
    TOPOLOGY_KEY,
    GOAL_KEY,
    AUTHORS_KEY,
)

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
    existing = latest_blackboard(conn, root).get(TOPOLOGY_KEY)
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
        key=TOPOLOGY_KEY,
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
