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
import hashlib
import json
import sqlite3
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

BLACKBOARD_PREFIX = "[swarm:blackboard] "
RESERVED_BLACKBOARD_KEYS = frozenset({"topology"})
_UNSET = object()


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


def _created_from_topology(
    data: dict[str, Any], *, expected_root_id: str
) -> SwarmCreated | None:
    worker_ids_raw = data.get("worker_ids")
    verifier_id = data.get("verifier_id")
    synthesizer_id = data.get("synthesizer_id")
    root_id = data.get("root_id")
    if (
        isinstance(root_id, str)
        and root_id == expected_root_id
        and isinstance(worker_ids_raw, list)
        and worker_ids_raw
        and all(
            isinstance(worker_id, str) and worker_id for worker_id in worker_ids_raw
        )
        and isinstance(verifier_id, str)
        and verifier_id
        and isinstance(synthesizer_id, str)
        and synthesizer_id
    ):
        worker_ids = list(worker_ids_raw)
        all_ids = [root_id, *worker_ids, verifier_id, synthesizer_id]
        if len(worker_ids) == len(set(worker_ids)) and len(all_ids) == len(
            set(all_ids)
        ):
            return SwarmCreated(
                root_id=root_id,
                worker_ids=worker_ids,
                verifier_id=verifier_id,
                synthesizer_id=synthesizer_id,
            )
    return None


def get_authoritative_topology(
    conn: sqlite3.Connection, root_id: str
) -> SwarmCreated | None:
    """Return DB-authoritative swarm topology, ignoring comment blackboard."""

    topology = kb.get_swarm_topology(conn, root_id)
    if topology is None:
        return None
    if "tenant" not in topology or "project_id" not in topology:
        return None
    created = _created_from_topology(topology, expected_root_id=root_id)
    if created is None or not _topology_binds_to_graph(
        conn,
        created,
        allow_untyped_synth=False,
        expected_tenant=topology.get("tenant", _UNSET),
        expected_project_id=topology.get("project_id", _UNSET),
    ):
        return None
    return created


def _topology_binds_to_graph(
    conn: sqlite3.Connection,
    created: SwarmCreated,
    *,
    allow_untyped_synth: bool,
    expected_tenant: object = _UNSET,
    expected_project_id: object = _UNSET,
) -> bool:
    """Validate task existence, scope isolation, and exact dependency edges."""

    all_ids = [
        created.root_id,
        *created.worker_ids,
        created.verifier_id,
        created.synthesizer_id,
    ]
    placeholders = ",".join("?" for _ in all_ids)
    rows = conn.execute(
        f"SELECT id, tenant, project_id FROM tasks WHERE id IN ({placeholders})",
        tuple(all_ids),
    ).fetchall()
    if {row["id"] for row in rows} != set(all_ids):
        return False
    if len({row["tenant"] for row in rows}) != 1:
        return False
    if len({row["project_id"] for row in rows}) != 1:
        return False
    root_row = next(row for row in rows if row["id"] == created.root_id)
    if expected_tenant is not _UNSET and root_row["tenant"] != expected_tenant:
        return False
    if (
        expected_project_id is not _UNSET
        and root_row["project_id"] != expected_project_id
    ):
        return False
    if any(
        kb.parent_ids(conn, worker_id) != [created.root_id]
        for worker_id in created.worker_ids
    ):
        return False
    if set(kb.parent_ids(conn, created.verifier_id)) != set(created.worker_ids):
        return False
    synth_parents = conn.execute(
        "SELECT parent_id, gate_kind FROM task_links WHERE child_id = ?",
        (created.synthesizer_id,),
    ).fetchall()
    if len(synth_parents) != 1:
        return False
    synth_parent = synth_parents[0]
    if synth_parent["parent_id"] != created.verifier_id or (
        synth_parent["gate_kind"] != "metadata_gate_pass"
        and not (allow_untyped_synth and synth_parent["gate_kind"] is None)
    ):
        return False
    return True


def _legacy_topology_from_comments(
    conn: sqlite3.Connection, root_id: str
) -> SwarmCreated | None:
    """Read the newest legacy topology comment without treating it as authority."""

    for comment in reversed(kb.list_comments(conn, root_id)):
        body = (comment.body or "").strip()
        if not body.startswith(BLACKBOARD_PREFIX):
            continue
        try:
            payload = json.loads(body[len(BLACKBOARD_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("key") != "topology":
            continue
        value = payload.get("value")
        if not isinstance(value, dict):
            return None
        return _created_from_topology(value, expected_root_id=root_id)
    return None


def migrate_legacy_swarm_topology(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    created_by: str = "swarm-topology-migration",
) -> SwarmCreated | None:
    """Migrate a structurally valid legacy comment topology atomically."""

    with kb.write_txn(conn):
        existing = get_authoritative_topology(conn, root_id)
        if existing is not None:
            return existing
        candidate = _legacy_topology_from_comments(conn, root_id)
        if candidate is None or not _topology_binds_to_graph(
            conn, candidate, allow_untyped_synth=True
        ):
            return None
        gate_row = conn.execute(
            "SELECT gate_kind FROM task_links WHERE parent_id = ? AND child_id = ?",
            (candidate.verifier_id, candidate.synthesizer_id),
        ).fetchone()
        if gate_row is None:
            return None
        if gate_row["gate_kind"] is None:
            kb.set_dependency_gate(
                conn,
                candidate.verifier_id,
                candidate.synthesizer_id,
                "metadata_gate_pass",
                actor=created_by,
            )
        root_scope = dict(
            conn.execute(
                "SELECT tenant, project_id FROM tasks WHERE id = ?", (root_id,)
            ).fetchone()
        )
        topology_payload = candidate.as_dict() | root_scope
        kb.store_swarm_topology(
            conn,
            root_id,
            topology_payload,
            created_by=created_by,
        )
        topology_sha256 = hashlib.sha256(
            json.dumps(topology_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        kb.append_event(
            conn,
            root_id,
            "swarm_topology_migrated",
            {
                "actor": created_by,
                "synthesizer_id": candidate.synthesizer_id,
                "verifier_id": candidate.verifier_id,
                "topology_sha256": topology_sha256,
            },
        )
        return candidate


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
    project_id: Optional[str] = None,
    created_by: str = "swarm-orchestrator",
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
) -> SwarmCreated:
    """Atomically create or recover a durable Kanban swarm graph."""

    with kb.write_txn(conn):
        return _create_swarm_in_txn(
            conn,
            goal=goal,
            workers=workers,
            verifier_assignee=verifier_assignee,
            synthesizer_assignee=synthesizer_assignee,
            root_title=root_title,
            verifier_title=verifier_title,
            synthesizer_title=synthesizer_title,
            tenant=tenant,
            project_id=project_id,
            created_by=created_by,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            priority=priority,
            idempotency_key=idempotency_key,
        )


def _create_swarm_in_txn(
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
    project_id: Optional[str] = None,
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

    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND tenant IS ? AND project_id IS ? "
            "AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
            (idempotency_key, tenant, project_id),
        ).fetchone()
        if row is not None:
            existing = get_authoritative_topology(conn, row["id"])
            if existing is None and kb.get_swarm_topology(conn, row["id"]) is None:
                existing = migrate_legacy_swarm_topology(
                    conn, row["id"], created_by=created_by
                )
            if existing is None:
                raise ValueError(
                    f"idempotent swarm root {row['id']} lacks valid authoritative topology"
                )
            return existing

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
        project_id=project_id,
        priority=priority,
        idempotency_key=idempotency_key,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
    )

    # If idempotency returned an existing non-archived root, do not duplicate the
    # swarm graph. Recover only from DB-authoritative topology; blackboard
    # comments are untrusted evidence and may be worker-forged.
    existing = get_authoritative_topology(conn, root)
    if existing is not None:
        return existing

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
            project_id=project_id,
            priority=spec.priority or priority,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            skills=spec.skills or None,
            max_runtime_seconds=spec.max_runtime_seconds,
        )
        worker_ids.append(worker_id)

    verifier_body = (
        "Review every worker handoff and blackboard update. Gate the swarm: "
        'complete only with metadata {"gate": "pass"} when evidence is '
        "sufficient; otherwise block with exact missing work." + context_suffix
    )
    verifier = kb.create_task(
        conn,
        title=verifier_title,
        body=verifier_body,
        assignee=verifier_assignee,
        created_by=created_by,
        parents=worker_ids,
        tenant=tenant,
        project_id=project_id,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["requesting-code-review"],
    )

    synthesizer_body = (
        "Synthesize the verified worker outputs into the final deliverable. "
        "Do not start until the verifier has passed the gate." + context_suffix
    )
    synthesizer = kb.create_task(
        conn,
        title=synthesizer_title,
        body=synthesizer_body,
        assignee=synthesizer_assignee,
        created_by=created_by,
        parents=[verifier],
        parent_gates={verifier: "metadata_gate_pass"},
        tenant=tenant,
        project_id=project_id,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["humanizer"],
    )

    created = SwarmCreated(root, worker_ids, verifier, synthesizer)
    kb.store_swarm_topology(
        conn,
        root,
        created.as_dict() | {"goal": goal, "tenant": tenant, "project_id": project_id},
        created_by=created_by,
    )
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
    payload = json.dumps(
        {"key": key, "value": value}, ensure_ascii=False, sort_keys=True
    )
    return kb.add_comment(
        conn, root_id, author=author, body=BLACKBOARD_PREFIX + payload
    )


def latest_blackboard(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    """Merge structured blackboard comments on a root card.

    Later comments replace earlier values for the same key. ``_authors`` records
    the author of the winning value for traceability. Reserved protocol keys are
    intentionally ignored here because comments are worker-controlled evidence,
    not authoritative swarm state.
    """

    merged: dict[str, Any] = {}
    authors: dict[str, str] = {}
    for comment in kb.list_comments(conn, root_id):
        body = comment.body or ""
        if not body.startswith(BLACKBOARD_PREFIX):
            continue
        try:
            payload = json.loads(body[len(BLACKBOARD_PREFIX) :])
        except json.JSONDecodeError:
            continue
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            continue
        if key in RESERVED_BLACKBOARD_KEYS:
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
    return SwarmWorkerSpec(
        profile=parts[0], title=parts[1], body=parts[1], skills=skills
    )
