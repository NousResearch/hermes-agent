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
from pathlib import Path
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


def _effective_workspace_kind(workspace_kind: str, project_id: Optional[str]) -> str:
    """Mirror ``kb.create_task``'s project-link promotion, up front.

    ``create_task`` upgrades a ``scratch`` task to ``worktree`` when it carries
    a resolvable ``project_id`` with a primary repo. That happens *after*
    :func:`create_swarm` has run its single-worker guard and pinned the shared
    checkout, so a ``--project`` swarm left on the default ``--workspace
    scratch`` would skip both and still hand every card its own
    ``<repo>/.worktrees/<task-id>``. Decide the kind the cards will actually
    get before either of those steps runs.

    An unresolvable project is left alone: ``create_task`` drops the dangling
    link and keeps the task on ``scratch``, so the swarm must too.
    """

    if workspace_kind != "scratch" or not project_id:
        return workspace_kind
    from hermes_cli import projects_db as _pdb

    try:
        with _pdb.connect_closing() as pconn:
            project = _pdb.get_project(pconn, str(project_id).strip())
    except Exception:
        return workspace_kind
    if project is not None and project.primary_path:
        return "worktree"
    return workspace_kind


def _board_default_repo() -> Optional[Path]:
    """Repo root that a bare ``worktree`` task would be anchored on."""

    try:
        default_workdir = (
            kb.read_board_metadata(kb.get_current_board()).get("default_workdir") or ""
        ).strip()
    except Exception:
        return None
    if not default_workdir:
        return None
    anchor = Path(default_workdir).expanduser()
    if not anchor.is_absolute():
        return None
    return kb._git_toplevel(anchor)


def _swarm_worktree_target(
    root_id: str,
    workspace_path: Optional[str],
    branch_name: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Pick one concrete checkout + branch for a non-project worktree swarm.

    ``kb.create_task`` only derives a path/branch for *project-linked*
    worktrees. Without a project the cards keep exactly what they were handed,
    and ``kb._resolve_worktree_workspace`` then improvises per card at dispatch:
    a distinct ``wt/<task-id>`` branch whenever no branch was set, and a
    distinct ``<repo>/.worktrees/<task-id>`` dir whenever the path is unset or
    points at a repo root. Either one is enough to open the verifier on a
    checkout the worker never touched, so anchor the whole swarm up front.

    A path that points *inside* a repo is already shared by every card, so it
    is kept verbatim and only the branch is pinned. When no repo can be found
    the path is left alone and dispatch keeps its current behaviour, including
    its existing "no anchor configured" error.
    """

    branch = branch_name or f"wt/{root_id}"
    if workspace_path is None:
        repo_root = _board_default_repo()
    else:
        candidate = Path(workspace_path).expanduser()
        repo_root = kb._git_toplevel(candidate)
        if repo_root is None or candidate.resolve(strict=False) != repo_root:
            return workspace_path, branch
    if repo_root is None:
        return workspace_path, branch
    return str(repo_root / ".worktrees" / root_id), branch


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
    project_id: Optional[str] = None,
    branch_name: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
) -> SwarmCreated:
    """Create a durable Kanban swarm graph.

    The returned graph is immediately dispatchable: the planning root is marked
    ``done`` with topology metadata, parallel workers are ``ready``, the verifier
    waits for every worker, and the synthesizer waits for the verifier.

    For ``workspace_kind="worktree"`` swarms (code pipelines, e.g. a single
    ``programmer`` worker handing off to a code-review ``verifier``), every
    downstream card must land in the *same* git worktree the worker committed
    to, or the verifier opens an empty checkout and reviews nothing.
    ``kb.create_task`` normally hands each project-linked task its own fresh
    ``<repo>/.worktrees/<task-id>`` dir, so left alone every card in the swarm
    would get a different worktree. To avoid that, one path/branch pair is
    settled before the downstream cards exist and reused verbatim for every
    remaining worker, the verifier, and the synthesizer:

    * ``project_id`` swarms take the first worker's *resolved* path/branch,
      which ``kb.create_task`` derived from the project's primary repo.
    * every other worktree swarm (explicit ``workspace_path``, or bare
      ``worktree`` anchored on the board's ``default_workdir``) is pinned to
      ``<repo>/.worktrees/<root-id>`` on ``wt/<root-id>`` unless the caller
      named a path inside a repo and/or a ``branch_name`` of its own.

    Note that a ``project_id`` promotes ``scratch`` to ``worktree`` inside
    ``kb.create_task``; :func:`_effective_workspace_kind` applies that here
    first so such swarms are guarded and pinned like any other worktree swarm.
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
    effective_ws_kind = _effective_workspace_kind(workspace_kind, project_id)
    if effective_ws_kind == "worktree" and len(worker_specs) > 1:
        raise ValueError(
            "swarm with workspace_kind='worktree' supports exactly one "
            "--worker: multiple parallel workers writing to the same git "
            "worktree/branch will clobber each other's changes. Use a "
            "single worker for code pipelines, or drop --workspace/--project "
            "to keep parallel workers on independent scratch dirs."
        )

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
    # One checkout for the whole swarm, so every later card (siblings, verifier,
    # synthesizer) lands where the worker committed instead of each getting its
    # own fresh `.worktrees/<task-id>` dir on its own `wt/<task-id>` branch.
    # Project swarms are settled after the first worker is created (below),
    # because only then has `create_task` derived the project path/branch.
    shared_ws_kind = effective_ws_kind
    shared_ws_path = workspace_path
    shared_branch = branch_name
    # An explicit ``workspace_path`` wins over the project link (``create_task``
    # only derives a path when it is given none), so only a project swarm that
    # named no path can be settled from the created row.
    project_derived = project_id is not None and workspace_path is None
    if shared_ws_kind == "worktree" and not project_derived:
        shared_ws_path, shared_branch = _swarm_worktree_target(
            root, shared_ws_path, shared_branch
        )

    worker_ids: list[str] = []
    for i, spec in enumerate(worker_specs):
        worker_id = kb.create_task(
            conn,
            title=spec.title,
            body=(spec.body or "") + context_suffix,
            assignee=spec.profile,
            created_by=created_by,
            parents=[root],
            tenant=tenant,
            priority=spec.priority or priority,
            workspace_kind=shared_ws_kind,
            workspace_path=shared_ws_path,
            branch_name=shared_branch,
            project_id=project_id if shared_ws_path is None else None,
            skills=spec.skills or None,
            max_runtime_seconds=spec.max_runtime_seconds,
        )
        worker_ids.append(worker_id)
        if i == 0 and shared_ws_kind == "worktree" and project_derived:
            # project_id-only worktree: kb.create_task just auto-derived a
            # fresh <repo>/.worktrees/<worker_id> dir + deterministic branch.
            # Lock that exact path/branch in for the verifier + synthesizer.
            created_worker = kb.get_task(conn, worker_id)
            shared_ws_kind = created_worker.workspace_kind or shared_ws_kind
            shared_ws_path = created_worker.workspace_path
            shared_branch = created_worker.branch_name

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
        workspace_kind=shared_ws_kind,
        workspace_path=shared_ws_path,
        branch_name=shared_branch,
        project_id=project_id if shared_ws_path is None else None,
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
        workspace_kind=shared_ws_kind,
        workspace_path=shared_ws_path,
        branch_name=shared_branch,
        project_id=project_id if shared_ws_path is None else None,
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
