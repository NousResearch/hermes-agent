"""Task creation phases and race-safe idempotency; public entry points live in kanban_db."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Iterable, Optional


def create_task_idempotent(
    conn: sqlite3.Connection, *, title: str, body: Optional[str] = None,
    assignee: Optional[str] = None, created_by: Optional[str] = None,
    workspace_kind: Optional[str] = None, workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None, tenant: Optional[str] = None, priority: int = 0,
    parents: Iterable[str] = (), triage: bool = False, idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None, skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None, model_override: Optional[str] = None,
    provider_override: Optional[str] = None, reasoning_effort: Optional[str] = None,
    goal_mode: bool = False, goal_max_turns: Optional[int] = None, initial_status: str = "running",
    session_id: Optional[str] = None, board: Optional[str] = None, project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    creator_task_id: Optional[str] = None,
    completion_contract: Optional[str] = None,
) -> tuple[str, bool]:
    """Create a task (optionally under ``parents``); returns ``(task_id, created)``.

    Status: ``ready`` unless a parent is not ``done`` (``todo``); ``triage=True``
    forces ``triage``; ``initial_status="blocked"`` parks it for human ops.
    ``idempotency_key``: an existing non-archived task with the key is returned
    with ``created=False`` instead of a duplicate — concurrency-safe, since a
    creator losing the partial-UNIQUE-index race also resolves to the winner's
    id. Callers that surface idempotency to their own clients (the REST
    adapter's 201-vs-200 split) need ``created``; everyone else uses
    :func:`create_task`. ``max_runtime_seconds``: cap before the dispatcher
    SIGTERMs and re-queues. ``model_override``/``provider_override`` pin the
    worker model (provider requires model); ``reasoning_effort`` is independent.
    ``creator_task_id``: inherit durable session/subscriptions independently of
    dependency edges; an explicit ``session_id`` still wins.
    ``project_source_task_id``: cross-profile fallback when ``project_id`` is not
    in the active profile's projects.db — see ``_resolve_project_link``.
    ``workspace_kind=None`` (omitted) inherits a project-scoped board's project;
    an explicit ``"scratch"`` or ``project_id=""`` is a request for no project.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_graph import initial_task_state, inherit_creator_origin
    from hermes_cli.kanban_pr_acceptance import validate_contract

    completion_contract = validate_contract(completion_contract)
    model_override, provider_override = _kb._validate_model_override(model_override, provider_override)
    reasoning_effort = _kb.normalize_reasoning_effort(reasoning_effort)
    assignee = _kb._canonical_assignee(assignee)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in _kb.VALID_INITIAL_STATUSES:
        raise ValueError(f"initial_status must be one of {sorted(_kb.VALID_INITIAL_STATUSES)}")
    project_id, project_obj, project_repo, workspace_kind, branch_name = _resolve_task_workspace(
        conn, project_id=project_id, project_source_task_id=project_source_task_id,
        workspace_kind=workspace_kind, workspace_path=workspace_path, branch_name=branch_name, board=board,
    )
    parents = tuple(p for p in parents if p)
    skills_list = _kb._normalize_task_skills(skills)

    idempotency_key = idempotency_key or None  # "" never deduped; keep it out of the unique index too
    # Idempotency fast path BEFORE the write txn (no lock held). The check alone is
    # racy, so the partial UNIQUE index on idempotency_key backstops it: the racing
    # loser's INSERT fails and is resolved to the winner's id below.
    if idempotency_key:
        row = _live_task_for_idempotency_key(conn, idempotency_key)
        if row:
            return row["id"], False

    now = int(time.time())

    # Only persistent kinds inherit the board ``default_workdir``: a scratch
    # task inheriting it would point cleanup at the user's source tree.
    if workspace_path is None and project_repo is None and workspace_kind in {"dir", "worktree"}:
        board_default = _kb._board_meta_for(board).get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _kb._new_task_id()
        try:
            # allow_nested: graph builders compose create_task under one outer
            # commit so the dispatcher never sees a half-built graph.
            with _kb.write_txn(conn, allow_nested=True):
                task_status, tenant = initial_task_state(conn, parents, initial_status, triage, tenant)
                # Project worktree: fresh dir under the repo + deterministic
                # branch, instead of the random ``wt/<id>`` worker fallback.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(project_repo, ".worktrees", task_id)
                    if not branch_name:
                        branch_name = _kb._project_branch_name(project_obj, task_id, title)

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort,
                        goal_mode, goal_max_turns, session_id, completion_contract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, title.strip(), body, assignee, task_status, priority,
                        created_by, now, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        _kb._opt_int(max_runtime_seconds),
                        json.dumps(skills_list) if skills_list is not None else None,
                        _kb._opt_int(max_retries), model_override, provider_override, reasoning_effort,
                        1 if goal_mode else 0, _kb._opt_int(goal_max_turns), session_id, completion_contract,
                    ),
                )
                for pid in parents:
                    _kb._link(conn, pid, task_id)
                _kb._append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "creator_task_id": creator_task_id,
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "project_id": project_id,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                    },
                )
                if task_status == "blocked":
                    _kb._append_event(
                        conn,
                        task_id,
                        "blocked",
                        {"reason": "initial_status", "status": "blocked", "actor": created_by or "user"},
                    )
                if task_status == "todo":
                    # Parked behind an open parent: record why, exactly as
                    # link_tasks does, so the board never shows an unexplained todo.
                    gating = [p for p in parents if _kb._task_status(conn, p) not in ("done", "archived")]
                    if gating:
                        _kb._append_event(
                            conn,
                            task_id,
                            "dependency_wait",
                            {"reason": "parent_not_done", "parent": gating[0]},
                        )
                # ACK-edge: the originating channel hears a child BLOCK, not just the fan-in.
                inherit_creator_origin(conn, task_id, creator_task_id, created_at=now)
                _kb._inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id, True
        except sqlite3.IntegrityError:
            # Losing the idempotency-key race (a concurrent creator with the same key
            # committed first, so the partial UNIQUE index rejected our row) must resolve
            # to the winner's row — retrying with a fresh id would fail on the same key.
            if idempotency_key:
                row = _live_task_for_idempotency_key(conn, idempotency_key)
                if row:
                    return row["id"], False
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


def _resolve_task_workspace(conn, *, project_id, project_source_task_id, workspace_kind,
                            workspace_path, branch_name, board):
    from hermes_cli import kanban_db as _kb

    # A project-scoped board anchors every new task to its project's repo
    # (deterministic worktree + branch) without each surface repeating it.
    # An explicit ``scratch`` (or ``project_id=""``) is a request for no project:
    # it must not be upgraded to a worktree in the board's repo (#106342).
    if project_id is None and workspace_kind != "scratch":
        try:
            project_id = (_kb._board_meta_for(board).get("project_id") or "").strip() or None
        except (ValueError, AttributeError):
            _kb._log.debug("kanban board project metadata unavailable", exc_info=True)
    if workspace_kind is None:
        workspace_kind = "scratch"
    if workspace_kind not in _kb.VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(_kb.VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    project_id, project_obj, project_repo, workspace_kind = _kb._resolve_project_link(
        conn, project_id, project_source_task_id, workspace_kind, workspace_path
    )
    return project_id, project_obj, project_repo, workspace_kind, branch_name


def _live_task_for_idempotency_key(conn: sqlite3.Connection, idempotency_key: str) -> Optional[sqlite3.Row]:
    """Live (non-archived) task row holding ``idempotency_key``. The tiebreak matches the survivor
    choice in ``kanban_db_connect._migrate_unique_idempotency_index`` so every lookup path picks the
    same row when legacy duplicates exist."""
    return conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC, id DESC LIMIT 1", (idempotency_key,),
    ).fetchone()

