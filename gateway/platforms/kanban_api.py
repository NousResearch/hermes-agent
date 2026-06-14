"""
Kanban HTTP endpoints for the OpenAI-compatible API server.

Exposes read/write access to the shared Hermes kanban board under
``/api/kanban/*`` so a headless control plane (the "master console") can
render and drive the board over the API server's existing bearer auth.

Design constraints (see docs/hermes-kanban-api.md):

- **Additive.** This module only adds handlers; it does not change the
  dispatcher, the CLI, or any existing API-server route. ``api_server``
  imports it lazily and registers the routes in ``connect()``.
- **Engine-routed writes.** Every create/assign/comment/status edit goes
  through the same ``hermes_cli.kanban_db`` functions the CLI uses, so
  claim-locks, idempotency, the failure counter, dependency gating, and
  the notify/event stream behave identically to a CLI-driven change. We
  never open our own sqlite connection — reads and writes run on the
  engine's ``connect_closing()`` connection inside its transaction
  helpers.
- **Shared board.** Handlers use ``kb.connect_closing()`` with no board
  argument, so they operate on whatever board the in-process dispatcher
  serves (the default ``<root>/kanban.db`` in the common single-board
  setup). A task created here is therefore picked up by the dispatcher
  exactly like a CLI-created one.

The handlers are plain ``async def f(adapter, request)`` functions rather
than adapter methods so the whole kanban surface lives in one module; the
adapter still owns auth (``_check_auth``), body parsing
(``_read_json_body``), and int parsing (``_parse_nonnegative_int``).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aiohttp import web

from hermes_cli import kanban_db as kb
from gateway.platforms.api_server import _openai_error

logger = logging.getLogger(__name__)

# Default and ceiling for the list endpoint's page size.
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 500
# Cap how many events the detail endpoint returns (most-recent slice).
_MAX_DETAIL_EVENTS = 50

# PATCH status targets we can route to a transition-aware engine helper.
# Everything else (todo/triage/running/review/done/scheduled) is driven by
# the dispatcher/worker lifecycle, not by an external editor, so we reject
# it rather than forcing an unsafe raw status flip.
_PATCH_STATUS_SUPPORTED = ("ready", "blocked", "archived")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _iso(epoch: Optional[int]) -> Optional[str]:
    """Convert an epoch-seconds column to an ISO-8601 UTC string (or None)."""
    if epoch is None:
        return None
    try:
        return (
            datetime.fromtimestamp(int(epoch), tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _updated_at(task: "kb.Task") -> Optional[str]:
    """Best-effort "last touched" timestamp.

    The schema has no ``updated_at`` column, so derive it from the latest
    available lifecycle timestamp.
    """
    latest = max(
        (t for t in (task.completed_at, task.started_at, task.created_at) if t),
        default=None,
    )
    return _iso(latest)


def _task_summary(conn, task: "kb.Task") -> Dict[str, Any]:
    """Compact board-card representation for the list endpoint."""
    current_run_status = None
    if task.current_run_id is not None:
        run = kb.get_run(conn, task.current_run_id)
        if run is not None:
            current_run_status = run.status
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "assignee": task.assignee,
        "tenant": task.tenant,
        "priority": task.priority,
        "created_at": _iso(task.created_at),
        "updated_at": _updated_at(task),
        "current_run_status": current_run_status,
    }


def _run_view(conn, task: "kb.Task") -> Optional[Dict[str, Any]]:
    """Dispatcher/run health block for the detail endpoint, or None."""
    latest = kb.latest_run(conn, task.id)
    if latest is None and not task.consecutive_failures and task.claim_lock is None:
        return None
    return {
        "status": latest.status if latest is not None else None,
        "consecutive_failures": task.consecutive_failures,
        "last_failure_error": task.last_failure_error,
        "claim_lock": task.claim_lock,
        "claim_expires": _iso(task.claim_expires),
        "last_heartbeat_at": _iso(task.last_heartbeat_at),
    }


def _task_detail(conn, task: "kb.Task") -> Dict[str, Any]:
    """Full task view: columns + comments + recent events + run health."""
    comments = [
        {
            "id": c.id,
            "author": c.author,
            "body": c.body,
            "created_at": _iso(c.created_at),
        }
        for c in kb.list_comments(conn, task.id)
    ]
    events = [
        {
            "kind": e.kind,
            "payload": e.payload,
            "run_id": e.run_id,
            "created_at": _iso(e.created_at),
        }
        for e in kb.list_events(conn, task.id)[-_MAX_DETAIL_EVENTS:]
    ]
    return {
        "id": task.id,
        "title": task.title,
        "body": task.body,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "created_by": task.created_by,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "branch_name": task.branch_name,
        "skills": list(task.skills) if task.skills else [],
        "max_retries": task.max_retries,
        "max_runtime_seconds": task.max_runtime_seconds,
        "model_override": task.model_override,
        "session_id": task.session_id,
        "workflow_template_id": task.workflow_template_id,
        "current_step_key": task.current_step_key,
        "result": task.result,
        "created_at": _iso(task.created_at),
        "started_at": _iso(task.started_at),
        "completed_at": _iso(task.completed_at),
        "updated_at": _updated_at(task),
        "comments": comments,
        "events": events,
        "run": _run_view(conn, task),
    }


def _err(message: str, *, status: int, code: Optional[str] = None,
         param: Optional[str] = None) -> "web.Response":
    return web.json_response(_openai_error(message, param=param, code=code), status=status)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def handle_list_tasks(adapter, request: "web.Request") -> "web.Response":
    """GET /api/kanban/tasks — filtered board listing."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    q = request.query
    tenant = q.get("tenant") or None
    assignee = q.get("assignee") or None
    status = q.get("status") or None
    if status is not None and status not in kb.VALID_STATUSES:
        return _err(
            f"status must be one of {sorted(kb.VALID_STATUSES)}",
            status=400, code="invalid_status", param="status",
        )
    limit = adapter._parse_nonnegative_int(q.get("limit"), default=_DEFAULT_LIST_LIMIT, maximum=_MAX_LIST_LIMIT)
    if limit == 0:
        limit = _DEFAULT_LIST_LIMIT
    offset = adapter._parse_nonnegative_int(q.get("offset"), default=0, maximum=1_000_000)

    try:
        with kb.connect_closing() as conn:
            # list_tasks has no offset arg; fetch through the requested
            # window and slice. Filters are AND-combined by the engine.
            rows = kb.list_tasks(
                conn,
                assignee=assignee,
                status=status,
                tenant=tenant,
                limit=offset + limit,
            )
            page = rows[offset:offset + limit]
            tasks = [_task_summary(conn, t) for t in page]
    except Exception:
        logger.exception("GET /api/kanban/tasks failed")
        return _err("Failed to list tasks", status=500, code="server_error")

    return web.json_response({"count": len(tasks), "tasks": tasks})


async def handle_get_task(adapter, request: "web.Request") -> "web.Response":
    """GET /api/kanban/tasks/{task_id} — full task detail."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    task_id = request.match_info["task_id"]
    try:
        with kb.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            if task is None:
                return _err(f"Task not found: {task_id}", status=404, code="task_not_found")
            detail = _task_detail(conn, task)
    except Exception:
        logger.exception("GET /api/kanban/tasks/%s failed", task_id)
        return _err("Failed to load task", status=500, code="server_error")
    return web.json_response({"task": detail})


async def handle_assignees(adapter, request: "web.Request") -> "web.Response":
    """GET /api/kanban/assignees — assignee roster with per-status counts."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    try:
        with kb.connect_closing() as conn:
            known = kb.known_assignees(conn)
    except Exception:
        logger.exception("GET /api/kanban/assignees failed")
        return _err("Failed to list assignees", status=500, code="server_error")

    out = [
        {
            "assignee": entry["name"],
            "total": sum(entry.get("counts", {}).values()),
            "by_status": entry.get("counts", {}),
            "on_disk": entry.get("on_disk", False),
        }
        for entry in known
    ]
    return web.json_response({"assignees": out})


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def handle_create_task(adapter, request: "web.Request") -> "web.Response":
    """POST /api/kanban/tasks — create a task the dispatcher will pick up."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err:
        return err

    title = body.get("title")
    if not isinstance(title, str) or not title.strip():
        return _err("'title' is required", status=400, code="missing_title", param="title")

    task_body = body.get("body")
    if task_body is not None and not isinstance(task_body, str):
        return _err("'body' must be a string", status=400, code="invalid_body", param="body")

    tenant = body.get("tenant")
    if tenant is not None and not isinstance(tenant, str):
        return _err("'tenant' must be a string", status=400, code="invalid_tenant", param="tenant")

    assignee = body.get("assignee")
    if assignee is not None and not isinstance(assignee, str):
        return _err("'assignee' must be a string", status=400, code="invalid_assignee", param="assignee")

    priority = body.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        return _err("'priority' must be an integer", status=400, code="invalid_priority", param="priority")

    skills = body.get("skills")
    if skills is not None:
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            return _err("'skills' must be an array of strings", status=400, code="invalid_skills", param="skills")

    idempotency_key = body.get("idempotency_key")
    if idempotency_key is not None and not isinstance(idempotency_key, str):
        return _err("'idempotency_key' must be a string", status=400, code="invalid_idempotency_key", param="idempotency_key")

    created_by = body.get("created_by")
    if created_by is not None and not isinstance(created_by, str):
        return _err("'created_by' must be a string", status=400, code="invalid_created_by", param="created_by")

    try:
        with kb.connect_closing() as conn:
            task_id = kb.create_task(
                conn,
                title=title,
                body=task_body,
                assignee=assignee,
                tenant=tenant,
                priority=priority,
                skills=skills,
                idempotency_key=idempotency_key or None,
                created_by=created_by or "api",
            )
            task = kb.get_task(conn, task_id)
            detail = _task_detail(conn, task) if task is not None else None
    except ValueError as exc:
        return _err(str(exc), status=400, code="invalid_request_error")
    except Exception:
        logger.exception("POST /api/kanban/tasks failed")
        return _err("Failed to create task", status=500, code="server_error")

    if detail is None:
        return _err("Task created but could not be reloaded", status=500, code="server_error")
    return web.json_response({"task": detail}, status=201)


async def handle_assign_task(adapter, request: "web.Request") -> "web.Response":
    """POST /api/kanban/tasks/{task_id}/assign — (re)assign to a profile."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    task_id = request.match_info["task_id"]
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    assignee = body.get("assignee")
    if not isinstance(assignee, str) or not assignee.strip():
        return _err("'assignee' is required", status=400, code="missing_assignee", param="assignee")

    try:
        with kb.connect_closing() as conn:
            try:
                ok = kb.assign_task(conn, task_id, assignee)
            except RuntimeError as exc:
                # Task is currently claimed/running — cannot reassign.
                return _err(str(exc), status=409, code="task_running")
            if not ok:
                return _err(f"Task not found: {task_id}", status=404, code="task_not_found")
            task = kb.get_task(conn, task_id)
            detail = _task_detail(conn, task) if task is not None else None
    except Exception:
        logger.exception("POST /api/kanban/tasks/%s/assign failed", task_id)
        return _err("Failed to assign task", status=500, code="server_error")

    return web.json_response({"task": detail})


async def handle_comment_task(adapter, request: "web.Request") -> "web.Response":
    """POST /api/kanban/tasks/{task_id}/comment — append a comment.

    Mirrors ``hermes kanban comment``: ``add_comment`` records a
    ``commented`` event, which the gateway's kanban-notifier watcher tails
    to wake subscribed requesters. No extra notify wiring is needed here.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    task_id = request.match_info["task_id"]
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    author = body.get("author")
    comment_body = body.get("body")
    if not isinstance(author, str) or not author.strip():
        return _err("'author' is required", status=400, code="missing_author", param="author")
    if not isinstance(comment_body, str) or not comment_body.strip():
        return _err("'body' is required", status=400, code="missing_body", param="body")

    try:
        with kb.connect_closing() as conn:
            if kb.get_task(conn, task_id) is None:
                return _err(f"Task not found: {task_id}", status=404, code="task_not_found")
            comment_id = kb.add_comment(conn, task_id, author, comment_body)
            created = {
                "id": comment_id,
                "task_id": task_id,
                "author": author.strip(),
                "body": comment_body.strip(),
                "created_at": _iso(_latest_comment_time(conn, comment_id)),
            }
    except ValueError as exc:
        return _err(str(exc), status=400, code="invalid_request_error")
    except Exception:
        logger.exception("POST /api/kanban/tasks/%s/comment failed", task_id)
        return _err("Failed to add comment", status=500, code="server_error")

    return web.json_response({"comment": created}, status=201)


def _latest_comment_time(conn, comment_id: int) -> Optional[int]:
    """Read back the created_at for a just-inserted comment (read-only)."""
    row = conn.execute(
        "SELECT created_at FROM task_comments WHERE id = ?", (comment_id,)
    ).fetchone()
    return row["created_at"] if row else None


async def handle_patch_task(adapter, request: "web.Request") -> "web.Response":
    """PATCH /api/kanban/tasks/{task_id} — edit title/body/priority/status.

    Metadata (title/body/priority) goes through ``update_task_metadata``.
    A ``status`` change is routed to the matching transition-aware engine
    helper (promote/block/archive); unsupported targets are rejected so we
    never bypass the dispatcher's status invariants with a raw flip.
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    task_id = request.match_info["task_id"]
    body, err = await adapter._read_json_body(request)
    if err:
        return err

    allowed = {"status", "priority", "title", "body"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        return _err(
            f"Unsupported fields: {', '.join(unknown)}",
            status=400, code="unsupported_field",
        )
    if not body:
        return _err("No fields to update", status=400, code="empty_patch")

    title = body.get("title")
    if "title" in body and (not isinstance(title, str) or not title.strip()):
        return _err("'title' must be a non-empty string", status=400, code="invalid_title", param="title")
    new_body = body.get("body")
    if "body" in body and new_body is not None and not isinstance(new_body, str):
        return _err("'body' must be a string", status=400, code="invalid_body", param="body")
    priority = body.get("priority")
    if "priority" in body and (isinstance(priority, bool) or not isinstance(priority, int)):
        return _err("'priority' must be an integer", status=400, code="invalid_priority", param="priority")

    status = body.get("status")
    if "status" in body:
        if not isinstance(status, str) or status not in kb.VALID_STATUSES:
            return _err(
                f"status must be one of {sorted(kb.VALID_STATUSES)}",
                status=400, code="invalid_status", param="status",
            )
        if status not in _PATCH_STATUS_SUPPORTED:
            return _err(
                f"status {status!r} cannot be set via the API; supported "
                f"targets are {list(_PATCH_STATUS_SUPPORTED)} (other statuses "
                "are driven by the dispatcher/worker lifecycle)",
                status=400, code="unsupported_status_transition", param="status",
            )

    try:
        with kb.connect_closing() as conn:
            if kb.get_task(conn, task_id) is None:
                return _err(f"Task not found: {task_id}", status=404, code="task_not_found")

            # Plain metadata first.
            meta_kwargs: Dict[str, Any] = {}
            if "title" in body:
                meta_kwargs["title"] = title
            if "body" in body:
                meta_kwargs["body"] = new_body
            if "priority" in body:
                meta_kwargs["priority"] = priority
            if meta_kwargs:
                kb.update_task_metadata(conn, task_id, **meta_kwargs)

            # Status transition via the matching engine helper.
            if "status" in body:
                transition_err = _apply_status(conn, task_id, status)
                if transition_err is not None:
                    return transition_err

            task = kb.get_task(conn, task_id)
            detail = _task_detail(conn, task) if task is not None else None
    except ValueError as exc:
        return _err(str(exc), status=400, code="invalid_request_error")
    except Exception:
        logger.exception("PATCH /api/kanban/tasks/%s failed", task_id)
        return _err("Failed to update task", status=500, code="server_error")

    return web.json_response({"task": detail})


def _apply_status(conn, task_id: str, status: str) -> Optional["web.Response"]:
    """Route a PATCH status target to its engine helper.

    Returns an error ``web.Response`` if the transition is rejected by the
    engine (e.g. invalid from the current state), else None on success.
    """
    if status == "archived":
        ok = kb.archive_task(conn, task_id)
    elif status == "blocked":
        ok = kb.block_task(conn, task_id, reason="blocked via API")
    elif status == "ready":
        ok, reason = kb.promote_task(conn, task_id, actor="api")
        if not ok:
            return _err(
                reason or f"cannot move {task_id} to ready",
                status=409, code="invalid_status_transition", param="status",
            )
        return None
    else:  # pragma: no cover - guarded by caller
        return _err("unsupported status", status=400, code="unsupported_status_transition", param="status")

    if not ok:
        return _err(
            f"cannot move {task_id} to {status} from its current state",
            status=409, code="invalid_status_transition", param="status",
        )
    return None


# ---------------------------------------------------------------------------
# Dispatcher visibility
# ---------------------------------------------------------------------------


def _dispatch_config() -> Dict[str, Optional[int]]:
    """Read the dispatch knobs the way the CLI/gateway dispatcher does.

    Mirrors ``hermes_cli.kanban._cmd_dispatch``: ``max_in_progress_per_profile``
    is the per-profile concurrency cap and ``failure_limit`` is the
    circuit-breaker trip count. Both fall back the same way the dispatcher
    falls back, so this view never disagrees with the live dispatcher.
    """
    max_per_profile: Optional[int] = None
    failure_limit = kb.DEFAULT_FAILURE_LIMIT
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}

        def _positive_int(value):
            if value is None:
                return None
            try:
                ival = int(value)
            except (TypeError, ValueError):
                return None
            return ival if ival >= 1 else None

        max_per_profile = _positive_int(kcfg.get("max_in_progress_per_profile"))
        cfg_limit = _positive_int(kcfg.get("failure_limit"))
        if cfg_limit is not None:
            failure_limit = cfg_limit
    except Exception:
        logger.debug("dispatch config load failed; using defaults", exc_info=True)
    return {"max_in_progress_per_profile": max_per_profile, "failure_limit": failure_limit}


def _effective_failure_limit(task: "kb.Task", default_limit: int) -> int:
    """Resolve a task's circuit-breaker trip count.

    Same precedence as ``_record_task_failure`` / ``recompute_ready``:
    per-task ``max_retries`` first, then the dispatcher-level
    ``kanban.failure_limit``, then ``DEFAULT_FAILURE_LIMIT``.
    """
    if task.max_retries is not None:
        return int(task.max_retries)
    return int(default_limit)


async def handle_dispatch_state(adapter, request: "web.Request") -> "web.Response":
    """GET /api/kanban/dispatch/state — dispatcher concurrency + benched tasks.

    Derived read-only from the engine; it does not touch the dispatcher.

    - ``per_profile``: one entry per assignee that currently has a task in
      ``running`` status — ``in_progress`` is that count, ``max_in_progress``
      is ``kanban.max_in_progress_per_profile`` (null when unconfigured).
    - ``blocked``: tasks the circuit breaker has benched, i.e. ``blocked``
      tasks whose ``consecutive_failures`` reached their effective failure
      limit (per-task ``max_retries`` or ``kanban.failure_limit``).
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    cfg = _dispatch_config()
    max_per_profile = cfg["max_in_progress_per_profile"]
    default_limit = cfg["failure_limit"]

    try:
        with kb.connect_closing() as conn:
            running = kb.list_tasks(conn, status="running")
            blocked_tasks = kb.list_tasks(conn, status="blocked")
    except Exception:
        logger.exception("GET /api/kanban/dispatch/state failed")
        return _err("Failed to load dispatch state", status=500, code="server_error")

    in_progress: Dict[str, int] = {}
    for t in running:
        key = t.assignee or "(unassigned)"
        in_progress[key] = in_progress.get(key, 0) + 1
    per_profile = [
        {
            "assignee": assignee,
            "in_progress": count,
            "max_in_progress": max_per_profile,
        }
        for assignee, count in sorted(in_progress.items())
    ]

    blocked = []
    for t in blocked_tasks:
        limit = _effective_failure_limit(t, default_limit)
        if t.consecutive_failures >= limit:
            blocked.append(
                {
                    "task_id": t.id,
                    "assignee": t.assignee,
                    "title": t.title,
                    "consecutive_failures": t.consecutive_failures,
                    "failure_limit": limit,
                    "last_failure_error": t.last_failure_error,
                }
            )

    return web.json_response({"per_profile": per_profile, "blocked": blocked})
