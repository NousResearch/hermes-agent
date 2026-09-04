"""Kanban dashboard plugin — backend API routes, mounted at /api/plugins/kanban/.

Every handler is a thin wrapper around ``hermes_cli.kanban_db`` (the same code paths the CLI
and gateway ``/kanban`` command use, so the surfaces cannot drift). The ``/events`` WebSocket
tails the append-only ``task_events`` table on a short poll (WAL reads run alongside the
dispatcher's write txns); it carries its credential in the query string (browsers can't set
``Authorization`` on an upgrade) and is gated by the dashboard's canonical WS auth check.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from fastapi import (
    APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect, status as http_status)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_diagnostics as kd
from hermes_cli.kanban_db import KANBAN_ATTACHMENT_MAX_BYTES, _collision_free_path, _safe_attachment_name

log = logging.getLogger(__name__)

router = APIRouter()

_BOARD_Q = Query(None, description="Kanban board slug (omit for current)")


# --- Connection / board helpers ---------------------------------------------

def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Authorize a WS upgrade via the dashboard's canonical gate (``web_server_chat._ws_auth_ok``:
    ``?token=`` / ``?ticket=`` / ``?internal=``) so this endpoint can never drift from core
    auth; accepts when the dashboard isn't importable (bare-FastAPI test harness)."""
    try:
        from hermes_cli import web_server_chat as _ws
    except Exception:
        return True
    return bool(_ws._ws_auth_ok(ws))


def _normalize_slug_or_400(slug: str) -> Optional[str]:
    with _value_error_400():
        return kanban_db._normalize_board_slug(slug)


def _resolve_board(board: Optional[str]) -> Optional[str]:
    """Validate/normalise a board slug query param (400 malformed, 404 unknown);
    ``None`` when omitted so ``kb.connect()`` falls through to the active board."""
    if board is None or board == "":
        return None
    normed = _normalize_slug_or_400(board)
    if normed and normed != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {normed!r} does not exist")
    return normed


def _existing_board_slug(slug: str) -> str:
    """Normalise a path slug and require the board to exist (400 / 404)."""
    normed = _normalize_slug_or_400(slug)
    if not normed or not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {slug!r} does not exist")
    return normed


def _conn(board: Optional[str] = None):
    """Connect to the already-normalised ``board`` (``None`` = active). ``init_db`` is
    idempotent; running it here lets a fresh install self-heal if POST /tasks arrives first."""
    try:
        kanban_db.init_db(board=board)
    except Exception as exc:
        log.warning("kanban init_db failed: %s", exc)
    return kbc.connect(board=board)


@contextmanager
def _board_conn(board: Optional[str]) -> Iterator[tuple[Optional[str], sqlite3.Connection]]:
    """Resolve the ``board`` query param, open a connection, close it on exit."""
    board = _resolve_board(board)
    with closing(_conn(board=board)) as conn:
        yield board, conn


def _with_board_pinned(board: Optional[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn`` with the board pinned context-locally, not via the process-global
    ``HERMES_KANBAN_BOARD`` env var (concurrent requests for different boards would cross-write)."""
    with kanban_db.scoped_current_board(_resolve_board(board) or kanban_db.DEFAULT_BOARD):
        return fn()


def _require(getter: Callable, conn: sqlite3.Connection, ident, label: str):
    obj = getter(conn, ident)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{label} {ident} not found")
    return obj


def _run_aux(board: Optional[str], module: str, fn: str, task_id: str, author: Optional[str]) -> Any:
    """Run a slow auxiliary-LLM task helper (``hermes_cli.<module>.<fn>``) with the board pinned;
    the module is imported lazily so a missing aux client can't break plugin load."""
    def _run():
        return getattr(importlib.import_module(f"hermes_cli.{module}"), fn)(task_id, author=(author or None))
    return _with_board_pinned(board, _run)


def _require_task(conn: sqlite3.Connection, task_id: str) -> kanban_db.Task:
    return _require(kanban_db.get_task, conn, task_id, "task")


def _require_run(conn: sqlite3.Connection, run_id: int) -> kanban_db.Run:
    return _require(kanban_db.get_run, conn, run_id, "run")


def _require_ok(ok: bool) -> None:
    """404 when a kanban_db mutator reports the task vanished mid-request."""
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=detail)


@contextmanager
def _map_errors(status: int, *types: type[BaseException]) -> Iterator[None]:
    """Map the given exception types to ``HTTPException(status, str(exc))``."""
    try:
        yield
    except types as e:
        raise HTTPException(status_code=status, detail=str(e))


_value_error_400 = partial(_map_errors, 400, ValueError)  # domain-layer validation refusals


@contextmanager
def _errors_to_500(prefix: str) -> Iterator[None]:
    """Map any unexpected exception to ``500 "<prefix>: <exc>"``; HTTPExceptions pass through."""
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{prefix}: {exc}")


# --- Serialization helpers --------------------------------------------------

# Dashboard columns, left-to-right ("archived" is a filter toggle, not a column). Keep in
# sync with kanban_db.VALID_STATUSES — a status missing here gets mis-bucketed into ``todo``.
BOARD_COLUMNS: list[str] = ["triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done"]

_CARD_SUMMARY_PREVIEW_CHARS = 200


def _task_dict(task: kanban_db.Task, *, latest_summary: Optional[str] = None) -> dict[str, Any]:
    d = asdict(task)
    # Derived age metrics so the UI can colour stale cards without client deltas.
    try:
        d["age"] = kanban_db.task_age(task)
    except Exception:
        d["age"] = {"created_age_seconds": None, "started_age_seconds": None, "time_to_complete_seconds": None}
    # Latest non-null run summary (workers hand off via ``task_runs.summary``, not ``tasks.result``).
    d["latest_summary"] = latest_summary
    return d


def _attachment_dict(a: kanban_db.Attachment) -> dict[str, Any]:
    """``stored_path`` is the absolute on-disk path workers read; UI downloads by ``id``."""
    return {
        "id": a.id, "task_id": a.task_id, "filename": a.filename, "content_type": a.content_type,
        "size": a.size, "uploaded_by": a.uploaded_by, "stored_path": a.stored_path, "created_at": a.created_at}


def _placeholders(ids: list) -> str:
    return ",".join(["?"] * len(ids))


def _compute_task_diagnostics(conn: sqlite3.Connection, task_ids: Optional[list[str]] = None) -> dict[str, list[dict]]:
    """``{task_id: [diagnostic_dict, ...]}`` (tasks with none omitted) via three aggregate
    queries (tasks, events, runs) — slurps the board; paginate if profiling shows a hotspot."""
    from hermes_cli.config import load_config

    if task_ids is not None and not task_ids:
        return {}
    diag_config = kd.config_from_runtime_config(load_config())
    if task_ids is not None:
        rows = conn.execute(f"SELECT * FROM tasks WHERE id IN ({_placeholders(task_ids)})", tuple(task_ids)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM tasks WHERE status != 'archived'").fetchall()
    if not rows:
        return {}
    row_ids = [r["id"] for r in rows]

    def _rows_by_task(table: str) -> dict[str, list]:
        by_task: dict[str, list] = {tid: [] for tid in row_ids}
        for row in conn.execute(
            f"SELECT * FROM {table} WHERE task_id IN ({_placeholders(row_ids)}) ORDER BY id", tuple(row_ids)):
            by_task.setdefault(row["task_id"], []).append(row)
        return by_task

    events_by_task = _rows_by_task("task_events")
    runs_by_task = _rows_by_task("task_runs")
    graph_by_task = kanban_db.task_graph_contexts(conn, row_ids)
    out: dict[str, list[dict]] = {}
    for r in rows:
        tid = r["id"]
        diags = kd.compute_task_diagnostics(
            r, events_by_task[tid], runs_by_task[tid], config=diag_config, graph=graph_by_task.get(tid))
        if diags:
            out[tid] = [d.to_dict() for d in diags]
    return out


def _warnings_summary_from_diagnostics(diagnostics: list[dict]) -> Optional[dict]:
    """Compact card badge summary ``{count, kinds, latest_at, highest_severity}``; None when empty."""
    if not diagnostics:
        return None
    kinds: dict[str, int] = {}
    count = latest = 0
    highest_idx, highest_sev = -1, None
    for d in diagnostics:
        n = d.get("count", 1)
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + n
        count += n
        latest = max(latest, d.get("last_seen_at") or 0)
        sev = d.get("severity")
        if sev in kd.SEVERITY_ORDER and kd.SEVERITY_ORDER.index(sev) > highest_idx:
            highest_idx, highest_sev = kd.SEVERITY_ORDER.index(sev), sev
    return {"count": count, "kinds": kinds, "latest_at": latest, "highest_severity": highest_sev}


def _attach_diagnostics(task_d: dict, diags: Optional[list[dict]]) -> None:
    """Full list in the payload (drawer renders without a second round-trip); card badge gets the summary."""
    if diags:
        task_d["diagnostics"] = diags
        task_d["warnings"] = _warnings_summary_from_diagnostics(diags)


def _links_for(conn: sqlite3.Connection, task_id: str) -> dict[str, list[str]]:
    """Return {'parents': [...], 'children': [...]} for a task."""
    def _ids(col: str, other: str) -> list[str]:
        return [r[col] for r in conn.execute(f"SELECT {col} FROM task_links WHERE {other} = ? ORDER BY {col}", (task_id,))]
    return {"parents": _ids("parent_id", "child_id"), "children": _ids("child_id", "parent_id")}


# --- GET /board -------------------------------------------------------------

@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None, description="Filter to a single tenant"),
    include_archived: bool = Query(False),
    board: Optional[str] = _BOARD_Q,
    workflow_template_id: Optional[str] = Query(None, description="Restrict to tasks using this workflow template id"),
    current_step_key: Optional[str] = Query(None, description="Restrict to tasks at this workflow step key")):
    """Full board grouped by status column; omitting ``board`` uses the active board
    (``HERMES_KANBAN_BOARD`` env → on-disk ``current`` pointer → ``default``)."""
    with _board_conn(board) as (board, conn):
        tasks = kanban_db.list_tasks(
            conn, tenant=tenant, include_archived=include_archived,
            workflow_template_id=workflow_template_id, current_step_key=current_step_key)
        # Link / comment / progress rollups are each one aggregate query rather than N per-task lookups.
        link_counts: dict[str, dict[str, int]] = {}
        for row in conn.execute("SELECT parent_id, child_id FROM task_links").fetchall():
            link_counts.setdefault(row["parent_id"], {"parents": 0, "children": 0})["children"] += 1
            link_counts.setdefault(row["child_id"], {"parents": 0, "children": 0})["parents"] += 1
        comment_counts: dict[str, int] = {
            r["task_id"]: r["n"] for r in conn.execute("SELECT task_id, COUNT(*) AS n FROM task_comments GROUP BY task_id")}
        progress: dict[str, dict[str, int]] = {}  # per parent: children done / total, rendered as "N/M"
        for row in conn.execute(
            "SELECT l.parent_id AS pid, t.status AS cstatus FROM task_links l JOIN tasks t ON t.id = l.child_id").fetchall():
            p = progress.setdefault(row["pid"], {"done": 0, "total": 0})
            p["total"] += 1
            p["done"] += row["cstatus"] == "done"
        diagnostics_per_task = _compute_task_diagnostics(conn, task_ids=None)
        latest_event_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM task_events").fetchone()["m"]
        columns: dict[str, list[dict]] = {c: [] for c in BOARD_COLUMNS}
        if include_archived:
            columns["archived"] = []
        # One window-function query for latest summaries (avoids N+1); cards get a
        # truncated preview, the full text comes from /tasks/:id.
        summary_map = kanban_db.latest_summaries(conn, [t.id for t in tasks])
        for t in tasks:
            full = summary_map.get(t.id)
            d = _task_dict(t, latest_summary=(full[:_CARD_SUMMARY_PREVIEW_CHARS] if full else None))
            d["link_counts"] = link_counts.get(t.id, {"parents": 0, "children": 0})
            d["comment_count"] = comment_counts.get(t.id, 0)
            d["progress"] = progress.get(t.id)  # None when the task has no children
            _attach_diagnostics(d, diagnostics_per_task.get(t.id))
            columns[t.status if t.status in columns else "todo"].append(d)

        # Per-column ordering (priority DESC, created_at ASC) comes from list_tasks.
        tenants = [r["tenant"] for r in conn.execute("SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant")]
        assignees = [r["assignee"] for r in conn.execute(
            "SELECT DISTINCT assignee FROM tasks WHERE assignee IS NOT NULL AND status != 'archived' ORDER BY assignee")]
        return {
            "columns": [{"name": name, "tasks": columns[name]} for name in columns], "tenants": tenants,
            "assignees": assignees, "latest_event_id": int(latest_event_id), "now": int(time.time())}


# --- GET /tasks/:id ---------------------------------------------------------

@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    board: Optional[str] = Query(None),
    run_state_type: Optional[str] = Query(None, description="With run_state_name: filter runs by column 'status' or 'outcome'"),
    run_state_name: Optional[str] = Query(None, description="With run_state_type: exact value for that run column")):
    with _board_conn(board) as (board, conn):
        if (run_state_type is None) ^ (run_state_name is None):
            raise HTTPException(status_code=400, detail="run_state_type and run_state_name must be passed together or omitted")
        if run_state_type not in (None, "status", "outcome"):
            raise HTTPException(status_code=400, detail="run_state_type must be 'status' or 'outcome'")
        task = _require_task(conn, task_id)
        # Drawer returns the FULL summary (cards on /board carry a 200-char preview).
        task_d = _task_dict(task, latest_summary=kanban_db.latest_summary(conn, task_id))
        links = _links_for(conn, task_id)
        child_summaries = kanban_db.latest_summaries(conn, links["children"])
        children = filter(None, (kanban_db.get_task(conn, cid) for cid in links["children"]))
        _attach_diagnostics(task_d, _compute_task_diagnostics(conn, task_ids=[task_id]).get(task_id) or [])
        return {
            "task": task_d,
            "comments": [asdict(c) for c in kanban_db.list_comments(conn, task_id)],
            "events": [asdict(e) for e in kanban_db.list_events(conn, task_id)],
            "attachments": [_attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)],
            "links": links,
            "child_results": [
                {"id": c.id, "title": c.title, "status": c.status, "latest_summary": child_summaries.get(c.id), "result": c.result}
                for c in children],
            "runs": [asdict(r) for r in kanban_db.list_runs(conn, task_id, state_type=run_state_type, state_name=run_state_name)]}


# --- POST /tasks ------------------------------------------------------------

class CreateTaskBody(BaseModel):
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    workspace_kind: str = "scratch"
    workspace_path: Optional[str] = None
    parents: list[str] = Field(default_factory=list)
    triage: bool = False
    idempotency_key: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    skills: Optional[list[str]] = None
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    reasoning_effort: Optional[str] = None  # none|minimal|…|ultra; None inherits the profile's level
    project_id: Optional[str] = None  # None inherits the board's scoped project (if any)


@router.post("/tasks")
def create_task(payload: CreateTaskBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn), _value_error_400():
        # CreateTaskBody field names match create_task's keyword parameters.
        task_id = kanban_db.create_task(conn, created_by="dashboard", board=board, **payload.model_dump())
        task = kanban_db.get_task(conn, task_id)
        body: dict[str, Any] = {"task": _task_dict(task) if task else None}
        # Dispatcher-presence warning so the UI can banner a ready+assigned task that would
        # otherwise sit idle (no gateway / dispatch_in_gateway=false); triage/todo are expected
        # to wait, unassigned tasks can't dispatch anyway. Probe the request's active home: the
        # dashboard backend may run under a different HERMES_HOME than the board's profile.
        if task and task.status == "ready" and task.assignee:
            try:
                from hermes_cli.kanban import _check_dispatcher_presence
                from hermes_constants import get_hermes_home
                running, message = _check_dispatcher_presence(hermes_home=get_hermes_home())
                if not running and message:
                    body["warning"] = message
            except Exception:
                pass  # probe failure must never block the create itself
        return body


# --- Attachments — upload / list / download / delete ------------------------
# Size cap, filename sanitiser, and collision resolver live in ``kanban_db`` so the
# dashboard, agent toolset, and CLI share one implementation.

@router.get("/tasks/{task_id}/attachments")
def list_task_attachments(task_id: str, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        return {"attachments": [_attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)]}


@router.post("/tasks/{task_id}/attachments")
async def upload_task_attachment(
    task_id: str,
    file: UploadFile = File(...),
    board: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Form(None)):
    """Store an upload under ``attachments_root(board)/<task_id>/`` (sanitised,
    collision-resolved name; ``_safe_attachment_name`` ValueError → 400) and record it."""
    with _board_conn(board) as (board, conn), _value_error_400():
        _require_task(conn, task_id)
        safe_name = _safe_attachment_name(file.filename or "")
        dest_dir = kanban_db.task_attachments_dir(task_id, board=board)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = _collision_free_path(dest_dir, safe_name)  # foo.pdf → foo (1).pdf …
        total = 0  # stream in chunks with a hard size cap so one upload can't fill the disk
        try:
            with open(dest_path, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > KANBAN_ATTACHMENT_MAX_BYTES:
                        out.close()
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413, detail=f"attachment exceeds {KANBAN_ATTACHMENT_MAX_BYTES // (1024 * 1024)} MB limit")
                    out.write(chunk)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"failed to store attachment: {exc}")
        att_id = kanban_db.add_attachment(
            conn, task_id, filename=dest_path.name, stored_path=str(dest_path.resolve()),
            content_type=file.content_type, size=total, uploaded_by=(uploaded_by or "dashboard"))
        att = kanban_db.get_attachment(conn, att_id)
        return {"attachment": _attachment_dict(att) if att else None}


@router.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        att = kanban_db.get_attachment(conn, attachment_id)
        if att is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        # Defense in depth against a tampered DB row: the blob must still live under the board's attachments root.
        root = kanban_db.attachments_root(board=board).resolve()
        try:
            stored = Path(att.stored_path).resolve()
            stored.relative_to(root)
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="attachment file unavailable")
        if not stored.is_file():
            raise HTTPException(status_code=404, detail="attachment file missing on disk")
        return FileResponse(path=str(stored), filename=att.filename, media_type=att.content_type or "application/octet-stream")


@router.delete("/attachments/{attachment_id}")
def remove_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        if kanban_db.delete_attachment(conn, attachment_id) is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        return {"ok": True, "id": attachment_id}


# --- PATCH /tasks/:id  and  POST /tasks/bulk ---------------------------------

class UpdateTaskBody(BaseModel):
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    result: Optional[str] = None
    block_reason: Optional[str] = None
    # Handoff fields forwarded to complete_task on -> 'done' (parity with ``hermes kanban complete``).
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    # In a PATCH ``None`` means "field not sent", so ``clear_*=True`` is the explicit clear signal.
    # ``reasoning_effort="none"`` is a VALUE (thinking off); it is cleared separately so
    # dropping a model override doesn't silently reset the depth.
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    clear_model_override: bool = False
    reasoning_effort: Optional[str] = None
    clear_reasoning_effort: bool = False


class BulkTaskBody(BaseModel):
    ids: list[str]
    status: Optional[str] = None
    assignee: Optional[str] = None  # "" or None = unassign
    priority: Optional[int] = None
    archive: bool = False
    result: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    reclaim_first: bool = False
    # Same semantics as UpdateTaskBody.
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    clear_model_override: bool = False
    reasoning_effort: Optional[str] = None
    clear_reasoning_effort: bool = False


class _StatusRejected(Exception):
    """A status the dashboard may not set via this path; the message is user-facing."""


_RUNNING_DIRECT_MSG = "Cannot set status to 'running' directly; use the dispatcher/claim path"


def _drag_to(conn, task_id: str, s: str) -> bool:
    """Drag-drop into ready/todo/triage: blocked/scheduled -> ready re-opens via ``unblock_task``;
    leaving ``review`` goes through ``reopen_review_task`` (stale-run recovery, parent re-gate,
    ``review_reopened`` event) instead of a raw write; ``triage`` needs no current-state query."""
    current = kanban_db.get_task(conn, task_id) if s != "triage" else None
    if s == "ready" and current and current.status in ("blocked", "scheduled"):
        return kanban_db.unblock_task(conn, task_id)
    if current is not None and current.status == "review":
        return kanban_db.reopen_review_task(conn, task_id)
    return _set_status_direct(conn, task_id, s)


# Status verb dispatch shared by PATCH /tasks/{id} and POST /tasks/bulk: (conn, task_id,
# payload) -> ok. ``review`` uses request_review (never a block, so it can't trip unblock-loop
# detection) with ``force=True``: a dashboard action is a human override of a live worker claim.
_STATUS_HANDLERS: dict[str, Any] = {
    "done": lambda conn, tid, p: kanban_db.complete_task(conn, tid, result=p.result, summary=p.summary, metadata=p.metadata),
    "blocked": lambda conn, tid, p: kanban_db.block_task(conn, tid, reason=getattr(p, "block_reason", None)),
    "scheduled": lambda conn, tid, p: kanban_db.schedule_task(conn, tid, reason=getattr(p, "block_reason", None)),
    "review": lambda conn, tid, p: kanban_db.request_review(
        conn, tid, summary=p.summary, metadata=p.metadata, reviewer=(p.assignee or None), force=True),
    "ready": lambda conn, tid, p: _drag_to(conn, tid, "ready"),
    "todo": lambda conn, tid, p: _drag_to(conn, tid, "todo"),
    "triage": lambda conn, tid, p: _drag_to(conn, tid, "triage")}


def _apply_status(conn, task_id: str, s: str, p, unknown_detail: str) -> bool:
    """Dispatch a status verb; raises ``_StatusRejected`` (user-facing message)
    for ``running`` or an unknown status (``unknown_detail``)."""
    if s == "running":
        raise _StatusRejected(_RUNNING_DIRECT_MSG)
    handler = _STATUS_HANDLERS.get(s)
    if handler is None:
        raise _StatusRejected(unknown_detail)
    return handler(conn, task_id, p)


def _set_priority(conn, task_id: str, priority: int, board: Optional[str]) -> None:
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (int(priority), task_id))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'reprioritized', ?, ?)",
            (task_id, json.dumps({"priority": int(priority)}), int(time.time())))
    # Mutation-boundary observer (post-commit): this direct-SQL write bypasses every kanban_db mutator.
    kanban_db.notify_task_updated(conn, task_id, ("priority",), board=board)


def _apply_model_override(conn, task_id: str, p) -> bool:
    """Raises ValueError/RuntimeError from kanban_db for the caller to map."""
    new_model = None if p.clear_model_override else (p.model_override or "").strip() or None
    return kanban_db.set_model_override(conn, task_id, new_model, provider=p.provider_override)


def _apply_reasoning_effort(conn, task_id: str, p) -> bool:
    return kanban_db.set_reasoning_effort(conn, task_id, None if p.clear_reasoning_effort else p.reasoning_effort)


# Override knobs shared by PATCH and bulk: (payload wants it?, apply, bulk refusal message).
_OVERRIDE_OPS = (
    (lambda p: p.clear_model_override or p.model_override is not None, _apply_model_override, "model override refused"),
    (lambda p: p.clear_reasoning_effort or p.reasoning_effort is not None, _apply_reasoning_effort, "reasoning override refused"),
)


def _patch_status(conn, task_id: str, payload: UpdateTaskBody, review_assignee_deferred: bool) -> None:
    """PATCH status phase: 400 on a rejected verb, 409 when the transition is refused
    (naming the blocking parent(s) for ``ready`` so the UI renders an actionable toast)."""
    s = payload.status
    if s == "archived":
        ok = kanban_db.archive_task(conn, task_id)
    else:
        with _map_errors(400, _StatusRejected):
            ok = _apply_status(conn, task_id, s, payload, f"unknown status: {s}")
        if s == "review" and ok and review_assignee_deferred and not payload.assignee:
            ok = kanban_db.assign_task(conn, task_id, None)
    if ok:
        return
    blockers = _parents_blocking_ready(conn, task_id) if s == "ready" else []
    if blockers:
        names = ", ".join(f"{p['title']!r} ({p['id']}, status={p['status']})" for p in blockers)
        raise _conflict(f"Cannot move to 'ready': blocked by parent(s) not done — {names}")
    raise _conflict(f"status transition to {s!r} not valid from current state")


def _patch_title_body(conn, task_id: str, payload: UpdateTaskBody, board: Optional[str]) -> None:
    """PATCH title/body phase: one UPDATE + ``edited`` event, then the post-commit observer
    (field names only — values never leave the DB via this payload)."""
    with kanban_db.write_txn(conn):
        sets, vals = [], []
        if payload.title is not None:
            if not payload.title.strip():
                raise HTTPException(status_code=400, detail="title cannot be empty")
            sets.append("title = ?")
            vals.append(payload.title.strip())
        if payload.body is not None:
            sets.append("body = ?")
            vals.append(payload.body)
        vals.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", vals)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'edited', NULL, ?)",
            (task_id, int(time.time())))
    kanban_db.notify_task_updated(
        conn, task_id, [f for f in ("title", "body") if getattr(payload, f) is not None], board=board)


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: UpdateTaskBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        # For a combined assignee+review patch, request_review must capture the
        # current implementer before the task is routed to the reviewer.
        review_assignee_deferred = payload.status == "review" and payload.assignee is not None
        if payload.assignee is not None and not review_assignee_deferred:
            with _map_errors(409, RuntimeError):
                _require_ok(kanban_db.assign_task(conn, task_id, payload.assignee or None))
        if payload.status is not None:
            _patch_status(conn, task_id, payload, review_assignee_deferred)
        for wanted, apply, _refused in _OVERRIDE_OPS:
            if wanted(payload):
                with _map_errors(400, ValueError, RuntimeError):
                    ok = apply(conn, task_id, payload)
                _require_ok(ok)
        if payload.priority is not None:
            _set_priority(conn, task_id, payload.priority, board)
        if payload.title is not None or payload.body is not None:
            _patch_title_body(conn, task_id, payload, board)
        updated = kanban_db.get_task(conn, task_id)
        return {"task": _task_dict(updated) if updated else None}


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        if not kanban_db.delete_task(conn, task_id):
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {"deleted": True, "task_id": task_id}


def _parents_blocking_ready(conn: sqlite3.Connection, task_id: str) -> list:
    """Parent rows (id, title, status) not ``done`` that block promotion to ``ready``.

    Used to enrich the 409 response from :func:`update_task` so the dashboard can show an actionable toast
    (#26744) instead of a silent no-op. Returns ``[]`` when nothing blocks the transition (e.g. no parents,
    or all parents already done).
    """
    rows = conn.execute(
        "SELECT t.id, t.title, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? AND t.status != 'done'",
        (task_id,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows]


def _set_status_direct(conn: sqlite3.Connection, task_id: str, new_status: str) -> bool:
    """Direct status write for drag-drop moves without a structured verb (todo<->ready,
    running<->ready) + a ``status`` event. Leaving ``running`` closes the run as 'reclaimed'
    so attempt history isn't orphaned; the worker is killed only AFTER the txn commits."""
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    effective_status = new_status
    with kanban_db.write_txn(conn):
        prev = conn.execute(
            "SELECT status, current_run_id, worker_pid, claim_lock FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if prev is None:
            return False
        if prev["status"] == "running" and new_status == "ready":
            resume_status = kanban_db._retry_status_for_run(conn, task_id, prev["current_run_id"])
            if resume_status == "review":
                effective_status = "review" if kanban_db._parents_satisfied(conn, task_id) else "todo"
        # Never promote to 'ready' unless all parents are done/archived — otherwise the
        # dispatcher spawns a child whose upstream work hasn't completed.
        if effective_status == "ready" and not kanban_db._parents_satisfied(conn, task_id):
            return False
        was_running = prev["status"] == "running"
        reopening_satisfied_parent = prev["status"] in {"done", "archived"} and effective_status not in {"done", "archived"}
        cur = conn.execute(
            "UPDATE tasks SET status = ?, "
            "  claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, "
            "  claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, "
            "  worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END "
            "WHERE id = ?",
            (effective_status,) * 4 + (task_id,))
        if cur.rowcount != 1:
            return False
        run_id = None
        if was_running and effective_status != "running" and prev["current_run_id"]:
            run_id = kanban_db._end_run(
                conn, task_id, outcome="reclaimed", status="reclaimed",
                summary=f"status changed to {effective_status} (dashboard/direct)")
            terminations.append((prev["worker_pid"], prev["claim_lock"]))
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, 'status', ?, ?)",
            (task_id, run_id, json.dumps({"status": effective_status, "requested_status": new_status}), int(time.time())))
        if reopening_satisfied_parent:
            # Domain-layer invalidation composes via a savepoint inside our txn and hands
            # back worker terminations to perform post-commit.
            result = kanban_db.invalidate_descendants_for_parent_reopen(conn, task_id, author="dashboard")
            terminations.extend(result["terminations"])
    for pid, claim_lock in terminations:
        kanban_db._terminate_reclaimed_worker(pid, claim_lock)
    # Re-opening something may have made children stale.
    if effective_status in {"done", "ready", "review"}:
        kanban_db.recompute_ready(conn)
    return True


# --- Comments / links -------------------------------------------------------

class CommentBody(BaseModel):
    body: str
    author: Optional[str] = "dashboard"


@router.post("/tasks/{task_id}/comments")
def add_comment(task_id: str, payload: CommentBody, board: Optional[str] = Query(None)):
    if not payload.body.strip():
        raise HTTPException(status_code=400, detail="body is required")
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        kanban_db.add_comment(conn, task_id, author=payload.author or "dashboard", body=payload.body)
        return {"ok": True}


class LinkBody(BaseModel):
    parent_id: str
    child_id: str


@router.post("/links")
def add_link(payload: LinkBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn), _value_error_400():
        kanban_db.link_tasks(conn, payload.parent_id, payload.child_id)
        return {"ok": True}


@router.delete("/links")
def delete_link(parent_id: str = Query(...), child_id: str = Query(...), board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        return {"ok": bool(kanban_db.unlink_tasks(conn, parent_id, child_id))}


def _bulk_apply_one(conn, tid: str, payload: BulkTaskBody, board: Optional[str], entry: dict) -> None:
    """Apply the bulk patch to one task, recording refusals in ``entry`` without aborting the
    remaining ops — except a rejected status verb (``_StatusRejected`` propagates)."""
    if payload.archive and not kanban_db.archive_task(conn, tid):
        entry.update(ok=False, error="archive refused")
    if payload.status is not None and not payload.archive:
        s = payload.status
        if not _apply_status(conn, tid, s, payload, f"unknown status {s!r}"):
            entry.update(ok=False, error=f"transition to {s!r} refused")
    if payload.assignee is not None:
        try:
            ok = (kanban_db.reassign_task(conn, tid, payload.assignee or None, reclaim_first=True) if payload.reclaim_first
                  else kanban_db.assign_task(conn, tid, payload.assignee or None))
            if not ok:
                entry.update(ok=False, error="assign refused")
        except RuntimeError as e:
            entry.update(ok=False, error=str(e))
    if payload.priority is not None:
        _set_priority(conn, tid, payload.priority, board)
    for wanted, apply, refused in _OVERRIDE_OPS:
        if wanted(payload):
            try:
                if not apply(conn, tid, payload):
                    entry.update(ok=False, error=refused)
            except (ValueError, RuntimeError) as e:
                entry.update(ok=False, error=str(e))


@router.post("/tasks/bulk")
def bulk_update(payload: BulkTaskBody, board: Optional[str] = Query(None)):
    """Apply the same patch to every id. Independent iteration — per-task
    failures don't abort siblings; returns per-id outcome for partials."""
    ids = [i for i in (payload.ids or []) if i]
    if not ids:
        raise HTTPException(status_code=400, detail="ids is required")
    results: list[dict] = []
    with _board_conn(board) as (board, conn):
        for tid in ids:
            entry: dict[str, Any] = {"id": tid, "ok": True}
            try:
                if kanban_db.get_task(conn, tid) is None:
                    entry.update(ok=False, error="not found")
                else:
                    _bulk_apply_one(conn, tid, payload, board, entry)
            except Exception as e:  # one bad id shouldn't kill the batch (incl. _StatusRejected)
                entry.update(ok=False, error=str(e))
            results.append(entry)
        return {"results": results}


# --- Diagnostics — fleet-wide distress signals (see kanban_diagnostics) ------

@router.get("/diagnostics")
def list_diagnostics(
    board: Optional[str] = _BOARD_Q,
    severity: Optional[str] = Query(None, description="Filter by severity: warning|error|critical")):
    """Tasks with an active diagnostic, highest severity first then most recent; also
    consumed by ``hermes kanban diagnostics`` when the dashboard runs."""
    with _board_conn(board) as (board, conn):
        diags_by_task = _compute_task_diagnostics(conn, task_ids=None)
        if severity and diags_by_task:
            diags_by_task = {
                tid: keep
                for tid, dl in diags_by_task.items()
                if (keep := [d for d in dl if kd.severity_at_or_above(d.get("severity"), severity)])}
        if not diags_by_task:
            return {"diagnostics": [], "count": 0}
        ids = list(diags_by_task.keys())
        rows = {r["id"]: r for r in conn.execute(
            f"SELECT id, title, status, assignee FROM tasks WHERE id IN ({_placeholders(ids)})", tuple(ids)).fetchall()}
        out = []
        for tid, dl in diags_by_task.items():
            r = rows.get(tid) or {"title": None, "status": None, "assignee": None}
            out.append({
                "task_id": tid, "task_title": r["title"], "task_status": r["status"], "task_assignee": r["assignee"],
                "diagnostics": dl})
        sev_idx = {s: i for i, s in enumerate(kd.SEVERITY_ORDER)}
        out.sort(key=lambda row: (
            -sev_idx.get(row["diagnostics"][0].get("severity"), -1), -(row["diagnostics"][0].get("last_seen_at") or 0)))
        return {"diagnostics": out, "count": sum(len(d["diagnostics"]) for d in out)}


# --- Worker visibility — active-worker list, per-run inspect/terminate -------

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]


@router.get("/workers/active")
def list_active_workers(board: Optional[str] = _BOARD_Q):
    """Every running worker: an open ``task_runs`` row with a ``worker_pid`` whose
    task is ``running``. Returns ``{workers, count, checked_at}``."""
    with _board_conn(board) as (board, conn):
        rows = conn.execute(
            "SELECT r.id AS run_id, r.task_id, t.title AS task_title, t.status AS task_status, "
            "t.assignee AS task_assignee, r.profile, r.worker_pid, r.started_at, r.claim_lock, "
            "r.claim_expires, r.last_heartbeat_at, r.max_runtime_seconds "
            "FROM task_runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE r.ended_at IS NULL AND r.worker_pid IS NOT NULL AND t.status = 'running' "
            "ORDER BY r.started_at ASC").fetchall()
        workers = [dict(row) for row in rows]
        return {"workers": workers, "count": len(workers), "checked_at": int(time.time())}


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: int, board: Optional[str] = _BOARD_Q):
    """``{run: {...}}`` with the same serialisation as ``GET /tasks/{id}``; 404 if unknown."""
    with _board_conn(board) as (board, conn):
        return {"run": asdict(_require_run(conn, run_id))}


@router.get("/runs/{run_id}/inspect")
def inspect_run_endpoint(run_id: int, board: Optional[str] = _BOARD_Q):
    """Live psutil stats for a run's worker; ``{alive: false, reason}`` when unavailable and
    access-denied reported inline rather than as a 500."""
    with _board_conn(board) as (board, conn):
        r = _require_run(conn, run_id)

    def _dead(reason: str, **extra) -> dict:
        return {"run_id": run_id, "alive": False, **extra, "reason": reason}

    if r.ended_at is not None:
        return _dead("run already ended")
    pid = r.worker_pid
    if pid is None:
        return _dead("no worker_pid recorded")
    if _psutil is None:
        return _dead("psutil not available", pid=pid)
    try:
        proc = _psutil.Process(pid)
        info = proc.as_dict(attrs=["cpu_percent", "memory_info", "num_threads", "status", "create_time", "cmdline"])
        try:
            num_fds = proc.num_fds()
        except AttributeError:  # POSIX-only
            num_fds = None
        mem = info.get("memory_info")
        return {
            "run_id": run_id, "alive": True, "pid": pid,
            "cpu_percent": info.get("cpu_percent"),
            "memory_rss_bytes": mem.rss if mem else None,
            "memory_vms_bytes": mem.vms if mem else None,
            "num_threads": info.get("num_threads"), "num_fds": num_fds,
            "status": info.get("status"), "create_time": info.get("create_time"), "cmdline": info.get("cmdline")}
    except _psutil.NoSuchProcess:
        return _dead("process not found", pid=pid)
    except _psutil.AccessDenied:
        return {"run_id": run_id, "alive": True, "pid": pid, "error": "access denied"}


class TerminateRunBody(BaseModel):
    reason: Optional[str] = None


@router.post("/runs/{run_id}/terminate")
def terminate_run_endpoint(run_id: int, payload: TerminateRunBody, board: Optional[str] = _BOARD_Q):
    """Terminate an in-flight run via ``reclaim_task`` (same SIGTERM->SIGKILL flow, bookkeeping
    and events as ``POST /tasks/{id}/reclaim``); 409 if already ended / not reclaimable.

    Closes the gap left by PR #28432, which shipped the read-only sibling endpoints (``/workers/active``,
    ``/runs/{run_id}``, ``/runs/{run_id}/inspect``) but no termination control surface.
    """
    with _board_conn(board) as (board, conn):
        r = _require_run(conn, run_id)
        if r.ended_at is not None:
            raise _conflict(f"run {run_id} already ended")
        if not kanban_db.reclaim_task(conn, r.task_id, reason=payload.reason):
            raise _conflict(f"cannot terminate run {run_id}: task {r.task_id} is no longer in a reclaimable state")
        return {"ok": True, "run_id": run_id, "task_id": r.task_id}


# --- Recovery actions — reclaim / specify / reassign / estimate -------------

class ReclaimBody(BaseModel):
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reclaim")
def reclaim_task_endpoint(task_id: str, payload: ReclaimBody, board: Optional[str] = Query(None)):
    """Release an active worker claim without waiting for the claim TTL
    (``hermes kanban reclaim <task_id> --reason ...``)."""
    with _board_conn(board) as (board, conn):
        if not kanban_db.reclaim_task(conn, task_id, reason=payload.reason):
            raise _conflict(f"cannot reclaim {task_id}: not in a claimable state (not running, or unknown id)")
        return {"ok": True, "task_id": task_id}


class SpecifyBody(BaseModel):
    """Only the author is configurable; model + prompt come from
    ``auxiliary.triage_specifier`` in config.yaml, same as the CLI."""

    author: Optional[str] = None


@router.post("/tasks/{task_id}/specify")
def specify_task_endpoint(task_id: str, payload: SpecifyBody, board: Optional[str] = Query(None)):
    """Flesh out a triage task via the auxiliary LLM (``hermes kanban specify``). Non-OK is NOT
    an HTTP error — the UI renders the reason inline. Sync ``def`` → runs in the threadpool."""
    outcome = _run_aux(board, "kanban_specify", "specify_task", task_id, payload.author)
    return {"ok": bool(outcome.ok), "task_id": outcome.task_id, "reason": outcome.reason, "new_title": outcome.new_title}


class ReassignBody(BaseModel):
    profile: Optional[str] = None  # "" or None = unassign
    reclaim_first: bool = False
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reassign")
def reassign_task_endpoint(task_id: str, payload: ReassignBody, board: Optional[str] = Query(None)):
    """Reassign to another profile, optionally reclaiming first
    (``hermes kanban reassign <task_id> <profile> [--reclaim]``)."""
    with _board_conn(board) as (board, conn):
        ok = kanban_db.reassign_task(
            conn, task_id, payload.profile or None, reclaim_first=bool(payload.reclaim_first), reason=payload.reason)
        if not ok:
            raise _conflict(
                f"cannot reassign {task_id}: unknown id, or still "
                "running (pass reclaim_first=true to release the claim first)")
        return {"ok": True, "task_id": task_id, "assignee": payload.profile or None}


# Estimate: rough token/complexity read via the auxiliary model. NOT a dollar cost.
_ESTIMATE_SYSTEM_PROMPT = (
    "You estimate how much work an autonomous coding agent will spend on a "
    "kanban task. Given the task title and description, respond with STRICT "
    "JSON only (no prose, no code fence):\n"
    '{"est_tokens": <integer total tokens across the whole run>, '
    '"complexity": "S"|"M"|"L", '
    '"rationale": "<one short sentence>"}\n'
    "Base the token figure on a realistic multi-turn agent run (reading files, "
    "tool calls, edits, retries) — not a single reply. S≈small/localized, "
    "M≈multi-file, L≈broad or ambiguous. Be honest that this is a rough guess.")


class EstimateBody(BaseModel):
    title: str = ""
    body: Optional[str] = None


@router.post("/estimate")
def estimate_text_endpoint(payload: EstimateBody):
    """Estimate from raw title/body (create dialog, before a task exists)."""
    return _run_estimate(payload.title, payload.body)


@router.post("/tasks/{task_id}/estimate")
def estimate_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    """Estimate for an existing task; ``{ok, est_tokens, complexity, rationale, model}``."""
    with _board_conn(board) as (board, conn):
        task = _require_task(conn, task_id)
    return _run_estimate(task.title, task.body)


def _cap(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _run_estimate(title: str, body: Optional[str]) -> dict:
    """Never raises — config/parse/API errors become ``{"ok": False, "reason"}`` so the UI renders them inline."""
    if not (title or "").strip():
        return {"ok": False, "reason": "a title is required to estimate"}
    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return {"ok": False, "reason": "auxiliary client unavailable"}
    user_msg = f"Title: {_cap(title, 400)}\n\nDescription:\n{_cap(body, 4000) or '(none)'}"
    try:
        resp = call_llm(
            task="kanban_estimator",
            messages=[{"role": "system", "content": _ESTIMATE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=300, timeout=60)
    except Exception as exc:
        return {"ok": False, "reason": f"LLM error: {type(exc).__name__}"}
    try:
        raw = (resp.choices[0].message.content or "").strip()
        model = getattr(resp, "model", None)
    except Exception:
        raw, model = "", None

    # Same tolerant JSON-blob extraction the specifier uses.
    try:
        m = None if raw.lstrip().startswith("{") else re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        parsed = obj if isinstance(obj, dict) else None
    except Exception:
        parsed = None
    if not parsed:
        return {"ok": False, "reason": "could not parse an estimate from the model"}
    try:
        est_tokens = int(parsed.get("est_tokens") or 0)
    except (TypeError, ValueError):
        est_tokens = 0
    complexity = str(parsed.get("complexity") or "").strip().upper()
    return {
        "ok": True, "est_tokens": est_tokens, "complexity": complexity if complexity in {"S", "M", "L"} else None,
        "rationale": str(parsed.get("rationale") or "").strip() or None, "model": model}


# --- Plugin config ----------------------------------------------------------

def _load_config_or_empty() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


@router.get("/config")
def get_config():
    """Kanban dashboard preferences from the ``dashboard.kanban`` config section."""
    k_cfg = (_load_config_or_empty().get("dashboard") or {}).get("kanban") or {}
    return {
        "default_tenant": k_cfg.get("default_tenant") or "",
        "lane_by_profile": bool(k_cfg.get("lane_by_profile", True)),
        "include_archived_by_default": bool(k_cfg.get("include_archived_by_default", False)),
        "render_markdown": bool(k_cfg.get("render_markdown", True))}


# --- Home-channel subscriptions (per-task, per-platform toggles) -------------
# Each gateway platform has at most one "home" (chat_id, thread_id, name); a toggle-on writes
# exactly the notify_subs row ``/kanban create`` would, so the gateway notifier needs no plumbing.

def _configured_home_channels() -> list[dict]:
    """Every platform with a home_channel, from the live GatewayConfig (so env overlays
    like ``TELEGRAM_HOME_CHANNEL`` are honored), sorted by platform."""
    try:
        from gateway.config import load_gateway_config
        gw_cfg = load_gateway_config()
    except Exception:
        return []
    result = [
        {"platform": platform.value, "chat_id": pcfg.home_channel.chat_id,
         "thread_id": pcfg.home_channel.thread_id or "", "name": pcfg.home_channel.name or "Home"}
        for platform, pcfg in gw_cfg.platforms.items() if pcfg and pcfg.home_channel]
    result.sort(key=lambda r: r["platform"])
    return result


def _active_profile_name() -> str:
    """Current Hermes profile name for notify-sub ownership."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _home_for_platform(platform: str, detail: str) -> dict:
    home = next((h for h in _configured_home_channels() if h["platform"] == platform), None)
    if not home:
        raise HTTPException(status_code=404, detail=detail)
    return home


@router.get("/home-channels")
def get_home_channels(task_id: Optional[str] = Query(None), board: Optional[str] = Query(None)):
    """Every platform with a home channel plus whether *task_id* (if given) is
    subscribed to it; without ``task_id`` every ``subscribed`` is false."""
    homes = _configured_home_channels()
    subscribed_homes: set[tuple[str, str, str]] = set()
    if task_id:
        with _board_conn(board) as (board, conn):
            subs = kbn.list_notify_subs(conn, task_id)
        subscribed_homes = {
            (str(sub.get("platform") or ""), str(sub.get("chat_id") or ""), str(sub.get("thread_id") or "")) for sub in subs}
    return {"home_channels": [
        {**home, "subscribed": (home["platform"], home["chat_id"], home["thread_id"]) in subscribed_homes} for home in homes]}


@router.post("/tasks/{task_id}/home-subscribe/{platform}")
def subscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Subscribe *task_id* to *platform*'s home channel. Idempotent at the DB
    layer; 404 when the platform has no home or the task doesn't exist."""
    home = _home_for_platform(
        platform,
        f"No home channel configured for platform {platform!r}. "
        f"Set one from the messenger via /sethome, or configure "
        f"gateway.platforms.{platform}.home_channel in config.yaml.")
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        kbn.add_notify_sub(
            conn, task_id=task_id, platform=platform, chat_id=home["chat_id"],
            thread_id=home["thread_id"] or None, notifier_profile=_active_profile_name())
        return {"ok": True, "task_id": task_id, "home_channel": home}


@router.delete("/tasks/{task_id}/home-subscribe/{platform}")
def unsubscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Remove any notify subscription on *task_id* matching *platform*'s home."""
    home = _home_for_platform(platform, f"No home channel configured for platform {platform!r}.")
    with _board_conn(board) as (board, conn):
        kbn.remove_notify_sub(
            conn, task_id=task_id, platform=platform, chat_id=home["chat_id"], thread_id=home["thread_id"] or None)
        return {"ok": True, "task_id": task_id, "home_channel": home}


# --- Stats / assignees / worker log / dispatch / model options ---------------

@router.get("/stats")
def get_stats(board: Optional[str] = Query(None)):
    """Per-status + per-assignee counts + oldest-ready age (HUD and router profiles)."""
    with _board_conn(board) as (board, conn):
        return kanban_db.board_stats(conn)


@router.get("/assignees")
def get_assignees(board: Optional[str] = Query(None)):
    """Union of on-disk profiles and assignees used on the board, so a fresh
    profile appears in the picker before it has any task."""
    with _board_conn(board) as (board, conn):
        return {"assignees": kanban_db.known_assignees(conn)}


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str, tail: Optional[int] = Query(None, ge=1, le=2_000_000), board: Optional[str] = Query(None)):
    """Worker stdout/stderr log. ``tail`` caps the response bytes; 404 if the
    task never spawned. On-disk log rotates at 2 MiB with one ``.log.1`` kept."""
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
    content = kanban_db.read_worker_log(task_id, tail_bytes=tail, board=board)
    log_path = kanban_db.worker_log_path(task_id, board=board)
    size = log_path.stat().st_size if log_path.exists() else 0
    return {
        "task_id": task_id, "path": str(log_path), "exists": content is not None,
        "size_bytes": size, "content": content or "", "truncated": bool(tail and size > tail)}


@router.post("/dispatch")
def dispatch(dry_run: bool = Query(False), max_n: int = Query(8, alias="max"), board: Optional[str] = Query(None)):
    """Dispatch nudge so the UI doesn't wait out the 60 s dispatcher tick."""
    with _board_conn(board) as (board, conn):
        result = kbd.dispatch_once(conn, dry_run=dry_run, max_spawn=max_n, board=board)
        try:
            return asdict(result)  # DispatchResult is a dataclass
        except TypeError:
            return {"result": str(result)}


@router.get("/model-options")
def model_options():
    """Providers + curated models for the override dropdown via ``inventory.build_models_payload``
    (same substrate as the Models page) so it can't offer a pair Hermes rejects. Skips pricing
    and custom-provider probes: a slow/offline local endpoint must not hang the drawer."""
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(
            load_picker_context(), explicit_only=True, canonical_order=True, probe_custom_providers=False)
        return {
            "providers": [
                {"slug": row.get("slug", ""), "label": row.get("label") or row.get("slug", ""),
                 "models": list(row.get("models") or [])}
                for row in payload.get("providers", [])
                if row.get("models")]}
    except Exception:
        log.exception("kanban model-options failed")
        return {"providers": []}  # empty catalog → the UI falls back to a free-text input


# --- Boards CRUD (multi-project support) --------------------------------------

class CreateBoardBody(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    default_workdir: Optional[str] = None
    # Project (id or slug) scoping the board: default_workdir mirrors its primary repo, tasks inherit it.
    project_id: Optional[str] = None
    switch: bool = False


class RenameBoardBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    # For both fields: ``None`` = leave unchanged; "" = clear; value = validate/resolve + set.
    default_workdir: Optional[str] = None
    project_id: Optional[str] = None


# Board transfer exchanges filesystem PATHS, not bytes (same contract as profile export/import):
# clients run the native save/open dialog on the machine hosting the backend.

class ExportBoardBody(BaseModel):
    output: str = ""  # empty → staging path under the kanban root
    attachments: bool = True
    logs: bool = False


class ImportBoardBody(BaseModel):
    archive: str  # path to a board .tar.gz on the backend's filesystem
    slug: Optional[str] = None  # override the archive's slug; collisions auto-suffix
    switch: bool = False


def _board_display_kwargs(p: BaseModel) -> dict[str, Any]:
    """Display-metadata fields shared by create_board / write_board_metadata."""
    return {"name": p.name, "description": p.description, "icon": p.icon, "color": p.color}


def _resolve_project(ref: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a project id/slug to ``(id, name, primary_path)``; ``(None,)*3``
    for a falsy ref, 400 when a non-empty ref doesn't resolve."""
    if not ref or not ref.strip():
        return None, None, None
    with _errors_to_500("projects unavailable"):
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            proj = pdb.get_project(pconn, ref.strip())
    if proj is None:
        raise HTTPException(status_code=400, detail=f"project {ref!r} does not exist")
    return proj.id, proj.name, (proj.primary_path or None)


def _projects_by_id() -> dict[str, Any]:
    """Map every project id -> Project (archived included) for annotation."""
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            return {p.id: p for p in pdb.list_projects(pconn, include_archived=True)}
    except Exception:
        return {}


def _board_counts(slug: str) -> dict[str, int]:
    """``{status: count}`` for a board; ``{}`` on a missing/empty DB."""
    try:
        if not kanban_db.kanban_db_path(board=slug).exists():
            return {}
        with closing(kbc.connect(board=slug)) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
            return {r["status"]: int(r["n"]) for r in rows}
    except Exception:
        return {}


def _default_workspace_kind(board: dict[str, Any]) -> str:
    """Recommend a non-destructive task workspace from board metadata."""
    workdir = str(board.get("default_workdir") or "").strip()
    if not workdir:
        return "scratch"
    try:
        return "worktree" if kbw._git_toplevel(Path(workdir)) else "dir"
    except (OSError, ValueError):
        return "dir"


def _annotate_board_meta(meta: dict) -> dict:
    meta["default_workspace_kind"] = _default_workspace_kind(meta)
    _, meta["project_name"], _ = _resolve_project(meta.get("project_id"))
    return meta


@router.get("/projects")
def list_kanban_projects():
    """Live (non-archived) projects available for board scoping."""
    with _errors_to_500("failed to list projects"):
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            projects = pdb.list_projects(pconn, include_archived=False)
    return {"projects": [
        {"id": p.id, "slug": p.slug, "name": p.name,
         "primary_path": p.primary_path or "", "icon": p.icon or "", "color": p.color or ""}
        for p in projects]}


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    """Every board on disk with task counts and the active slug."""
    boards = kanban_db.list_boards(include_archived=include_archived)
    current = kanban_db.get_current_board()
    proj_map = _projects_by_id()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_counts(b["slug"])
        # Live cards only — archived tasks are hidden from every default board view,
        # so counting them in the switcher badge would visibly disagree.
        b["total"] = sum(n for status, n in b["counts"].items() if status != "archived")
        b["default_workspace_kind"] = _default_workspace_kind(b)
        pid = b["project_id"] = b.get("project_id") or None
        proj = proj_map.get(pid) if pid else None
        b["project_name"] = proj.name if proj else None
    return {"boards": boards, "current": current}


def _validate_workdir(raw: str) -> str:
    """Board default_workdir must be an absolute, existing directory (400 otherwise)."""
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise HTTPException(status_code=400, detail="Project directory must be an absolute path.")
    if not requested.is_dir():
        raise HTTPException(status_code=400, detail="Project directory must be an existing directory.")
    return str(requested.resolve())


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    """Create a board. Idempotent — ``slug`` collision returns the existing one."""
    default_workdir = _validate_workdir(payload.default_workdir) if payload.default_workdir else None
    # A chosen project's primary repo becomes the default workdir unless one was passed explicitly.
    project_id, _pname, primary_path = _resolve_project(payload.project_id)
    if primary_path and not default_workdir:
        default_workdir = primary_path
    with _value_error_400():
        meta = kanban_db.create_board(
            payload.slug, default_workdir=default_workdir, project_id=project_id, **_board_display_kwargs(payload))
    if payload.switch:
        with _value_error_400():
            kanban_db.set_current_board(meta["slug"])
    return {"board": _annotate_board_meta(meta), "current": kanban_db.get_current_board()}


@router.patch("/boards/{slug}")
def rename_board(slug: str, payload: RenameBoardBody):
    """Update display metadata / default workdir / project scope (slug is immutable)."""
    normed = _existing_board_slug(slug)
    # write_board_metadata treats a falsy value as "clear", so pass "" through.
    default_workdir: Optional[str] = None
    if payload.default_workdir is not None:
        raw = payload.default_workdir.strip()
        default_workdir = _validate_workdir(raw) if raw else ""
    # A resolved project mirrors its repo into default_workdir unless the caller set it explicitly.
    project_id: Optional[str] = None
    if payload.project_id is not None:
        if payload.project_id.strip():
            project_id, _pname, primary_path = _resolve_project(payload.project_id)
            if primary_path and default_workdir is None:
                default_workdir = primary_path
        else:
            project_id = ""  # clear the scope
    meta = kanban_db.write_board_metadata(
        normed, default_workdir=default_workdir, project_id=project_id, **_board_display_kwargs(payload))
    return {"board": _annotate_board_meta(meta)}


@router.delete("/boards/{slug}")
def delete_board(slug: str, delete: bool = Query(False, description="Hard-delete instead of archive")):
    """Archive (default) or hard-delete a board."""
    with _value_error_400():
        res = kanban_db.remove_board(slug, archive=not delete)
    return {"result": res, "current": kanban_db.get_current_board()}


async def _run_transfer(fn, log_label: str):
    """Run a blocking kanban_transfer call off the event loop, mapping its errors
    to 404 (missing path) / 400 (invalid) / 500 (logged)."""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("%s failed", log_label)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/boards/{slug}/export")
async def export_board_endpoint(slug: str, body: ExportBoardBody):
    """Write ``slug`` to a portable archive; return the path written."""
    from hermes_cli import kanban_transfer

    output = (body.output or "").strip()
    if not output:
        staging = kanban_db.kanban_home() / "kanban" / "board-exports"
        try:
            staging.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create export directory: {exc}")
        output = str(staging / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz")
    return await _run_transfer(
        lambda: kanban_transfer.export_board(slug, output, include_attachments=body.attachments, include_logs=body.logs),
        f"POST /boards/{slug}/export")


@router.post("/boards/import")
async def import_board_endpoint(body: ImportBoardBody):
    """Import a board archive as a NEW board; return the landed board."""
    from hermes_cli import kanban_transfer

    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")
    result = await _run_transfer(
        lambda: kanban_transfer.import_board(archive, (body.slug or "").strip() or None, activate=body.switch),
        "POST /boards/import")
    return {**result, "current": kanban_db.get_current_board()}


@router.post("/boards/{slug}/switch")
def switch_board(slug: str):
    """Persist ``slug`` as the active board for CLI / slash-command parity
    (dashboard users pick boards client-side via localStorage)."""
    normed = _existing_board_slug(slug)
    kanban_db.set_current_board(normed)
    return {"current": normed}


# --- Profile metadata & description editing (kanban orchestrator) ------------

class DescribeBody(BaseModel):
    description: Optional[str] = None  # explicit user-authored text


class DescribeAutoBody(BaseModel):
    overwrite: bool = False


@router.get("/profiles")
def list_profile_roster():
    """Every installed profile with its description (profiles without one are
    still routable on name alone, just less precisely)."""
    with _errors_to_500("failed to list profiles"):
        from hermes_cli import profiles as profiles_mod
        profiles = profiles_mod.list_profiles()
    return {"profiles": [
        {"name": p.name, "is_default": bool(p.is_default), "model": p.model or "", "provider": p.provider or "",
         "description": p.description or "", "description_auto": bool(p.description_auto),
         "skill_count": int(p.skill_count or 0)}
        for p in profiles]}


@router.patch("/profiles/{profile_name}")
def update_profile_description(profile_name: str, payload: DescribeBody):
    """Set (``description_auto: false`` so the auto-describer won't overwrite it
    without ``--overwrite``) or clear (empty string) a profile's description."""
    with _errors_to_500("failed to update profile"):
        from hermes_cli import profiles as profiles_mod
        canon = profiles_mod.normalize_profile_name(profile_name)
        if canon == "default":
            from hermes_constants import get_hermes_home  # type: ignore
            profile_dir = Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
        if not profile_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"profile '{profile_name}' not found")
        text = (payload.description or "").strip()
        profiles_mod.write_profile_meta(profile_dir, description=text, description_auto=False)
    return {"ok": True, "profile": canon, "description": text}


@router.post("/profiles/{profile_name}/describe-auto")
def auto_describe_profile(profile_name: str, payload: DescribeAutoBody):
    """``hermes profile describe <name> --auto``: persist with ``description_auto: true``.
    Non-OK outcomes are NOT HTTP errors — the UI renders the reason inline."""
    with _errors_to_500("describer crashed"):
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(profile_name, overwrite=bool(payload.overwrite))
    return {"ok": bool(outcome.ok), "profile": outcome.profile_name, "reason": outcome.reason, "description": outcome.description}


# --- Decompose (built-in decomposer fan-out) ----------------------------------

class DecomposeBody(BaseModel):
    author: Optional[str] = None


@router.post("/tasks/{task_id}/decompose")
def decompose_task_endpoint(task_id: str, payload: DecomposeBody, board: Optional[str] = Query(None)):
    """Fan a triage task out into child tasks via the auxiliary LLM (``hermes kanban decompose``).
    Non-OK is NOT an HTTP error. Sync ``def`` → runs in the threadpool."""
    outcome = _run_aux(board, "kanban_decompose", "decompose_task", task_id, payload.author)
    return {
        "ok": bool(outcome.ok), "task_id": outcome.task_id, "reason": outcome.reason,
        "fanout": bool(outcome.fanout), "child_ids": outcome.child_ids or [], "new_title": outcome.new_title}


# --- Orchestration settings (kanban.orchestrator_profile / default_assignee /
#     auto_decompose / auto_promote_children) ----------------------------------

class OrchestrationSettingsBody(BaseModel):
    orchestrator_profile: Optional[str] = None
    default_assignee: Optional[str] = None
    auto_decompose: Optional[bool] = None
    auto_promote_children: Optional[bool] = None


_PROFILE_SETTINGS = ("orchestrator_profile", "default_assignee")


@router.get("/orchestration")
def get_orchestration_settings():
    """Current orchestration knobs from config.yaml plus the resolved effective
    values (fallbacks filled the same way the decomposer does)."""
    cfg = _load_config_or_empty()
    kanban_cfg = (cfg.get("kanban") or {}) if isinstance(cfg, dict) else {}
    explicit = {k: (kanban_cfg.get(k) or "").strip() for k in _PROFILE_SETTINGS}
    resolved = dict(explicit)
    try:
        from hermes_cli import profiles as profiles_mod
        active_default = profiles_mod.get_active_profile_name() or "default"
        for k, v in explicit.items():
            if not v or not profiles_mod.profile_exists(v):
                resolved[k] = active_default
    except Exception:
        active_default = "default"
        resolved = {k: v or active_default for k, v in resolved.items()}
    return {
        "orchestrator_profile": explicit["orchestrator_profile"],
        "default_assignee": explicit["default_assignee"],
        "auto_decompose": bool(kanban_cfg.get("auto_decompose", True)),
        "auto_promote_children": bool(kanban_cfg.get("auto_promote_children", True)),
        "resolved_orchestrator_profile": resolved["orchestrator_profile"],
        "resolved_default_assignee": resolved["default_assignee"],
        "active_profile": active_default}


def _validated_profile_name(raw: Optional[str], profiles_mod) -> str:
    """Strip a profile name; 400 if non-empty and unknown. Fails open when the lookup itself errors."""
    name = (raw or "").strip()
    if name and profiles_mod is not None:
        try:
            exists = profiles_mod.profile_exists(name)
        except Exception:
            exists = True
        if not exists:
            raise HTTPException(status_code=400, detail=f"profile '{name}' does not exist")
    return name


@router.put("/orchestration")
def set_orchestration_settings(payload: OrchestrationSettingsBody):
    """Update orchestration knobs in config.yaml. Only fields explicitly passed
    are written; empty profile strings clear the override."""
    with _errors_to_500("failed to load config"):
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
    kanban_section = cfg.setdefault("kanban", {})
    if not isinstance(kanban_section, dict):
        kanban_section = cfg["kanban"] = {}
    try:
        from hermes_cli import profiles as profiles_mod
    except Exception:
        profiles_mod = None  # type: ignore
    # Field order == write order (profiles validated first, then the booleans).
    for key, value in payload.model_dump(exclude_none=True).items():
        kanban_section[key] = _validated_profile_name(value, profiles_mod) if key in _PROFILE_SETTINGS else bool(value)
    with _errors_to_500("failed to save config"):
        save_config(cfg)
    return get_orchestration_settings()  # callers re-render from the resolved state


# --- WebSocket: /events?since=<event_id>&board=<slug> ------------------------

# Event tail poll interval: WAL + 300 ms polling is the simplest robust approach (negligible CPU).
_EVENT_POLL_SECONDS = 0.3


def _int_param(ws: WebSocket, name: str) -> int:
    try:
        return int(ws.query_params.get(name, "0"))
    except ValueError:
        return 0


def _ws_board(raw: Optional[str]) -> Optional[str]:
    try:
        return kanban_db._normalize_board_slug(raw) if raw else None
    except ValueError:
        return None


class _EventTail:
    """Per-socket ``task_events`` tailer. One SQLite connection, used/closed only on a
    dedicated single-thread executor (connections are thread-affine); reusing it avoids
    churning WAL/SHM sidecars while an idle dashboard polls."""

    def __init__(self, board: Optional[str]) -> None:
        self._board = board
        self._conn: Optional[sqlite3.Connection] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def _fetch(self, cursor: int) -> tuple[int, list[dict]]:
        if self._conn is None:
            self._conn = kbc.connect(board=self._board)
        rows = self._conn.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at "
            "FROM task_events WHERE id > ? ORDER BY id ASC LIMIT 200",
            (cursor,)).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                payload = json.loads(r["payload"]) if r["payload"] else None
            except Exception:
                payload = None
            out.append({**dict(r), "payload": payload})
        return (rows[-1]["id"] if rows else cursor), out

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def poll(self, cursor: int) -> tuple[int, list[dict]]:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kanban-events")
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._fetch, cursor)

    async def shutdown(self) -> None:
        if self._executor is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._close)
        except Exception as exc:
            log.warning("Kanban event stream connection cleanup failed: %s", exc)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    # Board is pinned at the handshake; the UI opens a new WS on board change
    # rather than reconciling two cursors mid-stream.
    tail = _EventTail(_ws_board(ws.query_params.get("board")))
    cursor = _int_param(ws, "since")
    try:
        while True:
            # Race receive() against the poll interval so a disconnect is detected even when no
            # events flow (else idle boards leak poll tasks). Other client messages are ignored.
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=_EVENT_POLL_SECONDS)
                if msg["type"] == "websocket.disconnect":
                    return
            except asyncio.TimeoutError:
                pass  # no client message — poll the DB
            cursor, events = await tail.poll(cursor)
            if events:
                await ws.send_json({"events": events, "cursor": cursor})
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return  # normal shutdown; CancelledError is a BaseException the handler below wouldn't quiet
    except Exception as exc:  # never crash the dashboard worker
        log.warning("Kanban event stream error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        await tail.shutdown()
