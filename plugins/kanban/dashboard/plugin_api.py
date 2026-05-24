"""Kanban dashboard plugin — backend API routes.

Mounted at /api/plugins/kanban/ by the dashboard plugin system.

This layer is intentionally thin: every handler is a small wrapper around
``hermes_cli.kanban_db`` or a direct SQL query. Writes use the same code
paths the CLI and gateway ``/kanban`` command use, so the three surfaces
cannot drift.

Live updates arrive via the ``/events`` WebSocket, which tails the
append-only ``task_events`` table on a short poll interval (WAL mode lets
reads run alongside the dispatcher's IMMEDIATE write transactions).

Security note
-------------
Plugin HTTP routes go through the dashboard's session-token auth middleware
(``web_server.auth_middleware``) just like core API routes — every
``/api/plugins/...`` request must present the session bearer token (or the
session cookie set when you load the dashboard HTML). The token is the
random per-process ``_SESSION_TOKEN`` printed at startup; the dashboard's
own pages inject it via ``window.__HERMES_SESSION_TOKEN__`` so logged-in
browsers don't have to handle it manually.

For the ``/events`` WebSocket we still require the session token as a
``?token=`` query parameter (browsers cannot set the ``Authorization``
header on an upgrade request), matching the established pattern used by
the in-browser PTY bridge in ``hermes_cli/web_server.py``.

This means ``hermes dashboard --host 0.0.0.0`` is safe to run on a LAN:
plugin routes are no longer an unauthenticated exception. The auth still
isn't multi-user — anyone who can read the printed URL+token gets full
dashboard access — but they can't ride along just because they can reach
the port.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect, status as http_status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from hermes_cli import kanban_db
from hermes_cli import kanban_diagnostics as kd

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth helper — WebSocket only (HTTP routes live behind the dashboard's
# existing plugin-bypass; this is documented above).
# ---------------------------------------------------------------------------

def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Authorize a WebSocket upgrade by delegating to the dashboard's canonical
    WS auth gate (``hermes_cli.web_server._ws_auth_ok``).

    Delegating (rather than re-implementing a ``_SESSION_TOKEN``-only check)
    means this endpoint transparently accepts whatever the core gate accepts
    in each mode:

      * loopback / ``--insecure``: legacy ``?token=<_SESSION_TOKEN>``
      * gated OAuth: single-use ``?ticket=`` (the browser SDK's
        ``buildWsUrl`` mints one per connect)
      * server-internal: the process-lifetime ``?internal=`` credential

    The previous bespoke check only understood ``_SESSION_TOKEN``, so the
    kanban live-events WS was rejected on every OAuth-gated deployment even
    though the rest of the dashboard worked. Routing through the shared gate
    also means this can never drift from core auth again.

    Imported lazily so the plugin still loads in test contexts where the
    dashboard ``web_server`` module isn't importable (e.g. the bare-FastAPI
    test harness); there we accept so the tail loop stays testable, matching
    the prior behaviour.
    """
    try:
        from hermes_cli import web_server as _ws
    except Exception:
        # No dashboard context (tests). Accept so the tail loop is still
        # testable; in production the dashboard module always imports
        # cleanly because it's the caller.
        return True
    return bool(_ws._ws_auth_ok(ws))


def _resolve_board(board: Optional[str]) -> Optional[str]:
    """Validate and normalise a board slug from a query param.

    Raises :class:`HTTPException` 400 on malformed slugs so the browser
    sees a clean error instead of a 500. Returns the normalised slug,
    or ``None`` when the caller omitted the param (which then falls
    through to the active board inside ``kb.connect()``).
    """
    if board is None or board == "":
        return None
    try:
        normed = kanban_db._normalize_board_slug(board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if normed and normed != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(normed):
        raise HTTPException(
            status_code=404,
            detail=f"board {normed!r} does not exist",
        )
    return normed


def _conn(board: Optional[str] = None):
    """Open a kanban_db connection, creating the schema on first use.

    Every handler that mutates the DB goes through this so the plugin
    self-heals on a fresh install (no user-visible "no such table"
    error if somebody hits POST /tasks before GET /board).
    ``init_db`` is idempotent.

    ``board`` is the query-param slug (already normalised by
    :func:`_resolve_board`). When ``None`` the active board is used
    via the resolution chain (env var → ``current`` file → ``default``).
    """
    try:
        kanban_db.init_db(board=board)
    except Exception as exc:
        log.warning("kanban init_db failed: %s", exc)
    return kanban_db.connect(board=board)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

# Columns shown by the dashboard, in left-to-right order. "archived" is
# available via a filter toggle rather than a visible column.
#
# Keep this in sync with kanban_db.VALID_STATUSES.  In particular,
# ``scheduled`` is a first-class waiting column used for time-based follow-ups;
# if it is omitted here, the board-level fallback below mis-buckets scheduled
# tasks into ``todo`` and makes the dashboard look like the Scheduled column
# disappeared.
BOARD_COLUMNS: list[str] = [
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done",
]


_CARD_SUMMARY_PREVIEW_CHARS = 200


def _task_dict(
    task: kanban_db.Task,
    *,
    latest_summary: Optional[str] = None,
) -> dict[str, Any]:
    d = asdict(task)
    # Add derived age metrics so the UI can colour stale cards without
    # computing deltas client-side.
    try:
        d["age"] = kanban_db.task_age(task)
    except Exception:
        d["age"] = {"created_age_seconds": None, "started_age_seconds": None, "time_to_complete_seconds": None}
    # Surface the latest non-null run summary so dashboards don't show
    # blank cards/drawers for tasks where the worker handed off via
    # ``task_runs.summary`` (the kanban-worker pattern) instead of
    # ``tasks.result``. ``None`` when no run has produced a summary yet.
    d["latest_summary"] = latest_summary
    # Keep body short on list endpoints; full body comes from /tasks/:id.
    return d


def _event_dict(event: kanban_db.Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "task_id": event.task_id,
        "kind": event.kind,
        "payload": event.payload,
        "created_at": event.created_at,
        "run_id": event.run_id,
    }


def _comment_dict(c: kanban_db.Comment) -> dict[str, Any]:
    return {
        "id": c.id,
        "task_id": c.task_id,
        "author": c.author,
        "body": c.body,
        "created_at": c.created_at,
    }


def _attachment_dict(a: kanban_db.Attachment) -> dict[str, Any]:
    """Serialise an Attachment for the drawer. ``stored_path`` is the
    absolute on-disk path workers read; the UI uses ``id`` for download."""
    return {
        "id": a.id,
        "task_id": a.task_id,
        "filename": a.filename,
        "content_type": a.content_type,
        "size": a.size,
        "uploaded_by": a.uploaded_by,
        "stored_path": a.stored_path,
        "created_at": a.created_at,
    }


def _run_dict(r: kanban_db.Run) -> dict[str, Any]:
    """Serialise a Run for the drawer's Run history section."""
    return {
        "id": r.id,
        "task_id": r.task_id,
        "profile": r.profile,
        "step_key": r.step_key,
        "status": r.status,
        "claim_lock": r.claim_lock,
        "claim_expires": r.claim_expires,
        "worker_pid": r.worker_pid,
        "max_runtime_seconds": r.max_runtime_seconds,
        "last_heartbeat_at": r.last_heartbeat_at,
        "started_at": r.started_at,
        "ended_at": r.ended_at,
        "outcome": r.outcome,
        "summary": r.summary,
        "metadata": r.metadata,
        "error": r.error,
    }


# Hallucination-warning event kinds — see complete_task() in kanban_db.py.
# completion_blocked_hallucination: kernel rejected created_cards with
#   phantom ids; task stays in prior state.
# suspected_hallucinated_references: prose scan found t_<hex> in summary
#   that doesn't resolve; completion succeeded, advisory only.
_WARNING_EVENT_KINDS = (
    "completion_blocked_hallucination",
    "suspected_hallucinated_references",
)


def _compute_task_diagnostics(
    conn: sqlite3.Connection,
    task_ids: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Run the diagnostic rule engine against every task (or a subset)
    and return ``{task_id: [diagnostic_dict, ...]}``.

    Tasks with no active diagnostics are omitted from the result.
    Uses ``hermes_cli.kanban_diagnostics`` — see that module for the
    rule definitions.
    """
    from hermes_cli import kanban_diagnostics as kd
    from hermes_cli.config import load_config

    diag_config = kd.config_from_runtime_config(load_config())

    # Build the candidate task list. We need each task's row + its
    # events + its runs. Doing N separate queries works but scales
    # poorly; do three aggregate queries instead.
    if task_ids is not None:
        if not task_ids:
            return {}
        placeholders = ",".join(["?"] * len(task_ids))
        rows = conn.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders})",
            tuple(task_ids),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status != 'archived'",
        ).fetchall()

    if not rows:
        return {}

    # Index events + runs by task id. For very large boards this will
    # slurp a lot — acceptable on the dashboard's typical working set
    # (hundreds of tasks), but we can add pagination / filtering later
    # if profiling shows it's a hotspot.
    row_ids = [r["id"] for r in rows]
    placeholders = ",".join(["?"] * len(row_ids))
    events_by_task: dict[str, list] = {tid: [] for tid in row_ids}
    for ev_row in conn.execute(
        f"SELECT * FROM task_events WHERE task_id IN ({placeholders}) ORDER BY id",
        tuple(row_ids),
    ).fetchall():
        events_by_task.setdefault(ev_row["task_id"], []).append(ev_row)
    runs_by_task: dict[str, list] = {tid: [] for tid in row_ids}
    for run_row in conn.execute(
        f"SELECT * FROM task_runs WHERE task_id IN ({placeholders}) ORDER BY id",
        tuple(row_ids),
    ).fetchall():
        runs_by_task.setdefault(run_row["task_id"], []).append(run_row)

    out: dict[str, list[dict]] = {}
    for r in rows:
        tid = r["id"]
        diags = kd.compute_task_diagnostics(
            r,
            events_by_task.get(tid, []),
            runs_by_task.get(tid, []),
            config=diag_config,
        )
        if diags:
            out[tid] = [d.to_dict() for d in diags]
    return out


def _warnings_summary_from_diagnostics(
    diagnostics: list[dict],
) -> Optional[dict]:
    """Compact summary for cards: {count, highest_severity, kinds,
    latest_at}. Replaces the old hallucination-only ``warnings`` object
    — same shape additions plus ``highest_severity`` so the UI can color
    badges per diagnostic severity.

    Returns None when ``diagnostics`` is empty.
    """
    if not diagnostics:
        return None
    from hermes_cli.kanban_diagnostics import SEVERITY_ORDER

    kinds: dict[str, int] = {}
    latest = 0
    highest_idx = -1
    highest_sev: Optional[str] = None
    count = 0
    for d in diagnostics:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + d.get("count", 1)
        count += d.get("count", 1)
        la = d.get("last_seen_at") or 0
        if la > latest:
            latest = la
        sev = d.get("severity")
        if sev in SEVERITY_ORDER:
            idx = SEVERITY_ORDER.index(sev)
            if idx > highest_idx:
                highest_idx = idx
                highest_sev = sev
    return {
        "count": count,
        "kinds": kinds,
        "latest_at": latest,
        "highest_severity": highest_sev,
    }


def _links_for(conn: sqlite3.Connection, task_id: str) -> dict[str, list[str]]:
    """Return {'parents': [...], 'children': [...]} for a task."""
    parents = [
        r["parent_id"]
        for r in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
            (task_id,),
        )
    ]
    children = [
        r["child_id"]
        for r in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
            (task_id,),
        )
    ]
    return {"parents": parents, "children": children}


# ---------------------------------------------------------------------------
# Selected-task summary tree helpers
# ---------------------------------------------------------------------------

_ARTIFACT_KEYS: dict[str, str] = {
    "workspace_path": "workspace",
    "diff_path": "diff",
    "output_path": "output",
    "file_path": "file",
    "folder_path": "folder",
    "artifact_path": "artifact",
    "artifact_paths": "artifact",
    "document_path": "document",
    "created_files": "created_file",
    "changed_files": "changed_file",
    "saved_files": "saved_file",
}
_IMPORTANT_COMMENT_TERMS = (
    "review-required", "handoff", "artifact", "output", "saved",
    "created", "changed_files", "diff_path", "blocked", "decision",
)
_SUMMARY_TREE_MAX_DEPTH = 50
_SUMMARY_TREE_MAX_NODES = 500


def _walk_descendant_graph(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    max_depth: int = _SUMMARY_TREE_MAX_DEPTH,
    max_nodes: int = _SUMMARY_TREE_MAX_NODES,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int], bool]:
    """Return breadth-first descendant order and parent→child edges."""
    order: list[str] = [root_id]
    depths: dict[str, int] = {root_id: 0}
    edges: list[dict[str, Any]] = []
    queue: list[str] = [root_id]
    expanded: set[str] = set()
    truncated = False

    while queue:
        parent_id = queue.pop(0)
        if parent_id in expanded:
            continue
        expanded.add(parent_id)
        parent_depth = depths.get(parent_id, 0)
        if parent_depth >= max_depth:
            if conn.execute("SELECT 1 FROM task_links WHERE parent_id = ? LIMIT 1", (parent_id,)).fetchone():
                truncated = True
            continue
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
            (parent_id,),
        ).fetchall()
        for row in rows:
            child_id = row["child_id"]
            child_depth = parent_depth + 1
            edges.append({
                "parent_id": parent_id,
                "child_id": child_id,
                "relation": "blocks",
                "depth": child_depth,
            })
            if child_id not in depths:
                if len(order) >= max_nodes:
                    truncated = True
                    continue
                depths[child_id] = child_depth
                order.append(child_id)
                queue.append(child_id)
    return order, edges, depths, truncated


def _rows_by_task_id(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not task_ids:
        return {}
    placeholders = ",".join(["?"] * len(task_ids))
    return {
        row["id"]: row
        for row in conn.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders})",
            tuple(task_ids),
        ).fetchall()
    }


def _links_for_many(conn: sqlite3.Connection, task_ids: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    # Two queries so the bind count stays at N: interpolating the id list
    # twice into one statement hits 1000 variables at the traversal cap
    # (_SUMMARY_TREE_MAX_NODES), over the 999 default
    # SQLITE_MAX_VARIABLE_NUMBER on SQLite < 3.32.
    parents = {tid: [] for tid in task_ids}
    children = {tid: [] for tid in task_ids}
    if not task_ids:
        return parents, children
    placeholders = ",".join(["?"] * len(task_ids))
    for row in conn.execute(
        f"SELECT parent_id, child_id FROM task_links WHERE child_id IN ({placeholders}) ORDER BY parent_id, child_id",
        tuple(task_ids),
    ).fetchall():
        parents[row["child_id"]].append(row["parent_id"])
    for row in conn.execute(
        f"SELECT parent_id, child_id FROM task_links WHERE parent_id IN ({placeholders}) ORDER BY parent_id, child_id",
        tuple(task_ids),
    ).fetchall():
        children[row["parent_id"]].append(row["child_id"])
    return parents, children


def _run_row_dict(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
    except Exception:
        metadata = None
    return {
        "id": int(row["id"]),
        "task_id": row["task_id"],
        "profile": row["profile"],
        "step_key": row["step_key"],
        "status": row["status"],
        "outcome": row["outcome"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "summary": row["summary"],
        "metadata": metadata,
        "error": row["error"],
    }


def _runs_for_many(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    runs = {tid: [] for tid in task_ids}
    if not task_ids:
        return runs
    placeholders = ",".join(["?"] * len(task_ids))
    for row in conn.execute(
        f"""
        SELECT * FROM task_runs
         WHERE task_id IN ({placeholders})
         ORDER BY task_id, COALESCE(ended_at, started_at) DESC, id DESC
        """,
        tuple(task_ids),
    ).fetchall():
        runs.setdefault(row["task_id"], []).append(_run_row_dict(row))
    return runs


def _comments_for_many(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    comments = {tid: [] for tid in task_ids}
    if not task_ids:
        return comments
    placeholders = ",".join(["?"] * len(task_ids))
    for row in conn.execute(
        f"SELECT * FROM task_comments WHERE task_id IN ({placeholders}) ORDER BY task_id, created_at ASC, id ASC",
        tuple(task_ids),
    ).fetchall():
        comments.setdefault(row["task_id"], []).append({
            "id": row["id"],
            "task_id": row["task_id"],
            "author": row["author"],
            "body": row["body"],
            "created_at": row["created_at"],
        })
    return comments


def _attachments_for_many(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Batch-load first-class task attachments (``task_attachments`` rows)
    for every traversed task, mirroring ``kanban_db.list_attachments``
    ordering. One query for the whole tree instead of N per-task calls."""
    attachments = {tid: [] for tid in task_ids}
    if not task_ids:
        return attachments
    placeholders = ",".join(["?"] * len(task_ids))
    for row in conn.execute(
        f"SELECT * FROM task_attachments WHERE task_id IN ({placeholders}) ORDER BY task_id, created_at ASC, id ASC",
        tuple(task_ids),
    ).fetchall():
        attachments.setdefault(row["task_id"], []).append({
            "id": row["id"],
            "task_id": row["task_id"],
            "filename": row["filename"],
            "stored_path": row["stored_path"],
            "content_type": row["content_type"],
            "size": row["size"] or 0,
            "uploaded_by": row["uploaded_by"],
            "created_at": row["created_at"],
        })
    return attachments


def _events_for_many(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    events = {tid: [] for tid in task_ids}
    if not task_ids:
        return events
    placeholders = ",".join(["?"] * len(task_ids))
    for row in conn.execute(
        f"SELECT * FROM task_events WHERE task_id IN ({placeholders}) ORDER BY task_id, created_at ASC, id ASC",
        tuple(task_ids),
    ).fetchall():
        try:
            payload = json.loads(row["payload"]) if row["payload"] else None
        except Exception:
            payload = None
        events.setdefault(row["task_id"], []).append({
            "id": row["id"],
            "task_id": row["task_id"],
            "kind": row["kind"],
            "payload": payload,
            "created_at": row["created_at"],
            "run_id": row["run_id"],
        })
    return events


def _blocked_state(
    task_status: str,
    *,
    runs: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if task_status != "blocked":
        return None
    latest_block = next((e for e in reversed(events) if e.get("kind") == "blocked"), None)
    blocked_runs = [r for r in runs if r.get("outcome") == "blocked" or r.get("status") == "blocked"]
    blocked_runs.sort(
        key=lambda r: (int(r.get("ended_at") or r.get("started_at") or 0), int(r.get("id") or 0)),
        reverse=True,
    )
    latest_blocked_run = blocked_runs[0] if blocked_runs else None
    reason = None
    source = None
    if latest_block and isinstance(latest_block.get("payload"), dict):
        raw_reason = latest_block["payload"].get("reason")
        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = raw_reason.strip()
            source = "event.payload.reason"
    if not reason and latest_blocked_run:
        raw_summary = latest_blocked_run.get("summary") or latest_blocked_run.get("error")
        if isinstance(raw_summary, str) and raw_summary.strip():
            reason = raw_summary.strip()
            source = "run.summary"
    since = latest_block.get("created_at") if latest_block else None
    relevant_comments = [c for c in comments if since is None or int(c.get("created_at") or 0) >= int(since)]
    latest_comment = relevant_comments[-1] if relevant_comments else (comments[-1] if comments else None)
    return {
        "is_blocked": True,
        "reason": reason,
        "missing_info": reason,
        "blocked_at": latest_block.get("created_at") if latest_block else None,
        "event_id": latest_block.get("id") if latest_block else None,
        "run_id": latest_block.get("run_id") if latest_block else (latest_blocked_run or {}).get("id"),
        "source": source,
        "latest_relevant_comment": latest_comment,
        "comment_prompt": "Add the missing info as a comment, then unblock when ready.",
    }


def _important_comments(comments: list[dict[str, Any]], *, comment_limit: int) -> list[dict[str, Any]]:
    """Newest comments whose body matches an importance term, capped at
    ``comment_limit``, returned in chronological order.

    No back-fill with ordinary comments: callers already get the recent
    tail in ``comments``, so padding this list would make the two fields
    redundant and leave the UI's recent-comments fallback unreachable.
    """
    if comment_limit <= 0:
        return []
    picked: list[dict[str, Any]] = []
    for c in reversed(comments):
        body = (c.get("body") or "").lower()
        if any(term in body for term in _IMPORTANT_COMMENT_TERMS):
            picked.append(c)
            if len(picked) >= comment_limit:
                break
    return list(reversed(picked))


def _safe_path_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\x00" in text or "\n" in text or len(text) > 4096:
        return None
    return text


def _path_is_within(path: str, base: str) -> bool:
    try:
        base_resolved = Path(os.path.expanduser(base)).resolve(strict=False)
        candidate_resolved = Path(os.path.expanduser(path)).resolve(strict=False)
        candidate_resolved.relative_to(base_resolved)
        return True
    except Exception:
        return False


def _artifact_record(
    raw_path: str,
    *,
    kind: str,
    source: str,
    workspace_path: Optional[str],
    workspace_kind: Optional[str],
    run_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    text = _safe_path_text(raw_path)
    if not text:
        return None

    resolved_path = None
    exists = None
    is_dir = None
    availability = "unknown"
    reason = None
    public_path: Optional[str] = text

    expanded = os.path.expanduser(text)
    is_absolute = os.path.isabs(expanded) or (len(expanded) > 2 and expanded[1:3] in (":/", ":\\"))
    if workspace_path and kind != "workspace":
        try:
            base = Path(os.path.expanduser(workspace_path)).resolve(strict=False)
            candidate = Path(expanded) if is_absolute else base / expanded
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(base)
            resolved_path = str(resolved)
            exists = resolved.exists()
            is_dir = resolved.is_dir() if exists else None
            availability = "available" if exists else "missing"
            if not exists:
                reason = "path does not exist"
        except Exception:
            availability = "outside_workspace"
            public_path = None
            resolved_path = None
            reason = "path outside allowed workspace"
    elif kind == "workspace" and text:
        try:
            resolved = Path(expanded).resolve(strict=False)
            resolved_path = str(resolved)
            exists = resolved.exists()
            is_dir = resolved.is_dir() if exists else None
            availability = "available" if exists else "missing"
        except Exception:
            availability = "unknown"
            reason = "path could not be normalized"
    elif is_absolute:
        availability = "outside_workspace"
        public_path = None
        reason = "absolute path; no workspace base"

    if workspace_kind == "scratch":
        availability = "scratch"
        public_path = None
        resolved_path = None
        reason = "scratch workspace is temporary; not a durable artifact"

    return {
        "path": public_path,
        "resolved_path": resolved_path,
        "label": public_path or kind.replace("_", " "),
        "kind": kind,
        "exists": exists,
        "is_dir": is_dir,
        "openable": False,
        "user_actionable": availability == "available" and workspace_kind != "scratch",
        "availability": availability,
        "source": source,
        "run_id": run_id,
        "reason": reason,
    }


def _values_for_artifact_key(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        text = _safe_path_text(value)
        if text:
            found.append(text)
    elif isinstance(value, list):
        for item in value:
            found.extend(_values_for_artifact_key(item))
    elif isinstance(value, dict):
        for key in ("path", "resolved_path", "file", "folder"):
            if key in value:
                found.extend(_values_for_artifact_key(value[key]))
        if not found:
            for item in value.values():
                found.extend(_values_for_artifact_key(item))
    return found


def _iter_artifact_metadata_values(value: Any, *, prefix: str = "run.metadata"):
    if isinstance(value, dict):
        for key, item in value.items():
            key_s = str(key)
            source = f"{prefix}.{key_s}"
            if key_s in _ARTIFACT_KEYS:
                for path_value in _values_for_artifact_key(item):
                    yield key_s, source, path_value
            if isinstance(item, (dict, list)):
                yield from _iter_artifact_metadata_values(item, prefix=source)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from _iter_artifact_metadata_values(item, prefix=prefix)


def _attachment_artifact_record(att: dict[str, Any]) -> dict[str, Any]:
    """Read-only artifact record for an uploaded task attachment.

    Attachments are first-class ``task_attachments`` rows whose blobs live
    under the board's attachments root — not the task workspace — so they
    skip the workspace containment check applied to run-metadata paths.
    ``attachment_id`` lets the UI reuse the existing authenticated
    ``GET /attachments/{id}`` download route; there is still no local-file
    opener (``openable`` stays False).
    """
    stored_path = att.get("stored_path")
    exists = bool(stored_path) and Path(stored_path).is_file()
    return {
        "path": att.get("filename"),
        "resolved_path": stored_path,
        "label": att.get("filename") or f"attachment {att.get('id')}",
        "kind": "attachment",
        "exists": exists,
        "is_dir": False if exists else None,
        "openable": False,
        "user_actionable": exists,
        "availability": "available" if exists else "missing",
        "source": "task.attachment",
        "run_id": None,
        "reason": None if exists else "attachment blob missing on disk",
        "attachment_id": att.get("id"),
        "size": att.get("size"),
        "content_type": att.get("content_type"),
        "uploaded_by": att.get("uploaded_by"),
    }


def _extract_artifacts(
    *,
    task_row: sqlite3.Row,
    runs: list[dict[str, Any]],
    attachments: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    workspace_path = task_row["workspace_path"]
    workspace_kind = task_row["workspace_kind"]

    def add(record: Optional[dict[str, Any]]) -> None:
        if not record:
            return
        key = (str(record.get("source")), str(record.get("kind")), str(record.get("resolved_path") or record.get("path") or record.get("label")))
        if key in seen:
            return
        seen.add(key)
        artifacts.append(record)

    if workspace_path:
        add(_artifact_record(
            workspace_path,
            kind="workspace",
            source="task.workspace_path",
            workspace_path=None,
            workspace_kind=workspace_kind,
        ))

    for att in attachments or []:
        add(_attachment_artifact_record(att))

    for run in runs:
        metadata = run.get("metadata")
        if not metadata:
            continue
        for artifact_key, source, path_value in _iter_artifact_metadata_values(metadata):
            add(_artifact_record(
                path_value,
                kind=_ARTIFACT_KEYS.get(artifact_key, "artifact"),
                source=source,
                workspace_path=workspace_path,
                workspace_kind=workspace_kind,
                run_id=run.get("id"),
            ))
    return artifacts


def _artifact_state(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    actionable = [a for a in artifacts if a.get("user_actionable")]
    return {
        "state": "available" if actionable else "absent",
        "has_user_actionable": bool(actionable),
        "user_actionable_count": len(actionable),
        "candidate_count": len(artifacts),
        "reason": None if actionable else (
            artifacts[0].get("reason") if artifacts else "no artifact metadata found"
        ),
    }


def _display_result(task_row: sqlite3.Row, latest_run: Optional[dict[str, Any]]) -> Optional[str]:
    if task_row["result"]:
        return task_row["result"]
    if latest_run and latest_run.get("summary"):
        return latest_run["summary"]
    if latest_run and latest_run.get("error"):
        outcome = latest_run.get("outcome") or latest_run.get("status") or "failed"
        return f"{outcome}: {latest_run['error']}"
    return None


def _summary_tree_payload(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    include_comments: bool = True,
    comment_limit: int = 3,
) -> dict[str, Any]:
    if kanban_db.get_task(conn, root_id) is None:
        raise HTTPException(status_code=404, detail=f"task {root_id} not found")
    comment_limit = max(0, min(int(comment_limit), 20))
    order, edges, depths, truncated = _walk_descendant_graph(conn, root_id)
    rows = _rows_by_task_id(conn, order)
    order = [tid for tid in order if tid in rows]
    parents, children = _links_for_many(conn, order)
    runs_by_task = _runs_for_many(conn, order)
    comments_by_task = _comments_for_many(conn, order)
    events_by_task = _events_for_many(conn, order)
    attachments_by_task = _attachments_for_many(conn, order)

    tasks: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for tid in order:
        row = rows[tid]
        task_runs = runs_by_task.get(tid, [])
        latest_run = task_runs[0] if task_runs else None
        all_comments = comments_by_task.get(tid, [])
        task_attachments = attachments_by_task.get(tid, [])
        artifacts = _extract_artifacts(
            task_row=row, runs=task_runs, attachments=task_attachments,
        )
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        tasks[tid] = {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "status": row["status"],
            "assignee": row["assignee"],
            "priority": row["priority"],
            "tenant": row["tenant"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "workspace_kind": row["workspace_kind"],
            "workspace_path": row["workspace_path"],
            "result": row["result"],
            "current_step_key": row["current_step_key"] if "current_step_key" in row.keys() else None,
            "workflow_template_id": row["workflow_template_id"] if "workflow_template_id" in row.keys() else None,
            "latest_summary": latest_run.get("summary") if latest_run else None,
            "display_result": _display_result(row, latest_run),
            "latest_run": latest_run,
            "run_count": len(task_runs),
            "comments": all_comments[-comment_limit:] if include_comments and comment_limit else [],
            "comment_count": len(all_comments),
            "important_comments": _important_comments(all_comments, comment_limit=comment_limit) if include_comments else [],
            "parents": parents.get(tid, []),
            "children": children.get(tid, []),
            "depth": depths.get(tid, 0),
            "artifacts": artifacts,
            "artifact_state": _artifact_state(artifacts),
            "attachment_count": len(task_attachments),
            "block": _blocked_state(
                row["status"],
                runs=task_runs,
                comments=all_comments,
                events=events_by_task.get(tid, []),
            ),
        }

    stats: dict[str, Any] = {
        "total": len(order),
        "max_depth": max((depths.get(tid, 0) for tid in order), default=0),
        "truncated": truncated,
    }
    for status_name in list(BOARD_COLUMNS) + ["archived"]:
        stats[status_name] = status_counts.get(status_name, 0)
    for status_name, count in status_counts.items():
        stats.setdefault(status_name, count)
    return {
        "root_id": root_id,
        "generated_at": int(time.time()),
        "tasks": tasks,
        "edges": [e for e in edges if e["parent_id"] in rows and e["child_id"] in rows],
        "roots": [root_id],
        "order": order,
        "stats": stats,
    }

# ---------------------------------------------------------------------------
# GET /board
# ---------------------------------------------------------------------------

@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None, description="Filter to a single tenant"),
    include_archived: bool = Query(False),
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
    workflow_template_id: Optional[str] = Query(
        None, description="Restrict to tasks using this workflow template id",
    ),
    current_step_key: Optional[str] = Query(
        None, description="Restrict to tasks at this workflow step key",
    ),
):
    """Return the full board grouped by status column.

    ``_conn()`` auto-initializes ``kanban.db`` on first call so a fresh
    install doesn't surface a "failed to load" error on the plugin tab.

    ``board`` selects which board to read from. Omitting it falls
    through to the active board (``HERMES_KANBAN_BOARD`` env → on-disk
    ``current`` pointer → ``default``).
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        tasks = kanban_db.list_tasks(
            conn,
            tenant=tenant,
            include_archived=include_archived,
            workflow_template_id=workflow_template_id,
            current_step_key=current_step_key,
        )
        # Pre-fetch link counts per task (cheap: one query).
        link_counts: dict[str, dict[str, int]] = {}
        for row in conn.execute(
            "SELECT parent_id, child_id FROM task_links"
        ).fetchall():
            link_counts.setdefault(row["parent_id"], {"parents": 0, "children": 0})[
                "children"
            ] += 1
            link_counts.setdefault(row["child_id"], {"parents": 0, "children": 0})[
                "parents"
            ] += 1

        # Comment + event counts (both cheap aggregates).
        comment_counts: dict[str, int] = {
            r["task_id"]: r["n"]
            for r in conn.execute(
                "SELECT task_id, COUNT(*) AS n FROM task_comments GROUP BY task_id"
            )
        }

        # Progress rollup: for each parent, how many children are done / total.
        # One pass over task_links joined with child status — cheaper than
        # N per-task queries and the plugin uses it to render "N/M".
        progress: dict[str, dict[str, int]] = {}
        for row in conn.execute(
            "SELECT l.parent_id AS pid, t.status AS cstatus "
            "FROM task_links l JOIN tasks t ON t.id = l.child_id"
        ).fetchall():
            p = progress.setdefault(row["pid"], {"done": 0, "total": 0})
            p["total"] += 1
            if row["cstatus"] == "done":
                p["done"] += 1

        # Diagnostics rollup for this board — see kanban_diagnostics.
        # We get the full structured list per task AND a compact
        # summary for the card badge (so cards don't carry the detail
        # text; the drawer fetches that via /tasks/:id or /diagnostics).
        diagnostics_per_task = _compute_task_diagnostics(conn, task_ids=None)

        latest_event_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM task_events"
        ).fetchone()["m"]

        columns: dict[str, list[dict]] = {c: [] for c in BOARD_COLUMNS}
        if include_archived:
            columns["archived"] = []

        # Batch-fetch the latest non-null run summary per task in one
        # window-function query (avoids N+1 ``latest_summary`` calls
        # for boards with hundreds of tasks). Truncated to a card-size
        # preview here — the full text is available via /tasks/:id.
        summary_map = kanban_db.latest_summaries(conn, [t.id for t in tasks])

        for t in tasks:
            full = summary_map.get(t.id)
            preview = (
                full[:_CARD_SUMMARY_PREVIEW_CHARS] if full else None
            )
            d = _task_dict(t, latest_summary=preview)
            d["link_counts"] = link_counts.get(t.id, {"parents": 0, "children": 0})
            d["comment_count"] = comment_counts.get(t.id, 0)
            d["progress"] = progress.get(t.id)  # None when the task has no children
            diags = diagnostics_per_task.get(t.id)
            if diags:
                # Full list goes into the payload so the drawer can render
                # without a second round-trip. The board-level badge only
                # needs the summary.
                d["diagnostics"] = diags
                d["warnings"] = _warnings_summary_from_diagnostics(diags)
            col = t.status if t.status in columns else "todo"
            columns[col].append(d)

        # Stable per-column ordering already applied by list_tasks
        # (priority DESC, created_at ASC), keep as-is.

        # List of known tenants for the UI filter dropdown.
        tenants = [
            r["tenant"]
            for r in conn.execute(
                "SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant"
            )
        ]
        # List of distinct assignees for the lane-by-profile sub-grouping.
        assignees = [
            r["assignee"]
            for r in conn.execute(
                "SELECT DISTINCT assignee FROM tasks WHERE assignee IS NOT NULL "
                "AND status != 'archived' ORDER BY assignee"
            )
        ]

        return {
            "columns": [
                {"name": name, "tasks": columns[name]} for name in columns.keys()
            ],
            "tenants": tenants,
            "assignees": assignees,
            "latest_event_id": int(latest_event_id),
            "now": int(time.time()),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GET /tasks/:id/summary-tree
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}/summary-tree")
def get_task_summary_tree(
    task_id: str,
    include_comments: bool = Query(True),
    comment_limit: int = Query(3, ge=0, le=20),
    depth: str = Query("all", description="Currently accepts 'all'; traversal is capped defensively."),
    board: Optional[str] = Query(None),
):
    _ = depth
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        return _summary_tree_payload(
            conn,
            task_id,
            include_comments=include_comments,
            comment_limit=comment_limit,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GET /tasks/:id
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    board: Optional[str] = Query(None),
    run_state_type: Optional[str] = Query(
        None, description="With run_state_name: filter runs by column 'status' or 'outcome'",
    ),
    run_state_name: Optional[str] = Query(
        None, description="With run_state_type: exact value for that run column",
    ),
):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        if (run_state_type is None) ^ (run_state_name is None):
            raise HTTPException(
                status_code=400,
                detail="run_state_type and run_state_name must be passed together or omitted",
            )
        if run_state_type is not None and run_state_type not in ("status", "outcome"):
            raise HTTPException(
                status_code=400,
                detail="run_state_type must be 'status' or 'outcome'",
            )
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        # Drawer/detail view returns the FULL summary (no truncation) so
        # operators can read the complete worker handoff without making
        # a second round-trip. Cards on /board carry a 200-char preview.
        full_summary = kanban_db.latest_summary(conn, task_id)
        task_d = _task_dict(task, latest_summary=full_summary)
        links = _links_for(conn, task_id)
        child_ids = links["children"]
        child_summaries = kanban_db.latest_summaries(conn, child_ids)
        child_results = []
        for child_id in child_ids:
            child = kanban_db.get_task(conn, child_id)
            if child is None:
                continue
            child_results.append({
                "id": child.id,
                "title": child.title,
                "status": child.status,
                "latest_summary": child_summaries.get(child.id),
                "result": child.result,
            })
        # Attach diagnostics so the drawer's Diagnostics section can
        # render recovery actions without a second round-trip.
        diags = _compute_task_diagnostics(conn, task_ids=[task_id])
        diag_list = diags.get(task_id) or []
        if diag_list:
            task_d["diagnostics"] = diag_list
            task_d["warnings"] = _warnings_summary_from_diagnostics(diag_list)
        return {
            "task": task_d,
            "comments": [_comment_dict(c) for c in kanban_db.list_comments(conn, task_id)],
            "events": [_event_dict(e) for e in kanban_db.list_events(conn, task_id)],
            "attachments": [_attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)],
            "links": links,
            "child_results": child_results,
            "runs": [
                _run_dict(r)
                for r in kanban_db.list_runs(
                    conn,
                    task_id,
                    state_type=run_state_type,
                    state_name=run_state_name,
                )
            ],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# POST /tasks
# ---------------------------------------------------------------------------

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


@router.post("/tasks")
def create_task(payload: CreateTaskBody, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task_id = kanban_db.create_task(
            conn,
            title=payload.title,
            body=payload.body,
            assignee=payload.assignee,
            created_by="dashboard",
            workspace_kind=payload.workspace_kind,
            workspace_path=payload.workspace_path,
            tenant=payload.tenant,
            priority=payload.priority,
            parents=payload.parents,
            triage=payload.triage,
            idempotency_key=payload.idempotency_key,
            max_runtime_seconds=payload.max_runtime_seconds,
            skills=payload.skills,
            goal_mode=payload.goal_mode,
            goal_max_turns=payload.goal_max_turns,
        )
        task = kanban_db.get_task(conn, task_id)
        body: dict[str, Any] = {"task": _task_dict(task) if task else None}
        # Surface a dispatcher-presence warning so the UI can show a
        # banner when a `ready` task would otherwise sit idle because no
        # gateway is running (or dispatch_in_gateway=false). Only emit
        # for ready+assigned tasks; triage/todo are expected to wait,
        # and unassigned tasks can't be dispatched regardless.
        if task and task.status == "ready" and task.assignee:
            try:
                from hermes_cli.kanban import _check_dispatcher_presence
                running, message = _check_dispatcher_presence()
                if not running and message:
                    body["warning"] = message
            except Exception:
                # Probe failure must never block the create itself.
                pass
        return body
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Attachments — upload / list / download / delete (#35338)
# ---------------------------------------------------------------------------

# Cap a single upload so a runaway request can't fill the disk. 25 MB
# comfortably covers PDFs, images, and source docs — the kanban use case.
_MAX_ATTACHMENT_BYTES = kanban_db.KANBAN_ATTACHMENT_MAX_BYTES


def _safe_attachment_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory components (``os.path.basename`` on both
    separators) so a malicious ``../../etc/passwd`` or ``C:\\x`` collapses
    to its leaf. Rejects empty / dotfile-only names. The result is only
    ever joined under the per-task attachments dir, never used verbatim
    as a path from the client.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    # Drop control chars and leading dots so we never write a dotfile or
    # a name with embedded NULs/newlines.
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '\x00').strip()
    name = name.lstrip(".").strip()
    if not name:
        raise HTTPException(status_code=400, detail="invalid attachment filename")
    return name[:200]


@router.get("/tasks/{task_id}/attachments")
def list_task_attachments(task_id: str, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {
            "attachments": [
                _attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)
            ]
        }
    finally:
        conn.close()


@router.post("/tasks/{task_id}/attachments")
async def upload_task_attachment(
    task_id: str,
    file: UploadFile = File(...),
    board: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Form(None),
):
    """Store an uploaded file for a task and record its metadata.

    The blob lands under ``attachments_root(board)/<task_id>/`` with a
    sanitised, collision-resolved name. The worker reads it via the
    absolute path surfaced in ``build_worker_context``.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")

        safe_name = _safe_attachment_name(file.filename or "")

        # Stream to disk with a hard size cap so a huge upload can't fill
        # the disk. Read in chunks; abort + clean up if the cap is hit.
        dest_dir = kanban_db.task_attachments_dir(task_id, board=board)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Resolve name collisions: foo.pdf → foo (1).pdf, foo (2).pdf, …
        stem, dot, ext = safe_name.partition(".")
        candidate = safe_name
        n = 1
        while (dest_dir / candidate).exists():
            candidate = f"{stem} ({n}){dot}{ext}"
            n += 1
        dest_path = dest_dir / candidate

        total = 0
        try:
            with open(dest_path, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_ATTACHMENT_BYTES:
                        out.close()
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"attachment exceeds {_MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit"
                            ),
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"failed to store attachment: {exc}")

        att_id = kanban_db.add_attachment(
            conn,
            task_id,
            filename=candidate,
            stored_path=str(dest_path.resolve()),
            content_type=file.content_type,
            size=total,
            uploaded_by=(uploaded_by or "dashboard"),
        )
        att = kanban_db.get_attachment(conn, att_id)
        return {"attachment": _attachment_dict(att) if att else None}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        att = kanban_db.get_attachment(conn, attachment_id)
        if att is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        # Confirm the blob still lives under the board's attachments root
        # before serving — defense in depth against a tampered DB row.
        root = kanban_db.attachments_root(board=board).resolve()
        try:
            stored = Path(att.stored_path).resolve()
            stored.relative_to(root)
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="attachment file unavailable")
        if not stored.is_file():
            raise HTTPException(status_code=404, detail="attachment file missing on disk")
        return FileResponse(
            path=str(stored),
            filename=att.filename,
            media_type=att.content_type or "application/octet-stream",
        )
    finally:
        conn.close()


@router.delete("/attachments/{attachment_id}")
def remove_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        att = kanban_db.delete_attachment(conn, attachment_id)
        if att is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        return {"ok": True, "id": attachment_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# PATCH /tasks/:id  (status / assignee / priority / title / body)
# ---------------------------------------------------------------------------

class UpdateTaskBody(BaseModel):
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    result: Optional[str] = None
    block_reason: Optional[str] = None
    # Structured handoff fields — forwarded to complete_task when status
    # transitions to 'done'. Dashboard parity with ``hermes kanban
    # complete --summary ... --metadata ...``.
    summary: Optional[str] = None
    metadata: Optional[dict] = None


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: UpdateTaskBody, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")

        # --- assignee ----------------------------------------------------
        if payload.assignee is not None:
            try:
                ok = kanban_db.assign_task(
                    conn, task_id, payload.assignee or None,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e))
            if not ok:
                raise HTTPException(status_code=404, detail="task not found")

        # --- status -------------------------------------------------------
        if payload.status is not None:
            s = payload.status
            ok = True
            if s == "done":
                ok = kanban_db.complete_task(
                    conn, task_id,
                    result=payload.result,
                    summary=payload.summary,
                    metadata=payload.metadata,
                )
            elif s == "blocked":
                ok = kanban_db.block_task(conn, task_id, reason=payload.block_reason)
            elif s == "scheduled":
                ok = kanban_db.schedule_task(conn, task_id, reason=payload.block_reason)
            elif s == "ready":
                # Re-open a blocked/scheduled task, or just an explicit status set.
                current = kanban_db.get_task(conn, task_id)
                if current and current.status in ("blocked", "scheduled"):
                    ok = kanban_db.unblock_task(conn, task_id)
                else:
                    # Direct status write for drag-drop (todo -> ready etc).
                    ok = _set_status_direct(conn, task_id, "ready")
            elif s == "archived":
                ok = kanban_db.archive_task(conn, task_id)
            elif s == "running":
                raise HTTPException(
                    status_code=400,
                    detail="Cannot set status to 'running' directly; use the dispatcher/claim path",
                )
            elif s in ("todo", "triage", "scheduled"):
                ok = _set_status_direct(conn, task_id, s)
            else:
                raise HTTPException(status_code=400, detail=f"unknown status: {s}")
            if not ok:
                # For ``ready``, name the blocking parent(s) so the dashboard
                # can render an actionable toast instead of a silent no-op.
                # See #26744.
                if s == "ready":
                    blockers = _parents_blocking_ready(conn, task_id)
                    if blockers:
                        names = ", ".join(
                            f"{p['title']!r} ({p['id']}, status={p['status']})"
                            for p in blockers
                        )
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"Cannot move to 'ready': blocked by parent(s) "
                                f"not done — {names}"
                            ),
                        )
                raise HTTPException(
                    status_code=409,
                    detail=f"status transition to {s!r} not valid from current state",
                )

        # --- priority -----------------------------------------------------
        if payload.priority is not None:
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET priority = ? WHERE id = ?",
                    (int(payload.priority), task_id),
                )
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, 'reprioritized', ?, ?)",
                    (task_id, json.dumps({"priority": int(payload.priority)}),
                     int(time.time())),
                )

        # --- title / body -------------------------------------------------
        if payload.title is not None or payload.body is not None:
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
                conn.execute(
                    f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", vals,
                )
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, 'edited', NULL, ?)",
                    (task_id, int(time.time())),
                )

        updated = kanban_db.get_task(conn, task_id)
        return {"task": _task_dict(updated) if updated else None}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DELETE /tasks/:id
# ---------------------------------------------------------------------------

@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        ok = kanban_db.delete_task(conn, task_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {"deleted": True, "task_id": task_id}
    finally:
        conn.close()


def _parents_blocking_ready(
    conn: sqlite3.Connection, task_id: str,
) -> list:
    """Return parent rows (``id``, ``title``, ``status``) that aren't ``done``
    and therefore prevent ``task_id`` from being promoted to ``ready``.

    Used to enrich the 409 response from :func:`update_task` so the
    dashboard can show an actionable toast (#26744) instead of a silent
    no-op.  Returns ``[]`` when nothing blocks the transition (e.g. no
    parents, or all parents already done).
    """
    rows = conn.execute(
        "SELECT t.id, t.title, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? AND t.status != 'done'",
        (task_id,),
    ).fetchall()
    return [
        {"id": r["id"], "title": r["title"], "status": r["status"]}
        for r in rows
    ]


def _set_status_direct(
    conn: sqlite3.Connection, task_id: str, new_status: str,
) -> bool:
    """Direct status write for drag-drop moves that aren't covered by the
    structured complete/block/unblock/archive verbs (e.g. todo<->ready,
    running<->ready). Appends a ``status`` event row for the live feed.

    When this transitions OFF ``running`` to anything other than the
    terminal verbs above (which own their own run closing), we close the
    active run with outcome='reclaimed' so attempt history isn't
    orphaned. ``running -> ready`` via drag-drop is the common case
    (user yanking a stuck worker back to the queue).
    """
    with kanban_db.write_txn(conn):
        # Snapshot current state so we know whether to close a run.
        prev = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if prev is None:
            return False

        # Guard: don't allow promoting to 'ready' unless all parents are done.
        # Prevents the dispatcher from spawning a child whose upstream work
        # hasn't completed (e.g. T4 dispatched while T3 is still blocked).
        if new_status == "ready":
            parent_statuses = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if parent_statuses and not all(
                p["status"] == "done" for p in parent_statuses
            ):
                return False

        was_running = prev["status"] == "running"
        reopening_satisfied_parent = (
            prev["status"] in {"done", "archived"}
            and new_status not in {"done", "archived"}
        )

        cur = conn.execute(
            "UPDATE tasks SET status = ?, "
            "  claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, "
            "  claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, "
            "  worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END "
            "WHERE id = ?",
            (new_status, new_status, new_status, new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        run_id = None
        if was_running and new_status != "running" and prev["current_run_id"]:
            run_id = kanban_db._end_run(
                conn, task_id,
                outcome="reclaimed", status="reclaimed",
                summary=f"status changed to {new_status} (dashboard/direct)",
            )
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'status', ?, ?)",
            (task_id, run_id, json.dumps({"status": new_status}), int(time.time())),
        )
        if reopening_satisfied_parent:
            # A parent leaving done/archived invalidates any direct child that
            # was sitting in ready solely because that parent used to satisfy
            # the dependency gate. Demote those children immediately so the
            # dashboard does not keep advertising stale-ready work.
            for row in conn.execute(
                "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
                (task_id,),
            ).fetchall():
                child_id = row["child_id"]
                demoted = conn.execute(
                    "UPDATE tasks SET status = 'todo' "
                    "WHERE id = ? AND status = 'ready'",
                    (child_id,),
                )
                if demoted.rowcount == 1:
                    conn.execute(
                        "INSERT INTO task_events (task_id, kind, payload, created_at) "
                        "VALUES (?, 'status', ?, ?)",
                        (
                            child_id,
                            json.dumps(
                                {
                                    "status": "todo",
                                    "reason": "parent_reopened",
                                    "parent": task_id,
                                }
                            ),
                            int(time.time()),
                        ),
                    )
    # If we re-opened something, children may have gone stale.
    if new_status in {"done", "ready"}:
        kanban_db.recompute_ready(conn)
    return True


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class CommentBody(BaseModel):
    body: str
    author: Optional[str] = "dashboard"


@router.post("/tasks/{task_id}/comments")
def add_comment(task_id: str, payload: CommentBody, board: Optional[str] = Query(None)):
    if not payload.body.strip():
        raise HTTPException(status_code=400, detail="body is required")
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        kanban_db.add_comment(
            conn, task_id, author=payload.author or "dashboard", body=payload.body,
        )
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

class LinkBody(BaseModel):
    parent_id: str
    child_id: str


@router.post("/links")
def add_link(payload: LinkBody, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        kanban_db.link_tasks(conn, payload.parent_id, payload.child_id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.delete("/links")
def delete_link(
    parent_id: str = Query(...),
    child_id: str = Query(...),
    board: Optional[str] = Query(None),
):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        ok = kanban_db.unlink_tasks(conn, parent_id, child_id)
        return {"ok": bool(ok)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bulk actions (multi-select on the board)
# ---------------------------------------------------------------------------

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


@router.post("/tasks/bulk")
def bulk_update(payload: BulkTaskBody, board: Optional[str] = Query(None)):
    """Apply the same patch to every id in ``payload.ids``.

    This is an *independent* iteration — per-task failures don't abort
    siblings. Returns per-id outcome so the UI can surface partials.
    """
    ids = [i for i in (payload.ids or []) if i]
    if not ids:
        raise HTTPException(status_code=400, detail="ids is required")
    results: list[dict] = []
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        for tid in ids:
            entry: dict[str, Any] = {"id": tid, "ok": True}
            try:
                task = kanban_db.get_task(conn, tid)
                if task is None:
                    entry.update(ok=False, error="not found")
                    results.append(entry)
                    continue
                if payload.archive:
                    if not kanban_db.archive_task(conn, tid):
                        entry.update(ok=False, error="archive refused")
                if payload.status is not None and not payload.archive:
                    s = payload.status
                    if s == "done":
                        ok = kanban_db.complete_task(
                            conn, tid,
                            result=payload.result,
                            summary=payload.summary,
                            metadata=payload.metadata,
                        )
                    elif s == "blocked":
                        ok = kanban_db.block_task(conn, tid)
                    elif s == "ready":
                        cur = kanban_db.get_task(conn, tid)
                        if cur and cur.status in ("blocked", "scheduled"):
                            ok = kanban_db.unblock_task(conn, tid)
                        else:
                            ok = _set_status_direct(conn, tid, "ready")
                    elif s == "running":
                        entry.update(
                            ok=False,
                            error=(
                                "Cannot set status to 'running' directly; "
                                "use the dispatcher/claim path"
                            ),
                        )
                        results.append(entry)
                        continue
                    elif s == "scheduled":
                        ok = kanban_db.schedule_task(conn, tid)
                    elif s in {"todo", "triage"}:
                        ok = _set_status_direct(conn, tid, s)
                    else:
                        entry.update(ok=False, error=f"unknown status {s!r}")
                        results.append(entry)
                        continue
                    if not ok:
                        entry.update(ok=False, error=f"transition to {s!r} refused")
                if payload.assignee is not None:
                    try:
                        if payload.reclaim_first:
                            ok = kanban_db.reassign_task(
                                conn, tid, payload.assignee or None,
                                reclaim_first=True,
                            )
                        else:
                            ok = kanban_db.assign_task(
                                conn, tid, payload.assignee or None,
                            )
                        if not ok:
                            entry.update(ok=False, error="assign refused")
                    except RuntimeError as e:
                        entry.update(ok=False, error=str(e))
                if payload.priority is not None:
                    with kanban_db.write_txn(conn):
                        conn.execute(
                            "UPDATE tasks SET priority = ? WHERE id = ?",
                            (int(payload.priority), tid),
                        )
                        conn.execute(
                            "INSERT INTO task_events (task_id, kind, payload, created_at) "
                            "VALUES (?, 'reprioritized', ?, ?)",
                            (tid, json.dumps({"priority": int(payload.priority)}),
                             int(time.time())),
                        )
            except Exception as e:  # defensive — one bad id shouldn't kill the batch
                entry.update(ok=False, error=str(e))
            results.append(entry)
        return {"results": results}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Diagnostics — fleet-wide distress signals (hallucinations, crashes,
# spawn failures, stuck-blocked). See hermes_cli.kanban_diagnostics for
# the rule engine.
# ---------------------------------------------------------------------------

@router.get("/diagnostics")
def list_diagnostics(
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
    severity: Optional[str] = Query(
        None,
        description="Filter by severity: warning|error|critical",
    ),
):
    """Return ``[{task_id, task_title, task_status, task_assignee,
    diagnostics: [...]}, ...]`` for every task on the board with at
    least one active diagnostic.

    Severity-filterable so the UI can render "just the critical ones"
    or the CLI can grep. Useful for the board-header attention strip
    AND for ``hermes kanban diagnostics`` which shells to this
    endpoint when the dashboard's running, or invokes the engine
    directly when it isn't.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        diags_by_task = _compute_task_diagnostics(conn, task_ids=None)
        if not diags_by_task:
            return {"diagnostics": [], "count": 0}

        # Narrow by severity if asked.
        if severity:
            filtered: dict[str, list[dict]] = {}
            for tid, dl in diags_by_task.items():
                keep = [d for d in dl if kd.severity_at_or_above(d.get("severity"), severity)]
                if keep:
                    filtered[tid] = keep
            diags_by_task = filtered
            if not diags_by_task:
                return {"diagnostics": [], "count": 0}

        # Pull the task rows we need in one query so we can include
        # titles/statuses without a per-task lookup.
        ids = list(diags_by_task.keys())
        placeholders = ",".join(["?"] * len(ids))
        rows = {
            r["id"]: r
            for r in conn.execute(
                f"SELECT id, title, status, assignee FROM tasks WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        }

        out = []
        for tid, dl in diags_by_task.items():
            r = rows.get(tid)
            out.append({
                "task_id": tid,
                "task_title": r["title"] if r else None,
                "task_status": r["status"] if r else None,
                "task_assignee": r["assignee"] if r else None,
                "diagnostics": dl,
            })
        # Sort: highest severity first, then most recent.
        from hermes_cli.kanban_diagnostics import SEVERITY_ORDER
        sev_idx = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        def _sort_key(row):
            top = row["diagnostics"][0]
            return (
                -sev_idx.get(top.get("severity"), -1),
                -(top.get("last_seen_at") or 0),
            )
        out.sort(key=_sort_key)

        return {
            "diagnostics": out,
            "count": sum(len(d["diagnostics"]) for d in out),
        }
    finally:
        conn.close()



# ---------------------------------------------------------------------------
# Worker visibility — cross-task active-worker list and per-run inspection
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]


@router.get("/workers/active")
def list_active_workers(
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Return every currently-running worker on the board.

    A worker is a ``task_runs`` row whose ``ended_at`` is NULL and whose
    ``worker_pid`` is non-NULL, belonging to a task with ``status='running'``.

    Returns ``{workers: [...], count: N, checked_at: <epoch>}``.  Each
    worker entry carries enough context for the dashboard to link back to
    its task without a second round-trip.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        rows = conn.execute(
            """
            SELECT
                r.id          AS run_id,
                r.task_id,
                t.title       AS task_title,
                t.status      AS task_status,
                t.assignee    AS task_assignee,
                r.profile,
                r.worker_pid,
                r.started_at,
                r.claim_lock,
                r.claim_expires,
                r.last_heartbeat_at,
                r.max_runtime_seconds
            FROM task_runs r
            JOIN tasks t ON t.id = r.task_id
            WHERE r.ended_at IS NULL
              AND r.worker_pid IS NOT NULL
              AND t.status = 'running'
            ORDER BY r.started_at ASC
            """,
        ).fetchall()
        workers = [
            {
                "run_id": row["run_id"],
                "task_id": row["task_id"],
                "task_title": row["task_title"],
                "task_status": row["task_status"],
                "task_assignee": row["task_assignee"],
                "profile": row["profile"],
                "worker_pid": row["worker_pid"],
                "started_at": row["started_at"],
                "claim_lock": row["claim_lock"],
                "claim_expires": row["claim_expires"],
                "last_heartbeat_at": row["last_heartbeat_at"],
                "max_runtime_seconds": row["max_runtime_seconds"],
            }
            for row in rows
        ]
        return {"workers": workers, "count": len(workers), "checked_at": int(time.time())}
    finally:
        conn.close()


@router.get("/runs/{run_id}")
def get_run_endpoint(
    run_id: int,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Direct lookup of a ``task_runs`` row by its integer id.

    Returns ``{run: {...}}`` using the same serialisation as the
    per-task run history embedded in ``GET /tasks/{task_id}``.
    404 when no such run exists.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        r = kanban_db.get_run(conn, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return {"run": _run_dict(r)}
    finally:
        conn.close()


@router.get("/runs/{run_id}/inspect")
def inspect_run_endpoint(
    run_id: int,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Live PID stats for a run's worker process via psutil.

    If the run has already ended, or has no recorded ``worker_pid``,
    returns ``{alive: false}`` with a human-readable ``reason``.

    When the process is live, returns CPU, memory, thread count, fd count,
    status, create_time, and cmdline.  ``access_denied`` is set when the
    OS refuses inspection rather than raising a 500.

    psutil availability: if psutil is not installed the endpoint still
    works but ``alive`` is always returned as ``false`` with
    ``reason="psutil not available"``.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        r = kanban_db.get_run(conn, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    finally:
        conn.close()

    if r.ended_at is not None:
        return {"run_id": run_id, "alive": False, "reason": "run already ended"}
    if r.worker_pid is None:
        return {"run_id": run_id, "alive": False, "reason": "no worker_pid recorded"}

    pid = r.worker_pid

    if _psutil is None:
        return {"run_id": run_id, "alive": False, "pid": pid, "reason": "psutil not available"}

    try:
        proc = _psutil.Process(pid)
        info = proc.as_dict(attrs=[
            "cpu_percent", "memory_info", "num_threads",
            "status", "create_time", "cmdline",
        ])
        # num_fds is POSIX-only; skip gracefully on Windows.
        try:
            num_fds = proc.num_fds()
        except AttributeError:
            num_fds = None
        mem = info.get("memory_info")
        return {
            "run_id": run_id,
            "alive": True,
            "pid": pid,
            "cpu_percent": info.get("cpu_percent"),
            "memory_rss_bytes": mem.rss if mem else None,
            "memory_vms_bytes": mem.vms if mem else None,
            "num_threads": info.get("num_threads"),
            "num_fds": num_fds,
            "status": info.get("status"),
            "create_time": info.get("create_time"),
            "cmdline": info.get("cmdline"),
        }
    except _psutil.NoSuchProcess:
        return {"run_id": run_id, "alive": False, "pid": pid, "reason": "process not found"}
    except _psutil.AccessDenied:
        return {"run_id": run_id, "alive": True, "pid": pid, "error": "access denied"}


class TerminateRunBody(BaseModel):
    reason: Optional[str] = None


@router.post("/runs/{run_id}/terminate")
def terminate_run_endpoint(
    run_id: int,
    payload: TerminateRunBody,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Terminate the worker process backing an in-flight run.

    Resolves ``run_id`` to its parent ``task_id`` and routes through
    :func:`kanban_db.reclaim_task` so the SIGTERM->SIGKILL flow,
    run-outcome bookkeeping, and event-log append all match what the
    existing ``POST /tasks/{task_id}/reclaim`` endpoint does.

    Responses:
      * 200 ``{"ok": true, "run_id": ..., "task_id": ...}`` on success.
      * 404 when ``run_id`` is unknown.
      * 409 when the run has already ended, or the task is no longer in
        a claimable state.

    Closes the gap left by PR #28432, which shipped the read-only
    sibling endpoints (``/workers/active``, ``/runs/{run_id}``,
    ``/runs/{run_id}/inspect``) but no termination control surface.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        r = kanban_db.get_run(conn, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        if r.ended_at is not None:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} already ended",
            )
        ok = kanban_db.reclaim_task(conn, r.task_id, reason=payload.reason)
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot terminate run {run_id}: task {r.task_id} is no "
                    "longer in a reclaimable state"
                ),
            )
        return {"ok": True, "run_id": run_id, "task_id": r.task_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recovery actions — reclaim a running claim, reassign to a new profile
# ---------------------------------------------------------------------------

class ReclaimBody(BaseModel):
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reclaim")
def reclaim_task_endpoint(
    task_id: str,
    payload: ReclaimBody,
    board: Optional[str] = Query(None),
):
    """Release an active worker claim on a running task.

    Used by the dashboard recovery popover when an operator wants to
    abort a stuck worker (e.g. one that keeps hallucinating card ids)
    without waiting for the claim TTL. Maps 1:1 to
    ``hermes kanban reclaim <task_id> --reason ...``.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        ok = kanban_db.reclaim_task(conn, task_id, reason=payload.reason)
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot reclaim {task_id}: not in a claimable state "
                    "(not running, or unknown id)"
                ),
            )
        return {"ok": True, "task_id": task_id}
    finally:
        conn.close()


class SpecifyBody(BaseModel):
    """Optional author override. Nothing else is configurable from the
    dashboard — model + prompt come from ``auxiliary.triage_specifier``
    in config.yaml, same as the CLI."""

    author: Optional[str] = None


@router.post("/tasks/{task_id}/specify")
def specify_task_endpoint(
    task_id: str,
    payload: SpecifyBody,
    board: Optional[str] = Query(None),
):
    """Flesh out a triage-column task via the auxiliary LLM and promote
    it to ``todo``. Maps 1:1 to ``hermes kanban specify <task_id>``.

    Returns the outcome shape used by the CLI: ``{ok, task_id, reason,
    new_title}``. A non-OK outcome is NOT an HTTP error — the UI renders
    the reason inline (e.g. "no auxiliary client configured") so the
    operator knows what to fix, and retries without a page reload.

    This endpoint runs in FastAPI's threadpool (sync ``def``) because
    the underlying LLM call can take tens of seconds to minutes on
    reasoning models, which would block the event loop if we used
    ``async def`` without an explicit ``run_in_executor``.
    """
    board = _resolve_board(board)
    # Pin the board for the duration of this call so the specifier module
    # (which calls ``kb.connect()`` with no args) hits the right DB. Use a
    # context-local override rather than mutating the process-global
    # HERMES_KANBAN_BOARD env var — this endpoint runs in FastAPI's
    # threadpool, so two concurrent requests for different boards would
    # otherwise race on the shared env var and cross-write (issue #38323).
    with kanban_db.scoped_current_board(board or kanban_db.DEFAULT_BOARD):
        # Import lazily so a missing auxiliary client at import time
        # doesn't break plugin load.
        from hermes_cli import kanban_specify  # noqa: WPS433 (intentional)

        outcome = kanban_specify.specify_task(
            task_id,
            author=(payload.author or None),
        )

    return {
        "ok": bool(outcome.ok),
        "task_id": outcome.task_id,
        "reason": outcome.reason,
        "new_title": outcome.new_title,
    }


class ReassignBody(BaseModel):
    profile: Optional[str] = None  # "" or None = unassign
    reclaim_first: bool = False
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reassign")
def reassign_task_endpoint(
    task_id: str,
    payload: ReassignBody,
    board: Optional[str] = Query(None),
):
    """Reassign a task to a different profile, optionally reclaiming first.

    Used by the dashboard recovery popover when an operator wants to
    retry a task with a different worker profile (e.g. switch to a
    smarter model after the assigned profile keeps hallucinating).
    Maps 1:1 to ``hermes kanban reassign <task_id> <profile> [--reclaim]``.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        ok = kanban_db.reassign_task(
            conn, task_id,
            payload.profile or None,
            reclaim_first=bool(payload.reclaim_first),
            reason=payload.reason,
        )
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot reassign {task_id}: unknown id, or still "
                    "running (pass reclaim_first=true to release the claim first)"
                ),
            )
        return {"ok": True, "task_id": task_id, "assignee": payload.profile or None}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Plugin config (read dashboard.kanban.* defaults from config.yaml)
# ---------------------------------------------------------------------------

@router.get("/config")
def get_config():
    """Return kanban dashboard preferences from ~/.hermes/config.yaml.

    Reads the ``dashboard.kanban`` section if present; defaults otherwise.
    Used by the UI to pre-select tenant filters, toggle markdown rendering,
    or set column-width preferences without a round-trip per page load.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    dash_cfg = (cfg.get("dashboard") or {})
    # dashboard.kanban may itself be a dict; fall back to {}.
    k_cfg = dash_cfg.get("kanban") or {}
    return {
        "default_tenant": k_cfg.get("default_tenant") or "",
        "lane_by_profile": bool(k_cfg.get("lane_by_profile", True)),
        "include_archived_by_default": bool(k_cfg.get("include_archived_by_default", False)),
        "render_markdown": bool(k_cfg.get("render_markdown", True)),
    }


# ---------------------------------------------------------------------------
# Home-channel subscriptions (per-task, per-platform toggles)
# ---------------------------------------------------------------------------
#
# Home channels are a first-class gateway concept — each configured platform
# can have exactly one (chat_id, thread_id, name) it considers "home". The
# dashboard surfaces these as per-task toggles so a user can opt a specific
# task into receiving terminal notifications (completed / blocked / gave_up)
# at their telegram/discord/slack home, without touching the CLI.
#
# The wire format mirrors kanban_db.add_notify_sub — (task_id, platform,
# chat_id, thread_id) — so toggle-on creates exactly the same row the
# `/kanban create` slash command would, and the existing gateway notifier
# watcher delivers events without any additional plumbing.


def _configured_home_channels() -> list[dict]:
    """Return every platform that has a home_channel set, fully hydrated.

    Reads the live GatewayConfig so env-var overlays (``TELEGRAM_HOME_CHANNEL``
    etc.) are honored alongside config.yaml. Returns platforms in a stable
    order and drops platforms without a home.
    """
    try:
        from gateway.config import load_gateway_config
    except Exception:
        return []
    try:
        gw_cfg = load_gateway_config()
    except Exception:
        return []
    result: list[dict] = []
    for platform, pcfg in gw_cfg.platforms.items():
        if not pcfg or not pcfg.home_channel:
            continue
        hc = pcfg.home_channel
        result.append({
            "platform": platform.value,
            "chat_id": hc.chat_id,
            "thread_id": hc.thread_id or "",
            "name": hc.name or "Home",
        })
    # Stable order for deterministic UI — platform name alphabetical.
    result.sort(key=lambda r: r["platform"])
    return result


def _active_profile_name() -> str:
    """Return the current Hermes profile name for notify-sub ownership."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _home_sub_matches(sub: dict, home: dict) -> bool:
    """True if a notify_subs row corresponds to the given home channel."""
    return (
        sub.get("platform") == home["platform"]
        and str(sub.get("chat_id", "")) == str(home["chat_id"])
        and str(sub.get("thread_id") or "") == str(home["thread_id"] or "")
    )


@router.get("/home-channels")
def get_home_channels(
    task_id: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
):
    """List every platform with a home channel, plus whether *task_id*
    (if given) is currently subscribed to that home.

    When ``task_id`` is omitted, every entry's ``subscribed`` is ``false``
    — useful for the "no task selected" state of the UI.
    """
    homes = _configured_home_channels()
    subscribed_homes: set[tuple[str, str, str]] = set()
    if task_id:
        board = _resolve_board(board)
        conn = _conn(board=board)
        try:
            subs = kanban_db.list_notify_subs(conn, task_id)
        finally:
            conn.close()
        for sub in subs:
            key = (
                str(sub.get("platform") or ""),
                str(sub.get("chat_id") or ""),
                str(sub.get("thread_id") or ""),
            )
            subscribed_homes.add(key)
    result = []
    for home in homes:
        key = (home["platform"], home["chat_id"], home["thread_id"])
        result.append({**home, "subscribed": key in subscribed_homes})
    return {"home_channels": result}


@router.post("/tasks/{task_id}/home-subscribe/{platform}")
def subscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Subscribe *task_id* to notifications routed to *platform*'s home channel.

    Idempotent — re-subscribing is a no-op at the DB layer. 404 if the
    platform has no home channel configured. 404 if the task doesn't exist.
    """
    homes = _configured_home_channels()
    home = next((h for h in homes if h["platform"] == platform), None)
    if not home:
        raise HTTPException(
            status_code=404,
            detail=f"No home channel configured for platform {platform!r}. "
                   f"Set one from the messenger via /sethome, or configure "
                   f"gateway.platforms.{platform}.home_channel in config.yaml.",
        )
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        kanban_db.add_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=home["chat_id"],
            thread_id=home["thread_id"] or None,
            notifier_profile=_active_profile_name(),
        )
        return {"ok": True, "task_id": task_id, "home_channel": home}
    finally:
        conn.close()


@router.delete("/tasks/{task_id}/home-subscribe/{platform}")
def unsubscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Remove any notify subscription on *task_id* that matches *platform*'s home."""
    homes = _configured_home_channels()
    home = next((h for h in homes if h["platform"] == platform), None)
    if not home:
        raise HTTPException(
            status_code=404,
            detail=f"No home channel configured for platform {platform!r}.",
        )
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        kanban_db.remove_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=home["chat_id"],
            thread_id=home["thread_id"] or None,
        )
        return {"ok": True, "task_id": task_id, "home_channel": home}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stats (per-profile / per-status counts + oldest-ready age)
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(board: Optional[str] = Query(None)):
    """Per-status + per-assignee counts + oldest-ready age.

    Designed for the dashboard HUD and for router profiles that need to
    answer "is this specialist overloaded?" without scanning the whole
    board themselves.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        return kanban_db.board_stats(conn)
    finally:
        conn.close()


@router.get("/assignees")
def get_assignees(board: Optional[str] = Query(None)):
    """Known profiles + per-profile task counts.

    Returns the union of ``~/.hermes/profiles/*`` on disk and every
    distinct assignee currently used on the board. The dashboard uses
    this to populate its assignee dropdown so a freshly-created profile
    appears in the picker before it's been given any task.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        return {"assignees": kanban_db.known_assignees(conn)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker log (read-only; file written by _default_spawn)
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}/log")
def get_task_log(
    task_id: str,
    tail: Optional[int] = Query(None, ge=1, le=2_000_000),
    board: Optional[str] = Query(None),
):
    """Return the worker's stdout/stderr log.

    ``tail`` caps the response size (bytes) so the dashboard drawer
    doesn't paginate megabytes into the browser. Returns 404 if the task
    has never spawned. The on-disk log is rotated at 2 MiB per
    ``_rotate_worker_log`` — a single ``.log.1`` is kept, no further
    generations, so disk usage per task is bounded at ~4 MiB.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
    finally:
        conn.close()
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    content = kanban_db.read_worker_log(task_id, tail_bytes=tail, board=board)
    log_path = kanban_db.worker_log_path(task_id, board=board)
    size = log_path.stat().st_size if log_path.exists() else 0
    return {
        "task_id": task_id,
        "path": str(log_path),
        "exists": content is not None,
        "size_bytes": size,
        "content": content or "",
        # Truncated when the on-disk file was larger than the tail cap.
        "truncated": bool(tail and size > tail),
    }


# ---------------------------------------------------------------------------
# Dispatch nudge (optional quick-path so the UI doesn't wait 60 s)
# ---------------------------------------------------------------------------

@router.post("/dispatch")
def dispatch(
    dry_run: bool = Query(False),
    max_n: int = Query(8, alias="max"),
    board: Optional[str] = Query(None),
):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        result = kanban_db.dispatch_once(
            conn, dry_run=dry_run, max_spawn=max_n, board=board,
        )
        # DispatchResult is a dataclass.
        try:
            return asdict(result)
        except TypeError:
            return {"result": str(result)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Boards CRUD (multi-project support)
# ---------------------------------------------------------------------------

class CreateBoardBody(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    default_workdir: Optional[str] = None
    switch: bool = False


class RenameBoardBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


def _board_counts(slug: str) -> dict[str, int]:
    """Return ``{status: count}`` for a board. Safe on an empty DB."""
    try:
        path = kanban_db.kanban_db_path(board=slug)
        if not path.exists():
            return {}
        conn = kanban_db.connect(board=slug)
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
            return {r["status"]: int(r["n"]) for r in rows}
        finally:
            conn.close()
    except Exception:
        return {}


def _default_workspace_kind(board: dict[str, Any]) -> str:
    """Recommend a non-destructive task workspace from board metadata."""
    workdir = str(board.get("default_workdir") or "").strip()
    if not workdir:
        return "scratch"
    try:
        return "worktree" if kanban_db._git_toplevel(Path(workdir)) else "dir"
    except (OSError, ValueError):
        return "dir"


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    """Return every board on disk with task counts and the active slug."""
    boards = kanban_db.list_boards(include_archived=include_archived)
    current = kanban_db.get_current_board()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_counts(b["slug"])
        b["total"] = sum(b["counts"].values())
        b["default_workspace_kind"] = _default_workspace_kind(b)
    return {"boards": boards, "current": current}


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    """Create a new board. Idempotent — ``slug`` collision returns existing."""
    default_workdir = None
    if payload.default_workdir:
        requested = Path(payload.default_workdir).expanduser()
        if not requested.is_absolute():
            raise HTTPException(
                status_code=400,
                detail="Project directory must be an absolute path.",
            )
        if not requested.is_dir():
            raise HTTPException(
                status_code=400,
                detail="Project directory must be an existing directory.",
            )
        default_workdir = str(requested.resolve())
    try:
        meta = kanban_db.create_board(
            payload.slug,
            name=payload.name,
            description=payload.description,
            icon=payload.icon,
            color=payload.color,
            default_workdir=default_workdir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if payload.switch:
        try:
            kanban_db.set_current_board(meta["slug"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    meta["default_workspace_kind"] = _default_workspace_kind(meta)
    return {"board": meta, "current": kanban_db.get_current_board()}


@router.patch("/boards/{slug}")
def rename_board(slug: str, payload: RenameBoardBody):
    """Update a board's display metadata (slug is immutable — create a new one to rename the directory)."""
    try:
        normed = kanban_db._normalize_board_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not normed or not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {slug!r} does not exist")
    meta = kanban_db.write_board_metadata(
        normed,
        name=payload.name,
        description=payload.description,
        icon=payload.icon,
        color=payload.color,
    )
    return {"board": meta}


@router.delete("/boards/{slug}")
def delete_board(slug: str, delete: bool = Query(False, description="Hard-delete instead of archive")):
    """Archive (default) or hard-delete a board."""
    try:
        res = kanban_db.remove_board(slug, archive=not delete)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"result": res, "current": kanban_db.get_current_board()}


@router.post("/boards/{slug}/switch")
def switch_board(slug: str):
    """Persist ``slug`` as the active board for subsequent CLI / slash calls.

    Dashboard users pick boards via a client-side ``localStorage`` — this
    endpoint is for ``/kanban boards switch`` parity so gateway slash
    commands and the CLI share the same current-board pointer.
    """
    try:
        normed = kanban_db._normalize_board_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not normed or not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {slug!r} does not exist")
    kanban_db.set_current_board(normed)
    return {"current": normed}


# ---------------------------------------------------------------------------
# WebSocket: /events?since=<event_id>
# ---------------------------------------------------------------------------

# Poll interval for the event tail loop. SQLite WAL + 300 ms polling is
# the simplest and most robust approach; it adds a fraction of a percent
# of CPU and has no shared state to synchronize across workers.
_EVENT_POLL_SECONDS = 0.3


# ---------------------------------------------------------------------------
# Profile metadata & description editing (consumed by the kanban orchestrator)
# ---------------------------------------------------------------------------

class DescribeBody(BaseModel):
    description: Optional[str] = None  # explicit user-authored text


class DescribeAutoBody(BaseModel):
    overwrite: bool = False


@router.get("/profiles")
def list_profile_roster():
    """Return every installed profile with its description.

    Consumed by the dashboard's settings panel (orchestrator picker)
    and the profile-description editing UI. Profiles without a
    description still appear here — they're routable on name alone,
    just less precisely.
    """
    try:
        from hermes_cli import profiles as profiles_mod
        profiles = profiles_mod.list_profiles()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to list profiles: {exc}")
    return {
        "profiles": [
            {
                "name": p.name,
                "is_default": bool(p.is_default),
                "model": p.model or "",
                "provider": p.provider or "",
                "description": p.description or "",
                "description_auto": bool(p.description_auto),
                "skill_count": int(p.skill_count or 0),
            }
            for p in profiles
        ],
    }


@router.patch("/profiles/{profile_name}")
def update_profile_description(profile_name: str, payload: DescribeBody):
    """Set or clear the description of a profile.

    Empty string clears the description; non-empty stores it as a
    user-authored description (``description_auto: false``) so the
    auto-describer won't overwrite it on a sweep without
    ``--overwrite``.
    """
    try:
        from hermes_cli import profiles as profiles_mod
        canon = profiles_mod.normalize_profile_name(profile_name)
        if canon == "default":
            from hermes_constants import get_hermes_home  # type: ignore
            from pathlib import Path as _Path
            profile_dir = _Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
        if not profile_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"profile '{profile_name}' not found")
        text = (payload.description or "").strip()
        profiles_mod.write_profile_meta(
            profile_dir,
            description=text,
            description_auto=False,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to update profile: {exc}")
    return {"ok": True, "profile": canon, "description": text}


@router.post("/profiles/{profile_name}/describe-auto")
def auto_describe_profile(profile_name: str, payload: DescribeAutoBody):
    """Generate a description for the named profile via the auxiliary
    LLM (``auxiliary.profile_describer``). Persists with
    ``description_auto: true`` so the dashboard can surface a "review"
    badge.

    Maps 1:1 to ``hermes profile describe <name> --auto``. Non-OK
    outcomes are NOT HTTP errors — the UI renders the reason inline
    (e.g. "no auxiliary client configured") so the operator can fix
    config and retry without a page reload.
    """
    try:
        from hermes_cli import profile_describer  # noqa: WPS433 (intentional)
        outcome = profile_describer.describe_profile(
            profile_name,
            overwrite=bool(payload.overwrite),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"describer crashed: {exc}")
    return {
        "ok": bool(outcome.ok),
        "profile": outcome.profile_name,
        "reason": outcome.reason,
        "description": outcome.description,
    }


# ---------------------------------------------------------------------------
# Decompose endpoint (built-in decomposer fan-out)
# ---------------------------------------------------------------------------

class DecomposeBody(BaseModel):
    author: Optional[str] = None


@router.post("/tasks/{task_id}/decompose")
def decompose_task_endpoint(
    task_id: str,
    payload: DecomposeBody,
    board: Optional[str] = Query(None),
):
    """Fan a triage-column task out into a graph of child tasks via the
    auxiliary LLM, routed to specialist profiles by description. Maps
    1:1 to ``hermes kanban decompose <task_id>``.

    Returns the outcome shape used by the CLI: ``{ok, task_id, reason,
    fanout, child_ids, new_title}``. A non-OK outcome is NOT an HTTP
    error — the UI renders the reason inline.

    Runs in FastAPI's threadpool (sync ``def``) because the LLM call
    can take minutes on reasoning models.
    """
    board = _resolve_board(board)
    # Context-local board pin (see specify endpoint above): this sync
    # endpoint runs in FastAPI's threadpool, so mutating the process-global
    # HERMES_KANBAN_BOARD env var would let concurrent requests for
    # different boards race and cross-write (issue #38323).
    with kanban_db.scoped_current_board(board or kanban_db.DEFAULT_BOARD):
        from hermes_cli import kanban_decompose  # noqa: WPS433 (intentional)
        outcome = kanban_decompose.decompose_task(
            task_id,
            author=(payload.author or None),
        )

    return {
        "ok": bool(outcome.ok),
        "task_id": outcome.task_id,
        "reason": outcome.reason,
        "fanout": bool(outcome.fanout),
        "child_ids": outcome.child_ids or [],
        "new_title": outcome.new_title,
    }


# ---------------------------------------------------------------------------
# Orchestration settings (kanban.orchestrator_profile / default_assignee /
# auto_decompose) — surfaced to the dashboard's settings panel
# ---------------------------------------------------------------------------

class OrchestrationSettingsBody(BaseModel):
    orchestrator_profile: Optional[str] = None
    default_assignee: Optional[str] = None
    auto_decompose: Optional[bool] = None
    auto_promote_children: Optional[bool] = None


@router.get("/orchestration")
def get_orchestration_settings():
    """Return the current kanban orchestration knobs from config.yaml
    plus the resolved effective values (filling in fallbacks)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    kanban_cfg = (cfg.get("kanban") or {}) if isinstance(cfg, dict) else {}
    explicit_orch = (kanban_cfg.get("orchestrator_profile") or "").strip()
    explicit_default = (kanban_cfg.get("default_assignee") or "").strip()
    auto_decompose = bool(kanban_cfg.get("auto_decompose", True))
    auto_promote_children = bool(kanban_cfg.get("auto_promote_children", True))

    # Resolve fallbacks the same way the decomposer does.
    resolved_orch = explicit_orch
    resolved_default = explicit_default
    try:
        from hermes_cli import profiles as profiles_mod
        active_default = profiles_mod.get_active_profile_name() or "default"
        if not resolved_orch or not profiles_mod.profile_exists(resolved_orch):
            resolved_orch = active_default
        if not resolved_default or not profiles_mod.profile_exists(resolved_default):
            resolved_default = active_default
    except Exception:
        active_default = "default"
        if not resolved_orch:
            resolved_orch = active_default
        if not resolved_default:
            resolved_default = active_default

    return {
        "orchestrator_profile": explicit_orch,
        "default_assignee": explicit_default,
        "auto_decompose": auto_decompose,
        "auto_promote_children": auto_promote_children,
        "resolved_orchestrator_profile": resolved_orch,
        "resolved_default_assignee": resolved_default,
        "active_profile": active_default,
    }


@router.put("/orchestration")
def set_orchestration_settings(payload: OrchestrationSettingsBody):
    """Update the kanban orchestration knobs in ~/.hermes/config.yaml.

    Each field is optional — only fields explicitly passed are
    written. ``orchestrator_profile`` / ``default_assignee`` accept
    empty strings to clear the override and fall back to the default
    profile.
    """
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load config: {exc}")

    kanban_section = cfg.setdefault("kanban", {})
    if not isinstance(kanban_section, dict):
        kanban_section = {}
        cfg["kanban"] = kanban_section

    # Validate any non-empty profile names exist before saving.
    try:
        from hermes_cli import profiles as profiles_mod
    except Exception:
        profiles_mod = None  # type: ignore

    if payload.orchestrator_profile is not None:
        name = (payload.orchestrator_profile or "").strip()
        if name and profiles_mod is not None:
            try:
                if not profiles_mod.profile_exists(name):
                    raise HTTPException(
                        status_code=400,
                        detail=f"profile '{name}' does not exist",
                    )
            except HTTPException:
                raise
            except Exception:
                pass  # fail open if the lookup itself errors
        kanban_section["orchestrator_profile"] = name

    if payload.default_assignee is not None:
        name = (payload.default_assignee or "").strip()
        if name and profiles_mod is not None:
            try:
                if not profiles_mod.profile_exists(name):
                    raise HTTPException(
                        status_code=400,
                        detail=f"profile '{name}' does not exist",
                    )
            except HTTPException:
                raise
            except Exception:
                pass
        kanban_section["default_assignee"] = name

    if payload.auto_decompose is not None:
        kanban_section["auto_decompose"] = bool(payload.auto_decompose)

    if payload.auto_promote_children is not None:
        kanban_section["auto_promote_children"] = bool(payload.auto_promote_children)

    try:
        save_config(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to save config: {exc}")

    # Echo back the resolved state (callers usually re-render from it).
    return get_orchestration_settings()


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    # Authorize the upgrade via the dashboard's canonical WS gate so the
    # correct credential is accepted in every mode (loopback token / gated
    # single-use ticket / server-internal credential). Browsers can't set
    # Authorization on a WS upgrade, so the credential rides in the query
    # string — the browser SDK's buildWsUrl() assembles it.
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    try:
        since_raw = ws.query_params.get("since", "0")
        try:
            cursor = int(since_raw)
        except ValueError:
            cursor = 0

        # Board selection — pinned at the WS handshake; re-subscribe to
        # switch boards. Changing boards mid-stream would require
        # reconciling two cursors, so the UI just opens a new WS on
        # board change.
        ws_board_raw = ws.query_params.get("board")
        try:
            ws_board = kanban_db._normalize_board_slug(ws_board_raw) if ws_board_raw else None
        except ValueError:
            ws_board = None

        def _fetch_new(cursor_val: int) -> tuple[int, list[dict]]:
            conn = kanban_db.connect(board=ws_board)
            try:
                rows = conn.execute(
                    "SELECT id, task_id, run_id, kind, payload, created_at "
                    "FROM task_events WHERE id > ? ORDER BY id ASC LIMIT 200",
                    (cursor_val,),
                ).fetchall()
                out: list[dict] = []
                new_cursor = cursor_val
                for r in rows:
                    try:
                        payload = json.loads(r["payload"]) if r["payload"] else None
                    except Exception:
                        payload = None
                    out.append({
                        "id": r["id"],
                        "task_id": r["task_id"],
                        "run_id": r["run_id"],
                        "kind": r["kind"],
                        "payload": payload,
                        "created_at": r["created_at"],
                    })
                    new_cursor = r["id"]
                return new_cursor, out
            finally:
                conn.close()

        while True:
            cursor, events = await asyncio.to_thread(_fetch_new, cursor)
            if events:
                await ws.send_json({"events": events, "cursor": cursor})
            await asyncio.sleep(_EVENT_POLL_SECONDS)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        # Normal shutdown path: dashboard process exit (Ctrl-C) cancels the
        # websocket task while it is sleeping in the poll loop.
        # CancelledError is a BaseException in 3.8+ so the bare Exception
        # handler below would not catch it; without this clause Uvicorn
        # surfaces the cancellation as an application traceback. Quiet it.
        return
    except Exception as exc:  # defensive: never crash the dashboard worker
        log.warning("Kanban event stream error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass
