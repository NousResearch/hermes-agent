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
import hmac
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status as http_status
from pydantic import BaseModel, Field

from hermes_cli import kanban_db

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth helper — WebSocket only (HTTP routes live behind the dashboard's
# existing plugin-bypass; this is documented above).
# ---------------------------------------------------------------------------

def _check_ws_token(provided: Optional[str]) -> bool:
    """Constant-time compare against the dashboard session token.

    Imported lazily so the plugin still loads in test contexts where the
    dashboard web_server module isn't importable (e.g. the bare-FastAPI
    test harness).
    """
    if not provided:
        return False
    try:
        from hermes_cli import web_server as _ws
    except Exception:
        # No dashboard context (tests). Accept so the tail loop is still
        # testable; in production the dashboard module always imports
        # cleanly because it's the caller.
        return True
    expected = getattr(_ws, "_SESSION_TOKEN", None)
    if not expected:
        return True
    return hmac.compare_digest(str(provided), str(expected))


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
BOARD_COLUMNS: list[str] = [
    "triage", "todo", "ready", "running", "blocked", "done",
]


_CARD_SUMMARY_PREVIEW_CHARS = 200


# Map current task.status → task_events.kind that records the most recent
# transition INTO that status. Used by ``_compute_entered_status_at`` to
# answer "when did this card enter its current column?" without adding a
# new ``status_changed_at`` column on the tasks table.
#
# ``todo`` is intentionally absent: a fresh task starts in ``todo`` with no
# dedicated entry event (``created_at`` is the natural fallback). Tasks
# rolled back to ``todo`` from ready (``claim_rejected``) or from archived
# (``restored``) get their entered-stage timestamp from those events instead.
_STATUS_TO_ENTRY_EVENT: dict[str, tuple[str, ...]] = {
    "ready":    ("promoted",),
    "running":  ("claimed",),
    "done":     ("completed",),
    "blocked":  ("blocked",),
    "archived": ("archived",),
    "todo":     ("restored", "claim_rejected"),
    # "triage" — no dedicated event; falls back to created_at.
}


def _entered_status_at_for_task(
    conn: sqlite3.Connection, task_id: str, status: str, created_at: int,
) -> int:
    """Return the timestamp when ``task_id`` entered its current ``status``.

    Why: Dashboard cards need a per-stage timestamp distinct from
    ``created_at`` so operators can spot tasks stuck in ``running`` /
    ``blocked`` independently of how old the ticket is.
    What: Looks up the latest ``task_events.created_at`` whose ``kind``
    matches the status's entry event; falls back to ``created_at`` when
    no such event exists (e.g. fresh ``todo`` tasks, or ``triage``).
    Test: Promote a task to ready, expect entered_status_at == promoted
    event timestamp; claim it, expect entered_status_at == claimed
    timestamp (NOT promoted).
    """
    kinds = _STATUS_TO_ENTRY_EVENT.get(status)
    if not kinds:
        return created_at
    placeholders = ",".join("?" for _ in kinds)
    row = conn.execute(
        f"SELECT MAX(created_at) AS ts FROM task_events "
        f"WHERE task_id = ? AND kind IN ({placeholders})",
        (task_id, *kinds),
    ).fetchone()
    ts = row["ts"] if row else None
    return int(ts) if ts is not None else int(created_at)


def _batch_entered_status_at(
    conn: sqlite3.Connection, tasks: list[kanban_db.Task],
) -> dict[str, int]:
    """Batch version of :func:`_entered_status_at_for_task` for list endpoints.

    Why: ``GET /board`` serializes hundreds of tasks; calling the
    per-task helper would be N+1 against ``task_events``. One grouped
    aggregate query keeps the cost flat regardless of board size.
    What: For each task, picks MAX(created_at) over the events matching
    its status's entry kinds. Falls back to ``task.created_at`` when no
    matching event exists.
    Test: Build a board with 3 tasks in ready/running/done, call this,
    assert each maps to its own status's event timestamp.
    """
    if not tasks:
        return {}
    # Collect every (task_id, kind) pair we care about so the WHERE clause
    # filters tightly. We don't pre-filter by status because a task that
    # just transitioned still has older events of the previous kinds —
    # filtering on the *current* status's kinds is what gives us the
    # "entered this column" semantics.
    interest: dict[str, tuple[str, ...]] = {}
    for t in tasks:
        kinds = _STATUS_TO_ENTRY_EVENT.get(t.status)
        if kinds:
            interest[t.id] = kinds

    out: dict[str, int] = {t.id: int(t.created_at) for t in tasks}
    if not interest:
        return out

    # Build one IN-clause over the union of all kinds we need so we can
    # do a single scan, then filter per-task client-side. This is cheaper
    # than one query per task even for boards in the thousands.
    all_kinds = sorted({k for ks in interest.values() for k in ks})
    all_ids = list(interest.keys())
    id_ph = ",".join("?" for _ in all_ids)
    kind_ph = ",".join("?" for _ in all_kinds)
    rows = conn.execute(
        f"SELECT task_id, kind, MAX(created_at) AS ts "
        f"FROM task_events "
        f"WHERE task_id IN ({id_ph}) AND kind IN ({kind_ph}) "
        f"GROUP BY task_id, kind",
        (*all_ids, *all_kinds),
    ).fetchall()
    per_task_max: dict[str, int] = {}
    for r in rows:
        tid = r["task_id"]
        kind = r["kind"]
        # Skip rows whose kind isn't in *this* task's interest set
        # (we widened the IN-clause for batch efficiency above).
        if kind not in interest.get(tid, ()):
            continue
        ts = int(r["ts"])
        cur = per_task_max.get(tid)
        if cur is None or ts > cur:
            per_task_max[tid] = ts
    for tid, ts in per_task_max.items():
        out[tid] = ts
    return out


def _task_dict(
    task: kanban_db.Task,
    *,
    latest_summary: Optional[str] = None,
    entered_status_at: Optional[int] = None,
) -> dict[str, Any]:
    d = asdict(task)
    # Add derived age metrics so the UI can colour stale cards without
    # computing deltas client-side.
    try:
        d["age"] = kanban_db.task_age(task)
    except Exception:
        d["age"] = {"created_age_seconds": None, "started_age_seconds": None, "time_to_complete_seconds": None}
    # Per-stage timestamp: when did this card enter its current column?
    # ``entered_status_at`` is precomputed by the batch helper on list
    # endpoints; falls back to ``None`` here (single-task callers can
    # compute it themselves via /tasks/:id below).
    d["entered_status_at"] = entered_status_at
    # Derived age in seconds for the current stage, alongside the
    # existing created/started/complete ages. ``None`` when we don't
    # have entered_status_at yet (caller didn't pass it).
    if entered_status_at is not None:
        try:
            age_block = d.get("age") or {}
            age_block["entered_status_age_seconds"] = max(
                0, int(time.time()) - int(entered_status_at)
            )
            d["age"] = age_block
        except Exception:
            pass
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
# GET /board
# ---------------------------------------------------------------------------

@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None, description="Filter to a single tenant"),
    include_archived: bool = Query(False),
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
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
            conn, tenant=tenant, include_archived=include_archived
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
        # Batch-fetch entered-status timestamps in one aggregate query.
        entered_at_map = _batch_entered_status_at(conn, tasks)

        for t in tasks:
            full = summary_map.get(t.id)
            preview = (
                full[:_CARD_SUMMARY_PREVIEW_CHARS] if full else None
            )
            d = _task_dict(
                t,
                latest_summary=preview,
                entered_status_at=entered_at_map.get(t.id),
            )
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
# GET /tasks/archived  — MUST be registered before /tasks/{task_id} so the
# static path doesn't get swallowed by the path-param catch-all below.
# Powers the dashboard Archive Viewer (companion to /boards/archived).
# ---------------------------------------------------------------------------

@router.get("/tasks/archived")
def list_archived_tasks(
    board: Optional[str] = Query(
        None,
        description="Board slug to filter on (omit for the active board)",
    ),
    all_boards: bool = Query(
        False,
        description="Span every active board instead of just the selected one",
    ),
):
    """Return every task with status='archived' on the chosen board(s).

    Why: Dashboard Archive Viewer needs a list of archived tasks across
    one (or all) active boards so users can spot accidental archives and
    restore them.
    What: When ``all_boards`` is True, iterates every active board and
    concatenates results, tagging each task with ``board_slug``. Otherwise
    queries just the resolved board.
    Test: Archive a task on board A, call with ``all_boards=true``,
    expect that task in the returned list with the right ``board_slug``.
    """
    out: list[dict] = []
    if all_boards:
        boards = kanban_db.list_boards(include_archived=False)
        slugs = [b["slug"] for b in boards]
    else:
        slugs = [_resolve_board(board) or kanban_db.get_current_board()]
    for slug in slugs:
        if not kanban_db.board_exists(slug):
            continue
        conn = kanban_db.connect(board=slug)
        try:
            tasks = kanban_db.list_tasks(conn, include_archived=True, status="archived")
            entered_at_map = _batch_entered_status_at(conn, tasks)
            for t in tasks:
                d = _task_dict(t, entered_status_at=entered_at_map.get(t.id))
                d["board_slug"] = slug
                out.append(d)
        except Exception as exc:
            log.warning("list_archived_tasks: board %s failed: %s", slug, exc)
        finally:
            conn.close()
    return {"archived_tasks": out, "count": len(out)}


# ---------------------------------------------------------------------------
# GET /tasks/:id
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}")
def get_task(task_id: str, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        # Drawer/detail view returns the FULL summary (no truncation) so
        # operators can read the complete worker handoff without making
        # a second round-trip. Cards on /board carry a 200-char preview.
        full_summary = kanban_db.latest_summary(conn, task_id)
        entered_at = _entered_status_at_for_task(
            conn, task.id, task.status, task.created_at,
        )
        task_d = _task_dict(
            task, latest_summary=full_summary, entered_status_at=entered_at,
        )
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
            "links": _links_for(conn, task_id),
            "runs": [_run_dict(r) for r in kanban_db.list_runs(conn, task_id)],
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
        )
        task = kanban_db.get_task(conn, task_id)
        if task is not None:
            entered_at = _entered_status_at_for_task(
                conn, task.id, task.status, task.created_at,
            )
            body: dict[str, Any] = {
                "task": _task_dict(task, entered_status_at=entered_at),
            }
        else:
            body = {"task": None}
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
            elif s == "ready":
                # Re-open a blocked task, or just an explicit status set.
                current = kanban_db.get_task(conn, task_id)
                if current and current.status == "blocked":
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
            elif s in {"todo", "triage"}:
                ok = _set_status_direct(conn, task_id, s)
            else:
                raise HTTPException(status_code=400, detail=f"unknown status: {s}")
            if not ok:
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
        if updated is not None:
            entered_at = _entered_status_at_for_task(
                conn, updated.id, updated.status, updated.created_at,
            )
            return {
                "task": _task_dict(updated, entered_status_at=entered_at),
            }
        return {"task": None}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DELETE /tasks/:id  — hard-delete a task and all its derived rows.
#
# This is destructive and irreversible: archive (PATCH status=archived) is
# the soft-delete path for tasks the user wants to keep in history. Use
# DELETE only when the task is truly garbage (e.g. an accidental create,
# a duplicate of another card, or a crashed task that's already been
# replaced) and you don't want it cluttering audit/event queries either.
#
# Tables touched (no FK cascades in v1 schema, so explicit deletes):
#   - tasks            (the row itself)
#   - task_links       (parent_id = ? OR child_id = ?)
#   - task_comments    (task_id = ?)
#   - task_events      (task_id = ?)
#   - task_runs        (task_id = ?)
#   - kanban_notify_subs (task_id = ?)  — gateway notification subscriptions
# ---------------------------------------------------------------------------

@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(
    task_id: str,
    board: Optional[str] = Query(None),
    force: bool = Query(
        False,
        description=(
            "Override the status-based integrity rule. Default behaviour "
            "(force=false) refuses hard-delete when the task is not in "
            "{'archived', 'done'} so the dashboard can't accidentally "
            "wipe live work."
        ),
    ),
):
    """Hard-delete a task (with terminal-status integrity rule).

    Why: Soft-delete (archive) is the right path for tasks the user
    wants to keep in history; this endpoint is for true garbage. The
    integrity rule (only 'archived' or 'done') stops the dashboard from
    accidentally yanking a running/blocked task and orphaning its
    workspace.
    What: Refuses with 409 unless the task's status is 'archived' or
    'done', OR ``force=true`` is supplied (escape hatch for legacy
    callers and operator override).
    Test: Try to DELETE a 'ready' task — expect 409. DELETE a 'done'
    task — expect 204.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")

        # Integrity rule: only allow hard delete on terminal-state tasks.
        if not force and task.status not in {"archived", "done"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": (
                        f"task {task_id} has status {task.status!r}; hard-delete "
                        "is only allowed on tasks in 'archived' or 'done' "
                        "state. Archive the task first, or pass force=true to "
                        "override."
                    ),
                    "current_status": task.status,
                },
            )

        # Close any active run with outcome='reclaimed' so the run record is
        # consistent before we drop everything; matters if the dispatcher
        # later joins task_runs in audit queries that survive deletes.
        if task.status == "running" and task.current_run_id:
            kanban_db._end_run(
                conn, task_id,
                outcome="reclaimed", status="reclaimed",
                summary="task hard-deleted from dashboard while run was active",
            )

        with kanban_db.write_txn(conn):
            conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? OR child_id = ?",
                (task_id, task_id),
            )
            conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
            # kanban_notify_subs may not exist on older boards — guard.
            try:
                conn.execute(
                    "DELETE FROM kanban_notify_subs WHERE task_id = ?",
                    (task_id,),
                )
            except sqlite3.OperationalError:
                pass
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cur.rowcount != 1:
                # Concurrent delete or row vanished between our get_task and
                # the actual DELETE — surface as 404 so the UI removes the
                # card either way.
                raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        # Returning None with status_code=204 means FastAPI emits no body.
        return None
    finally:
        conn.close()


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
                        if cur and cur.status == "blocked":
                            ok = kanban_db.unblock_task(conn, tid)
                        else:
                            ok = _set_status_direct(conn, tid, "ready")
                    elif s in {"todo", "running", "triage"}:
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
                keep = [d for d in dl if d.get("severity") == severity]
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
    # (which calls ``kb.connect()`` with no args) hits the right DB.
    prev_env = os.environ.get("HERMES_KANBAN_BOARD")
    try:
        os.environ["HERMES_KANBAN_BOARD"] = board or kanban_db.DEFAULT_BOARD
        # Import lazily so a missing auxiliary client at import time
        # doesn't break plugin load.
        from hermes_cli import kanban_specify  # noqa: WPS433 (intentional)

        outcome = kanban_specify.specify_task(
            task_id,
            author=(payload.author or None),
        )
    finally:
        if prev_env is None:
            os.environ.pop("HERMES_KANBAN_BOARD", None)
        else:
            os.environ["HERMES_KANBAN_BOARD"] = prev_env

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


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    """Return every board on disk with task counts and the active slug."""
    boards = kanban_db.list_boards(include_archived=include_archived)
    current = kanban_db.get_current_board()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_counts(b["slug"])
        b["total"] = sum(b["counts"].values())
    return {"boards": boards, "current": current}


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    """Create a new board. Idempotent — ``slug`` collision returns existing."""
    try:
        meta = kanban_db.create_board(
            payload.slug,
            name=payload.name,
            description=payload.description,
            icon=payload.icon,
            color=payload.color,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if payload.switch:
        try:
            kanban_db.set_current_board(meta["slug"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
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


# ---------------------------------------------------------------------------
# Board lifecycle: archive / restore / hard-delete + archive viewer
#
# Integrity rules (enforced here AND mirrored in the dashboard UI):
#
#   * "default" board cannot be archived or hard-deleted (kanban_db enforces
#     this at the function level — we surface the error as 409).
#   * Cannot archive the currently-active board — user must ``switch`` first.
#   * Archive of a board with non-archived tasks requires ``cascade=true``;
#     otherwise refuses with 409 so the user has to decide explicitly.
#   * Hard-delete of a board requires zero tasks (active OR archived).
#   * Hard-delete of a task requires status in {archived, done}.
#   * Restoring a board re-creates the on-disk directory; refuses if a
#     board with the same slug already exists in the active set.
#
# Every integrity refusal returns HTTP 409 with a JSON body containing a
# clear ``error`` field so the dashboard can render the message verbatim.
# ---------------------------------------------------------------------------


def _board_task_status_counts(slug: str) -> dict[str, int]:
    """Return ``{status: count}`` for a board's task table.

    Why: Integrity checks (cascade, hard-delete, restore) need to know
    how many active and archived tasks the board has — separate from
    :func:`_board_counts` which is also used as a UI hint.
    What: Opens the board's DB, runs a GROUP BY on ``tasks.status``.
    Test: Create 2 todo tasks + 1 archived, expect ``{"todo": 2,
    "archived": 1}`` for that board.
    """
    try:
        path = kanban_db.kanban_db_path(board=slug)
        if not path.exists():
            return {}
        conn = kanban_db.connect(board=slug)
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status",
            ).fetchall()
            return {r["status"]: int(r["n"]) for r in rows}
        finally:
            conn.close()
    except Exception:
        return {}


def _non_archived_total(status_counts: dict[str, int]) -> int:
    """Sum of every status that is NOT 'archived' (== "live" task count)."""
    return sum(n for s, n in status_counts.items() if s != "archived")


@router.get("/boards/archived")
def list_archived_boards():
    """Return every archived board directory with parsed metadata.

    Why: Powers the dashboard's Archive Viewer tab so users can find
    boards they archived earlier (by accident or intentionally).
    What: Wraps :func:`kanban_db.list_archived_boards` and adds a
    ``can_restore`` hint per row (false when an active board with the
    same slug already exists).
    Test: Archive one board, GET /boards/archived → 1 entry with
    ``can_restore=true``.
    """
    rows = kanban_db.list_archived_boards()
    active_slugs = {b["slug"] for b in kanban_db.list_boards(include_archived=False)}
    for r in rows:
        r["can_restore"] = r["slug"] not in active_slugs
    return {"archived_boards": rows, "count": len(rows)}


# NOTE: ``GET /tasks/archived`` is registered earlier in the file (just
# before ``GET /tasks/{task_id}``) so the static path matches before the
# path-param catch-all. See ``list_archived_tasks`` near the top.


@router.post("/boards/{slug}/archive")
def archive_board_endpoint(
    slug: str,
    cascade: bool = Query(
        False,
        description="Also archive every non-archived task on this board",
    ),
):
    """Archive a board (move on-disk to ``boards/_archived/``).

    Why: Replaces the legacy ``DELETE /boards/{slug}`` archive path with
    an explicit, structured verb so the dashboard UI can distinguish
    archive (recoverable) from hard delete (destructive) without relying
    on a query-param flag.
    What: Enforces integrity rules (not 'default', not active, no live
    tasks unless ``cascade=true``), then calls :func:`kanban_db.remove_board`
    with ``archive=True``. When ``cascade=true``, archives every
    non-archived task first.
    Test: Archive a non-default board with tasks and cascade=true,
    assert all tasks transition to status='archived' and the dir moves
    to ``boards/_archived/``.
    """
    try:
        normed = kanban_db._normalize_board_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not normed:
        raise HTTPException(status_code=400, detail="board slug is required")
    if normed == kanban_db.DEFAULT_BOARD:
        raise HTTPException(
            status_code=409,
            detail={"error": "the 'default' board cannot be archived"},
        )
    if not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {normed!r} does not exist")
    if kanban_db.get_current_board() == normed:
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    f"cannot archive {normed!r} while it is the active board — "
                    "switch to another board first"
                ),
            },
        )

    counts = _board_task_status_counts(normed)
    live = _non_archived_total(counts)
    if live > 0 and not cascade:
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    f"board {normed!r} has {live} non-archived task(s); pass "
                    "cascade=true to archive them along with the board"
                ),
                "live_task_count": live,
                "counts": counts,
            },
        )

    cascaded_ids: list[str] = []
    if live > 0 and cascade:
        conn = kanban_db.connect(board=normed)
        try:
            tasks = kanban_db.list_tasks(conn, include_archived=False)
            for t in tasks:
                if t.status != "archived":
                    try:
                        if kanban_db.archive_task(conn, t.id):
                            cascaded_ids.append(t.id)
                    except Exception as exc:
                        log.warning("cascade archive failed for %s: %s", t.id, exc)
        finally:
            conn.close()

    try:
        res = kanban_db.remove_board(normed, archive=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)})
    return {
        "result": res,
        "cascade_archived_tasks": cascaded_ids,
        "current": kanban_db.get_current_board(),
    }


@router.post("/boards/{slug}/restore")
def restore_board_endpoint(slug: str):
    """Restore an archived board back to the active list.

    Why: One-click undo for accidental archives.
    What: Moves ``boards/_archived/<slug>-<ts>/`` → ``boards/<slug>/``.
    Refuses (409) if an active board with the same slug already exists.
    Test: Archive a board, call this endpoint, assert ``board_exists``
    returns True for the slug.
    """
    try:
        res = kanban_db.restore_board(slug)
    except ValueError as exc:
        msg = str(exc)
        # "already exists" → 409 conflict; otherwise 404 (no such archive).
        status_code = 409 if "already exists" in msg else 404
        raise HTTPException(status_code=status_code, detail={"error": msg})
    return {"result": res}


@router.delete("/boards/{slug}")
def delete_board(
    slug: str,
    delete: bool = Query(
        False,
        description="Hard-delete (destructive) instead of archive. Default = archive.",
    ),
    cascade: bool = Query(
        False,
        description=(
            "Archive mode: also archive every non-archived task. "
            "Hard-delete mode: opt-in to destroy a non-empty board "
            "(directory + every task, active and archived, are purged)."
        ),
    ),
):
    """Archive (default) or hard-delete a board.

    Why: Backwards-compatible entry point for the existing dashboard
    bundle which still emits ``DELETE /boards/{slug}``. New code should
    prefer ``POST /boards/{slug}/archive`` (explicit) or this endpoint
    with ``delete=true`` (hard delete; ``cascade=true`` opts in to
    destroying a non-empty board outright).
    What: With ``delete=false`` (default) → behaves like the archive
    endpoint above (cascade rules included). With ``delete=true`` →
    refuses on a non-empty board unless ``cascade=true``; on success
    purges every on-disk trace (active dir + every matching
    ``_archived/<slug>-*/`` entry, including all tasks inside).
    Test: Try ``delete=true`` on a non-empty board — expect 409 with
    ``task_count`` in detail. Same call with ``cascade=true`` → 200 and
    the entire board (active + archived dirs) is removed.
    """
    try:
        normed = kanban_db._normalize_board_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not normed:
        raise HTTPException(status_code=400, detail="board slug is required")
    if normed == kanban_db.DEFAULT_BOARD:
        raise HTTPException(
            status_code=409,
            detail={"error": "the 'default' board cannot be removed"},
        )

    if not delete:
        # Archive path — defer to the explicit archive endpoint logic.
        return archive_board_endpoint(slug=normed, cascade=cascade)

    # Hard-delete path. Auto-switch to default if it's the active board —
    # the dashboard always intends to remove the current board, so a 409
    # here just adds friction. We DON'T do this for the legacy archive
    # path because archive is the recoverable action.
    if kanban_db.get_current_board() == normed:
        kanban_db.set_current_board(kanban_db.DEFAULT_BOARD)

    # Non-empty board requires explicit cascade=true opt-in. The integrity
    # rule (refuse-by-default) stays intentional — accidental clicks on a
    # board with content must NOT silently destroy data. Cascade is the
    # power-user / dashboard-confirmed path.
    counts = _board_task_status_counts(normed)
    total = sum(counts.values())
    if total > 0 and not cascade:
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    f"board {normed!r} has {total} task(s) (active or archived); "
                    "pass cascade=true to hard-delete the board AND every "
                    "task on it (irreversible). Consider archive (POST "
                    "/boards/{slug}/archive?cascade=true) for a "
                    "recoverable delete."
                ),
                "task_count": total,
                "counts": counts,
            },
        )

    try:
        res = kanban_db.hard_delete_board(normed)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)})
    return {"result": res, "current": kanban_db.get_current_board()}


@router.post("/tasks/{task_id}/restore")
def restore_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    """Restore an archived task back to status='todo'.

    Why: One-click undo for archived tasks from the dashboard Archive
    Viewer.
    What: Calls :func:`kanban_db.restore_task`; refuses (409) if the task
    isn't currently archived.
    Test: Archive a task, call this endpoint, assert the task's status
    is 'todo' afterwards.
    """
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        if task.status != "archived":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": (
                        f"task {task_id} is not archived (current status: "
                        f"{task.status!r}); nothing to restore"
                    ),
                },
            )
        ok = kanban_db.restore_task(conn, task_id)
        if not ok:
            raise HTTPException(
                status_code=409,
                detail={"error": f"failed to restore task {task_id}"},
            )
        updated = kanban_db.get_task(conn, task_id)
        if updated is not None:
            entered_at = _entered_status_at_for_task(
                conn, updated.id, updated.status, updated.created_at,
            )
            return {
                "ok": True,
                "task": _task_dict(updated, entered_status_at=entered_at),
            }
        return {"ok": True, "task": None}
    finally:
        conn.close()


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


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    # Enforce the dashboard session token as a query param — browsers can't
    # set Authorization on a WS upgrade. This matches how the PTY bridge
    # authenticates in hermes_cli/web_server.py.
    token = ws.query_params.get("token")
    if not _check_ws_token(token):
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
