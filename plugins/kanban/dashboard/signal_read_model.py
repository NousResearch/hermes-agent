"""Bounded, sanitized Kanban snapshot and event contracts for observers."""
from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, status
from fastapi.responses import JSONResponse, Response

from hermes_cli import kanban_db, kanban_db_connect as kbc

router = APIRouter()

_SIGNAL_CAPABILITIES = (
    "kanban.read_model.signal.v1",
    "kanban.events.signal.v1",
)
_SIGNAL_SCHEMA = "kanban.read_model.signal.v1"
_SIGNAL_EVENTS_SCHEMA = "kanban.events.signal.v1"
_SIGNAL_TASK_LIMIT = 500
_SIGNAL_LINK_LIMIT = 2000
_SIGNAL_RUN_LIMIT = 1000
_SIGNAL_AUTH_QUERY_KEYS = frozenset({"token", "ticket", "internal"})
# Consumed by the dashboard's ``_plugin_route_secret_scope`` dependency (the SDK's
# fetchJSON/authedFetch/buildWsUrl append it to every /api/plugins/ URL). Known
# here only so the allowlist accepts it; it never selects or shapes signal data.
_SIGNAL_SCOPE_QUERY_KEYS = frozenset({"profile"})


class SignalUnavailable(RuntimeError):
    """The board exists but its signal state cannot be served: a legacy schema
    without the persisted event generation, or a database not readable now."""


def supports_signal_view() -> bool:
    """Stable capability probe for external dashboard plugins."""
    return True


def _no_store_headers(*, generation: str | None = None, cursor: int | None = None):
    headers = {
        "Cache-Control": "private, no-store",
        "X-Kanban-Capabilities": ", ".join(_SIGNAL_CAPABILITIES),
    }
    if generation is not None:
        headers["X-Kanban-Event-Generation"] = generation
    if cursor is not None:
        headers["X-Kanban-Event-Cursor"] = str(cursor)
    return headers


def _strict_query_values(query_params, *, allowed: set[str], required: set[str]):
    pairs = list(query_params.multi_items())
    counts: dict[str, int] = {}
    for key, _value in pairs:
        counts[key] = counts.get(key, 0) + 1
    unknown = sorted(set(counts) - allowed)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(required - set(counts))
    if unknown or duplicates or missing:
        problems = []
        if unknown:
            problems.append(f"unknown query parameters: {', '.join(unknown)}")
        if duplicates:
            problems.append(f"duplicate query parameters: {', '.join(duplicates)}")
        if missing:
            problems.append(f"missing query parameters: {', '.join(missing)}")
        raise ValueError("; ".join(problems))
    return {key: query_params.get(key) for key in counts}


def _resolve_signal_board(raw: str | None) -> str:
    """Require an existing board expressed as its exact canonical slug."""
    if not raw:
        raise ValueError("board is required")
    try:
        board = kanban_db._normalize_board_slug(raw)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not board or board != raw:
        raise ValueError("board must be a canonical slug")
    if not kanban_db.board_exists(board):
        raise FileNotFoundError(f"board {board!r} does not exist")
    # An archived board leaves a tombstone board.json at its slug; like the
    # plugin's _resolve_board it is gone, never reopened or recreated (#43243).
    if kanban_db.read_board_metadata(board).get("archived"):
        raise FileNotFoundError(f"board {board!r} is archived")
    if not kanban_db.kanban_db_path(board=board).is_file():
        raise FileNotFoundError(f"board {board!r} has no database")
    return board


def _open_signal_connection(board: str) -> sqlite3.Connection:
    """Existing-only, query-only connection whose signal state is materialized.

    Missing databases raise ``FileNotFoundError``; a schema an older binary
    wrote (no ``kanban_metadata``) raises ``SignalUnavailable``. Neither path
    creates, initializes, or migrates anything.
    """
    try:
        conn = kbc._connect_existing_query_only(board=board)
    except FileNotFoundError:
        raise
    except (RuntimeError, sqlite3.DatabaseError) as exc:
        raise SignalUnavailable(f"board {board!r} signal state is unavailable") from exc
    try:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kanban_metadata'"
        ).fetchone()
        if present is None:
            raise SignalUnavailable(
                f"board {board!r} signal state is not materialized"
            )
        return conn
    except BaseException:
        conn.close()
        raise


def _signal_generation(conn: sqlite3.Connection) -> str:
    try:
        return kanban_db.event_generation(conn)
    except RuntimeError as exc:
        raise SignalUnavailable("event generation is not materialized") from exc


def _signal_close_code(exc: BaseException) -> int:
    """WebSocket close code for a signal stream that cannot continue."""
    if isinstance(exc, FileNotFoundError):
        return status.WS_1008_POLICY_VIOLATION
    return status.WS_1011_INTERNAL_ERROR


def _signal_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _signal_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:@-]+", text):
        return "[REDACTED]"
    return text


def _signal_task(row: sqlite3.Row) -> dict:
    status = str(row["status"])
    block_kind = row["block_kind"]
    return {
        "id": _signal_label(row["id"]),
        "status": status if status in kanban_db.VALID_STATUSES else "unknown",
        "priority": _signal_int(row["priority"]) or 0,
        "assignee": _signal_label(row["assignee"]),
        "created_at": _signal_int(row["created_at"]),
        "started_at": _signal_int(row["started_at"]),
        "completed_at": _signal_int(row["completed_at"]),
        "current_run_id": _signal_int(row["current_run_id"]),
        "block_kind": (
            str(block_kind) if block_kind in kanban_db.VALID_BLOCK_KINDS else None
        ),
    }


def _build_signal_read_model(board: str) -> dict:
    conn = _open_signal_connection(board)
    try:
        # Keep generation, cursor, tasks, links, and runs on one WAL snapshot.
        # A concurrent GC must not rotate generation between SELECT statements.
        conn.execute("BEGIN")
        task_rows = conn.execute(
            "SELECT id, status, priority, assignee, created_at, "
            "started_at, completed_at, current_run_id, block_kind "
            "FROM tasks WHERE status != 'archived' "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (_SIGNAL_TASK_LIMIT + 1,),
        ).fetchall()
        visible_tasks = task_rows[:_SIGNAL_TASK_LIMIT]
        link_rows = conn.execute(
            "WITH visible AS ("
            " SELECT id FROM tasks WHERE status != 'archived'"
            " ORDER BY created_at ASC, id ASC LIMIT ?"
            ") SELECT l.parent_id, l.child_id FROM task_links l "
            "JOIN visible p ON p.id = l.parent_id "
            "JOIN visible c ON c.id = l.child_id "
            "ORDER BY l.parent_id ASC, l.child_id ASC LIMIT ?",
            (_SIGNAL_TASK_LIMIT, _SIGNAL_LINK_LIMIT + 1),
        ).fetchall()
        run_rows = conn.execute(
            "WITH visible AS ("
            " SELECT id FROM tasks WHERE status != 'archived'"
            " ORDER BY created_at ASC, id ASC LIMIT ?"
            ") SELECT r.id, r.task_id, r.profile, r.status, r.started_at, "
            "r.ended_at, r.outcome FROM task_runs r "
            "JOIN visible v ON v.id = r.task_id "
            "ORDER BY r.id ASC LIMIT ?",
            (_SIGNAL_TASK_LIMIT, _SIGNAL_RUN_LIMIT + 1),
        ).fetchall()
        cursor_row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS cursor FROM task_events"
        ).fetchone()
        generation = _signal_generation(conn)
        cursor = int(cursor_row["cursor"])
        truncated = (
            len(task_rows) > _SIGNAL_TASK_LIMIT
            or len(link_rows) > _SIGNAL_LINK_LIMIT
            or len(run_rows) > _SIGNAL_RUN_LIMIT
        )
        return {
            "schema": _SIGNAL_SCHEMA,
            "capabilities": list(_SIGNAL_CAPABILITIES),
            "board": board,
            "generation": generation,
            "cursor": cursor,
            "generated_at": int(time.time()),
            "truncated": truncated,
            "tasks": [_signal_task(row) for row in visible_tasks],
            "links": [
                {
                    "source": _signal_label(row["parent_id"]),
                    "target": _signal_label(row["child_id"]),
                    "kind": "parent-child",
                }
                for row in link_rows[:_SIGNAL_LINK_LIMIT]
            ],
            "runs": [
                {
                    "id": _signal_int(row["id"]),
                    "task_id": _signal_label(row["task_id"]),
                    "profile": _signal_label(row["profile"]),
                    "status": _signal_label(row["status"]),
                    "started_at": _signal_int(row["started_at"]),
                    "ended_at": _signal_int(row["ended_at"]),
                    "outcome": _signal_label(row["outcome"]),
                }
                for row in run_rows[:_SIGNAL_RUN_LIMIT]
            ],
        }
    finally:
        if conn.in_transaction:
            conn.rollback()
        conn.close()


def _signal_http_response(request: Request):
    try:
        values = _strict_query_values(
            request.query_params,
            allowed={"board"} | _SIGNAL_SCOPE_QUERY_KEYS,
            required={"board"},
        )
        board = _resolve_signal_board(values["board"])
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=str(exc), headers=_no_store_headers()
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=str(exc), headers=_no_store_headers()
        ) from exc
    try:
        model = _build_signal_read_model(board)
    except FileNotFoundError as exc:
        # The board was archived or removed after validation.
        raise HTTPException(
            status_code=404, detail=str(exc), headers=_no_store_headers()
        ) from exc
    except (SignalUnavailable, sqlite3.OperationalError) as exc:
        raise HTTPException(
            status_code=503,
            detail="kanban signal state is unavailable",
            headers=_no_store_headers(),
        ) from exc
    headers = _no_store_headers(
        generation=model["generation"], cursor=model["cursor"]
    )
    if request.method == "HEAD":
        return Response(status_code=200, headers=headers)
    return JSONResponse(model, headers=headers)


@router.get("/capabilities")
def get_capabilities():
    return JSONResponse(
        {"capabilities": list(_SIGNAL_CAPABILITIES)},
        headers=_no_store_headers(),
    )


@router.get("/read-model/signal")
def get_signal_read_model(request: Request):
    return _signal_http_response(request)


@router.head("/read-model/signal")
def head_signal_read_model(request: Request):
    return _signal_http_response(request)


def _parse_signal_ws_query(ws: WebSocket) -> tuple[str, int, str | None]:
    allowed = (
        {"view", "board", "since", "generation"}
        | _SIGNAL_AUTH_QUERY_KEYS
        | _SIGNAL_SCOPE_QUERY_KEYS
    )
    values = _strict_query_values(
        ws.query_params,
        allowed=allowed,
        required={"view", "board", "since"},
    )
    if values["view"] != "signal":
        raise ValueError("unsupported event view")
    # The canonical gate already admitted this upgrade, possibly through the
    # native ``X-Hermes-Gateway-Ticket`` header or the ``hermes-gateway-ticket.``
    # subprotocol, so no query credential is required here. Several distinct
    # query credentials are still refused as ambiguous.
    if len(set(values) & _SIGNAL_AUTH_QUERY_KEYS) > 1:
        raise ValueError("ambiguous WebSocket credentials")
    board = _resolve_signal_board(values["board"])
    since = values["since"] or ""
    if not re.fullmatch(r"0|[1-9][0-9]*", since):
        raise ValueError("since must be a canonical non-negative integer")
    cursor = int(since)
    if cursor > 9_223_372_036_854_775_807:
        raise ValueError("since exceeds SQLite integer range")
    generation = values.get("generation")
    if generation is not None and not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise ValueError("generation must be 32 lowercase hexadecimal characters")
    if cursor > 0 and generation is None:
        raise ValueError("generation is required when since is greater than zero")
    return board, cursor, generation


def _signal_event_frame(
    conn: sqlite3.Connection,
    cursor: int,
    *,
    initial: bool,
    expected_generation: str | None = None,
) -> dict:
    try:
        # Generation, bounds, and events must describe one WAL snapshot; the
        # read transaction ends with every poll so the reused connection
        # never pins an old snapshot (or blocks checkpoints) between polls.
        conn.execute("BEGIN")
        generation = _signal_generation(conn)
        bounds = conn.execute(
            "SELECT MIN(id) AS min_id, COALESCE(MAX(id), 0) AS max_id "
            "FROM task_events"
        ).fetchone()
        min_id = _signal_int(bounds["min_id"])
        max_id = int(bounds["max_id"])
        reset = (
            (expected_generation is not None and generation != expected_generation)
            or cursor > max_id
            or (min_id is not None and cursor < min_id - 1)
        )
        events: list[dict] = []
        new_cursor = max_id if reset else cursor
        if not reset:
            rows = conn.execute(
                "SELECT id, task_id, run_id, kind, created_at "
                "FROM task_events WHERE id > ? ORDER BY id ASC LIMIT 200",
                (cursor,),
            ).fetchall()
            for row in rows:
                events.append(
                    {
                        "id": int(row["id"]),
                        "task_id": _signal_label(row["task_id"]),
                        "run_id": _signal_int(row["run_id"]),
                        "kind": _signal_label(row["kind"]),
                        "created_at": _signal_int(row["created_at"]),
                    }
                )
                new_cursor = int(row["id"])
        return {
            "schema": _SIGNAL_EVENTS_SCHEMA,
            "initial": initial,
            "generation": generation,
            "reset": reset,
            "cursor": new_cursor,
            "events": events,
        }
    finally:
        if conn.in_transaction:
            conn.rollback()


class _SignalEventReader:
    """Per-socket signal reader: ONE existing-only, query-only connection,
    opened, used, and closed only on a dedicated single-thread executor
    (sqlite3 connections are thread-affine). Reusing it avoids churning the
    WAL/SHM sidecars on every poll, like the legacy ``_EventTail``."""

    def __init__(self, board: str) -> None:
        self._board = board
        self._conn: sqlite3.Connection | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kanban-signal"
        )

    def _frame(
        self, cursor: int, initial: bool, expected_generation: str | None
    ) -> dict:
        if self._conn is None:
            self._conn = _open_signal_connection(self._board)
        return _signal_event_frame(
            self._conn,
            cursor,
            initial=initial,
            expected_generation=expected_generation,
        )

    def _close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    async def frame(
        self, cursor: int, *, initial: bool, expected_generation: str | None
    ) -> dict:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            partial(self._frame, cursor, initial, expected_generation),
        )

    async def aclose(self) -> None:
        # Queued behind any in-flight poll on the connection's own thread.
        closing = self._executor.submit(self._close)
        try:
            await asyncio.shield(asyncio.wrap_future(closing))
        finally:
            # A cancelled handler (server shutdown, or the ASGI server cancelling
            # after disconnect) stops awaiting, but the connection must not
            # outlive it: wait for the queued close, at most one poll behind.
            self._executor.shutdown(wait=True)


async def _stream_signal_events(
    ws: WebSocket,
    board: str,
    cursor: int,
    expected_generation: str | None,
    *,
    poll_seconds: float,
):
    reader = _SignalEventReader(board)
    try:
        frame = await reader.frame(
            cursor, initial=True, expected_generation=expected_generation
        )
        await ws.send_json(frame)
        cursor = int(frame["cursor"])
        generation = str(frame["generation"])
        while True:
            try:
                message = await asyncio.wait_for(
                    ws.receive(), timeout=poll_seconds
                )
                if message["type"] == "websocket.disconnect":
                    return
            except TimeoutError:
                pass
            frame = await reader.frame(
                cursor, initial=False, expected_generation=generation
            )
            if frame["reset"] or frame["events"]:
                await ws.send_json(frame)
                cursor = int(frame["cursor"])
                generation = str(frame["generation"])
    finally:
        await reader.aclose()
