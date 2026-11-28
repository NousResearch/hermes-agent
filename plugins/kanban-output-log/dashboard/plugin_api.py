"""Kanban Output Log — a read-only transcript feed for a Kanban card's worker.

WHAT THIS PROVIDES
------------------
One read-only route, ``GET /api/plugins/kanban-output-log/tasks/{task_id}/output``, returning
a tail of the transcript of the worker currently (or most recently) assigned to a Kanban
card. The Desktop half of this same plugin
(``apps/desktop/src/plugins/kanban-output-log/``) polls it and renders an "Output log" section
inside the card drawer, so an operator can watch a running card work instead of guessing
whether it is wedged.

WHY IT READS ``state.db`` RATHER THAN THE CARD'S WORKER LOG
-----------------------------------------------------------
A card's per-task worker log (``<board>/logs/<task_id>.log``) holds only the ``Session:``
line, the exit trailer and the final reply for an owner-admitted worker, and it does not grow
while the worker runs — issues #144334 and #143004 track that. The worker's actual activity
(assistant text, thinking, tool calls, tool results) is persisted in the *owning profile's*
``state.db``, keyed by the worker session id. This route resolves

    board -> task -> current run's owner-admitted session -> owning profile's state.db -> transcript tail

so the record Hermes already keeps is reachable from the surface that shows the card.

HOW THE SESSION IS RESOLVED
---------------------------
Most authoritative first, so a profile rename or a re-assignment cannot point the reader at
the wrong session:

1. the ``owner_admitted`` event of the task's *current run* (``task_events``), whose
   payload carries both the live ``session_id`` and the ``state.db`` the owner wrote it to;
2. that run's ``task_runs.metadata.worker_session_id``;
3. only on legacy cards without a run, the ``Session: <id>`` per-task log line.
A newly claimed run with no session shows no output, never the previous run's transcript.

READ-ONLY, THROUGH THE STORAGE INTERFACE
----------------------------------------
This module never writes a Hermes database.

* The **profile store** is read through ``persistence_assembly.open_persistence_reader`` —
  the storage interface's owned, read-only handle, also used by the achievements plugin.
  It refuses to create, migrate or repair the store, and permits reading a different
  profile's ``state.db`` without opening a writable handle beside that profile's owner.
* The **transcript** comes from ``messages.read_transcript(..., TranscriptView.DISPLAY)``, so
  the feed shows the same logical messages every other display surface shows: compaction
  archives deduped to the surviving row, rewind-dropped rows excluded. Hand-rolling this over
  the raw ``messages`` table would have to re-derive that projection and would drift from it.
* The **board** database is opened directly on its file with the codebase's
  ``sqlite_support.read_only_db_uri`` (``mode=ro``) plus ``PRAGMA query_only=ON``; only
  ``SELECT``s run. It is deliberately *not* opened through ``kanban_db_connect.connect``,
  whose ``mode=ro`` branch only fires for fenced dispatched descendants: a server-side call
  there would ``mkdir`` the board directory, take ``<db>.init.lock`` and run the
  schema/migrations. A missing or uninitialised board is reported (404/409), never created.

Read-only also means the *envelope* is read-only: opening a WAL database with ``mode=ro``
makes SQLite materialise empty ``-wal``/``-shm`` sidecars when no process currently holds that
database open. That is a property of SQLite's read-only open, not a write to any table, and it
only arises for a quiescent board — an active board already has them.

THE CONTENT IS RETURNED VERBATIM; CROSS-PROFILE PRIVACY NEEDS REVIEW
-----------------------------------------------------------------
Entry text is capped by length but otherwise unredacted, including thinking, tool arguments
and tool output. The board identifies the worker's owning profile; the authenticated dashboard
route reads that profile's store even when a different profile is active in the viewer. This
is a deliberate limitation of this draft, not an assurance that secrets are absent. A future
server-side redaction-on-by-default policy needs a maintainer decision.

FAILURE MODES
-------------
Every path answers with a stated ``state`` instead of a 500, and no state creates or mutates
anything:

* ``ok``               — entries returned;
* ``no_session``       — queued / never started: no worker session is recorded;
* ``session_missing``  — a session id is recorded but that session is gone from the store;
* ``no_messages``      — the session exists but has no display rows recorded yet;
* ``no_db``            — the owning profile's ``state.db`` is absent or could not be read.

A malformed ``board`` is a 400, an unknown/archived board or unknown task a 404, and an
out-of-range ``tail``/``before`` a 422 — all from the board library's own validation.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)

router = APIRouter()

_BOARD_Q = Query(None, description="Kanban board slug (omit for the current board)")

# Feed-line caps. The browser truncates further for display; these bound the payload so one
# huge tool argument or tool result cannot bloat a 5 s poll.
_LINE_CHARS = 600
_TOOL_ARG_CHARS = 240
_TAIL_DEFAULT = 200
_TAIL_MAX = 2000

# The `Session: <id>` line the worker client prints to its per-task log — the last-resort
# fallback when the board carries neither an owner-admitted marker nor a run's
# worker_session_id. Anchored per line, so prose mentioning a session cannot match.
_LOG_SESSION_RE = re.compile(r"^Session:\s*(\S+)\s*$", re.MULTILINE)

# A profile name safe to join into ``profiles/<name>/state.db``. ``.`` and ``..`` match this
# pattern, so they are rejected explicitly at the call site.
_PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# --- small helpers -----------------------------------------------------------

def _decode_json(raw: Any) -> Any:
    """Decode a JSON text column; tolerate a value that is already decoded."""
    if not raw:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _clamp(text: Any, limit: int) -> str:
    """Trim to *limit* characters, keeping the head of the string."""
    s = "" if text is None else str(text)
    s = s.rstrip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _one_line(text: Any, limit: int) -> str:
    """Collapse all whitespace so a multi-line argument/result reads as one line."""
    return _clamp(" ".join(str(text or "").split()), limit)


def _tool_calls(raw: Any) -> list[dict[str, Any]]:
    """A decoded ``tool_calls`` value -> ``[{name, arguments}]`` (tolerant of odd shapes).

    The store hands ``tool_calls`` over already decoded, so this only tolerates the shapes a
    provider can produce, not JSON text.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        fn_raw = call.get("function")
        fn: dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else {}
        name = fn.get("name") or call.get("name") or call.get("tool_name")
        args = fn.get("arguments", call.get("arguments"))
        if args is not None and not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        out.append({"name": name, "arguments": args})
    return out


def _connect_board_readonly(path: Path) -> sqlite3.Connection:
    """A genuinely read-only connection to the *board* database.

    ``read_only_db_uri`` is the codebase's ``mode=ro`` URI builder — it uses ``as_uri()`` so a
    ``?`` or ``#`` in the home path cannot truncate the URI and silently open the wrong (empty)
    database. ``query_only`` is a second, in-connection lock: code that later reaches this
    handle still cannot write through it.
    """
    from sqlite_support import read_only_db_uri

    conn = sqlite3.connect(read_only_db_uri(path), uri=True)  # health: allow HX015 -- the board database, never a profile state.db
    conn.execute("PRAGMA query_only=ON")
    return conn


# --- board / profile resolution ---------------------------------------------

def _open_board(board: Optional[str]) -> tuple[Optional[str], sqlite3.Connection]:
    """Normalise *board* and open a read-only connection to its database.

    Raises 400 for a malformed slug, 404 for a board that does not exist or is archived, 404
    when its database is not on disk, and 409 when it is on disk but is not a usable kanban
    database. Never creates, migrates or locks anything.
    """
    from hermes_cli import kanban_db

    normed: Optional[str] = None
    if board:
        try:
            normed = kanban_db._normalize_board_slug(board)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if normed and normed != kanban_db.DEFAULT_BOARD:
            if not kanban_db.board_exists(normed):
                raise HTTPException(status_code=404, detail=f"board {normed!r} does not exist")
            if kanban_db.read_board_metadata(normed).get("archived"):
                raise HTTPException(status_code=404, detail=f"board {normed!r} is archived")

    path = kanban_db.kanban_db_path(board=normed)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"board {normed or kanban_db.DEFAULT_BOARD!r} has no database on disk",
        )
    try:
        conn = _connect_board_readonly(path)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=409, detail=f"board database is not readable: {exc}")
    conn.row_factory = sqlite3.Row
    conn.text_factory = kanban_db._lossy_text
    try:
        initialised = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks' LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        initialised = False
    if not initialised:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=f"board {normed or kanban_db.DEFAULT_BOARD!r} is not initialised",
        )
    return normed, conn


def _profile_state_db(profile: Optional[str]) -> Optional[Path]:
    """The owning profile's ``state.db`` path (``profiles/<name>/state.db``, or the root
    ``state.db`` for the default profile), or ``None`` for a name that cannot be a profile.

    *profile* comes from board data, so it is validated before it is joined into a path: only
    ``[A-Za-z0-9._-]`` is accepted, and ``.``/``..`` (or anything with a path separator)
    yields ``None``.
    """
    try:
        from hermes_constants import get_default_hermes_root

        root = Path(get_default_hermes_root())
    except (ImportError, OSError):
        log.warning("kanban-output-log: no Hermes root, so no profile store", exc_info=True)
        return None
    if not profile or profile == "default":
        return root / "state.db"
    if profile in (".", "..") or not _PROFILE_RE.match(profile):
        return None
    return root / "profiles" / profile / "state.db"


def _validated_state_db(db: Any) -> Optional[Path]:
    """*db* (a ``task_events.payload.db`` value) only when it is a ``state.db`` under the
    Hermes root, else ``None``.

    The payload is attacker-influenceable (any writer of an ``owner_admitted`` event), so an
    absolute path taken from it must not be allowed to steer this reader at an arbitrary file.
    Accepting only a ``state.db`` beneath the Hermes root keeps the payload's useful property —
    it names the exact store the owner used — without trusting it as a path.
    """
    if not isinstance(db, str) or not db:
        return None
    try:
        from hermes_constants import get_default_hermes_root

        root = Path(get_default_hermes_root()).resolve()
        candidate = Path(db).expanduser().resolve()
    except (ImportError, OSError, ValueError):
        return None
    if candidate.name != "state.db":
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _resolve_worker_session(
    conn: sqlite3.Connection, task_id: str, assignee: Optional[str], board: Optional[str],
    current_run_id: Optional[int],
):
    """``(session_id, state_db, profile, run_id, source)`` for the current attempt.

    Match the core's current-run admission fence (``gateway/session_kanban.py`` and
    ``kanban_owner_recovery._owner_receipt``). That helper is private core, not a
    published plugin API; the read-only query here preserves the same run scope.
    The task log is not run-scoped and is safe only on legacy cards with no run.
    """
    if current_run_id is not None:
        run = conn.execute(
            "SELECT id, profile, metadata FROM task_runs WHERE id=? AND task_id=?",
            (current_run_id, task_id),
        ).fetchone()
        if run is None:
            return None
        profile = run["profile"] or assignee
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='owner_admitted' ORDER BY id DESC LIMIT 1",
            (task_id, current_run_id),
        ).fetchone()
        if row is not None:
            payload = _decode_json(row["payload"]) or {}
            sid = payload.get("session_id") if isinstance(payload, dict) else None
            db_path = _validated_state_db(payload.get("db")) if isinstance(payload, dict) else None
            if isinstance(sid, str) and sid and db_path is not None:
                return sid, db_path, profile, current_run_id, "owner_admitted"
        md = _decode_json(run["metadata"]) or {}
        sid = md.get("worker_session_id") if isinstance(md, dict) else None
        if isinstance(sid, str) and sid:
            return sid, _profile_state_db(profile), profile, current_run_id, "run_metadata"
        return None

    from hermes_cli import kanban_db

    content = kanban_db.read_worker_log(task_id, board=board)
    if content:
        m = _LOG_SESSION_RE.search(content)
        if m:
            return m.group(1), _profile_state_db(assignee), assignee, None, "worker_log"

    return None


# --- transcript read (through the storage interface) -------------------------

def _read_transcript(state_db: Path, session_id: str, tail: int, before: Optional[int]):
    """``(rows_chronological, total_logical_messages, has_earlier)`` or a state string.

    ``total`` counts every logical message a display read of the session shows, not just the
    window, so the client can render "N of M" and know whether a pager is worth offering.

    The newest window is a single paged read — that is the path a 5 s poll takes. Paging
    further back needs the cursor's index in the display sequence, which the interface exposes
    no direct lookup for, so that path reads the session's display rows once and slices; it
    runs on a user click, never on the poll.
    """
    from persistence.conversation.messages import TranscriptRead, TranscriptView
    from persistence_assembly import open_persistence_reader

    if not state_db.is_file():
        return "no_db"
    try:
        with open_persistence_reader(state_db) as store:
            total = store.messages.display_count(session_id)
            if total == 0:
                # A session row that is gone reads differently from one with no rows yet.
                gone = store.sessions.get(session_id) is None
                return "session_missing" if gone else "no_messages"
            if before is None:
                rows = store.messages.read_transcript(
                    session_id, TranscriptRead(view=TranscriptView.DISPLAY, latest=True, limit=tail))
                has_earlier = total > len(rows)
            else:
                everything = store.messages.read_transcript(
                    session_id, TranscriptRead(view=TranscriptView.DISPLAY))
                cursor = next((i for i, r in enumerate(everything) if r.id >= before), len(everything))
                start = max(0, cursor - tail)
                rows = everything[start:cursor]
                has_earlier = start > 0
    except Exception:
        # A panel that cannot read a transcript states that; it must not 500 the drawer. The
        # traceback is logged so the cause is diagnosable.
        log.warning("kanban-output-log: could not read a worker transcript", exc_info=True)
        return "no_db"

    return list(rows), int(total), has_earlier


def _entries_from_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Expand transcript rows into ordered feed lines (one row can yield several).

    ``seq`` disambiguates the lines a single row produces — thinking, then text, then one per
    tool call — so the client can key each line without collisions.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        role = (r.role or "").lower()
        ts = r.timestamp
        mid = r.id
        if role == "assistant":
            reasoning = r.reasoning_content or r.reasoning
            if reasoning and str(reasoning).strip():
                out.append({"id": mid, "seq": 0, "kind": "thinking", "role": role,
                            "text": _clamp(reasoning, _LINE_CHARS), "ts": ts})
            if r.content and str(r.content).strip():
                out.append({"id": mid, "seq": 1, "kind": "text", "role": role,
                            "text": _clamp(r.content, _LINE_CHARS), "ts": ts})
            for i, call in enumerate(_tool_calls(r.tool_calls)):
                out.append({"id": mid, "seq": 2 + i, "kind": "tool_call", "role": role,
                            "tool_name": call.get("name"),
                            "text": _one_line(call.get("arguments"), _TOOL_ARG_CHARS), "ts": ts})
        elif role == "tool":
            out.append({"id": mid, "seq": 0, "kind": "tool_result", "role": role,
                        "tool_name": r.tool_name, "text": _one_line(r.content, _LINE_CHARS), "ts": ts})
        elif r.compressed_summary:
            out.append({"id": mid, "seq": 0, "kind": "summary", "role": role,
                        "text": _clamp(r.content, _LINE_CHARS), "ts": ts})
        else:
            out.append({"id": mid, "seq": 0, "kind": role or "message", "role": role,
                        "text": _clamp(r.content, _LINE_CHARS), "ts": ts})
    return out


def _empty(task_id: str, state: str, **extra: Any) -> dict[str, Any]:
    """The shape every non-``ok`` answer shares, so the client has one thing to render."""
    return {"task_id": task_id, "state": state, "session_id": None, "profile": None,
            "entries": [], "total": 0, "returned": 0, "has_earlier": False, "next_before": None,
            "started_at": None, "last_at": None, **extra}


# --- GET /tasks/{task_id}/output --------------------------------------------

@router.get("/tasks/{task_id}/output")
def get_task_output(
    task_id: str,
    board: Optional[str] = _BOARD_Q,
    tail: int = Query(_TAIL_DEFAULT, ge=1, le=_TAIL_MAX,
                      description="Message rows to return (the newest ones)"),
    before: Optional[int] = Query(None, ge=1,
                                  description="Return rows older than this message id (paging)"),
):
    """A tail of the task worker's transcript, shaped into feed lines.

    The module docstring holds the resolution order, the read-only guarantees and the meaning
    of every ``state`` value. ``next_before`` is the cursor to hand back as ``before`` for
    "Load earlier". The owning profile is named by ``profile``; its absolute ``state.db`` path
    is never disclosed.
    """
    normed, conn = _open_board(board)
    try:
        task_row = conn.execute(
            "SELECT id, assignee, status, current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task_row is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")

        assignee = task_row["assignee"]
        resolved = _resolve_worker_session(conn, task_id, assignee, normed, task_row["current_run_id"])
        if resolved is None:
            return _empty(task_id, "no_session",
                          task_status=task_row["status"], assignee=assignee)
        session_id, db_path, profile, run_id, source = resolved
        context = {"session_id": session_id, "profile": profile, "run_id": run_id,
                   "source": source, "task_status": task_row["status"], "assignee": assignee}

        if db_path is None or not Path(db_path).is_file():
            return _empty(task_id, "no_db", **context)

        result = _read_transcript(Path(db_path), session_id, int(tail), before)
        if isinstance(result, str):  # a stated state, not a window
            return _empty(task_id, result, **context)
        rows, total, has_earlier = result
        entries = _entries_from_rows(rows)

        return {
            "task_id": task_id,
            "state": "ok",
            "total": total,
            "returned": len(entries),
            "has_earlier": has_earlier,
            "next_before": rows[0].id,
            "started_at": rows[0].timestamp,
            "last_at": rows[-1].timestamp,
            "entries": entries,
            **context,
        }
    finally:
        conn.close()
