"""Per-board kanban pause — a resumable stop for NEW dispatch on one board.

``hermes kanban pause`` writes a sentinel at ``<board_dir>/PAUSED``; ``hermes
kanban resume`` removes it. While it exists the dispatcher spawns nothing on
that board (auto-decompose, ready-queue fan-out and review dispatch all keep
off) and in-flight workers are never touched — a paused board drains, it does
not kill. Same contract as the machine-wide emergency stop
(:mod:`agent.estop`), scoped to one board so an operator can hold a single
board while other boards keep working.

The sentinel body is optional JSON ``{"reason", "paused_at"}``; a corrupt or
empty file still counts as paused (fail safe, e.g. ``touch .../PAUSED``).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import suppress
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Optional

PAUSE_FILE = "PAUSED"


def _kb():
    # Late-bound: kanban_db imports this module's callers, so a top-level
    # import would be circular (same pattern as kanban_db_boards).
    from hermes_cli import kanban_db

    return kanban_db


def resolve_slug(board: Optional[str] = None) -> str:
    """The board *board* names, resolving ``None`` the way ``connect()`` does.

    ``board_dir(None)`` answers for ``default`` while ``connect(None)`` opens the *current*
    board, so every pause entry point that took the raw argument watched a different board
    than the work it guards: ``hermes kanban pause`` on a switched board wrote the sentinel to
    ``default``, and a dashboard ``POST /pause`` without ``?board`` paused ``default`` while its
    own response named the current board. One resolver, used everywhere here, is what makes a
    bare ``pause()`` mean "the board I am looking at".
    """
    if board:
        return _kb()._slug_or_default(board)
    return _kb().get_current_board()


def pause_path(board: Optional[str] = None) -> Path:
    """``<board_dir>/PAUSED`` — the per-board pause sentinel."""
    return _kb().board_dir(resolve_slug(board)) / PAUSE_FILE


def is_paused(board: Optional[str] = None) -> bool:
    """True while *board* is paused. Fail SAFE (True) on stat errors: a board
    we cannot inspect must not flood the provider."""
    try:
        return pause_path(board).exists()
    except OSError:
        return True


def board_of_connection(conn: Any) -> Optional[str]:
    """Slug of the board a live connection serves, or ``None`` when it isn't a board DB.

    Read from the file the connection actually opened, not from the argument the caller
    happened to pass: ``connect()`` resolves ``board=None`` to the *current* board, while
    ``board_dir(None)`` answers for ``default``. Keying the sentinel off the argument
    therefore watched a different board than the tick wrote to.
    """
    try:
        rows = conn.execute("PRAGMA database_list").fetchall() or ()
    except (sqlite3.Error, AttributeError):  # pragma: no cover - closed/odd connection
        return None
    path = next((Path(row[2]) for row in rows if row[1] == "main" and row[2]), None)
    if path is None:
        return None
    kb = _kb()
    try:
        parts = path.resolve().parent.resolve().relative_to(kb.boards_root().resolve()).parts
    except (ValueError, OSError):
        # The default board's DB lives at the kanban root, and a HERMES_KANBAN_DB override can
        # point anywhere: neither names a named board, so the sentinel they fall back to is the
        # default one.
        return kb.DEFAULT_BOARD
    # ``boards/<slug>/…`` at any depth is that board (workspaces keep DBs in subdirectories).
    return parts[0] if parts else kb.DEFAULT_BOARD


def is_paused_for(conn: Any) -> bool:
    """Is the board this *connection* serves paused?

    The dispatcher's guard. Callers pass ``board=None`` (the CLI and the daemon both do),
    so the sentinel has to be resolved the way the connection resolved it.
    """
    return is_paused(board_of_connection(conn))


def get_state(board: Optional[str] = None) -> Optional[dict[str, Any]]:
    """``{"board", "reason", "paused_at"}`` when paused, else ``None``.

    An unreadable/corrupt body still reports paused with both fields ``None``.
    """
    if not is_paused(board):
        return None
    slug = resolve_slug(board)
    state: dict[str, Any] = {"board": slug, "reason": None, "paused_at": None}
    with suppress(OSError, ValueError, AttributeError):
        raw = json.loads(pause_path(board).read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            state["reason"] = raw.get("reason") or None
            state["paused_at"] = raw.get("paused_at") or None
    return state


def pause(board: Optional[str] = None, reason: Optional[str] = None) -> dict[str, Any]:
    """Pause *board* (idempotent). Re-pausing refreshes reason/timestamp.

    Returns the state, so callers can echo it back without a second read.
    """
    slug = resolve_slug(board)
    if slug != _kb().DEFAULT_BOARD:
        if not _kb().board_exists(slug):
            raise ValueError(f"unknown board {slug!r}")
    path = pause_path(slug)
    payload = {
        "board": slug,
        "paused_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": (reason or "").strip() or None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        # Best effort: an empty/partial sentinel still pauses (fail safe).
        with suppress(OSError):
            path.touch(exist_ok=True)
    return payload


def resume(board: Optional[str] = None) -> bool:
    """Lift the pause. Returns True when a sentinel was removed."""
    try:
        pause_path(board).unlink()
        return True
    except (OSError, AttributeError):
        return False


def paused_boards() -> list[str]:
    """Slugs of every board currently paused (metadata-aware, never raises).

    Never raises is a contract, not a convenience: this feeds the CLI listing and the
    dashboard's status line, so an unreadable board index must read as "nothing known to
    be paused" rather than take the caller down with it.
    """
    try:
        from hermes_cli import kanban_db_boards as boards

        slugs = [entry["slug"] for entry in boards.list_boards(include_archived=False)]
    except (ImportError, OSError, sqlite3.Error, KeyError, TypeError):
        return []
    return [slug for slug in slugs if is_paused(slug)]
