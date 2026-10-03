"""Read-only, profile-scoped idle analysis for ``subc`` ticket drafts.

This module deliberately has no scheduler registration and no write path.  A caller may
invoke it while idle (or on demand), review the returned drafts, and choose whether to
create an issue.  Database failures are treated as no observation, never as a reason to
repair or initialize a store.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

logger = logging.getLogger(__name__)

_FAILURE_REASONS = ("error", "agent_error", "content_filter")


@dataclass(frozen=True, slots=True)
class TicketDraft:
    """A proposed ticket only; producing one never creates or changes a ticket."""

    kind: str
    title: str
    body: str
    dedup_key: str
    occurrences: int


def _read_only_connection(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite database without allowing SQLite to create side files."""
    return sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)


def _failure_counts(state_db: Path, profile_name: str) -> list[tuple[str, int]]:
    if not state_db.is_file() or not profile_name.strip():
        return []
    try:
        conn = _read_only_connection(state_db)
        try:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(
                """
                SELECT lower(trim(m.finish_reason)) AS reason, COUNT(DISTINCT s.id) AS occurrences
                  FROM sessions AS s
                  JOIN messages AS m ON m.session_id = s.id
                 WHERE s.profile_name = ?
                   AND m.active = 1
                   AND m.role = 'assistant'
                   AND lower(trim(m.finish_reason)) IN (?, ?, ?)
                 GROUP BY lower(trim(m.finish_reason))
                """,
                (profile_name.strip(), *_FAILURE_REASONS),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        logger.debug("subc skipped unreadable session history: %s", state_db, exc_info=True)
        return []
    return [(str(reason), int(occurrences)) for reason, occurrences in rows]


def _ticket_keys(kanban_db: Path) -> set[str]:
    """Read known subc keys from an existing board; absent or incompatible boards are empty."""
    if not kanban_db.is_file():
        return set()
    try:
        conn = _read_only_connection(kanban_db)
        try:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute("SELECT title, body FROM tasks").fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        logger.debug("subc skipped unreadable kanban board: %s", kanban_db, exc_info=True)
        return set()
    return {
        token
        for title, body in rows
        for token in (str(title or "") + "\n" + str(body or "")).split()
        if token.startswith("subc:")
    }


def analyze_idle_time(
    state_db: Path, *, profile_name: str, existing_ticket_keys: Iterable[str] = (), min_occurrences: int = 3,
) -> list[TicketDraft]:
    """Return deduplicated repeated-failure drafts for one profile, failing open on all reads."""
    if min_occurrences < 2:
        raise ValueError("min_occurrences must be at least 2")
    known = {str(key) for key in existing_ticket_keys}
    drafts: list[TicketDraft] = []
    for reason, occurrences in _failure_counts(Path(state_db), profile_name):
        key = f"subc:failure:{reason}"
        if occurrences < min_occurrences or key in known:
            continue
        drafts.append(TicketDraft(
            kind="nightmare",
            title=f"Nightmare: {reason} recurred across {occurrences} sessions",
            body=(
                f"Detected {occurrences} distinct failed sessions in profile `{profile_name}` "
                f"with finish reason `{reason}`. Review the session history before acting.\n\n{key}"
            ),
            dedup_key=key,
            occurrences=occurrences,
        ))
    return drafts


def analyze_profile_idle_time(
    *, profile_home: Path, profile_name: str, min_occurrences: int = 3, kanban_db: Path | None = None,
) -> list[TicketDraft]:
    """Profile-level seam used by idle callers; it reads, but never initializes, either store."""
    home = Path(profile_home)
    if kanban_db is None:
        try:
            from hermes_cli.kanban_db import kanban_db_path

            kanban_db = kanban_db_path()
        except Exception:  # The analysis still has useful session-only observations.
            kanban_db = home / "kanban.db"
    return analyze_idle_time(
        home / "state.db",
        profile_name=profile_name,
        existing_ticket_keys=_ticket_keys(Path(kanban_db)),
        min_occurrences=min_occurrences,
    )
