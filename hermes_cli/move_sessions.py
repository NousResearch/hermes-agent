"""How a received move writes conversations: through the store's owner, never beside it.

A running gateway is the only writer of its ``state.db``, so every write goes through its
revision-fenced ``session.mutate`` (``import``, ``sidebar`` for archive, ``delete``). A stopped
install has no owner; the writer then takes the owner lock itself (``exclusive_maintenance``).

:class:`OwnerSessions` takes the mutate call as a function, so the REST route (through
``_mutate_session_request``) and ``hermes move receive`` (through the gateway client) share it.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

# (session_id, request_id, operation, payload, expected_revision, expected_generation) -> result
Mutate = Callable[[str, str, str, dict, int, Optional[int]], dict]


def visible_session_ids(home: Path) -> list[str]:
    """The conversations the target's sidebar shows today (what a replace archives): not
    archived, not hidden plumbing, not a child row of another session."""
    return [row[0] for row in _read(home, "SELECT id FROM sessions WHERE archived = 0 AND hidden = 0 "
                                          "AND parent_session_id IS NULL "
                                          "AND source NOT IN ('tool', 'subagent', 'cron')", ())]


def _read(home: Path, sql: str, args: tuple) -> list[tuple]:
    from hermes_cli.sqlite_safe_read import connect_tracked
    db_path = home / "state.db"
    if not db_path.is_file():
        return []
    try:
        con = connect_tracked(f"file:{db_path}?mode=ro", tracking_path=db_path, uri=True)
        try:
            return list(con.execute(sql, args))
        finally:
            con.close()
    except sqlite3.Error:
        return []


class OwnerSessions:
    """Session writes through the store's owner (a live gateway)."""

    def __init__(self, home: Path, mutate: Mutate, *, move_id: str = ""):
        self._home, self._mutate, self._move_id = home, mutate, move_id

    def _counters(self, session_id: str) -> Optional[tuple[int, Optional[int]]]:
        rows = _read(self._home, "SELECT runtime_revision, runtime_generation FROM sessions WHERE id = ?",
                     (session_id,))
        return (int(rows[0][0]), rows[0][1]) if rows else None

    def import_session(self, session: dict[str, Any]) -> dict[str, Any]:
        sid = session["id"]
        try:
            return self._mutate(sid, f"move-{self._move_id}-{sid}", "import", {"sessions": [session]}, 0, None)
        except RevisionConflict:
            return {"imported": 0, "skipped": 1}

    def set_archived(self, session_id: str, archived: bool) -> None:
        counters = self._counters(session_id)
        if counters is not None:
            self._mutate(session_id, uuid.uuid4().hex, "sidebar", {"archived": archived}, counters[0], None)

    def delete(self, session_id: str) -> None:
        counters = self._counters(session_id)
        if counters is not None:
            self._mutate(session_id, uuid.uuid4().hex, "delete", {}, counters[0], counters[1])


class RevisionConflict(RuntimeError):
    """The owner refused a write because the row changed (or, for an import, already exists)."""


class OfflineSessions:
    """Session writes for a stopped install: the owner lock, then the store directly."""

    def __init__(self, home: Path):
        self._home = home

    def _db(self):
        from hermes_state import SessionDB
        return SessionDB(db_path=self._home / "state.db")

    def _locked(self, fn):
        from gateway.runtime_ownership import exclusive_maintenance
        with exclusive_maintenance([self._home]):
            db = self._db()
            try:
                return fn(db)
            finally:
                db.close()

    def import_session(self, session: dict[str, Any]) -> dict[str, Any]:
        result = self._locked(lambda db: db.import_sessions([session]))
        if not result.get("ok"):
            raise ValueError("; ".join(str(e.get("error")) for e in result.get("errors") or []))
        return result

    def set_archived(self, session_id: str, archived: bool) -> None:
        self._locked(lambda db: db.set_session_archived(session_id, archived))

    def delete(self, session_id: str) -> None:
        self._locked(lambda db: db.delete_sessions([session_id]))
