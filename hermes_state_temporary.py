"""Temporary (incognito) chats for SessionDB.

A temporary chat is an ordinary session row whose ``model_config`` carries a ``temporary`` owner
record. The row exists only so resume, REST transcript hydration and compression keep working while
the chat is open; it is born hidden, excluded from every listing and search, and deleted when its
owner closes it. A crash skips that delete, so :meth:`sweep_orphaned_temporary_sessions` removes
rows whose owning process is provably gone (or, when the owner cannot be probed, idle past
``TEMPORARY_ORPHAN_MAX_IDLE_SECONDS``).
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("hermes_state")

T = TypeVar("T")

TEMPORARY_MODEL_CONFIG_KEY = "temporary"
# Unverifiable owners (another host / PID namespace, no psutil) fall back to idle age. Matches the
# retention window hosted assistants advertise for their temporary chats.
TEMPORARY_ORPHAN_MAX_IDLE_SECONDS = 72 * 3600


def not_temporary_sql(alias: str = "s") -> str:
    """SQL predicate admitting only non-temporary rows of ``sessions`` aliased *alias*. Tolerates a
    NULL or malformed ``model_config`` (json_type on NULL is NULL)."""
    col = f"{alias}.model_config"
    return f"json_type(CASE WHEN json_valid({col}) THEN {col} END, '$.{TEMPORARY_MODEL_CONFIG_KEY}') IS NULL"


def inherit_temporary_marker(model_config: Optional[dict[str, Any]], parent_model_config: Any) -> tuple[Optional[dict[str, Any]], bool]:
    """(child model_config, is temporary): a row derived from a temporary parent carries its marker."""
    from hermes_state_sessions import _parse_model_config

    owner = _parse_model_config(parent_model_config).get(TEMPORARY_MODEL_CONFIG_KEY)
    if owner is None:
        return model_config, False
    return {**(model_config or {}), TEMPORARY_MODEL_CONFIG_KEY: owner}, True


def _process_create_time(pid: int) -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except psutil.Error:
        return None


def _pid_namespace() -> Optional[str]:
    from hermes_state_pidns import pid_namespace_id
    return pid_namespace_id()


def temporary_owner_record() -> dict[str, Any]:
    """The ``model_config.temporary`` value naming THIS process as the chat's owner."""
    pid = os.getpid()
    return {
        "pid": pid, "create_time": _process_create_time(pid), "pidns": _pid_namespace(),
        "host": socket.gethostname(), "created_at": time.time(),
    }


def owner_is_gone(owner: Any) -> Optional[bool]:
    """True when the recorded owner process is provably dead, False when provably alive, None when
    this process cannot tell (other host or PID namespace, malformed record, no psutil)."""
    if not isinstance(owner, dict):
        return None
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return None
    if owner.get("host") != socket.gethostname() or owner.get("pidns") != _pid_namespace():
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        create_time = owner.get("create_time")
        if create_time is not None and abs(float(proc.create_time()) - float(create_time)) >= 2.0:
            return True  # pid reused by a different process
        return proc.status() == getattr(psutil, "STATUS_ZOMBIE", "zombie")
    except psutil.NoSuchProcess:
        return True
    except (psutil.Error, TypeError, ValueError):  # access denied, malformed create_time
        return None


class SessionTemporaryMixin:
    """See module docstring — mixin for SessionDB (temporary chats)."""

    def mark_session_temporary(self, session_id: str, owner: Optional[dict[str, Any]] = None) -> bool:
        """Flag an existing row (and its compression lineage) temporary and hidden, owned by *owner*
        (default: this process — a backend that resumes the chat after a restart takes it over, so
        the orphan sweep never reclaims a chat that is open again). False when the row is missing."""
        if not session_id:
            return False
        self.patch_session_model_config(session_id, {TEMPORARY_MODEL_CONFIG_KEY: owner or temporary_owner_record()})
        return self.set_session_hidden(session_id, True)

    def get_session_temporary_owner(self, session_id: str) -> Optional[dict[str, Any]]:
        """The owner record when *session_id* is a temporary chat, else None."""
        owner = self.get_session_model_config_value(session_id, TEMPORARY_MODEL_CONFIG_KEY)
        return owner if isinstance(owner, dict) else ({} if owner else None)

    def is_session_temporary(self, session_id: str) -> bool:
        return bool(session_id) and self.get_session_temporary_owner(session_id) is not None

    def delete_temporary_session(
        self, session_id: str, sessions_dir: Optional[Path] = None, *, exclude_active_write_guards: bool = False,
    ) -> bool:
        """Delete *session_id* with its whole compression chain — only if it IS a temporary chat, so a
        stale or forged id can never erase a normal conversation. True when rows were removed. With
        *exclude_active_write_guards* a row under a turn lease raises ``SessionActiveWriteGuardError``."""
        if not self.is_session_temporary(session_id):
            return False
        return self._scrubbed(lambda: self.delete_session(
            session_id, sessions_dir=sessions_dir, include_compression_chain=True,
            exclude_active_write_guards=exclude_active_write_guards))

    def _scrubbed(self, delete: Callable[[], T]) -> T:
        """Run *delete* so the removed rows are unrecoverable from the database files, as if never
        written: the writer connection runs ``secure_delete=ON`` (every freed page is zero-filled,
        including segment pages an FTS merge freed while the chat was open), FTS5 ``secure-delete``
        drops the terms from the live index segments (not just a tombstone until the next merge),
        and a final
        checkpoint copies the zeroed pages over the old ones and empties the WAL that still holds
        the chat's earlier page images — as close to never written as a shared ``state.db`` row gets."""
        fts_tables = self._fts5_tables()
        self._set_fts_secure_delete(fts_tables, True)
        try:
            return delete()
        finally:
            self._set_fts_secure_delete(fts_tables, False)
            self._checkpoint_and_empty_wal()

    def _fts5_tables(self) -> list[str]:
        if not getattr(self, "_fts_enabled", True):
            return []
        rows = self._read_all("SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%USING fts5%'")
        return [r["name"] for r in rows]

    def _set_fts_secure_delete(self, fts_tables: list[str], on: bool) -> None:
        def _do(conn) -> None:
            for name in fts_tables:  # names read from sqlite_master, not caller input
                conn.execute(f'INSERT INTO "{name}"("{name}", rank) VALUES (\'secure-delete\', ?)', (int(on),))
        try:
            if fts_tables:
                self._execute_write(_do)
        except sqlite3.Error as exc:  # pre-3.42 SQLite has no FTS5 secure-delete; the row delete still runs
            logger.debug("FTS5 secure-delete unavailable: %s", exc)

    def _checkpoint_and_empty_wal(self) -> None:
        """PASSIVE first (never blocks writers); reset the WAL only once nothing is left to copy, so
        the TRUNCATE never does the bulk-frame work behind #45383. A reader holding an old snapshot
        keeps the WAL; its frames are reused from the start once that reader moves on."""
        try:
            with self._lock:
                if self._conn is None:
                    return
                busy, log_frames, copied = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if not busy and log_frames == copied:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            logger.debug("temporary-chat WAL checkpoint skipped: %s", exc)

    def _temporary_session_rows(self) -> list[dict[str, Any]]:
        rows = self._read_all(
            f"""SELECT s.id, s.model_config, s.started_at,
                       COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id),
                                s.started_at) AS last_active
                FROM sessions s WHERE NOT ({not_temporary_sql('s')})"""
        )
        return [dict(r) for r in rows]

    def sweep_orphaned_temporary_sessions(
        self, sessions_dir: Optional[Path] = None, *, keep_ids: Optional[set[str]] = None,
        now: Optional[float] = None,
    ) -> list[str]:
        """Delete temporary chats whose owner process is gone (crash, kill -9, power loss). Rows in
        *keep_ids* or under an active turn lease are never touched. Returns the deleted ids."""
        from hermes_state_errors import SessionActiveWriteGuardError
        from hermes_state_sessions import _parse_model_config

        now = time.time() if now is None else now
        keep = set(keep_ids or ())
        deleted: list[str] = []
        for row in self._temporary_session_rows():
            sid = row["id"]
            if sid in keep:
                continue
            owner = _parse_model_config(row.get("model_config")).get(TEMPORARY_MODEL_CONFIG_KEY)
            gone = owner_is_gone(owner)
            if gone is False:
                continue
            if gone is None and now - float(row.get("last_active") or 0) < TEMPORARY_ORPHAN_MAX_IDLE_SECONDS:
                continue
            try:
                if self._scrubbed(lambda: self.delete_session(
                        sid, sessions_dir=sessions_dir, exclude_active_write_guards=True,
                        include_compression_chain=True)):
                    deleted.append(sid)
            except SessionActiveWriteGuardError:
                continue
            except Exception:
                logger.debug("temporary-session sweep failed for %s", sid, exc_info=True)
        if deleted:
            logger.info("Removed %d orphaned temporary chat(s)", len(deleted))
        return deleted


__all__ = [
    "TEMPORARY_MODEL_CONFIG_KEY", "TEMPORARY_ORPHAN_MAX_IDLE_SECONDS", "SessionTemporaryMixin",
    "inherit_temporary_marker", "not_temporary_sql", "owner_is_gone", "temporary_owner_record",
]
