"""Capability-gated ``PluginContext`` surface: versioned read-only access to
``state.db`` (sessions and transcripts) and ``kanban.db`` (tasks).

Plugins used to read both databases with raw SQL against hand-resolved paths:
coupled to internal schema names, able to open ``kanban.db`` in read-write
mode (creating or migrating a board from an observer), and reading ``state.db``
outside the owning profile's home. This facade replaces that with versioned
calls that return plain JSON dicts:

- ``state.db`` is read through the shared registry-owned ``SessionDB`` of this
  manager's profile home, using only its public read methods.
- ``kanban.db`` is opened read-only (``mode=ro``) and never initialized or
  migrated: a missing board or schema is a clean ``None``/empty result, never
  a created file.
- Every response carries its contract version (``schema_version`` for
  ``state.db``, ``api_version`` for kanban — kanban has no schema-version
  marker today, so the facade versions its own field list instead).

The methods are generic plugin surface: they name no plugin, no team and no
product concept. They need the ``state.read`` capability; without it they
return ``None`` / empty results and log once (fail closed, like
``side_question``). Reads are profile-wide, not limited to the session that
fired the hook — the consent description says so.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugins_loader import _plugin_home_scope

logger = logging.getLogger(__name__)

STATE_READ_CAPABILITY = "state.read"

# Contract version of the kanban DTOs below. Kanban has no schema-version
# marker; the facade guarantees the listed fields exist whatever additive
# migrations the schema has applied.
KANBAN_API_VERSION = 1

_READ_MESSAGE_LIMIT_MAX = 100
_READ_TASK_LIMIT_MAX = 100

# Closed field list for a kanban task DTO. Deliberately excludes internals
# (claim_lock, workspace_path, worker_pid) and free-text bulk (body).
_TASK_DTO_FIELDS = (
    "id",
    "title",
    "status",
    "assignee",
    "priority",
    "session_id",
    "created_at",
    "started_at",
    "completed_at",
    "result",
    "block_kind",
    "block_recurrences",
    "completion_contract",
    "current_step_key",
)


class PluginStateReadCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``manifest``, ``_manager``, ``plugin_id``)."""

    def _state_read_granted(self) -> bool:
        """Consent read from THIS manager's profile home: plugins call from their own threads,
        where the active profile may be the launch profile's."""
        try:
            from hermes_cli.config import load_config

            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:  # health: allow BLE001 -- fail closed: unreadable consent state never grants
            return False
        return plugin_capability_granted(
            self.plugin_id, STATE_READ_CAPABILITY, config=cfg
        )

    def _state_read_denied(self) -> bool:
        """Fail-closed denial: log once, report not-granted."""
        if not getattr(self, "_state_read_denial_logged", False):
            self._state_read_denial_logged = True
            logger.info(
                "Plugin %s called a state-read method without the %s capability; returning empty",
                self.plugin_id,
                STATE_READ_CAPABILITY,
            )
        return False

    def _acquire_session_db(self):  # type: ignore[no-untyped-def] -- SessionDB imported lazily
        """Shared registry-owned SessionDB for this manager's profile home."""
        from hermes_state_registry import acquire

        return acquire(self._manager.home_path / "state.db")

    def _release_session_db(self, db) -> None:  # type: ignore[no-untyped-def]
        from hermes_state_registry import release

        release(db)

    def _state_schema_version(self) -> Optional[int]:
        """``schema_version`` of this profile's state.db, read-only; None when absent."""
        path = self._manager.home_path / "state.db"
        if not path.is_file():
            return None
        try:
            conn = sqlite3.connect(
                f"file:{path.resolve().as_posix()}?mode=ro", uri=True
            )
            try:
                row = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        return int(row[0]) if row else None

    def read_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Session row for *session_id* (resolved to the live continuation tip first),
        as returned by ``SessionDB.get_session``; None when unknown or not granted."""
        if not self._state_read_granted():
            self._state_read_denied()
            return None
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("read_session requires a non-empty session_id")
        db = self._acquire_session_db()
        try:
            target = db.resolve_resume_session_id(session_id)
            return db.get_session(target)
        finally:
            self._release_session_db(db)

    def resolve_session(self, session_id: str) -> str:
        """Resolve *session_id* to the live continuation tip (compression-aware).

        When the capability is not granted, returns *session_id* unchanged: the
        caller learns nothing new, and no resolution is performed."""
        if not self._state_read_granted():
            self._state_read_denied()
            return session_id
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("resolve_session requires a non-empty session_id")
        db = self._acquire_session_db()
        try:
            return db.resolve_resume_session_id(session_id)
        finally:
            self._release_session_db(db)

    def read_session_messages(
        self,
        session_id: str,
        *,
        limit: int = 50,
        after_id: Optional[int] = None,
        include_compacted: bool = False,
    ) -> dict[str, Any]:
        """Messages of *session_id* (resolved to the live tip), keyset-paged.

        Returns ``{"schema_version", "session_id", "messages", "next_after_id"}``;
        ``next_after_id`` is the last returned message id when a further page may
        exist, else None. ``limit`` is clamped to 1..100. ``after_id`` and
        ``include_compacted`` cannot be combined (the core's keyset reader forbids
        it); an empty dict with no messages is returned when not granted.
        """
        if not self._state_read_granted():
            self._state_read_denied()
            return {
                "schema_version": None,
                "session_id": session_id,
                "messages": [],
                "next_after_id": None,
            }
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("read_session_messages requires a non-empty session_id")
        if after_id is not None and include_compacted:
            raise ValueError("after_id is incompatible with include_compacted")
        limit = max(1, min(int(limit), _READ_MESSAGE_LIMIT_MAX))
        db = self._acquire_session_db()
        try:
            target = db.resolve_resume_session_id(session_id)
            messages = db.get_messages(
                target,
                include_compacted=include_compacted,
                limit=limit,
                after_id=after_id,
            )
        finally:
            self._release_session_db(db)
        next_after = None
        if messages and len(messages) >= limit:
            last_id = messages[-1].get("id")
            next_after = int(last_id) if last_id is not None else None
        return {
            "schema_version": self._state_schema_version(),
            "session_id": target,
            "messages": messages,
            "next_after_id": next_after,
        }

    def _open_kanban_ro(self, board: Optional[str]):  # type: ignore[no-untyped-def]
        """Read-only kanban connection for *board*; None when the board is absent
        or has no schema. Never creates, initializes or migrates anything."""
        from hermes_cli import kanban_db as _kb

        path = _kb.kanban_db_path(board=board)
        if not path.is_file():
            return None
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.text_factory = _kb._lossy_text
        try:
            present = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()[0]
        except sqlite3.Error:
            conn.close()
            return None
        if not present:
            conn.close()
            return None
        return conn

    def _task_to_dict(self, task: Any, board: Optional[str]) -> dict[str, Any]:
        """Closed-field DTO for a kanban ``Task``."""
        dto = {field: getattr(task, field, None) for field in _TASK_DTO_FIELDS}
        dto["board"] = board or "default"
        return dto

    def read_kanban_task(
        self,
        task_id: str,
        *,
        board: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """One kanban task as a closed-field DTO, or None when unknown, when the
        board is absent, or when not granted. Never creates a board."""
        if not self._state_read_granted():
            self._state_read_denied()
            return None
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("read_kanban_task requires a non-empty task_id")
        from hermes_cli import kanban_db as _kb

        conn = self._open_kanban_ro(board)
        if conn is None:
            logger.debug(
                "Plugin %s: kanban board %r absent; read_kanban_task returns None",
                self.plugin_id,
                board or "default",
            )
            return None
        try:
            task = _kb.get_task(conn, task_id)
        finally:
            conn.close()
        if task is None:
            return None
        return self._task_to_dict(task, board)

    def list_kanban_tasks(
        self,
        *,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        board: Optional[str] = None,
    ) -> dict[str, Any]:
        """Kanban tasks filtered by status/assignee/session, newest-first by the
        core's default order. Returns ``{"api_version", "tasks"}``; empty tasks
        when the board is absent or not granted. ``limit`` is clamped to 1..100.
        An unknown ``status`` raises ``ValueError`` (the core validates it)."""
        if not self._state_read_granted():
            self._state_read_denied()
            return {"api_version": KANBAN_API_VERSION, "tasks": []}
        limit = max(1, min(int(limit), _READ_TASK_LIMIT_MAX))
        from hermes_cli import kanban_db as _kb

        conn = self._open_kanban_ro(board)
        if conn is None:
            return {"api_version": KANBAN_API_VERSION, "tasks": []}
        try:
            tasks = _kb.list_tasks(
                conn,
                status=status,
                assignee=assignee,
                session_id=session_id,
                limit=limit,
            )
        finally:
            conn.close()
        return {
            "api_version": KANBAN_API_VERSION,
            "tasks": [self._task_to_dict(task, board) for task in tasks],
        }

    def read_kanban_task_summary(
        self,
        task_id: str,
        *,
        board: Optional[str] = None,
    ) -> Optional[str]:
        """Newest non-empty run summary for *task_id*, or None when there is none,
        the task/board is absent, or not granted."""
        if not self._state_read_granted():
            self._state_read_denied()
            return None
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("read_kanban_task_summary requires a non-empty task_id")
        from hermes_cli import kanban_db as _kb

        conn = self._open_kanban_ro(board)
        if conn is None:
            return None
        try:
            return _kb.latest_summary(conn, task_id)
        finally:
            conn.close()
