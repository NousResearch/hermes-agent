"""
hermes_storage/sqlite_backend.py — Local SQLite implementation of StorageBackend.

Wraps hermes_state.SessionDB behind the async StorageBackend protocol using
asyncio.to_thread so SQLite's blocking I/O doesn't stall the event loop.

Design decisions:
- conversation_id == session_id (a UUID string) in SQLite mode.  The format
  matches what SessionDB.create_session / ensure_session already produce.
- get_or_create_conversation uses identity fields to build a deterministic
  session source string ("sqlite_backend:<platform>:<team_id>:<user_id>:<channel_id>[:<thread_id>]")
  and stores/looks up by user_id for fast retrieval.
- search_sessions delegates to SessionDB's FTS5 search path, filtered by
  user_id from identity to maintain per-user isolation.
- All async wrappers use asyncio.to_thread to keep the event loop unblocked;
  the underlying SessionDB is thread-safe (it uses threading.Lock internally).

Assumptions:
- HERMES_MODE != "saas" when this backend is selected.  Callers should not
  try to enforce cross-tenant RLS with this backend — it's single-process dev.
- db_path defaults to SessionDB's DEFAULT_DB_PATH (same file local CLI uses).
  Tests should pass a tmpdir-scoped path to avoid polluting the real DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hermes_state import SessionDB

if TYPE_CHECKING:
    from hermes_identity import HermesIdentity

logger = logging.getLogger(__name__)


class SQLiteBackend:
    """
    Async wrapper around SessionDB satisfying the StorageBackend protocol.

    Usage (local dev)::

        backend = SQLiteBackend()
        conv_id = await backend.get_or_create_conversation(identity, "#general", None)
        await backend.append_message(conv_id, "user", "Hello, Hermes!")
        history = await backend.get_conversation_history(conv_id)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Args:
            db_path: Override default DB path.  Pass a tmp path in tests.
        """
        self._db_path = db_path
        self._db: Optional[SessionDB] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_db(self) -> SessionDB:
        """Lazily initialise SessionDB (thread-safe via SessionDB's internal lock)."""
        if self._db is None:
            kwargs: dict = {}
            if self._db_path is not None:
                kwargs["db_path"] = self._db_path
            self._db = SessionDB(**kwargs)
        return self._db

    async def close(self) -> None:
        """Close the SQLite connection pool."""
        if self._db is not None:
            await asyncio.to_thread(self._db.close)
            self._db = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_source(identity: "HermesIdentity", channel_id: str, thread_id: Optional[str]) -> str:
        """
        Build a deterministic session source tag from identity + channel context.

        Format: "hermes_saas:<platform>:<team_id>:<channel_id>[:<thread_id>]"
        This is stored in sessions.source so we can look it up later.
        """
        parts = ["hermes_saas", identity.platform, identity.team_id, channel_id]
        if thread_id:
            parts.append(thread_id)
        return ":".join(parts)

    # ------------------------------------------------------------------
    # StorageBackend protocol
    # ------------------------------------------------------------------

    async def get_or_create_conversation(
        self,
        identity: "HermesIdentity",
        channel_id: str,
        thread_id: Optional[str],
    ) -> str:
        """
        Return the session_id for this (identity, channel, thread) combination.

        Looks up an existing session by (source, user_id).  Creates a new one
        if not found.  Idempotent — repeated calls with the same args return
        the same session_id.
        """
        source = self._session_source(identity, channel_id, thread_id)

        def _find_or_create() -> str:
            db = self._get_db()
            # Check for existing session with this source + user_id.
            row = db._conn.execute(
                "SELECT id FROM sessions WHERE source = ? AND user_id = ? "
                "AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
                (source, identity.user_id),
            ).fetchone()
            if row:
                return str(row["id"])
            # Create a new session.
            session_id = str(uuid.uuid4())
            db.create_session(session_id, source=source, user_id=identity.user_id)
            logger.info(
                "SQLiteBackend: created new conversation %s for %s/%s/%s",
                session_id, identity.platform, identity.team_id, identity.user_id,
            )
            return session_id

        return await asyncio.to_thread(_find_or_create)

    async def append_raw_event(
        self,
        tenant_id: str,
        conversation_id: str | None,
        event_kind: str,
        platform_message_id: str | None,
        raw_payload: dict,
    ) -> str | None:
        """
        Plan 007-A no-op for local mode.

        Compliance-grade raw_events audit is saas-only. Local sessions don't
        need cross-restart audit (data lives on disk anyway). Plan 006
        (Workflow Observability) writes its own events.db locally for the
        dev observability use case.
        """
        logger.debug(
            "SQLiteBackend.append_raw_event no-op (saas-only): kind=%s msg_id=%s",
            event_kind, platform_message_id,
        )
        return None

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Append one message to the conversation.  Returns a string message ID.

        Delegates to SessionDB.append_message which handles FTS5 indexing.
        tool_calls dict is serialised to JSON for the tool_calls column.
        metadata is logged but not persisted to SQLite (SQLite schema has no
        generic metadata column — it uses fixed columns like token_count).
        """

        def _append() -> str:
            db = self._get_db()
            # SessionDB.append_message takes role + content as positional args.
            # tool_calls must be a list or dict (not pre-serialised).
            row_id = db.append_message(
                conversation_id,
                role,
                content,
                tool_calls=tool_calls,
            )
            if metadata:
                # Log metadata fields not currently persisted in the SQLite schema.
                logger.debug(
                    "SQLiteBackend: message %s metadata (not persisted): %s",
                    row_id, metadata,
                )
            return str(row_id)

        return await asyncio.to_thread(_append)

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return the last *limit* messages for the conversation, oldest first.

        Each dict has at minimum {"role", "content"}.  Tool calls are included
        when present (decoded from JSON).
        """

        def _get() -> list[dict]:
            db = self._get_db()
            raw = db.get_messages(conversation_id)
            # get_messages returns all; apply limit from the tail.
            if len(raw) > limit:
                raw = raw[-limit:]
            result = []
            for msg in raw:
                entry: dict = {
                    "role": msg.get("role", ""),
                    "content": msg.get("content") or "",
                }
                if msg.get("tool_calls"):
                    tc = msg["tool_calls"]
                    if isinstance(tc, str):
                        try:
                            tc = json.loads(tc)
                        except json.JSONDecodeError:
                            pass
                    entry["tool_calls"] = tc
                result.append(entry)
            return result

        return await asyncio.to_thread(_get)

    async def search_sessions(
        self,
        query: str,
        identity: "HermesIdentity",
        limit: int = 5,
    ) -> list[dict]:
        """
        FTS5 search across sessions belonging to this identity.

        Uses SessionDB.list_sessions_rich to find sessions for this user,
        then runs a basic text search.  Returns list of dicts with
        {"conversation_id", "snippet"}.

        Assumption: SQLite FTS5 is single-user; cross-user isolation comes
        from filtering by user_id (no RLS needed for local mode).
        """

        def _search() -> list[dict]:
            db = self._get_db()
            # Build source prefix to narrow the search to this user's sessions.
            # We search all channels/threads for the user.
            source_prefix = f"hermes_saas:{identity.platform}:{identity.team_id}:"
            rows = db._conn.execute(
                """
                SELECT s.id, m.content
                FROM sessions s
                JOIN messages m ON m.session_id = s.id
                WHERE s.user_id = ? AND s.source LIKE ?
                  AND m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                (identity.user_id, source_prefix + "%", f"%{query}%", limit),
            ).fetchall()
            seen: set[str] = set()
            results: list[dict] = []
            for row in rows:
                conv_id = str(row[0])
                if conv_id not in seen:
                    seen.add(conv_id)
                    snippet = (row[1] or "")[:200]
                    results.append({"conversation_id": conv_id, "snippet": snippet})
            return results

        return await asyncio.to_thread(_search)
