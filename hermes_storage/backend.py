"""
hermes_storage/backend.py — Storage backend protocol for SaaS-mode Hermes.

Defines the interface that both SQLiteBackend (local dev) and NeonBackend
(SaaS/cloud) must satisfy.  Uses runtime_checkable Protocol so isinstance()
checks work at the factory layer.

Design decisions:
- All methods are async: NeonBackend needs await; SQLiteBackend wraps sync
  SQLite calls with asyncio.to_thread to keep the event loop unblocked.
- conversation_id is always a string (UUID format in Neon; session_id in SQLite).
  Callers treat it as an opaque token — don't assume format.
- search_sessions returns list[dict] with at minimum {"conversation_id", "snippet"}
  keys so callers can present results without format knowledge.
- identity: HermesIdentity is forward-referenced as a string to avoid a
  circular import if callers import both hermes_identity and hermes_storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hermes_identity import HermesIdentity


@runtime_checkable
class StorageBackend(Protocol):
    """
    Minimal async interface for Hermes conversation storage.

    Implementations:
      - SQLiteBackend (hermes_storage/sqlite_backend.py) — local dev
      - NeonBackend   (hermes_storage/neon_backend.py)   — SaaS / cloud

    Selection: hermes_storage.get_backend() reads HERMES_MODE at startup.
    """

    async def get_or_create_conversation(
        self,
        identity: "HermesIdentity",
        channel_id: str,
        thread_id: str | None,
    ) -> str:
        """
        Return the conversation_id for this (identity, channel, thread) tuple.

        Creates a new conversation row if none exists.  Idempotent — calling
        twice with the same arguments returns the same ID.

        Returns: opaque conversation_id string (UUID in Neon; session UUID in SQLite).
        """
        ...

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Append one message to the conversation.

        Args:
            conversation_id: From get_or_create_conversation.
            role: One of "user", "assistant", "tool".
            content: Text payload.
            tool_calls: Optional dict of tool-call objects (OpenAI format).
            metadata: Optional dict of per-message metadata (tokens, latency, etc.).

        Returns: opaque message_id string.
        """
        ...

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return the last *limit* messages for this conversation, oldest first.

        Each dict contains at minimum: {"role": str, "content": str}.
        Tool calls and metadata are included when present.
        """
        ...

    async def search_sessions(
        self,
        query: str,
        identity: "HermesIdentity",
        limit: int = 5,
    ) -> list[dict]:
        """
        Full-text search across conversations visible to this identity.

        Returns: list of dicts with at minimum {"conversation_id": str, "snippet": str}.
        Results are ordered by relevance (most relevant first).
        """
        ...

    async def append_raw_event(
        self,
        tenant_id: str,
        conversation_id: str | None,
        event_kind: str,
        platform_message_id: str | None,
        raw_payload: dict,
    ) -> str | None:
        """
        Append one raw audit row to the compliance event log (Plan 007-A).

        Append-only. Best-effort: implementations MUST swallow internal failures
        (return None) so the audit path never blocks the user-facing flow.

        Args:
            tenant_id: UUID string of the tenant (caller resolves).
            conversation_id: Optional conversation UUID (None for events that
                arrive before a conversation is established, e.g. dropped Slack
                events).
            event_kind: One of "slack_inbound", "slack_outbound",
                "tool_call_request", "tool_call_response". Backend enforces
                via check constraint.
            platform_message_id: Slack ts for inbound/outbound; deterministic
                hash for tool calls. Used as idempotency key. May be None.
            raw_payload: The full event dict. Caller is responsible for secret
                redaction BEFORE handing the payload off.

        Returns:
            UUID string of the inserted row on success, None on idempotent
            duplicate or any failure (audit path must never raise).

        Saas vs local:
            NeonBackend writes to the raw_events table with RLS enforcement.
            SQLiteBackend no-ops with a debug log — local-mode compliance audit
            is out of scope (track in events.db via Plan 006 instead).
        """
        ...

    async def close(self) -> None:
        """Release connection pool / file handles.  Safe to call multiple times."""
        ...
