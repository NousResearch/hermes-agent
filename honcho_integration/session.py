"""Honcho-based session management for conversation history."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from honcho_integration.client import get_honcho_client

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)


@dataclass
class HonchoSession:
    """
    A conversation session backed by Honcho.

    Provides a local message cache that syncs to Honcho's
    AI-native memory system for user modeling.
    """

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history for LLM context."""
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class HonchoSessionManager:
    """
    Manages conversation sessions using Honcho.

    Runs alongside hermes' existing SQLite state and file-based memory,
    adding persistent cross-session user modeling via Honcho's AI-native memory.
    """

    def __init__(
        self,
        honcho: Honcho | None = None,
        context_tokens: int | None = None,
        config: Any | None = None,
    ):
        """
        Initialize the session manager.

        Args:
            honcho: Optional Honcho client. If not provided, uses the singleton.
            context_tokens: Max tokens for context() calls (None = Honcho default).
            config: HonchoClientConfig from global config (provides peer_name, ai_peer, etc.).
        """
        self._honcho = honcho
        self._context_tokens = context_tokens
        self._config = config
        self._cache: dict[str, HonchoSession] = {}
        self._peers_cache: dict[str, Any] = {}
        self._sessions_cache: dict[str, Any] = {}

    @property
    def honcho(self) -> Honcho:
        """Get the Honcho client, initializing if needed."""
        if self._honcho is None:
            self._honcho = get_honcho_client()
        return self._honcho

    def _get_or_create_peer(self, peer_id: str) -> Any:
        """
        Get or create a Honcho peer.

        Peers are lazy -- no API call until first use.
        Observation settings are controlled per-session via SessionPeerConfig.
        """
        if peer_id in self._peers_cache:
            return self._peers_cache[peer_id]

        peer = self.honcho.peer(peer_id)
        self._peers_cache[peer_id] = peer
        return peer

    def _get_or_create_honcho_session(
        self, session_id: str, user_peer: Any, assistant_peer: Any
    ) -> tuple[Any, list]:
        """
        Get or create a Honcho session with peers configured.

        Returns:
            Tuple of (honcho_session, existing_messages).
        """
        if session_id in self._sessions_cache:
            logger.debug("Honcho session '%s' retrieved from cache", session_id)
            return self._sessions_cache[session_id], []

        session = self.honcho.session(session_id)

        # Configure peer observation settings
        from honcho.session import SessionPeerConfig
        user_config = SessionPeerConfig(observe_me=True, observe_others=True)
        ai_config = SessionPeerConfig(observe_me=False, observe_others=True)

        session.add_peers([(user_peer, user_config), (assistant_peer, ai_config)])

        # Load existing messages via context() - single call for messages + metadata
        existing_messages = []
        try:
            ctx = session.context(summary=True, tokens=self._context_tokens)
            existing_messages = ctx.messages or []

            # Verify chronological ordering
            if existing_messages and len(existing_messages) > 1:
                timestamps = [m.created_at for m in existing_messages if m.created_at]
                if timestamps and timestamps != sorted(timestamps):
                    logger.warning(
                        "Honcho messages not chronologically ordered for session '%s', sorting",
                        session_id,
                    )
                    existing_messages = sorted(
                        existing_messages,
                        key=lambda m: m.created_at or datetime.min,
                    )

            if existing_messages:
                logger.info(
                    "Honcho session '%s' retrieved (%d existing messages)",
                    session_id, len(existing_messages),
                )
            else:
                logger.info("Honcho session '%s' created (new)", session_id)
        except Exception as e:
            logger.warning(
                "Honcho session '%s' loaded (failed to fetch context: %s)",
                session_id, e,
            )

        self._sessions_cache[session_id] = session
        return session, existing_messages

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize an ID to match Honcho's pattern: ^[a-zA-Z0-9_-]+"""
        return re.sub(r'[^a-zA-Z0-9_-]', '-', id_str)

    def get_or_create(self, key: str) -> HonchoSession:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            logger.debug("Local session cache hit: %s", key)
            return self._cache[key]

        # Use peer names from global config when available
        if self._config and self._config.peer_name:
            user_peer_id = self._sanitize_id(self._config.peer_name)
        else:
            # Fallback: derive from session key
            parts = key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "default"
            chat_id = parts[1] if len(parts) > 1 else key
            user_peer_id = self._sanitize_id(f"user-{channel}-{chat_id}")

        assistant_peer_id = (
            self._config.ai_peer if self._config else "hermes-assistant"
        )

        # Sanitize session ID for Honcho
        honcho_session_id = self._sanitize_id(key)

        # Get or create peers
        user_peer = self._get_or_create_peer(user_peer_id)
        assistant_peer = self._get_or_create_peer(assistant_peer_id)

        # Get or create Honcho session
        honcho_session, existing_messages = self._get_or_create_honcho_session(
            honcho_session_id, user_peer, assistant_peer
        )

        # Convert Honcho messages to local format
        local_messages = []
        for msg in existing_messages:
            role = "assistant" if msg.peer_id == assistant_peer_id else "user"
            local_messages.append({
                "role": role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                "_synced": True,  # Already in Honcho
            })

        # Create local session wrapper with existing messages
        session = HonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=honcho_session_id,
            messages=local_messages,
        )

        self._cache[key] = session
        return session

    def save(self, session: HonchoSession) -> None:
        """
        Save messages to Honcho.

        Syncs only new (unsynced) messages from the local cache.
        """
        if not session.messages:
            return

        # Get the Honcho session and peers
        user_peer = self._get_or_create_peer(session.user_peer_id)
        assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
        honcho_session = self._sessions_cache.get(session.honcho_session_id)

        if not honcho_session:
            honcho_session, _ = self._get_or_create_honcho_session(
                session.honcho_session_id, user_peer, assistant_peer
            )

        # Only send new messages (those without a '_synced' flag)
        new_messages = [m for m in session.messages if not m.get("_synced")]

        if not new_messages:
            return

        honcho_messages = []
        for msg in new_messages:
            peer = user_peer if msg["role"] == "user" else assistant_peer
            honcho_messages.append(peer.message(msg["content"]))

        try:
            honcho_session.add_messages(honcho_messages)
            for msg in new_messages:
                msg["_synced"] = True
            logger.debug("Synced %d messages to Honcho for %s", len(honcho_messages), session.key)
        except Exception as e:
            for msg in new_messages:
                msg["_synced"] = False
            logger.error("Failed to sync messages to Honcho: %s", e)

        # Update cache
        self._cache[session.key] = session

    def delete(self, key: str) -> bool:
        """Delete a session from local cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def new_session(self, key: str) -> HonchoSession:
        """
        Create a new session, preserving the old one for user modeling.

        Creates a fresh session with a new ID while keeping the old
        session's data in Honcho for continued user modeling.
        """
        import time

        # Remove old session from caches (but don't delete from Honcho)
        old_session = self._cache.pop(key, None)
        if old_session:
            self._sessions_cache.pop(old_session.honcho_session_id, None)

        # Create new session with timestamp suffix
        timestamp = int(time.time())
        new_key = f"{key}:{timestamp}"

        # get_or_create will create a fresh session
        session = self.get_or_create(new_key)

        # Cache under both original key and timestamped key
        self._cache[key] = session
        self._cache[new_key] = session

        logger.info("Created new session for %s (honcho: %s)", key, session.honcho_session_id)
        return session

    def get_user_context(self, session_key: str, query: str) -> str:
        """
        Query Honcho's dialectic chat for user context.

        Args:
            session_key: The session key to get context for.
            query: Natural language question about the user.

        Returns:
            Honcho's response about the user.
        """
        session = self._cache.get(session_key)
        if not session:
            return "No session found for this context."

        user_peer = self._get_or_create_peer(session.user_peer_id)

        try:
            return user_peer.chat(query)
        except Exception as e:
            logger.error("Failed to get user context from Honcho: %s", e)
            return f"Unable to retrieve user context: {e}"

    def get_prefetch_context(self, session_key: str, user_message: str | None = None) -> dict[str, str]:
        """
        Pre-fetch user context using Honcho's context() method.

        Single API call that returns the user's representation
        and peer card, using semantic search based on the user's message.

        Args:
            session_key: The session key to get context for.
            user_message: The user's message for semantic search.

        Returns:
            Dictionary with 'representation' and 'card' keys.
        """
        session = self._cache.get(session_key)
        if not session:
            return {}

        honcho_session = self._sessions_cache.get(session.honcho_session_id)
        if not honcho_session:
            return {}

        try:
            ctx = honcho_session.context(
                summary=False,
                tokens=self._context_tokens,
                peer_target=session.user_peer_id,
                search_query=user_message,
            )
            # peer_card is list[str] in SDK v2, join for prompt injection
            card = ctx.peer_card or []
            card_str = "\n".join(card) if isinstance(card, list) else str(card)
            return {
                "representation": ctx.peer_representation or "",
                "card": card_str,
            }
        except Exception as e:
            logger.warning("Failed to fetch context from Honcho: %s", e)
            return {}

    def migrate_local_history(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        """
        Upload local session history to Honcho as a file.

        Used when Honcho activates mid-conversation to preserve prior context.

        Args:
            session_key: The session key (e.g., "telegram:123456").
            messages: Local messages (dicts with role, content, timestamp).

        Returns:
            True if upload succeeded, False otherwise.
        """
        sanitized = self._sanitize_id(session_key)
        honcho_session = self._sessions_cache.get(sanitized)
        if not honcho_session:
            logger.warning("No Honcho session cached for '%s', skipping migration", session_key)
            return False

        # Resolve user peer for attribution
        parts = session_key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else session_key
        user_peer_id = self._sanitize_id(f"user-{channel}-{chat_id}")
        user_peer = self._peers_cache.get(user_peer_id)
        if not user_peer:
            logger.warning("No user peer cached for '%s', skipping migration", user_peer_id)
            return False

        content_bytes = self._format_migration_transcript(session_key, messages)
        first_ts = messages[0].get("timestamp") if messages else None

        try:
            honcho_session.upload_file(
                file=("prior_history.txt", content_bytes, "text/plain"),
                peer=user_peer,
                metadata={"source": "local_jsonl", "count": len(messages)},
                created_at=first_ts,
            )
            logger.info("Migrated %d local messages to Honcho for %s", len(messages), session_key)
            return True
        except Exception as e:
            logger.error("Failed to upload local history to Honcho for %s: %s", session_key, e)
            return False

    @staticmethod
    def _format_migration_transcript(session_key: str, messages: list[dict[str, Any]]) -> bytes:
        """Format local messages as an XML transcript for Honcho file upload."""
        timestamps = [m.get("timestamp", "") for m in messages]
        time_range = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"

        lines = [
            "<prior_conversation_history>",
            "<context>",
            "This conversation history occurred BEFORE the Honcho memory system was activated.",
            "These messages are the preceding elements of this conversation session and should",
            "be treated as foundational context for all subsequent interactions. The user and",
            "assistant have already established rapport through these exchanges.",
            "</context>",
            "",
            f'<transcript session_key="{session_key}" message_count="{len(messages)}"',
            f'           time_range="{time_range}">',
            "",
        ]
        for msg in messages:
            ts = msg.get("timestamp", "?")
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{ts}] {role}: {content}")

        lines.append("")
        lines.append("</transcript>")
        lines.append("</prior_conversation_history>")

        return "\n".join(lines).encode("utf-8")

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        """
        Upload MEMORY.md and USER.md to Honcho as files.

        Used when Honcho activates on an instance that already has locally
        consolidated memory. Backwards compatible -- skips if files don't exist.

        Args:
            session_key: The session key to associate files with.
            memory_dir: Path to the memories directory (~/.hermes/memories/).

        Returns:
            True if at least one file was uploaded, False otherwise.
        """
        from pathlib import Path
        memory_path = Path(memory_dir)

        if not memory_path.exists():
            return False

        sanitized = self._sanitize_id(session_key)
        honcho_session = self._sessions_cache.get(sanitized)
        if not honcho_session:
            logger.warning("No Honcho session cached for '%s', skipping memory migration", session_key)
            return False

        # Resolve user peer for attribution
        parts = session_key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else session_key
        user_peer_id = self._sanitize_id(f"user-{channel}-{chat_id}")
        user_peer = self._peers_cache.get(user_peer_id)
        if not user_peer:
            logger.warning("No user peer cached for '%s', skipping memory migration", user_peer_id)
            return False

        uploaded = False
        files = [
            ("MEMORY.md", "consolidated_memory.md", "Long-term agent notes and preferences"),
            ("USER.md", "user_profile.md", "User profile and preferences"),
        ]

        for filename, upload_name, description in files:
            filepath = memory_path / filename
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue

            wrapped = (
                f"<prior_memory_file>\n"
                f"<context>\n"
                f"This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                f"{description}. Treat as foundational context for this user.\n"
                f"</context>\n"
                f"\n"
                f"{content}\n"
                f"</prior_memory_file>\n"
            )

            try:
                honcho_session.upload_file(
                    file=(upload_name, wrapped.encode("utf-8"), "text/plain"),
                    peer=user_peer,
                    metadata={"source": "local_memory", "original_file": filename},
                )
                logger.info("Uploaded %s to Honcho for %s", filename, session_key)
                uploaded = True
            except Exception as e:
                logger.error("Failed to upload %s to Honcho: %s", filename, e)

        return uploaded

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all cached sessions."""
        return [
            {
                "key": s.key,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": len(s.messages),
            }
            for s in self._cache.values()
        ]
