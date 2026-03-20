"""QMD-based session management for conversation history.

QMDSessionManager provides local vector-based memory management as an
alternative to Honcho's cloud-backed user modeling. This is a strict
OR relationship — when QMD is enabled, Honcho is bypassed entirely.

Features:
  - Local FAISS indexing for semantic memory search
  - Anticipatory context (FlowState) for zero-latency memory retrieval
  - Configurable embedding models (including NousResearch integration point)
  - Async write queue for non-blocking memory persistence

Memory modes:
  - local: Store in QMD only (default)
  - hybrid: QMD + Hermes native memory (future)
  - (Note: cloud sync like Honcho is not supported in QMD mode)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qmd_integration.client import QMDClient, get_qmd_client

logger = logging.getLogger(__name__)

# Sentinel to signal the async writer thread to shut down
_ASYNC_SHUTDOWN = object()


@dataclass
class QMDSession:
    """A conversation session backed by QMD vector store.

    Provides local message cache that syncs to QMD's FAISS index
    for semantic memory retrieval.
    """

    key: str  # session key (usually channel:chat_id)
    session_id: str  # QMD session identifier
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history for LLM context."""
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_conversation_text(self, max_messages: int = 20) -> str:
        """Get recent conversation as plain text for embedding."""
        history = self.get_history(max_messages)
        lines = []
        for msg in history:
            role = msg["role"].upper() if msg["role"] != "system" else "SYSTEM"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now(timezone.utc)


class QMDSessionManager:
    """Manages conversation sessions using QMD vector store.

    This is a strict alternative to HonchoSessionManager. When QMD is
    enabled in config, Honcho is completely bypassed. No hybrid mode.

    Features:
      - Local FAISS indexing (no cloud dependency)
      - Anticipatory context via FlowState
      - Configurable embedding models
      - Async write queue for non-blocking persistence
    """

    def __init__(
        self,
        client: QMDClient | None = None,
        config: Any | None = None,
    ):
        """Initialize the QMD session manager.

        Args:
            client: QMD HTTP client. If not provided, uses singleton.
            config: QMDClientConfig for session settings.
        """
        self._client = client
        self._config = config
        self._cache: dict[str, QMDSession] = {}
        self._qmd_ready = False

        # Write frequency state
        write_frequency = (config.write_frequency if config else "async")
        self._write_frequency = write_frequency
        self._turn_counter = 0

        # Anticipatory context settings
        self._anticipatory_enabled = (
            config.anticipatory_enabled if config else True
        )
        self._anticipatory_max_results = (
            config.anticipatory_max_results if config else 3
        )

        # Async write queue — started lazily on first enqueue
        self._async_queue: queue.Queue | None = None
        self._async_thread: threading.Thread | None = None
        if write_frequency == "async":
            self._async_queue = queue.Queue()
            self._async_thread = threading.Thread(
                target=self._async_writer_loop,
                name="qmd-async-writer",
                daemon=True,
            )
            self._async_thread.start()

        # Check QMD server availability
        self._check_qmd_server()

    def _check_qmd_server(self) -> bool:
        """Check if QMD server is running and accessible."""
        try:
            client = self.client
            if client.is_ready():
                self._qmd_ready = True
                logger.info("QMD server is ready at %s", client.config.server_url)
                return True
        except Exception as e:
            logger.debug("QMD server not available: %s", e)

        logger.warning(
            "QMD server not available. Memory features disabled. "
            "Start QMD server with: qmd server"
        )
        self._qmd_ready = False
        return False

    @property
    def client(self) -> QMDClient:
        """Get the QMD client, initializing if needed."""
        if self._client is None:
            self._client = get_qmd_client(self._config)
        return self._client

    @property
    def is_ready(self) -> bool:
        """Check if QMD session manager is ready."""
        return self._qmd_ready

    def get_or_create(self, key: str) -> QMDSession:
        """Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            logger.debug("QMD session cache hit: %s", key)
            return self._cache[key]

        # Resolve session name based on strategy
        if self._config:
            session_id = self._config.resolve_session_name(session_id=key)
        else:
            session_id = f"qmd-{key}"

        session = QMDSession(
            key=key,
            session_id=session_id,
        )

        self._cache[key] = session
        return session

    def _flush_session(self, session: QMDSession) -> bool:
        """Internal: write messages to QMD synchronously."""
        if not session.messages:
            return True

        if not self._qmd_ready:
            return False

        try:
            # Get unsynced messages
            new_messages = [
                m for m in session.messages if not m.get("_synced")
            ]
            if not new_messages:
                return True

            # Prepare batch for QMD
            memories = []
            for msg in new_messages:
                # Truncate if needed
                content = msg["content"]
                if self._config and len(content) > self._config.memory_char_limit:
                    content = content[: self._config.memory_char_limit]

                memories.append({
                    "content": content,
                    "role": msg.get("role", "agent"),
                    "session_id": session.session_id,
                })

            # Batch add to QMD
            self.client.add_memory_batch(memories)

            # Mark as synced
            for msg in new_messages:
                msg["_synced"] = True

            logger.debug(
                "Synced %d messages to QMD for %s",
                len(new_messages),
                session.key,
            )
            self._cache[session.key] = session
            return True

        except Exception as e:
            logger.error("Failed to sync messages to QMD: %s", e)
            return False

    def _async_writer_loop(self) -> None:
        """Background daemon thread: drains the async write queue."""
        while True:
            try:
                item = self._async_queue.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break

                first_error: Exception | None = None
                try:
                    success = self._flush_session(item)
                except Exception as e:
                    success = False
                    first_error = e

                if success:
                    continue

                if first_error is not None:
                    logger.warning(
                        "QMD async write failed, retrying once: %s",
                        first_error,
                    )
                else:
                    logger.warning("QMD async write failed, retrying once")

                time.sleep(2)

                try:
                    retry_success = self._flush_session(item)
                except Exception as e2:
                    logger.error(
                        "QMD async write retry failed, dropping batch: %s",
                        e2,
                    )
                    continue

                if not retry_success:
                    logger.error(
                        "QMD async write retry failed, dropping batch",
                    )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("QMD async writer error: %s", e)

    def save(self, session: QMDSession) -> None:
        """Save messages to QMD, respecting write_frequency.

        Write frequency modes:
          "async"   — enqueue for background thread (zero blocking)
          "turn"    — flush synchronously every turn
          "session" — defer until flush_all() is called explicitly
          N (int)   — flush every N turns
        """
        self._turn_counter += 1
        wf = self._write_frequency

        if wf == "async":
            if self._async_queue is not None:
                self._async_queue.put(session)
        elif wf == "turn":
            self._flush_session(session)
        elif wf == "session":
            # Accumulate; caller must call flush_all() at session end
            pass
        elif isinstance(wf, int) and wf > 0:
            if self._turn_counter % wf == 0:
                self._flush_session(session)

    def flush_all(self) -> None:
        """Flush all pending messages for all cached sessions.

        Called at session end for "session" write_frequency, or to force
        a sync before process exit regardless of mode.
        """
        for session in list(self._cache.values()):
            try:
                self._flush_session(session)
            except Exception as e:
                logger.error("QMD flush_all error for %s: %s", session.key, e)

        # Drain async queue synchronously if it exists
        if self._async_queue is not None:
            while not self._async_queue.empty():
                try:
                    item = self._async_queue.get_nowait()
                    if item is not _ASYNC_SHUTDOWN:
                        self._flush_session(item)
                except queue.Empty:
                    break

    def shutdown(self) -> None:
        """Gracefully shut down the async writer thread."""
        if self._async_queue is not None and self._async_thread is not None:
            self.flush_all()
            self._async_queue.put(_ASYNC_SHUTDOWN)
            self._async_thread.join(timeout=10)

    def delete(self, key: str) -> bool:
        """Delete a session from local cache (not from QMD)."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def new_session(self, key: str) -> QMDSession:
        """Create a new session, preserving the old one in QMD.

        Creates a fresh session with a new ID while keeping the old
        session's data in QMD for continued memory access.
        """
        # Remove old session from caches
        old_session = self._cache.pop(key, None)

        # Create new session with timestamp suffix
        timestamp = int(time.time())
        new_key = f"{key}:{timestamp}"

        # Create new session
        session = self.get_or_create(new_key)

        # Cache under the original key so callers find it by the expected name
        self._cache[key] = session

        logger.info(
            "Created new QMD session for %s (qmd_id: %s)",
            key,
            session.session_id,
        )
        return session

    def query_memory(
        self,
        session_key: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query QMD memories for a session.

        Args:
            session_key: The session key to query against.
            query: Search query text.
            top_k: Number of results (uses config default if not provided).

        Returns:
            List of matching memories with scores.
        """
        if not self._qmd_ready:
            return []

        session = self._cache.get(session_key)
        session_id = session.session_id if session else None

        k = top_k or self._config.top_k if self._config else 5

        try:
            return self.client.query_memories(
                query=query,
                top_k=k,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("QMD memory query failed: %s", e)
            return []

    def get_anticipatory_context(
        self,
        session_key: str,
        recent_conversation: str,
    ) -> list[dict[str, Any]]:
        """Get anticipatory context for the current conversation.

        This is the FlowState feature — returns context that's predicted
        to be relevant based on conversation state, before the agent asks.

        Args:
            session_key: The session key.
            recent_conversation: Recent conversation text.

        Returns:
            List of anticipatory memories with scores.
        """
        if not self._qmd_ready or not self._anticipatory_enabled:
            return []

        try:
            result = self.client.get_anticipatory_context(
                recent_conversation=recent_conversation,
                lite_mode=self._config.lite_mode if self._config else False,
            )
            return result.get("context", [])
        except Exception as e:
            logger.debug("Anticipatory context fetch failed: %s", e)
            return []

    def get_recent_memories(
        self,
        session_key: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent memories for a session.

        Args:
            session_key: The session key.
            limit: Maximum memories to return.

        Returns:
            List of recent memories.
        """
        if not self._qmd_ready:
            return []

        session = self._cache.get(session_key)
        session_id = session.session_id if session else None

        try:
            return self.client.get_recent_memories(
                limit=limit,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("QMD recent memories fetch failed: %s", e)
            return []
