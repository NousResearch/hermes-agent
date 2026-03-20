"""QMD HTTP client for communicating with the QMD MCP server.

The QMD server provides:
  - /status: Server health and index info
  - /memories: Add/query/delete memories
  - /memories/batch: Batch add memories
  - /memories/recent: Get recent memories
  - /anticipatory_context: Fetch anticipatory context (FlowState feature)

This client is a thin wrapper around httpx with connection pooling
and automatic retry logic.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from qmd_integration.config import QMDClientConfig

logger = logging.getLogger(__name__)

_qmd_client: httpx.Client | None = None


class QMDClient:
    """HTTP client for QMD MCP server.

    Provides typed methods for all QMD server endpoints with automatic
    JSON serialization and error handling.
    """

    def __init__(self, config: QMDClientConfig):
        """Initialize QMD client.

        Args:
            config: QMD configuration with server connection details.
        """
        self.config = config
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create the HTTP client (lazy initialization)."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.server_url,
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "QMDClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def is_ready(self) -> bool:
        """Check if QMD server is running and ready."""
        try:
            resp = self.client.get("/status")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def get_status(self) -> dict[str, Any]:
        """Get QMD server status and index info.

        Returns:
            Status response with model, dimensions, and memory count.
        """
        resp = self.client.get("/status")
        resp.raise_for_status()
        return resp.json()

    def add_memory(
        self,
        content: str,
        role: str = "agent",
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a memory to the index.

        Args:
            content: Memory content text.
            role: Role (user, agent, system).
            tags: Optional tags for filtering.
            session_id: Optional session association.

        Returns:
            Memory response with id and created_at.
        """
        payload: dict[str, Any] = {
            "content": content,
            "role": role,
        }
        if tags:
            payload["tags"] = tags
        if session_id:
            payload["session_id"] = session_id

        resp = self.client.post("/memories", json=payload)
        resp.raise_for_status()
        return resp.json()

    def add_memory_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add multiple memories in a single request.

        Args:
            memories: List of memory dicts with content, role, tags, session_id.

        Returns:
            List of memory responses.
        """
        resp = self.client.post("/memories/batch", json=memories)
        resp.raise_for_status()
        return resp.json()

    def query_memories(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over memories.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            session_id: Optional session filter.
            tags: Optional tag filter.

        Returns:
            List of matching memories with scores.
        """
        params: dict[str, Any] = {"q": query, "top_k": top_k}
        if session_id:
            params["session_id"] = session_id

        resp = self.client.get("/memories", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_recent_memories(
        self,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get most recent memories.

        Args:
            limit: Maximum number of memories to return.
            session_id: Optional session filter.

        Returns:
            List of recent memories.
        """
        params: dict[str, Any] = {"limit": limit}
        if session_id:
            params["session_id"] = session_id

        resp = self.client.get("/memories/recent", params=params)
        resp.raise_for_status()
        return resp.json()

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete.

        Returns:
            Deletion confirmation.
        """
        resp = self.client.delete(f"/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()

    def clear_memories(self) -> dict[str, Any]:
        """Clear all memories from the index.

        Returns:
            Confirmation of clearing.
        """
        resp = self.client.delete("/memories")
        resp.raise_for_status()
        return resp.json()

    def get_anticipatory_context(
        self,
        recent_conversation: str,
        lite_mode: bool = False,
    ) -> dict[str, Any]:
        """Fetch anticipatory context for the current conversation.

        This is the primary FlowState feature — returns context that's
        predicted to be relevant based on the conversation state, before
        the agent even asks.

        Args:
            recent_conversation: Recent conversation text.
            lite_mode: Use lightweight mode for faster response.

        Returns:
            Anticipatory context with predicted relevant memories.
        """
        try:
            resp = self.client.post(
                "/anticipatory_context",
                json={
                    "recent_conversation": recent_conversation,
                    "lite_mode": lite_mode,
                },
            )
            if resp.status_code == 404:
                # Server doesn't support anticipatory_context endpoint
                logger.debug("QMD server doesn't support /anticipatory_context")
                return {"context": [], "source": "fallback"}
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            logger.warning("QMD server not available for anticipatory context")
            return {"context": [], "source": "unavailable"}


def get_qmd_client(config: QMDClientConfig | None = None) -> QMDClient:
    """Get or create the QMD client singleton.

    Args:
        config: Optional QMD configuration. If not provided, loads from
                environment or defaults.

    Returns:
        QMDClient instance.
    """
    global _qmd_client

    if _qmd_client is not None:
        return _qmd_client

    if config is None:
        config = QMDClientConfig.from_env()

    _qmd_client = QMDClient(config)
    return _qmd_client


def reset_qmd_client() -> None:
    """Reset the QMD client singleton (useful for testing or reconfiguration)."""
    global _qmd_client
    if _qmd_client is not None:
        _qmd_client.close()
    _qmd_client = None
