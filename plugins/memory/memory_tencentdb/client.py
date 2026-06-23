"""MemoryTencentdbSdkClient — HTTP client for the memory-tencentdb Gateway.

Wraps all Gateway API endpoints with timeout, retry, and error handling.
Thread-safe — can be shared across prefetch/sync threads.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds


class MemoryTencentdbSdkClient:
    """HTTP client for the memory-tencentdb Gateway sidecar."""

    def __init__(self, base_url: str = "http://127.0.0.1:8420", timeout: int = DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path: str, body: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Make a POST request to the Gateway."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning("memory-tencentdb Gateway %s returned %d: %s", path, e.code, body_text[:500])
            raise
        except Exception as e:
            logger.debug("memory-tencentdb Gateway %s failed: %s", path, e)
            raise

    def _get(self, path: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Make a GET request to the Gateway."""
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("memory-tencentdb Gateway GET %s failed: %s", path, e)
            raise

    # -- API methods ----------------------------------------------------------

    def health(self, timeout: int = 3) -> Dict[str, Any]:
        """Check if the Gateway is healthy."""
        return self._get("/health", timeout=timeout)

    def recall(self, query: str, session_key: str, user_id: str = "") -> Dict[str, Any]:
        """Recall memories for a query (prefetch)."""
        body: Dict[str, Any] = {"query": query, "session_key": session_key}
        if user_id:
            body["user_id"] = user_id
        return self._post("/recall", body)

    def capture(
        self,
        user_content: str,
        assistant_content: str,
        session_key: str,
        session_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        """Capture a conversation turn (sync_turn)."""
        body: Dict[str, Any] = {
            "user_content": user_content,
            "assistant_content": assistant_content,
            "session_key": session_key,
        }
        if session_id:
            body["session_id"] = session_id
        if user_id:
            body["user_id"] = user_id
        return self._post("/capture", body)

    def search_memories(self, query: str, limit: int = 5, type_filter: str = "", scene: str = "") -> Dict[str, Any]:
        """Search L1 structured memories."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if type_filter:
            body["type"] = type_filter
        if scene:
            body["scene"] = scene
        return self._post("/search/memories", body)

    def search_conversations(self, query: str, limit: int = 5, session_key: str = "") -> Dict[str, Any]:
        """Search L0 raw conversations."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if session_key:
            body["session_key"] = session_key
        return self._post("/search/conversations", body)

    def end_session(self, session_key: str, user_id: str = "") -> Dict[str, Any]:
        """End a session and trigger flush."""
        body: Dict[str, Any] = {"session_key": session_key}
        if user_id:
            body["user_id"] = user_id
        return self._post("/session/end", body)

    def seed(
        self,
        data: Any,
        session_key: str = "",
        strict_round_role: bool = False,
        auto_fill_timestamps: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Batch seed historical conversations into the memory pipeline.

        Args:
            data: Seed input — Format A ``{"sessions": [...]}`` or Format B ``[...]``.
            session_key: Fallback session key when input sessions lack one.
            strict_round_role: Require each round to have both user and assistant.
            auto_fill_timestamps: Auto-fill missing timestamps (default True).
            config_override: Plugin config overrides (deep-merged).
            timeout: Request timeout in seconds (seed can be slow, default 300s).

        Returns:
            Summary dict with sessions_processed, rounds_processed, etc.
        """
        body: Dict[str, Any] = {"data": data}
        if session_key:
            body["session_key"] = session_key
        if strict_round_role:
            body["strict_round_role"] = True
        if not auto_fill_timestamps:
            body["auto_fill_timestamps"] = False
        if config_override:
            body["config_override"] = config_override
        return self._post("/seed", body, timeout=timeout)
