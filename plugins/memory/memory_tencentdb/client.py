"""MemoryTencentdbSdkClient — HTTP client for the memory-tencentdb Gateway.

Wraps all Gateway API endpoints with timeout, retry, and error handling.
Thread-safe — can be shared across prefetch/sync threads.

v3 migration: all data-plane endpoints now use /v3/* paths with
team_id / agent_id / user_id tenancy isolation.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds


class MemoryTencentdbSdkClient:
    """HTTP client for the memory-tencentdb Gateway sidecar."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8420",
        timeout: int = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
        service_id: str = "default",
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = (api_key or "").strip() or None
        self._service_id = service_id or "default"

    def _build_headers(self, *, content_type: bool) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = "application/json"
        # Always send Bearer token: if api_key is configured use it,
        # otherwise send "local" so parseV2Auth doesn't reject the request
        # (Gateway with auth=disabled ignores the token value).
        headers["Authorization"] = f"Bearer {self._api_key or 'local'}"
        headers["x-tdai-service-id"] = self._service_id
        return headers

    def _post(self, path: str, body: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Make a POST request to the Gateway and unwrap the v3 envelope."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._build_headers(content_type=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                return self._unwrap_v3(raw, path)
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
        req = urllib.request.Request(
            url,
            headers=self._build_headers(content_type=False),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("memory-tencentdb Gateway GET %s failed: %s", path, e)
            raise

    @staticmethod
    def _unwrap_v3(raw: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Extract data from v3 envelope {code, message, data}.
        
        Returns the raw dict on non-zero code so callers can inspect code/message.
        """
        code = raw.get("code", -1)
        if code != 0:
            msg = raw.get("message", "unknown")
            logger.warning("memory-tencentdb Gateway %s returned code=%d: %s", path, code, msg)
        return raw

    # -- API methods ----------------------------------------------------------

    def health(self, timeout: int = 3) -> Dict[str, Any]:
        """Check if the Gateway is healthy."""
        return self._get("/health", timeout=timeout)

    # ── v3: conversation (L0) ────────────────────────────────────────────────

    def conversation_add(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Add conversation messages to L0 (v3 /conversation/add).
        
        Args:
            messages: list of {role, content, timestamp}.
            session_id: business-side session id.
            team_id / agent_id / user_id: tenancy isolation.
        """
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages,
        }
        return self._post("/v3/conversation/add", body)

    def conversation_search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Search L0 conversations (v3 /conversation/search)."""
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "query": query,
            "limit": limit,
        }
        if session_id:
            body["session_id"] = session_id
        return self._post("/v3/conversation/search", body)

    # ── v3: atomic (L1) ─────────────────────────────────────────────────────

    def atomic_search(
        self,
        query: str,
        *,
        limit: int = 5,
        type_filter: str = "",
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Search L1 structured memories (v3 /atomic/search)."""
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "query": query,
            "limit": limit,
        }
        if type_filter:
            body["type"] = type_filter
        return self._post("/v3/atomic/search", body)

    # ── v3: scenario (L2) ────────────────────────────────────────────────────

    def scenario_ls(
        self,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """List L2 scene blocks (v3 /scenario/ls)."""
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
        }
        return self._post("/v3/scenario/ls", body)

    def scenario_read(
        self,
        path: str,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Read a L2 scene block (v3 /scenario/read)."""
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "path": path,
        }
        return self._post("/v3/scenario/read", body)

    # ── v3: core (L3) ────────────────────────────────────────────────────────

    def core_read(
        self,
        *,
        team_id: str = "default",
        agent_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Read L3 persona / user core (v3 /core/read)."""
        body: Dict[str, Any] = {
            "team_id": team_id,
            "agent_id": agent_id,
            "user_id": user_id,
        }
        return self._post("/v3/core/read", body)

    # ── v1 legacy (kept for backward compat; deprecated) ─────────────────────

    def recall(self, query: str, session_key: str, user_id: str = "") -> Dict[str, Any]:
        """[DEPRECATED] v1 recall — replaced by atomic_search + core_read + scenario_ls."""
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
        """[DEPRECATED] v1 capture — replaced by conversation_add."""
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
        """[DEPRECATED] v1 search_memories — replaced by atomic_search."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if type_filter:
            body["type"] = type_filter
        if scene:
            body["scene"] = scene
        return self._post("/search/memories", body)

    def search_conversations(self, query: str, limit: int = 5, session_key: str = "") -> Dict[str, Any]:
        """[DEPRECATED] v1 search_conversations — replaced by conversation_search."""
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if session_key:
            body["session_key"] = session_key
        return self._post("/search/conversations", body)

    def end_session(self, session_key: str, user_id: str = "") -> Dict[str, Any]:
        """[DEPRECATED] v1 end_session — v3 pipeline handles this automatically.
        
        Kept as no-op for backward compatibility; callers should remove this call.
        """
        logger.debug("memory-tencentdb end_session: v3 pipeline auto-handles session end, no-op")
        return {"status": "ok"}

    def seed(
        self,
        data: Any,
        session_key: str = "",
        strict_round_role: bool = False,
        auto_fill_timestamps: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Batch seed historical conversations into the memory pipeline."""
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
