"""DreamCycle memory provider plugin template for Hermes."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "http://127.0.0.1:8765"
_DEFAULT_LIMIT = 5


class DreamcycleMemoryProvider(MemoryProvider):
    """Context prefetch + turn sync integration against DreamCycle /v1/memory/*."""

    def __init__(self) -> None:
        self._base_url = os.getenv("DREAMCYCLE_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")
        self._api_key = os.getenv("DREAMCYCLE_API_KEY", "").strip()
        self._source = os.getenv("DREAMCYCLE_SOURCE", "hermes")
        self._namespace = os.getenv("DREAMCYCLE_NAMESPACE", "").strip()
        self._user_id = os.getenv("DREAMCYCLE_USER_ID", "").strip()
        self._trace_id = os.getenv("DREAMCYCLE_TRACE_ID", "").strip()
        self._session_id = ""
        self._timeout = float(os.getenv("DREAMCYCLE_HTTP_TIMEOUT", "8"))

    @property
    def name(self) -> str:
        return "dreamcycle"

    def is_available(self) -> bool:
        return bool(self._base_url and self._api_key and self._namespace and self._user_id)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id or ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        response = self._post_json(
            "/v1/memory/search",
            {
                "query": query,
                "limit": _DEFAULT_LIMIT,
                "role": "assistant",
            },
            session_id or self._session_id,
        )
        memories = response.get("memories", []) if isinstance(response, dict) else []
        if not memories:
            return ""
        lines = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: List[Dict[str, Any]] | None = None,
    ) -> None:
        uc = (user_content or "").strip()
        ac = (assistant_content or "").strip()
        if not uc or not ac:
            return

        self._post_json(
            "/v1/memory/turns",
            {
                "user_content": uc,
                "assistant_content": ac,
                "source": self._source,
                "conversation_id": session_id or self._session_id,
                "trace_id": self._trace_id,
                "importance": 0.55,
                "success": True,
                "data_classification": "public",
                "metadata": {
                    "provider": "hermes",
                    "namespace": self._namespace,
                    "messages_count": len(messages or []),
                },
            },
            session_id or self._session_id,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        return json.dumps({"error": "DreamCycle provider has no tools"})

    def _post_json(self, path: str, payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + path,
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
                "x-dreamcycle-conversation-id": session_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"raw": raw}
        except urllib.error.HTTPError as exc:
            logger.warning("DreamCycle HTTP error on %s: %s", path, exc)
            return {}
        except Exception as exc:
            logger.warning("DreamCycle request failed on %s: %s", path, exc)
            return {}


def register(ctx):
    provider = DreamcycleMemoryProvider()
    ctx.register_memory_provider(provider)
