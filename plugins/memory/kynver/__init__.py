"""Kynver AgentOS memory provider for Hermes.

This provider is intentionally additive: Hermes' built-in local memory remains
the source of truth for the built-in ``memory`` tool, while this provider mirrors
explicit durable writes and recalls Kynver AgentOS memory when configured as the
external ``memory.provider``. It does not replace terminal/file/browser tooling
or any local Hermes store.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.kynver_agentos_bridge import (
    KynverAgentOSClient,
    KynverAgentOSError,
    agentos_available,
)
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_SOURCE_ID = "hermes:forge"
_DEFAULT_SEARCH_LIMIT = 5

SEARCH_SCHEMA = {
    "name": "kynver_memory_search",
    "description": (
        "Search Kynver AgentOS memory for source-backed, cross-runtime context. "
        "Use this when local Hermes memory may be incomplete or when the work should "
        "align with Kynver's runtime-agnostic AgentOS state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language memory query."},
            "k": {"type": "integer", "description": "Max results, default 5, max 20."},
        },
        "required": ["query"],
    },
}

WRITE_SCHEMA = {
    "name": "kynver_memory_write",
    "description": (
        "Write a durable memory to Kynver AgentOS. This complements Hermes local "
        "memory; use for durable facts, decisions, preferences, runbooks, and source maps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "key": {"type": "string", "description": "Optional stable memory slug/key."},
            "memoryType": {
                "type": "string",
                "description": "Kynver memory type, e.g. fact, decision, preference, lesson, runbook.",
            },
        },
        "required": ["content"],
    },
}


def _coerce_items(payload: Any) -> list[dict[str, Any]]:
    """Extract memory-like rows from common Kynver/MCP response envelopes."""

    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            return _coerce_items(json.loads(payload))
        except Exception:
            return [{"content": payload}]
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"content": str(item)} for item in payload]
    if not isinstance(payload, dict):
        return [{"content": str(payload)}]

    for key in ("structuredContent", "result", "data"):
        if key in payload:
            nested = _coerce_items(payload[key])
            if nested:
                return nested

    for key in ("memories", "results", "items", "memory"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"content": str(item)} for item in value]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return [{"content": value}]

    if any(key in payload for key in ("content", "text", "memory", "slug", "key", "id")):
        return [payload]
    return []


def _item_content(item: dict[str, Any]) -> str:
    for key in ("content", "text", "memory", "summary"):
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _safe_error(exc: Exception) -> str:
    """Return a bounded, credential-redacted error string."""

    text = str(exc)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret|password)=([^\s&]+)", r"\1=[REDACTED]", text)
    return text[:500]


def _threat_message(content: str) -> Optional[str]:
    """Reuse Hermes' strict memory scan before promoting text to Kynver."""

    try:
        from tools.threat_patterns import first_threat_message

        return first_threat_message(content, scope="strict")
    except Exception:
        logger.debug("Kynver memory threat scan unavailable", exc_info=True)
        return None


def _format_prefetch(items: list[dict[str, Any]]) -> str:
    lines = ["# Kynver AgentOS memory"]
    count = 0
    for item in items[:_DEFAULT_SEARCH_LIMIT]:
        content = _item_content(item)
        if not content:
            continue
        count += 1
        ref = item.get("key") or item.get("slug") or item.get("id") or item.get("sourceId")
        suffix = f" [{ref}]" if ref else ""
        lines.append(f"- {content}{suffix}")
    if count == 0:
        return ""
    return "\n".join(lines)


class KynverMemoryProvider(MemoryProvider):
    """External MemoryProvider backed by Kynver AgentOS."""

    def __init__(self, client: Optional[Any] = None):
        self._client = client
        self._active = client is not None
        self._write_enabled = False
        self._session_id = ""
        self._platform = ""
        self._agent_identity = ""
        self._agent_workspace = ""

    @property
    def name(self) -> str:
        return "kynver"

    def is_available(self) -> bool:
        if self._client is not None:
            return True
        return agentos_available()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._platform = str(kwargs.get("platform") or "")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._agent_workspace = str(kwargs.get("agent_workspace") or "")
        agent_context = str(kwargs.get("agent_context") or "primary")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        if self._client is None:
            self._client = KynverAgentOSClient()
            self._active = bool(self._client and self._client.config.enabled)
        else:
            # Test doubles and future compatible clients may only implement the
            # small get/post bridge surface. Treat injected clients as active so
            # the provider stays decoupled from a concrete transport class.
            config = getattr(self._client, "config", None)
            self._active = bool(getattr(config, "enabled", True))

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return (
            "# Kynver AgentOS memory\n"
            "Kynver memory is active as an additive external provider. Keep Hermes local memory intact; "
            "use Kynver for runtime-agnostic, source-backed memory and cross-agent continuity."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._client or not (query or "").strip():
            return ""
        try:
            payload = self._client.post("/memory/search", {"query": query.strip(), "k": _DEFAULT_SEARCH_LIMIT})
            return _format_prefetch(_coerce_items(payload))
        except Exception:
            logger.debug("Kynver AgentOS memory prefetch failed", exc_info=True)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Deliberately no-op for now. Automatic whole-turn capture can pollute durable
        # Kynver memory; explicit memory writes and session logs are safer first slices.
        return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, WRITE_SCHEMA]

    def _write_memory(
        self,
        content: str,
        *,
        key: str = "",
        memory_type: str = "fact",
        target: str = "memory",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self._active or not self._client:
            raise KynverAgentOSError("Kynver AgentOS memory provider is not active")
        clean = (content or "").strip()
        if not clean:
            raise ValueError("content is required")
        threat = _threat_message(clean)
        if threat:
            raise ValueError(threat)
        meta = dict(metadata or {})
        meta.setdefault("target", target)
        meta.setdefault("hermesSessionId", self._session_id)
        if self._platform:
            meta.setdefault("platform", self._platform)
        if self._agent_identity:
            meta.setdefault("agentIdentity", self._agent_identity)
        if self._agent_workspace:
            meta.setdefault("agentWorkspace", self._agent_workspace)
        body: dict[str, Any] = {
            "content": clean,
            "memoryType": memory_type or "fact",
            "sourceId": _SOURCE_ID,
            "metadata": meta,
        }
        if key:
            body["key"] = key
        return self._client.post("/memory", body)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action != "add" or not self._write_enabled or not (content or "").strip():
            return
        try:
            self._write_memory(content, target=target, memory_type="preference" if target == "user" else "fact", metadata=metadata)
        except Exception:
            logger.debug("Kynver AgentOS on_memory_write failed", exc_info=True)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "kynver_memory_search":
            if not self._active or not self._client:
                return tool_error("Kynver AgentOS memory provider is not active")
            query = str(args.get("query") or "").strip()
            if not query:
                return tool_error("query is required")
            try:
                k = max(1, min(20, int(args.get("k", _DEFAULT_SEARCH_LIMIT) or _DEFAULT_SEARCH_LIMIT)))
            except Exception:
                k = _DEFAULT_SEARCH_LIMIT
            try:
                payload = self._client.post("/memory/search", {"query": query, "k": k}) if self._client else {}
                memories = _coerce_items(payload)[:k]
                return json.dumps({"memories": memories, "count": len(memories)})
            except Exception as exc:
                return tool_error(f"Kynver memory search failed: {_safe_error(exc)}")

        if tool_name == "kynver_memory_write":
            content = str(args.get("content") or "").strip()
            if not content:
                return tool_error("content is required")
            try:
                result = self._write_memory(
                    content,
                    key=str(args.get("key") or "").strip(),
                    memory_type=str(args.get("memoryType") or "fact").strip() or "fact",
                    metadata={"write_origin": "kynver_memory_write_tool"},
                )
                return json.dumps({
                    "saved": True,
                    "id": (result or {}).get("id", "") if isinstance(result, dict) else "",
                    "key": (result or {}).get("key", "") if isinstance(result, dict) else "",
                })
            except Exception as exc:
                return tool_error(f"Kynver memory write failed: {_safe_error(exc)}")

        return tool_error(f"Kynver memory provider does not handle tool '{tool_name}'")


def register(ctx) -> None:
    ctx.register_memory_provider(KynverMemoryProvider())
