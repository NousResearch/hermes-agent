"""
memory_tencentdb_v2 — Hermes MemoryProvider backed by TencentDB Agent Memory v2 API.

Uses the official `tencentdb_agent_memory` Python SDK to communicate with the
Memory Gateway. Supports both local sidecar mode (127.0.0.1:8420) and remote
service mode (endpoint + api_key + service_id).

Key differences from v1 `memory_tencentdb`:
  - Uses v2 REST API (`/v2/*`) with structured envelope responses
  - Uses `tencentdb_agent_memory.MemoryClient` SDK instead of raw urllib
  - Supports Bearer token authentication for multi-tenant service mode
  - Exposes `tdai_read_scene` tool for on-demand L2 scene reading
  - No local Gateway subprocess management (expects external Gateway)

Environment variables:
  TDAI_MEMORY_ENDPOINT    — Gateway URL (default: http://127.0.0.1:8420)
  TDAI_MEMORY_API_KEY     — API key for authentication (optional for local)
  TDAI_MEMORY_SERVICE_ID  — Service/Space ID (optional for local)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import json
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger("memory_tencentdb_v2")

# ════════════════════════════════════════════
# SDK import (lazy to allow graceful degradation)
# ════════════════════════════════════════════

_sdk_available = False
_MemoryClient = None
_AsyncMemoryClient = None
_TDAMError = None

try:
    from tencentdb_agent_memory import MemoryClient, AsyncMemoryClient, TDAMError
    _MemoryClient = MemoryClient
    _AsyncMemoryClient = AsyncMemoryClient
    _TDAMError = TDAMError
    _sdk_available = True
except ImportError:
    logger.warning(
        "tencentdb_agent_memory SDK not installed. "
        "Install the wheel package from the Hermes adapter README."
    )

# Decay/consolidation helpers from the hardened client (optional). Import
# gracefully: if tencentdb_client.py is not importable from this environment,
# recall falls back to raw relevance order — the provider must never break.
_tdai_decay_available = False
try:
    # Resolve the shared client (<hermes-agent>/tools/tencentdb_client.py).
    # The provider file may be loaded from the bundled hermes-agent tree OR a
    # copied/hardlinked tree under ~/.memory-tencentdb, so we walk up from
    # __file__ first, then fall back to known hermes-agent roots.
    import sys as _sys
    import os as _os
    _home = _os.path.expanduser("~")
    _candidates = []
    _cur = _os.path.dirname(_os.path.abspath(__file__))
    while True:
        _candidates.append(_os.path.join(_cur, "tools", "tencentdb_client.py"))
        _parent = _os.path.dirname(_cur)
        if _parent == _cur:
            break
        _cur = _parent
    # Known hermes-agent install roots (fallback for copied plugin trees).
    _candidates.append(_os.path.join(_home, ".hermes", "hermes-agent", "tools", "tencentdb_client.py"))
    _candidates.append(_os.path.join(_home, "hermes-agent", "tools", "tencentdb_client.py"))
    _tools = None
    for _cand in _candidates:
        if _os.path.isfile(_cand):
            _tools = _os.path.dirname(_cand)
            break
    if _tools and _tools not in _sys.path:
        _sys.path.insert(0, _tools)
    from tencentdb_client import (
        apply_decay as _tdai_apply_decay,
        consolidate_hits as _tdai_consolidate,
    )
    _tdai_decay_available = True
except Exception:
    logger.warning(
        "tencentdb_client decay helpers not importable; "
        "prefetch will use raw relevance order."
    )


class MemoryTencentdbV2Provider(MemoryProvider):
    """
    Hermes MemoryProvider implementation using TencentDB Agent Memory v2 API.
    """

    NAME = "memory_tencentdb_v2"

    def __init__(self):
        super().__init__()
        self._client: Optional[Any] = None
        self._session_id: str = ""
        self._endpoint: str = ""
        self._available: bool = False
        self._lock = threading.Lock()

        # Circuit breaker
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0
        self._max_failures: int = 5
        self._circuit_timeout: float = 60.0  # seconds

    # ════════════════════════════════════════════
    # Identity
    # ════════════════════════════════════════════

    @property
    def name(self) -> str:
        return self.NAME

    # ════════════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════════════

    def is_available(self) -> bool:
        return _sdk_available

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._endpoint = os.environ.get("TDAI_MEMORY_ENDPOINT", "http://127.0.0.1:8420")
        api_key = os.environ.get("TDAI_MEMORY_API_KEY", "")
        service_id = os.environ.get("TDAI_MEMORY_SERVICE_ID", "")

        if not _sdk_available:
            logger.error("tencentdb_agent_memory SDK not available, cannot initialize")
            return

        try:
            self._client = _MemoryClient(
                endpoint=self._endpoint,
                api_key=api_key or "local",
                service_id=service_id or "default",
            )
            self._available = True
            logger.info(
                f"Initialized: endpoint={self._endpoint}, "
                f"service_id={service_id or 'default'}, "
                f"session_id={session_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            self._available = False

    def shutdown(self) -> None:
        if self._client and self._session_id:
            try:
                # Flush pipeline buffers for this session
                self._safe_call("on_session_end", lambda: None)
            except Exception:
                pass
        self._client = None
        self._available = False
        logger.info("Shutdown complete")

    # ════════════════════════════════════════════
    # Core: Recall (prefetch)
    # ════════════════════════════════════════════

    def prefetch(self, query: str, session_id: str = "") -> Optional[Dict[str, Any]]:
        """
        Search L1 memories + L3 core for the given query.
        Returns dict with 'prepend_context' and 'append_system_context'.
        """
        sid = session_id or self._session_id

        def _do():
            # Search L1 memories (atomic) — SDK returns {"items": [...]}
            memories_result = self._client.search_atomic(query=query, limit=5)
            memories = memories_result.get("items", memories_result.get("results", []))

            # Apply temporal decay + consolidation (hardened client helpers).
            # Frequently-relevant memories stay fresh; stale ones fade below
            # the floor; near-duplicates collapse into one representative. This
            # runs inside the always-on recall path, not just the manual skill.
            if _tdai_decay_available:
                try:
                    memories = _tdai_apply_decay(memories, min_score=0.0)
                    memories = _tdai_consolidate(memories, threshold=0.6)
                except Exception:
                    logger.warning("prefetch decay/consolidation failed; using raw order")

            # Read L3 core (formerly persona)
            core_text = ""
            try:
                core_result = self._client.read_core()
                core_text = core_result.get("content", "")
            except Exception:
                pass

            # Read L2 scene navigation (list scenarios)
            scene_nav = ""
            try:
                scenarios = self._client.list_scenarios()
                entries = scenarios.get("entries", [])
                if entries:
                    lines = []
                    for s in entries:
                        name = s.get("path", "").replace("scene_blocks/", "").replace(".md", "")
                        lines.append(f"- Scene: {name} ({s.get('size', 0)} bytes)")
                    scene_nav = "Available scenes:\n" + "\n".join(lines)
            except Exception:
                pass

            # Build recall context
            prepend = ""
            if memories:
                memory_lines = []
                for m in memories:
                    content = m.get("content", "")
                    mtype = m.get("type", "unknown")
                    memory_lines.append(f"- [{mtype}] {content}")
                prepend = (
                    "<relevant-memories>\n"
                    "以下是当前对话召回的相关记忆，仅作为参考：\n\n"
                    + "\n".join(memory_lines)
                    + "\n</relevant-memories>"
                )

            append_parts = []
            if core_text:
                append_parts.append(f"<user-core>\n{core_text}\n</user-core>")
            if scene_nav:
                append_parts.append(f"<scene-navigation>\n{scene_nav}\n</scene-navigation>")

            return {
                "prepend_context": prepend,
                "append_system_context": "\n\n".join(append_parts) if append_parts else "",
            }

        return self._safe_call("prefetch", _do)

    # ════════════════════════════════════════════
    # Core: Capture (sync_turn)
    # ════════════════════════════════════════════

    def sync_turn(self, user_content: str, assistant_content: str, session_id: str = "") -> None:
        """Write a conversation turn to L0 via v2 API."""
        sid = session_id or self._session_id
        # v2 schema requires ISO 8601 datetime strings for timestamps.
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # Make user message slightly earlier than assistant for ordering
        user_ts = now.replace(microsecond=max(0, now.microsecond - 1000)).isoformat().replace("+00:00", "Z")
        assistant_ts = now.isoformat().replace("+00:00", "Z")

        def _do():
            messages = [
                {"role": "user", "content": user_content, "timestamp": user_ts},
                {"role": "assistant", "content": assistant_content, "timestamp": assistant_ts},
            ]
            self._client.add_conversation(session_id=sid, messages=messages)

        self._safe_call("sync_turn", _do)

    # ════════════════════════════════════════════
    # Mirror explicit memory-tool writes to TencentDB
    # ════════════════════════════════════════════

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory-tool writes into TencentDB L0.

        The base class no-ops this; without this override the explicit durable
        memory entries (agent notes + user profile) stay only in local Hermes
        storage. We push them through the same durable L0 capture path as
        per-turn sync_turn writes so every memory entry is recoverable from
        TencentDB. Best-effort via _safe_call — never breaks the write.
        """
        if not content:
            return

        def _do():
            # Tag the payload so a restore/backfill can distinguish
            # tool-memory entries from raw conversation turns.
            tag = "hermes-memory-tool"
            meta = dict(metadata or {})
            meta["action"] = action
            meta["target"] = target
            payload = (
                f"[{tag}] action={action} target={target} "
                f"meta={json.dumps(meta, ensure_ascii=False)} "
                f"content={content}"
            )
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            messages = [
                {"role": "user", "content": payload, "timestamp": now},
            ]
            self._client.add_conversation(
                session_id=self._session_id or "hermes-memory-tool", messages=messages
            )

        self._safe_call("on_memory_write", _do)

    # ════════════════════════════════════════════
    # Core: Session end
    # ════════════════════════════════════════════

    def on_session_end(self, messages: Optional[List] = None) -> None:
        """Signal session end to flush pipeline buffers."""
        # v2 API doesn't have explicit session/end — the pipeline handles it
        # via timer-based flush. This is a no-op for v2.
        logger.debug(f"Session end signaled for {self._session_id}")

    # ════════════════════════════════════════════
    # Tools: Agent-callable
    # ════════════════════════════════════════════

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose tools for the Agent to call."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "tdai_memory_search",
                    "description": (
                        "Search through the user's long-term memories. "
                        "Returns relevant memory records ranked by relevance."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results (default 5)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tdai_conversation_search",
                    "description": (
                        "Search through past conversation history. "
                        "Returns relevant messages ranked by relevance."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results (default 5)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tdai_read_scene",
                    "description": (
                        "Read a scene block's full content by its name. "
                        "Use when you see a scene listed in Scene Navigation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scene_id": {
                                "type": "string",
                                "description": "Scene name (e.g. 'travel-plan')",
                            },
                        },
                        "required": ["scene_id"],
                    },
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Handle Agent tool calls."""
        if tool_name == "tdai_memory_search":
            return self._handle_memory_search(args)
        elif tool_name == "tdai_conversation_search":
            return self._handle_conversation_search(args)
        elif tool_name == "tdai_read_scene":
            return self._handle_read_scene(args)
        return None

    def _handle_memory_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        limit = args.get("limit", 5)

        def _do():
            result = self._client.search_atomic(query=query, limit=limit)
            items = result.get("items", result.get("results", []))
            if not items:
                return "No memories found for this query."
            lines = []
            for m in items:
                lines.append(f"- [{m.get('type', '?')}] {m.get('content', '')}")
            return "\n".join(lines)

        return self._safe_call("memory_search", _do) or "Memory search failed."

    def _handle_conversation_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        limit = args.get("limit", 5)

        def _do():
            result = self._client.search_conversation(query=query, limit=limit)
            items = result.get("messages", result.get("results", []))
            if not items:
                return "No conversations found for this query."
            lines = []
            for m in items:
                role = m.get("role", "?")
                content = m.get("content", "")
                lines.append(f"[{role}] {content}")
            return "\n".join(lines)

        return self._safe_call("conversation_search", _do) or "Conversation search failed."

    def _handle_read_scene(self, args: Dict[str, Any]) -> str:
        scene_id = args.get("scene_id", "")
        if not scene_id:
            return "Error: scene_id is required"

        path = scene_id if scene_id.endswith(".md") else f"{scene_id}.md"

        def _do():
            result = self._client.read_scenario(path=path)
            content = result.get("content", "")
            if not content:
                return f"Scene '{scene_id}' is empty or not found."
            return content

        return self._safe_call("read_scene", _do) or f"Failed to read scene '{scene_id}'."

    # ════════════════════════════════════════════
    # Prompt
    # ════════════════════════════════════════════

    def system_prompt_block(self) -> str:
        return ""  # Core profile and scene nav injected via prefetch

    # ════════════════════════════════════════════
    # Circuit breaker + safe call wrapper
    # ════════════════════════════════════════════

    def _safe_call(self, label: str, fn):
        """Execute fn with circuit breaker protection."""
        if not self._client:
            logger.warning(f"[{label}] Client not initialized")
            return None

        # Check circuit breaker
        if self._consecutive_failures >= self._max_failures:
            if time.time() < self._circuit_open_until:
                logger.warning(f"[{label}] Circuit open, skipping call")
                return None
            else:
                logger.info(f"[{label}] Circuit half-open, retrying")

        try:
            result = fn()
            with self._lock:
                self._consecutive_failures = 0
            return result
        except Exception as e:
            with self._lock:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._circuit_open_until = time.time() + self._circuit_timeout
                    logger.error(
                        f"[{label}] Circuit OPEN after {self._consecutive_failures} failures "
                        f"(timeout={self._circuit_timeout}s): {e}"
                    )
                else:
                    logger.warning(f"[{label}] Failed ({self._consecutive_failures}/{self._max_failures}): {e}")
            return None
