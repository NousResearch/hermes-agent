"""EverOS memory plugin — MemoryProvider interface for the EverMind EverOS memory system.

EverOS provides structured memory extraction from conversations, intelligent retrieval
with multiple search methods (keyword, vector, hybrid, RRF, agentic), and progressive
user profile building. It achieves 93% reasoning accuracy on the LoCoMo benchmark.

API Reference: https://github.com/EverMind-AI/EverOS
Requires: EverOS server running (Docker) at configured URL (default: localhost:1995).

Config via environment variables:
  EVEROS_URL       — EverOS API base URL (default: http://localhost:1995)
  EVEROS_USER_ID   — User identifier (default: hermes-user)

Or via $HERMES_HOME/everos.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# HTTP helper (no external deps)
# ---------------------------------------------------------------------------

def _http_request(url: str, method: str = "GET", data: dict = None, timeout: int = 10) -> dict:
    """Make an HTTP request and return parsed JSON response."""
    body = json.dumps(data).encode("utf-8") if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/everos.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "url": os.environ.get("EVEROS_URL", "http://localhost:1995"),
        "user_id": os.environ.get("EVEROS_USER_ID", "hermes-user"),
    }

    config_path = get_hermes_home() / "everos.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "everos_search",
    "description": (
        "Search memories stored in EverOS by meaning. Supports keyword, vector, "
        "hybrid, and agentic retrieval. Returns grouped results ranked by relevance. "
        "Use this to recall past conversations, user preferences, facts, and events."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in stored memories.",
            },
            "method": {
                "type": "string",
                "enum": ["keyword", "vector", "hybrid", "rrf", "agentic"],
                "description": "Retrieval method. 'hybrid' is a good default. 'agentic' is slowest but most thorough.",
            },
            "memory_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Memory types to search: episodic_memory, profile, foresight, event_log.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max result groups to return (default: 10, max: 50).",
            },
        },
        "required": ["query"],
    },
}

RECALL_SCHEMA = {
    "name": "everos_recall",
    "description": (
        "Fetch stored memories by type — user profile, episodic memories, or event logs. "
        "Use at conversation start to load user context, or when you need the full profile "
        "rather than a semantic search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_type": {
                "type": "string",
                "enum": ["profile", "episodic_memory", "foresight", "event_log"],
                "description": "Type of memories to fetch (default: episodic_memory).",
            },
            "limit": {
                "type": "integer",
                "description": "Max memories to return (default: 10, max: 100).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class EverOSMemoryProvider(MemoryProvider):
    """EverOS memory provider with structured extraction and multi-method retrieval.

    Ingests conversations via sync_turn(), auto-extracts episodic memories,
    user profiles, and foresight predictions. Provides semantic search and
    typed recall as agent tools.
    """

    def __init__(self):
        self._config = _load_config()
        self._session_id = ""
        self._agent_context = "primary"
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_cache: Dict[str, str] = {}
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_until = 0.0

    @property
    def name(self) -> str:
        return "everos"

    # -- Core lifecycle -------------------------------------------------------

    def is_available(self) -> bool:
        """Check if EverOS server is reachable. NO network calls in check."""
        url = os.environ.get("EVEROS_URL", self._config.get("url", ""))
        return bool(url)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize EverOS connection and verify server health."""
        self._session_id = session_id
        self._config = _load_config()
        self._agent_context = kwargs.get("agent_context", "primary")

        # Verify server is reachable
        base_url = self._config["url"].rstrip("/")
        health = _http_request(f"{base_url}/health", timeout=5)
        if "error" in health:
            logger.warning(
                "EverOS server not reachable at %s: %s. "
                "Plugin will retry on first use.",
                base_url, health["error"],
            )
        else:
            logger.info("EverOS connected: %s", health.get("status", "ok"))

    def system_prompt_block(self) -> str:
        """Static info about EverOS for the system prompt."""
        return (
            "EverOS memory is active. Use everos_search to find past memories by "
            "meaning, or everos_recall to fetch memories by type (profile, episodic). "
            "Memories are auto-extracted from conversations."
        )

    # -- Prefetch / recall context --------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch results for the upcoming turn."""
        cache_key = session_id or "default"
        return self._prefetch_cache.pop(cache_key, "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background recall for the next turn."""
        cache_key = session_id or "default"

        def _fetch():
            try:
                results = self._search(query, method="hybrid", top_k=5)
                if results:
                    self._prefetch_cache[cache_key] = self._format_search_results(results)
            except Exception as e:
                logger.debug("EverOS prefetch failed: %s", e)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()

    # -- Sync turns (ingest) --------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to EverOS. Non-blocking."""
        # Skip writes for non-primary contexts (cron, subagent, flush)
        if self._agent_context != "primary":
            return

        def _sync():
            if not self._check_breaker():
                return
            try:
                self._ingest_message(user_content, role="user")
                self._ingest_message(assistant_content, role="assistant")
                self._consecutive_failures = 0
            except Exception as e:
                self._record_failure(e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True)
        self._sync_thread.start()

    # -- Tool schemas and dispatch --------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose everos_search and everos_recall tools."""
        return [SEARCH_SCHEMA, RECALL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool calls for everos_search and everos_recall."""
        if tool_name == "everos_search":
            return self._handle_search(args)
        elif tool_name == "everos_recall":
            return self._handle_recall(args)
        return tool_error(f"Unknown EverOS tool: {tool_name}")

    # -- Optional hooks -------------------------------------------------------

    def on_session_end(self, messages: list) -> None:
        """Final extraction flush at session end."""
        if self._agent_context != "primary":
            return
        # Ingest the full conversation summary if available
        if messages:
            last_user = ""
            last_asst = ""
            for msg in reversed(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                if role == "user" and not last_user:
                    last_user = content
                elif role == "assistant" and not last_asst:
                    last_asst = content
                if last_user and last_asst:
                    break
            if last_user or last_asst:
                self.sync_turn(last_user, last_asst)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to EverOS."""
        if self._agent_context != "primary" or not self._check_breaker():
            return
        try:
            self._ingest_message(
                f"[Memory {action}] target={target}: {content}",
                role="assistant",
            )
        except Exception as e:
            logger.debug("EverOS memory mirror failed: %s", e)

    def shutdown(self) -> None:
        """Wait for pending sync to complete."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    # -- Config ---------------------------------------------------------------

    def get_config_schema(self) -> list:
        return [
            {
                "key": "url",
                "description": "EverOS server URL",
                "default": "http://localhost:1995",
                "required": True,
            },
            {
                "key": "user_id",
                "description": "User identifier in EverOS",
                "default": "hermes-user",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/everos.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "everos.json"
        config_path.write_text(json.dumps(values, indent=2))

    # -- Internal API methods -------------------------------------------------

    def _base_url(self) -> str:
        return self._config["url"].rstrip("/")

    def _ingest_message(self, content: str, role: str = "user") -> dict:
        """Send a single message to EverOS for memory extraction."""
        url = f"{self._base_url()}/api/v1/memories"
        payload = {
            "message_id": f"{self._session_id}_{int(time.time() * 1000)}",
            "create_time": datetime.now(timezone.utc).isoformat(),
            "sender": self._config["user_id"],
            "content": content,
            "role": role,
            "sender_name": self._config["user_id"],
        }
        return _http_request(url, method="POST", data=payload, timeout=15)

    def _search(self, query: str, method: str = "hybrid",
                memory_types: list = None, top_k: int = 10) -> dict:
        """Search EverOS memories."""
        url = f"{self._base_url()}/api/v1/memories/search"
        payload = {
            "query": query,
            "user_id": self._config["user_id"],
            "retrieve_method": method,
            "top_k": top_k,
        }
        if memory_types:
            payload["memory_types"] = memory_types
        return _http_request(url, method="GET", data=payload, timeout=15)

    def _fetch(self, memory_type: str = "episodic_memory", limit: int = 10) -> dict:
        """Fetch memories by type."""
        url = f"{self._base_url()}/api/v1/memories"
        payload = {
            "user_id": self._config["user_id"],
            "memory_type": memory_type,
            "limit": limit,
        }
        return _http_request(url, method="GET", data=payload, timeout=15)

    # -- Tool handlers --------------------------------------------------------

    def _handle_search(self, args: dict) -> str:
        """Handle everos_search tool call."""
        query = args.get("query", "")
        method = args.get("method", "hybrid")
        memory_types = args.get("memory_types")
        top_k = args.get("top_k", 10)

        if not query:
            return json.dumps({"error": "query is required"})

        if not self._check_breaker():
            return json.dumps({"error": "EverOS temporarily unavailable (circuit breaker active)"})

        try:
            result = self._search(query, method=method, memory_types=memory_types, top_k=top_k)
            if "error" in result:
                self._record_failure(result["error"])
                return json.dumps({"error": f"EverOS search failed: {result['error']}"})
            self._consecutive_failures = 0
            return json.dumps(result)
        except Exception as e:
            self._record_failure(e)
            return json.dumps({"error": f"EverOS search error: {e}"})

    def _handle_recall(self, args: dict) -> str:
        """Handle everos_recall tool call."""
        memory_type = args.get("memory_type", "episodic_memory")
        limit = args.get("limit", 10)

        if not self._check_breaker():
            return json.dumps({"error": "EverOS temporarily unavailable (circuit breaker active)"})

        try:
            result = self._fetch(memory_type=memory_type, limit=limit)
            if "error" in result:
                self._record_failure(result["error"])
                return json.dumps({"error": f"EverOS recall failed: {result['error']}"})
            self._consecutive_failures = 0
            return json.dumps(result)
        except Exception as e:
            self._record_failure(e)
            return json.dumps({"error": f"EverOS recall error: {e}"})

    # -- Formatting -----------------------------------------------------------

    @staticmethod
    def _format_search_results(result: dict) -> str:
        """Format search results for injection as context."""
        lines = ["[EverOS recalled memories]"]
        memories = result.get("result", {}).get("memories", [])
        if not memories:
            return ""
        for group in memories[:5]:
            if isinstance(group, dict):
                summary = group.get("summary", group.get("episode", ""))
                if summary:
                    lines.append(f"- {summary}")
        return "\n".join(lines) if len(lines) > 1 else ""

    # -- Circuit breaker ------------------------------------------------------

    def _check_breaker(self) -> bool:
        """Check if circuit breaker allows API calls."""
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            if time.time() < self._breaker_until:
                return False
            # Cooldown expired, allow one attempt
            return True
        return True

    def _record_failure(self, error) -> None:
        """Record a failure for the circuit breaker."""
        self._consecutive_failures += 1
        logger.warning("EverOS failure #%d: %s", self._consecutive_failures, error)
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_until = time.time() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "EverOS circuit breaker OPEN — pausing calls for %ds",
                _BREAKER_COOLDOWN_SECS,
            )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by the memory plugin discovery system."""
    ctx.register_memory_provider(EverOSMemoryProvider())
