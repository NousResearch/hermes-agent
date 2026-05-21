"""OpenMemory (self-hosted mem0) memory plugin — MemoryProvider interface.

Direct httpx client for the OpenMemory Community Edition API (e.g. Unraid
mem0-aio container). No dependency on the mem0ai cloud SDK.

Config via environment variables:
  OPENMEMORY_HOST      — Base URL of the OpenMemory API (required)
                         Example: http://192.168.188.22:3000/openmemory-api
  OPENMEMORY_API_KEY   — Optional API key if the instance requires auth
  OPENMEMORY_USER_ID   — User identifier (default: hermes-user)
  OPENMEMORY_AGENT_ID  — Agent identifier (default: hermes)

Or via $HERMES_HOME/openmemory.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/openmemory.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "host": os.environ.get("OPENMEMORY_HOST", ""),
        "api_key": os.environ.get("OPENMEMORY_API_KEY", ""),
        "user_id": os.environ.get("OPENMEMORY_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("OPENMEMORY_AGENT_ID", "hermes"),
        "rerank": True,
    }

    config_path = get_hermes_home() / "openmemory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


PROFILE_SCHEMA = {
    "name": "openmemory_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "openmemory_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "openmemory_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


class OpenMemoryProvider(MemoryProvider):
    """Self-hosted OpenMemory provider via direct HTTP API calls."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._host = ""
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "openmemory"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("host"))

    def save_config(self, values, hermes_home):
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "openmemory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "host", "description": "OpenMemory base URL (e.g. http://host:3000/openmemory-api)", "required": True, "env_var": "OPENMEMORY_HOST"},
            {"key": "api_key", "description": "Optional API key", "secret": True, "env_var": "OPENMEMORY_API_KEY"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Lazy-init httpx client."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError:
                raise RuntimeError("httpx not installed. Run: pip install httpx")
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Token {self._api_key}"
            self._client = httpx.Client(
                base_url=self._host.rstrip("/"),
                headers=headers,
                timeout=30.0,
            )
            return self._client

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "OpenMemory circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._host = self._config.get("host", "")
        self._api_key = self._config.get("api_key", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def _read_params(self) -> Dict[str, Any]:
        return {"user_id": self._user_id}

    def _write_params(self) -> Dict[str, Any]:
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# OpenMemory\n"
            f"Active. User: {self._user_id}. Host: {self._host}.\n"
            "Use openmemory_search to find memories, openmemory_conclude to store facts, "
            "openmemory_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## OpenMemory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                payload = {
                    "query": query,
                    **self._read_params(),
                }
                if self._rerank:
                    payload["rerank"] = True
                resp = client.post("/api/v1/memories/filter/", json=payload)
                resp.raise_for_status()
                results = self._unwrap_results(resp.json())
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("OpenMemory prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="openmemory-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                # OpenMemory CE expects flat text, not a messages array
                text = f"User: {user_content}\nAssistant: {assistant_content}"
                payload = {"text": text, **self._write_params()}
                resp = client.post("/api/v1/memories/", json=payload)
                resp.raise_for_status()
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("OpenMemory sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="openmemory-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "OpenMemory API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "openmemory_profile":
            try:
                resp = client.get("/api/v1/memories/", params=self._read_params())
                resp.raise_for_status()
                memories = self._unwrap_results(resp.json())
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "openmemory_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                payload = {
                    "query": query,
                    **self._read_params(),
                }
                if rerank:
                    payload["rerank"] = True
                if top_k:
                    payload["top_k"] = top_k
                resp = client.post("/api/v1/memories/filter/", json=payload)
                resp.raise_for_status()
                results = self._unwrap_results(resp.json())
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "openmemory_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                payload = {"text": conclusion, **self._write_params()}
                resp = client.post("/api/v1/memories/", json=payload)
                resp.raise_for_status()
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def register(ctx) -> None:
    """Register OpenMemory as a memory provider plugin."""
    ctx.register_memory_provider(OpenMemoryProvider())
