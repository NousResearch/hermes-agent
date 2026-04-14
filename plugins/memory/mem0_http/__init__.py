"""Mem0 HTTP memory plugin — MemoryProvider interface.

Direct HTTP integration with the Mem0 Platform API.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_BASE_URL      — API base URL (default: https://api.mem0.ai)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0_http.json.
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

_DEFAULT_BASE_URL = "https://api.mem0.ai"

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0_http.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "base_url": os.environ.get("MEM0_BASE_URL", _DEFAULT_BASE_URL),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "version": "v2",
    }

    config_path = get_hermes_home() / "mem0_http.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v not in (None, "")})
        except Exception:
            pass

    return config


PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the current Mem0 identity. "
        "Use at conversation start for a profile overview."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search Mem0 memories by meaning. Returns relevant facts ranked by similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {
                "type": "boolean",
                "description": "Enable reranking for precision (default: false).",
            },
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the current Mem0 identity. Stored verbatim."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


class _Client:
    """Minimal Mem0 HTTP client."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        json_body: Dict[str, Any] | None = None,
        timeout: float = 8.0,
    ) -> Any:
        import requests

        response = requests.request(
            method.upper(),
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            headers=self._headers(),
            timeout=timeout,
        )
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        if not response.ok:
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("error") or payload
            else:
                message = payload
            raise RuntimeError(
                f"Mem0 {method.upper()} {path} failed ({response.status_code}): {message}"
            )
        return payload

    @staticmethod
    def _scope_and(filters: Dict[str, Any]) -> Dict[str, Any]:
        clauses = []
        for key, value in filters.items():
            if value:
                clauses.append({key: value})
        if not clauses:
            return {}
        if len(clauses) == 1:
            return {"OR": clauses}
        return {"AND": clauses}

    def add_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        user_id: str,
        agent_id: str,
        version: str,
        infer: bool = True,
    ) -> Any:
        payload = {
            "messages": messages,
            "user_id": user_id,
            "version": version,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        if not infer:
            payload["infer"] = False
        return self.request("POST", "/v1/memories/", json_body=payload, timeout=10.0)

    def search(
        self,
        query: str,
        *,
        filters: Dict[str, Any],
        rerank: bool,
        top_k: int,
    ) -> Any:
        payload = {
            "query": query,
            "filters": self._scope_and(filters),
            "rerank": rerank,
            "top_k": top_k,
        }
        return self.request("POST", "/v2/memories/search/", json_body=payload, timeout=8.0)

    def get_all(self, *, filters: Dict[str, Any], version: str) -> Any:
        params = {k: v for k, v in filters.items() if v}
        if version:
            params["version"] = version
        try:
            return self.request("GET", "/v1/memories/", params=params, timeout=8.0)
        except Exception:
            return self.request("GET", "/v1/memories", params=params, timeout=8.0)


class Mem0HttpMemoryProvider(MemoryProvider):
    """Mem0 memory provider implemented via direct HTTP calls."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._base_url = _DEFAULT_BASE_URL
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._version = "v2"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0_http"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        from pathlib import Path

        config_path = Path(hermes_home) / "mem0_http.json"
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
            {
                "key": "api_key",
                "description": "Mem0 Platform API key",
                "secret": True,
                "required": True,
                "env_var": "MEM0_API_KEY",
                "url": "https://app.mem0.ai",
            },
            {
                "key": "base_url",
                "description": "Mem0 API base URL",
                "default": _DEFAULT_BASE_URL,
                "env_var": "MEM0_BASE_URL",
            },
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {
                "key": "rerank",
                "description": "Enable reranking for recall",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = _Client(self._api_key, self._base_url)
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
                "Mem0 HTTP circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._base_url = self._config.get("base_url", _DEFAULT_BASE_URL)
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = str(self._config.get("rerank", True)).lower() != "false"
        self._version = self._config.get("version", "v2")

    def _read_scope(self) -> Dict[str, Any]:
        scope = {"user_id": self._user_id}
        if self._agent_id:
            scope["agent_id"] = self._agent_id
        return scope

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 HTTP Memory\n"
            f"Active. User: {self._user_id}. Agent: {self._agent_id or 'unset'}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        if isinstance(response, dict):
            return response.get("results", response.get("memories", []))
        if isinstance(response, list):
            return response
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(
                    client.search(
                        query=query,
                        filters=self._read_scope(),
                        rerank=self._rerank,
                        top_k=5,
                    )
                )
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {line}" for line in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 HTTP prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mem0-http-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                client.add_messages(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    version=self._version,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 HTTP sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="mem0-http-sync"
        )
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(
                    client.get_all(filters=self._read_scope(), version=self._version)
                )
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(
                    client.search(
                        query=query,
                        filters=self._read_scope(),
                        rerank=rerank,
                        top_k=top_k,
                    )
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {"memory": r.get("memory", ""), "score": r.get("score", 0)}
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        if tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add_messages(
                    [{"role": "user", "content": conclusion}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    version=self._version,
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 HTTP as a memory provider plugin."""
    ctx.register_memory_provider(Mem0HttpMemoryProvider())
