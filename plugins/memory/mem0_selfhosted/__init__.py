"""Mem0 Self-Hosted memory plugin — MemoryProvider interface.

Talks to the self-hosted Mem0 FastAPI server directly via its REST API
(X-API-Key header, /memories, /search — NOT the cloud SDK's /v1/* paths).

Configuration
-------------
Secret (lives in $HERMES_HOME/.env or environment):
  MEM0_SELFHOSTED_API_KEY   — Server API key (X-API-Key header)

Behavioral settings (live in $HERMES_HOME/mem0_selfhosted.json, set via
`hermes memory setup`):
  base_url    — Server base URL, e.g. "http://localhost:8888"
  user_id     — Canonical user identifier (default: "hermes-user")
  agent_id    — Agent identifier (default: "hermes")
  timeout     — HTTP timeout in seconds (default: 30)

The matching MEM0_SELFHOSTED_BASE_URL / MEM0_SELFHOSTED_USER_ID /
MEM0_SELFHOSTED_AGENT_ID environment variables are still read as a
backward-compatible fallback.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_DEFAULT_USER_ID = "hermes-user"


def _is_client_error_status(status: Optional[int]) -> bool:
    """4xx (except 429) is a client-side issue — should NOT trip the breaker."""
    if status is None:
        return False
    if status == 429:
        return False
    return 400 <= status < 500


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0_selfhosted.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "base_url": os.environ.get("MEM0_SELFHOSTED_BASE_URL", ""),
        "api_key": os.environ.get("MEM0_SELFHOSTED_API_KEY", ""),
        "agent_id": os.environ.get("MEM0_SELFHOSTED_AGENT_ID", "hermes"),
        "timeout": 30,
    }
    env_user_id = os.environ.get("MEM0_SELFHOSTED_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mem0_selfhosted.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text())
            if isinstance(file_cfg, dict):
                for k, v in file_cfg.items():
                    # Only override if file has a truthy value; empty strings
                    # in the file should NOT wipe an env-provided secret.
                    if v not in (None, ""):
                        config[k] = v
        except Exception as e:
            logger.warning("Failed to parse mem0_selfhosted.json: %s", e)
    return config


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class _Mem0ServerClient:
    """Thin httpx client for the self-hosted Mem0 FastAPI server."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        import httpx
        base_url = (base_url or "").rstrip("/")
        if not base_url:
            raise ValueError("base_url is required for mem0_selfhosted")
        if not api_key:
            raise ValueError("api_key is required for mem0_selfhosted")
        self._base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "hermes-mem0-selfhosted/1.0",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def ping(self) -> None:
        """Quick reachability check. Uses /configure (auth-gated, small payload)."""
        r = self._client.get("/configure")
        r.raise_for_status()

    def add(
        self,
        messages: List[Dict[str, str]],
        *,
        user_id: str,
        agent_id: str,
        infer: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        payload: Dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            payload["metadata"] = metadata
        r = self._client.post("/memories", json=payload)
        r.raise_for_status()
        return r.json()

    def get_all(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        top_k: int = 100,
    ) -> dict:
        params: Dict[str, Any] = {"user_id": user_id, "top_k": top_k}
        if agent_id:
            params["agent_id"] = agent_id
        r = self._client.get("/memories", params=params)
        r.raise_for_status()
        body = r.json()
        # Fork returns {"results": [...]} or {"results": [...], "count": N}
        if isinstance(body, dict):
            results = body.get("results", [])
            count = body.get("count", len(results))
            return {"results": results, "count": count}
        return {"results": body if isinstance(body, list) else [], "count": 0}

    def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list:
        filters: Dict[str, Any] = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id
        payload: Dict[str, Any] = {
            "query": query,
            "filters": filters,
            "top_k": top_k,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        r = self._client.post("/search", json=payload)
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict) and "results" in body:
            return body["results"] or []
        if isinstance(body, list):
            return body
        return []

    def update(self, memory_id: str, text: str) -> dict:
        r = self._client.put(f"/memories/{memory_id}", json={"text": text})
        r.raise_for_status()
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        r = self._client.delete(f"/memories/{memory_id}")
        r.raise_for_status()
        return {"result": "Memory deleted.", "memory_id": memory_id}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

LIST_SCHEMA = {
    "name": "mem0_list",
    "description": "List all stored memories for the current user. Use at conversation start for a full overview.",
    "parameters": {
        "type": "object",
        "properties": {
            "top_k": {"type": "integer", "description": "Max results (default 100, max 500)."},
        },
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": "Search memories by semantic query. Returns memories ranked by relevance.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default 10, max 50)."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": "Store a durable fact about the user. Stored verbatim (no LLM extraction).",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": "Update an existing memory's text by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": "Delete a memory by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0SelfHostedMemoryProvider(MemoryProvider):
    """Mem0 self-hosted server memory provider (REST client for the fork).

    Uses X-API-Key + /memories + /search endpoints. Independent of the
    cloud `mem0ai` SDK.
    """

    def __init__(self):
        self._config: Optional[dict] = None
        self._client: Optional[_Mem0ServerClient] = None
        self._base_url = ""
        self._api_key = ""
        self._user_id = _DEFAULT_USER_ID
        self._agent_id = "hermes"
        self._channel = "cli"
        self._init_error: Optional[str] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()
        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return "mem0_selfhosted"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("base_url") and cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0_selfhosted.json."""
        from pathlib import Path
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "mem0_selfhosted.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "base_url", "description": "Mem0 server base URL (e.g. http://your-host:8888)", "required": True},
            {"key": "api_key", "description": "Server API key (X-API-Key)", "secret": True, "required": True, "env_var": "MEM0_SELFHOSTED_API_KEY"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
        ]

    def _create_client(self) -> Optional[_Mem0ServerClient]:
        try:
            timeout = float(self._config.get("timeout", 30)) if self._config else 30.0
            return _Mem0ServerClient(self._base_url, self._api_key, timeout=timeout)
        except Exception as e:
            logger.error("Mem0 self-hosted client init failed: %s", e)
            self._init_error = str(e)
            return None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._base_url = self._config.get("base_url", "")
        self._api_key = self._config.get("api_key", "")
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        self._channel = kwargs.get("platform") or "cli"
        self._client = self._create_client()
        if self._client and not self._atexit_registered:
            atexit.register(self._shutdown_client)
            self._atexit_registered = True

    def _read_filters_user(self) -> str:
        return self._user_id

    def _write_metadata(self) -> Dict[str, Any]:
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory (self-hosted)\n"
            f"Active. Server: {self._base_url}. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_add to store facts, "
            "mem0_list for a full overview, mem0_update and mem0_delete to manage by ID."
        )

    # -- Circuit breaker helpers --------------------------------------------

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures += 1
            count = self._consecutive_failures
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            logger.warning(
                "Mem0 self-hosted circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _handle_http_error(self, exc: Exception) -> None:
        """Only trip the breaker for real server/network problems, not 4xx."""
        try:
            import httpx
        except Exception:
            self._record_failure()
            return
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response is not None else None
            if _is_client_error_status(status):
                return
        self._record_failure()

    # -- Prefetch + sync -----------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return ""
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory (self-hosted)\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._client is None or self._is_breaker_open():
            return

        def _run():
            client = self._client
            if client is None:
                return
            try:
                results = client.search(
                    query=query,
                    user_id=self._read_filters_user(),
                    top_k=5,
                )
                if results:
                    lines = [
                        (r.get("memory") or "").strip()
                        for r in results
                        if r.get("memory")
                    ]
                    lines = [l for l in lines if l]
                    if lines:
                        with self._prefetch_lock:
                            self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._handle_http_error(e)
                logger.debug("mem0_selfhosted prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mem0-selfhosted-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        """Send turn to server for LLM-based fact extraction (non-blocking)."""
        if self._client is None or self._is_breaker_open():
            return
        if not user_content and not assistant_content:
            return

        def _sync():
            client = self._client
            if client is None:
                return
            try:
                turn_messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(
                    turn_messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                    metadata=self._write_metadata(),
                )
                self._record_success()
            except Exception as e:
                self._handle_http_error(e)
                logger.warning("mem0_selfhosted sync failed: %s", e)

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(
                target=_sync, daemon=True, name="mem0-selfhosted-sync"
            )
            self._sync_thread.start()

    # -- Tool schemas + dispatch --------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [LIST_SCHEMA, SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._client is None:
            err = self._init_error or "unknown error"
            return json.dumps({"error": f"mem0_selfhosted backend not initialized: {err}. Check base_url and api_key."})

        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 self-hosted temporarily unavailable (multiple consecutive failures). "
                         f"Will retry automatically. Check that {self._base_url} is reachable."
            })

        try:
            if tool_name == "mem0_list":
                top_k = max(1, min(int(args.get("top_k", 100)), 500))
                response = self._client.get_all(
                    user_id=self._user_id, agent_id=None, top_k=top_k,
                )
                self._record_success()
                results = response.get("results", [])
                if not results:
                    return json.dumps({"result": "No memories stored yet."})
                items = [
                    {"id": m.get("id"), "memory": m.get("memory", "")}
                    for m in results
                ]
                return json.dumps({
                    "results": items,
                    "count": response.get("count", len(items)),
                })

            elif tool_name == "mem0_search":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = max(1, min(int(args.get("top_k", 10)), 50))
                results = self._client.search(
                    query, user_id=self._user_id, top_k=top_k,
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [
                    {
                        "id": r.get("id"),
                        "memory": r.get("memory", ""),
                        "score": r.get("score", 0),
                    }
                    for r in results
                ]
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "mem0_add":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                result = self._client.add(
                    [{"role": "user", "content": content}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                    metadata=self._write_metadata(),
                )
                self._record_success()
                new_ids = []
                if isinstance(result, dict):
                    for r in result.get("results") or []:
                        if isinstance(r, dict) and r.get("id"):
                            new_ids.append(r["id"])
                return json.dumps({
                    "result": "Fact stored.",
                    "memory_ids": new_ids,
                })

            elif tool_name == "mem0_update":
                memory_id = args.get("memory_id", "")
                text = args.get("text", "")
                if not memory_id:
                    return tool_error("Missing required parameter: memory_id")
                if not text:
                    return tool_error("Missing required parameter: text")
                result = self._client.update(memory_id, text)
                self._record_success()
                return json.dumps(result)

            elif tool_name == "mem0_delete":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return tool_error("Missing required parameter: memory_id")
                result = self._client.delete(memory_id)
                self._record_success()
                return json.dumps(result)

            return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            try:
                import httpx
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code if e.response is not None else None
                    if _is_client_error_status(status):
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            detail = e.response.text if e.response is not None else ""
                        return tool_error(f"mem0_selfhosted {status}: {detail or str(e)}")
            except Exception:
                pass
            self._handle_http_error(e)
            return tool_error(f"mem0_selfhosted {tool_name} failed: {e}")

    def _shutdown_client(self):
        try:
            if self._client:
                self._client.close()
                self._client = None
        except Exception:
            pass

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_client()


def register(ctx) -> None:
    """Register mem0_selfhosted as a memory provider plugin."""
    ctx.register_memory_provider(Mem0SelfHostedMemoryProvider())
