"""mem9 (mnemos) memory plugin — MemoryProvider interface.

Persistent, shared memory for AI agents backed by TiDB Cloud vector index.
Server-side fact extraction from conversation turns, semantic search with
relative-age metadata, and five CRUD tools (store, search, get, update, delete).

The plugin talks directly to the mnemos REST API — no Python SDK needed.

Config via environment variables:
  MEM9_API_KEY       — API key (doubles as tenant ID for v1alpha2 auth)
  MEM9_API_URL       — Server URL (default: https://api.mem9.ai)
  MEM9_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem9.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: pause API calls after consecutive failures.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_DEFAULT_API_URL = "https://api.mem9.ai"
_DEFAULT_AGENT_ID = "hermes"
_REQUEST_TIMEOUT = 8.0


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------

class _Mem9Client:
    """Lightweight REST client for the mnemos v1alpha2 API.

    Uses ``httpx`` (already a core dependency) for HTTP.  All methods are
    synchronous — background threading is handled by the provider layer.

    The ``agent_id`` is sent as the ``X-Mnemo-Agent-Id`` header and is what
    the server uses to set ``Memory.Source``.  Per-user isolation works by
    setting this header to a user-specific value (see ``Mem9MemoryProvider``).
    """

    def __init__(self, api_url: str, api_key: str, agent_id: str):
        import httpx

        self._base = api_url.rstrip("/")
        self._api_key = api_key
        self._agent_id = agent_id
        self._http = httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "X-Mnemo-Agent-Id": agent_id,
            },
        )

    # -- memories CRUD -------------------------------------------------------

    @staticmethod
    def _safe_json(resp) -> dict:
        """Parse JSON from response, returning {} for non-JSON bodies."""
        try:
            return resp.json()
        except Exception:
            return {}

    def store(self, content: str, *, tags: List[str] = None,
              session_id: str = "") -> dict:
        """Store a memory. Server returns 202 with ``{"status":"accepted"}``."""
        body: dict = {"content": content}
        if tags:
            body["tags"] = tags
        if session_id:
            body["session_id"] = session_id
        resp = self._http.post(f"{self._base}/v1alpha2/mem9s/memories", json=body)
        resp.raise_for_status()
        return self._safe_json(resp)

    def ingest(self, messages: List[dict], *, session_id: str = "") -> dict:
        """Ingest a conversation turn for server-side fact extraction (202)."""
        body: dict = {"messages": messages}
        if session_id:
            body["session_id"] = session_id
        resp = self._http.post(f"{self._base}/v1alpha2/mem9s/memories", json=body)
        resp.raise_for_status()
        return self._safe_json(resp)

    def search(self, query: str, *, limit: int = 10, tags: str = "") -> dict:
        """Search memories by semantic similarity.

        User scoping is handled by the ``X-Mnemo-Agent-Id`` header set at
        client construction time — the server uses this to derive the source
        filter internally.
        """
        params: dict = {"q": query, "limit": str(limit)}
        if tags:
            params["tags"] = tags
        resp = self._http.get(
            f"{self._base}/v1alpha2/mem9s/memories", params=params,
        )
        resp.raise_for_status()
        return resp.json()

    def get(self, memory_id: str) -> Optional[dict]:
        """Get a single memory by ID.

        By-ID operations are scoped at the tenant level (API key).
        Per-user isolation is enforced at the ``X-Mnemo-Agent-Id`` header
        level — the server sets Memory.Source from this header.
        """
        resp = self._http.get(
            f"{self._base}/v1alpha2/mem9s/memories/{memory_id}",
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def update(self, memory_id: str, content: str = "",
               tags: List[str] = None) -> Optional[dict]:
        """Update a memory's content or tags."""
        body: dict = {}
        if content:
            body["content"] = content
        if tags is not None:
            body["tags"] = tags
        resp = self._http.put(
            f"{self._base}/v1alpha2/mem9s/memories/{memory_id}",
            json=body,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted, False if not found."""
        resp = self._http.delete(
            f"{self._base}/v1alpha2/mem9s/memories/{memory_id}",
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def close(self) -> None:
        self._http.close()

    # -- autoprovision -------------------------------------------------------

    @staticmethod
    def autoprovision(api_url: str = _DEFAULT_API_URL) -> dict:
        """Create a new mem9 tenant via POST /v1alpha1/mem9s.

        Returns ``{"id": "..."}`` on success.  The returned ``id``
        doubles as both tenant ID and API key for v1alpha2 header auth.
        """
        import httpx

        resp = httpx.post(
            f"{api_url.rstrip('/')}/v1alpha1/mem9s",
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem9.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "api_url": os.environ.get("MEM9_API_URL", _DEFAULT_API_URL),
        "api_key": os.environ.get("MEM9_API_KEY", ""),
        "agent_id": os.environ.get("MEM9_AGENT_ID", _DEFAULT_AGENT_ID),
    }

    config_path = get_hermes_home() / "mem9.json"
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

STORE_SCHEMA = {
    "name": "mem9_store",
    "description": (
        "Store a memory in mem9. Use for explicit facts, preferences, decisions, "
        "or project context worth remembering across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization.",
            },
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "mem9_search",
    "description": (
        "Search memories by semantic similarity. Returns relevant memories "
        "ranked by score with relative age."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results (default: 10, max: 50).", "maximum": 50},
        },
        "required": ["query"],
    },
}

GET_SCHEMA = {
    "name": "mem9_get",
    "description": "Get a specific memory by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Memory ID to retrieve."},
        },
        "required": ["id"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem9_update",
    "description": "Update an existing memory's content or tags.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Memory ID to update."},
            "content": {"type": "string", "description": "New content (optional)."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New tags (optional).",
            },
        },
        "required": ["id"],
    },
}

DELETE_SCHEMA = {
    "name": "mem9_delete",
    "description": "Delete a memory by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem9MemoryProvider(MemoryProvider):
    """mem9 (mnemos) persistent memory with semantic search."""

    def __init__(self):
        self._config: dict = {}
        self._client: Optional[_Mem9Client] = None
        self._client_lock = threading.Lock()
        self._session_id = ""
        self._user_id = ""
        self._agent_id = _DEFAULT_AGENT_ID
        # Background threads
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        # Circuit breaker (protected by _breaker_lock for thread safety)
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem9"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "mem9.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "mem9 API key",
                "secret": True,
                "required": True,
                "env_var": "MEM9_API_KEY",
                "url": "https://app.mem9.ai",
            },
            {
                "key": "api_url",
                "description": "mem9 server URL",
                "default": _DEFAULT_API_URL,
                "env_var": "MEM9_API_URL",
            },
            {
                "key": "agent_id",
                "description": "Agent identifier",
                "default": _DEFAULT_AGENT_ID,
            },
        ]

    # -- setup wizard --------------------------------------------------------

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Interactive setup: autoprovision a tenant or enter an existing key."""
        import getpass
        import sys

        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _curses_select, _write_env_vars

        print("\n  Configuring mem9 memory:\n")

        items = [
            ("Auto-provision", "Create a free tenant automatically (recommended)"),
            ("Manual", "Enter an existing API key from app.mem9.ai"),
        ]
        choice = _curses_select("  Setup mode", items, default=0)

        env_path = Path(hermes_home) / ".env"
        env_writes: dict = {}
        provider_config: dict = {}
        api_key = ""

        if choice == 0:
            # Autoprovision
            api_url = _DEFAULT_API_URL
            print("\n  Provisioning mem9 tenant...")
            try:
                result = _Mem9Client.autoprovision(api_url)
                api_key = result.get("id", "")
                if not api_key:
                    print("  ✗ Provisioning failed: no tenant ID returned.")
                    return
                env_writes["MEM9_API_KEY"] = api_key
                print(f"  ✓ Tenant created: {api_key[:12]}...")
            except Exception as e:
                print(f"  ✗ Provisioning failed: {e}")
                print("  Try manual setup or check https://app.mem9.ai")
                return
        else:
            # Manual key entry
            print("\n  Get your API key at https://app.mem9.ai\n")
            existing_key = os.environ.get("MEM9_API_KEY", "")
            if existing_key:
                masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
                sys.stdout.write(f"  API key (current: {masked}, blank to keep): ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
                if not api_key:
                    api_key = existing_key
            else:
                sys.stdout.write("  API key: ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["MEM9_API_KEY"] = api_key

        if not api_key:
            print("  ✗ No API key — setup cancelled.")
            return

        # Optional: custom server URL
        val = input(f"  Server URL [{_DEFAULT_API_URL}]: ").strip()
        if val:
            provider_config["api_url"] = val

        # Optional: agent ID
        val = input(f"  Agent ID [{_DEFAULT_AGENT_ID}]: ").strip()
        if val:
            provider_config["agent_id"] = val

        # Connection test
        test_url = provider_config.get("api_url", _DEFAULT_API_URL)
        print("\n  Testing connection...")
        try:
            client = _Mem9Client(test_url, api_key,
                                 provider_config.get("agent_id", _DEFAULT_AGENT_ID))
            client.search("test", limit=1)
            client.close()
            print("  ✓ Connection successful")
        except Exception as e:
            print(f"  ✗ Connection test failed: {e}")
            print("  Config NOT saved. Fix your key/URL and try again.")
            return

        # Persist only after successful connection test
        config.setdefault("memory", {})["provider"] = "mem9"
        save_config(config)

        if provider_config:
            self.save_config(provider_config, hermes_home)

        if env_writes:
            _write_env_vars(env_path, env_writes)

        print(f"\n  ✓ mem9 activated")
        if env_writes:
            print(f"  API key saved to .env")
        print(f"  Start a new session to activate.\n")

    # -- lifecycle -----------------------------------------------------------

    def _get_client(self) -> Optional[_Mem9Client]:
        """Lazily create the HTTP client on first use.

        This avoids connection-pool leaks if the provider is replaced or
        an error occurs before shutdown() is called.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            api_key = self._config.get("api_key", "")
            api_url = self._config.get("api_url", _DEFAULT_API_URL)
            if not api_key:
                return None
            # Use _user_id as X-Mnemo-Agent-Id so the server sets
            # Memory.Source per user — this is the real isolation mechanism.
            self._client = _Mem9Client(api_url, api_key, self._user_id)
            return self._client

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._session_id = session_id
        self._agent_id = self._config.get("agent_id", _DEFAULT_AGENT_ID)
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to agent_identity (CLI profile), then agent_id.
        # This value becomes the X-Mnemo-Agent-Id header which the server
        # uses to set Memory.Source — the real per-user isolation mechanism.
        self._user_id = (
            kwargs.get("user_id")
            or kwargs.get("agent_identity")
            or self._agent_id
        )

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            if self._client:
                self._client.close()
                self._client = None

    # -- circuit breaker -----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
                logger.warning(
                    "mem9 circuit breaker tripped after %d failures. "
                    "Pausing API calls for %ds.",
                    self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
                )

    # -- system prompt / prefetch / sync -------------------------------------

    def system_prompt_block(self) -> str:
        if not self._config.get("api_key"):
            return ""
        return (
            "# mem9 Memory\n"
            f"Active. Agent: {self._user_id}.\n"
            "Use mem9_search to find memories, mem9_store to save facts, "
            "mem9_get/mem9_update/mem9_delete for CRUD."
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        client = self._get_client()
        if not client or self._is_breaker_open():
            return

        def _run():
            try:
                result = client.search(query[:200], limit=5)
                memories = result.get("memories") or []
                if memories:
                    lines = []
                    for m in memories:
                        content = m.get("content", "")
                        age = m.get("relative_age", "")
                        prefix = f"[{age}] " if age else ""
                        lines.append(f"- {prefix}{content}")
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("mem9 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mem9-prefetch",
        )
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## mem9 Memory\n{result}"

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Send the turn to mem9 for server-side fact extraction."""
        client = self._get_client()
        if not client or self._is_breaker_open():
            return

        def _sync():
            try:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.ingest(messages, session_id=self._session_id)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("mem9 sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="mem9-sync",
        )
        self._sync_thread.start()

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [STORE_SCHEMA, SEARCH_SCHEMA, GET_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if not self._get_client():
            return tool_error("mem9 is not configured")

        if self._is_breaker_open():
            return json.dumps({
                "error": "mem9 API temporarily unavailable. Will retry automatically.",
            })

        dispatch = {
            "mem9_store": self._tool_store,
            "mem9_search": self._tool_search,
            "mem9_get": self._tool_get,
            "mem9_update": self._tool_update,
            "mem9_delete": self._tool_delete,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}")
        return handler(args)

    def _tool_store(self, args: dict) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return tool_error("Missing required parameter: content")
        tags = args.get("tags") or []
        try:
            result = self._client.store(
                content, tags=tags, session_id=self._session_id,
            )
            self._record_success()
            return json.dumps({"stored": True, "id": result.get("id", "")})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Failed to store: {e}")

    def _tool_search(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        limit = min(int(args.get("limit", 10) or 10), 50)
        try:
            result = self._client.search(query, limit=limit)
            self._record_success()
            memories = result.get("memories") or []
            items = []
            for m in memories:
                entry: dict = {
                    "id": m.get("id", ""),
                    "content": m.get("content", ""),
                }
                if m.get("score") is not None:
                    entry["score"] = m["score"]
                if m.get("relative_age"):
                    entry["age"] = m["relative_age"]
                if m.get("tags"):
                    entry["tags"] = m["tags"]
                items.append(entry)
            if not items:
                return json.dumps({"result": "No relevant memories found."})
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Search failed: {e}")

    def _tool_get(self, args: dict) -> str:
        memory_id = (args.get("id") or "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: id")
        try:
            memory = self._client.get(memory_id)
            self._record_success()
            if memory is None:
                return json.dumps({"error": "Memory not found."})
            return json.dumps(memory)
        except Exception as e:
            self._record_failure()
            return tool_error(f"Get failed: {e}")

    def _tool_update(self, args: dict) -> str:
        memory_id = (args.get("id") or "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: id")
        content = (args.get("content") or "").strip()
        tags = args.get("tags")
        if not content and tags is None:
            return tool_error("Provide content or tags to update.")
        try:
            result = self._client.update(memory_id, content=content, tags=tags)
            self._record_success()
            if result is None:
                return json.dumps({"error": "Memory not found."})
            return json.dumps({"updated": True, "id": memory_id})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Update failed: {e}")

    def _tool_delete(self, args: dict) -> str:
        memory_id = (args.get("id") or "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: id")
        try:
            deleted = self._client.delete(memory_id)
            self._record_success()
            if not deleted:
                return json.dumps({"error": "Memory not found."})
            return json.dumps({"deleted": True, "id": memory_id})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Delete failed: {e}")


def register(ctx) -> None:
    """Register mem9 as a memory provider plugin."""
    ctx.register_memory_provider(Mem9MemoryProvider())
