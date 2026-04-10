"""Cognee V2 memory plugin — MemoryProvider interface.

Knowledge graph memory with session-aware storage, auto-routing recall,
forget capability, and session-end graph improvement via the Cognee V2 API.

Cognee stores conversation turns in a lightweight session cache per turn,
then bridges that data into the permanent knowledge graph at session end
via improve(). This replaces the V1 pattern of add()+cognify() every turn.

Config via environment variables:
  LLM_API_KEY           — LLM provider API key (required for local mode)
  LLM_MODEL             — LLM model name (default: provider default)
  COGNEE_BASE_URL       — Cognee server URL (if using server mode, legacy)
  COGNEE_API_KEY        — Cognee server API key (if using server mode)
  COGNEE_SERVICE_URL    — Cognee Cloud/remote instance URL (V2 serve() mode)

Or via $HERMES_HOME/cognee.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Circuit breaker settings
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/cognee.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        # Existing V1 keys (backward compatible)
        "llm_api_key": os.environ.get("LLM_API_KEY", ""),
        "llm_model": os.environ.get("LLM_MODEL", ""),
        "base_url": os.environ.get("COGNEE_BASE_URL", ""),
        "api_key": os.environ.get("COGNEE_API_KEY", ""),
        "dataset": os.environ.get("COGNEE_DATASET", "hermes"),
        "search_type": "GRAPH_COMPLETION",
        "top_k": 5,
        "auto_cognify": True,  # kept for backward compat, ignored by V2
        # V2 keys
        "auto_route": True,
        "session_prefix": "hermes",
        "improve_on_end": True,
        "serve_url": os.environ.get("COGNEE_SERVICE_URL", ""),
    }

    config_path = get_hermes_home() / "cognee.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Async helper — Cognee's API is fully async
# ---------------------------------------------------------------------------

class _AsyncBridge:
    """Run async cognee calls from sync code via a dedicated event loop thread."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _ensure_loop(self):
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="cognee-event-loop",
            )
            self._thread.start()

    def run(self, coro, timeout: float = 60):
        """Submit a coroutine to the background loop and wait for the result."""
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def run_nowait(self, coro):
        """Submit a coroutine without waiting for the result."""
        self._ensure_loop()
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self):
        with self._lock:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            self._loop = None
            self._thread = None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "cognee_recall",
    "description": (
        "Search the Cognee knowledge graph and session memory for relevant "
        "information. Auto-routing picks the best search strategy unless you "
        "specify one. Session memory is searched first, falling through to "
        "the permanent knowledge graph."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search for.",
            },
            "search_type": {
                "type": "string",
                "description": (
                    "Override auto-routing with a specific strategy. Options: "
                    "GRAPH_COMPLETION (LLM-reasoned from graph), "
                    "GRAPH_COMPLETION_COT (chain-of-thought reasoning), "
                    "GRAPH_COMPLETION_CONTEXT_EXTENSION (extended context), "
                    "GRAPH_SUMMARY_COMPLETION (summaries + graph), "
                    "RAG_COMPLETION (traditional RAG), CHUNKS (raw text), "
                    "CHUNKS_LEXICAL (keyword search), SUMMARIES, TEMPORAL, "
                    "FEELING_LUCKY (auto-selects best)."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 5, max: 20).",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Where to search: 'auto' (session then graph, default), "
                    "'session' (session cache only), 'graph' (permanent graph only)."
                ),
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "cognee_remember",
    "description": (
        "Permanently store data in the Cognee knowledge graph. Runs the full "
        "add + cognify + improve pipeline. Use for important facts, decisions, "
        "or content the user wants to persist across sessions. "
        "NOT needed for routine conversation logging (that is automatic)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content to permanently store.",
            },
            "dataset": {
                "type": "string",
                "description": "Dataset name (default: hermes).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "cognee_forget",
    "description": (
        "Delete data from the Cognee knowledge graph. Can target a specific "
        "dataset or delete all data. Use when the user asks to remove or "
        "clear memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Dataset name to delete (deletes entire dataset).",
            },
            "everything": {
                "type": "boolean",
                "description": "If true, delete ALL data across all datasets.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# SearchType resolution
# ---------------------------------------------------------------------------

_SEARCH_TYPE_MAP = None


def _get_search_type_map() -> dict:
    """Lazy-load the SearchType enum mapping."""
    global _SEARCH_TYPE_MAP
    if _SEARCH_TYPE_MAP is not None:
        return _SEARCH_TYPE_MAP
    try:
        from cognee.modules.search.types import SearchType
        _SEARCH_TYPE_MAP = {member.name: member for member in SearchType}
    except Exception:
        _SEARCH_TYPE_MAP = {}
    return _SEARCH_TYPE_MAP


def _resolve_search_type(search_type_str: str):
    """Resolve a search type string to the cognee SearchType enum."""
    mapping = _get_search_type_map()
    key = search_type_str.upper().strip()
    if key in mapping:
        return mapping[key]
    # Fallback to GRAPH_COMPLETION
    from cognee.modules.search.types import SearchType
    return SearchType.GRAPH_COMPLETION


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class CogneeMemoryProvider(MemoryProvider):
    """Cognee V2 knowledge graph memory with session-aware storage and auto-routing."""

    _IDENTITY_EMAIL = "hermes-agent@cognee.local"
    _IDENTITY_PASSWORD = "hermes-agent-plugin"

    def __init__(self):
        self._config: dict = {}
        self._bridge = _AsyncBridge()
        self._initialized = False
        self._session_id = ""
        self._session_cognee_id = ""
        self._dataset = "hermes"
        self._top_k = 5
        self._auto_route = True
        self._improve_on_end = True
        self._serve_url = ""
        self._remote_mode = False
        self._user = None  # Cognee User object for this integration
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "cognee"

    def is_available(self) -> bool:
        cfg = _load_config()
        has_local = bool(cfg.get("llm_api_key"))
        has_server = bool(cfg.get("base_url")) and bool(cfg.get("api_key"))
        has_serve = bool(cfg.get("serve_url"))
        return has_local or has_server or has_serve

    def get_config_schema(self):
        return [
            {
                "key": "llm_api_key",
                "description": "LLM API key (for local mode — e.g. OpenAI key)",
                "secret": True,
                "required": False,
                "env_var": "LLM_API_KEY",
            },
            {
                "key": "base_url",
                "description": "Cognee server URL (for server mode, legacy)",
                "required": False,
                "env_var": "COGNEE_BASE_URL",
            },
            {
                "key": "api_key",
                "description": "Cognee server API key (for server mode)",
                "secret": True,
                "required": False,
                "env_var": "COGNEE_API_KEY",
            },
            {
                "key": "serve_url",
                "description": "Cognee Cloud/remote instance URL (V2 mode)",
                "required": False,
                "env_var": "COGNEE_SERVICE_URL",
            },
            {
                "key": "llm_model",
                "description": "LLM model name",
                "required": False,
                "env_var": "LLM_MODEL",
            },
            {
                "key": "dataset",
                "description": "Default dataset name",
                "default": "hermes",
            },
            {
                "key": "auto_route",
                "description": "Auto-select search strategy in recall (true/false)",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "improve_on_end",
                "description": "Bridge session to permanent graph on session end",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/cognee.json."""
        config_path = Path(hermes_home) / "cognee.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -- Circuit breaker ----------------------------------------------------

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
                "Cognee circuit breaker tripped after %d consecutive failures. "
                "Pausing calls for %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    # -- Identity & configuration -------------------------------------------

    async def _ensure_identity(self):
        """Create or retrieve the hermes-agent identity in Cognee."""
        from cognee.modules.users.methods import create_user, get_user_by_email

        user = await get_user_by_email(self._IDENTITY_EMAIL)
        if user:
            return user

        try:
            user = await create_user(
                email=self._IDENTITY_EMAIL,
                password=self._IDENTITY_PASSWORD,
                is_verified=True,
                is_active=True,
            )
            logger.info("Cognee identity created: %s (id=%s)", self._IDENTITY_EMAIL, user.id)
            return user
        except Exception:
            # Race condition or other error — try fetch again
            user = await get_user_by_email(self._IDENTITY_EMAIL)
            if user:
                return user
            # Last resort: default user
            from cognee.modules.users.methods import get_default_user
            return await get_default_user()

    def _configure_cognee(self):
        """Apply configuration to the cognee library (local mode)."""
        try:
            import cognee

            if self._config.get("llm_api_key"):
                cognee.config.set_llm_api_key(self._config["llm_api_key"])
            if self._config.get("llm_model"):
                cognee.config.set_llm_model(self._config["llm_model"])
        except Exception as e:
            logger.debug("Cognee configuration failed: %s", e)

    # -- Lifecycle methods --------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._session_id = session_id
        self._dataset = self._config.get("dataset", "hermes")
        self._top_k = int(self._config.get("top_k", 5))

        prefix = self._config.get("session_prefix", "hermes")
        self._session_cognee_id = f"{prefix}_{session_id}"

        auto_route = self._config.get("auto_route", True)
        self._auto_route = auto_route if isinstance(auto_route, bool) else str(auto_route).lower() == "true"

        improve = self._config.get("improve_on_end", True)
        self._improve_on_end = improve if isinstance(improve, bool) else str(improve).lower() == "true"

        self._serve_url = self._config.get("serve_url", "")

        # Remote/cloud mode via V2 serve()
        if self._serve_url:
            try:
                api_key = self._config.get("api_key", "")
                self._bridge.run(self._do_serve(self._serve_url, api_key))
                self._remote_mode = True
                logger.info("Cognee connected to remote instance: %s", self._serve_url)
            except Exception as e:
                logger.warning("Cognee remote connection failed, falling back to local: %s", e)
                self._remote_mode = False

        # Configure cognee library if in local mode (no remote, no legacy server)
        if not self._remote_mode and not self._config.get("base_url"):
            if self._config.get("llm_api_key"):
                self._configure_cognee()

        # Create integration identity (hermes-agent@cognee.local)
        try:
            self._user = self._bridge.run(self._ensure_identity())
            logger.info("Cognee identity: %s (id=%s)", self._IDENTITY_EMAIL, self._user.id)
        except Exception as e:
            logger.warning("Cognee identity creation failed, using default: %s", e)
            self._user = None

        self._initialized = True

    def system_prompt_block(self) -> str:
        if self._remote_mode:
            mode = "remote"
        elif self._config.get("base_url"):
            mode = "server"
        else:
            mode = "local"
        return (
            "# Cognee Knowledge Graph Memory\n"
            f"Active ({mode} mode). Dataset: {self._dataset}.\n"
            "Use cognee_recall to query the knowledge graph and session memory "
            "(auto-routes to best search strategy), cognee_remember to permanently "
            "store important content, cognee_forget to delete memories."
        )

    # -- Async helpers (V2 API) ---------------------------------------------

    async def _do_serve(self, url: str, api_key: str):
        """Connect to a remote Cognee instance via V2 serve()."""
        import cognee
        kwargs = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        await cognee.serve(**kwargs)

    async def _do_disconnect(self):
        """Disconnect from remote Cognee instance."""
        import cognee
        await cognee.disconnect()

    async def _do_recall(self, query: str, search_type_str: Optional[str],
                         top_k: int, scope: str) -> list:
        """Search via V2 recall() with session awareness and auto-routing."""
        import cognee

        kwargs: Dict[str, Any] = {"top_k": top_k, "auto_route": self._auto_route}
        if self._user is not None:
            kwargs["user"] = self._user

        if scope == "session":
            # Session-only: pass session_id, no datasets or query_type
            kwargs["session_id"] = self._session_cognee_id
        elif scope == "graph":
            # Graph-only: pass datasets, optional query_type
            kwargs["datasets"] = [self._dataset]
            if search_type_str:
                kwargs["query_type"] = _resolve_search_type(search_type_str)
        else:
            # Auto: pass session_id (recall searches session first, falls through to graph)
            kwargs["session_id"] = self._session_cognee_id

        # Explicit search_type override (except for session-only scope)
        if search_type_str and scope != "session":
            kwargs["query_type"] = _resolve_search_type(search_type_str)

        return await cognee.recall(query_text=query, **kwargs)

    async def _do_remember_session(self, content: str):
        """Store content in session cache only (fast, no graph pipeline)."""
        import cognee
        kwargs: Dict[str, Any] = {
            "data": content,
            "dataset_name": self._dataset,
            "session_id": self._session_cognee_id,
        }
        if self._user is not None:
            kwargs["user"] = self._user
        return await cognee.remember(**kwargs)

    async def _do_remember_permanent(self, content: str, dataset: str):
        """Store content permanently via full add+cognify+improve pipeline."""
        import cognee
        kwargs: Dict[str, Any] = {
            "data": content,
            "dataset_name": dataset,
            "self_improvement": True,
            "session_ids": [self._session_cognee_id],
        }
        if self._user is not None:
            kwargs["user"] = self._user
        return await cognee.remember(**kwargs)

    async def _do_forget(self, dataset: Optional[str] = None,
                         everything: bool = False) -> dict:
        """Delete data via V2 forget()."""
        import cognee
        kwargs: Dict[str, Any] = {"everything": everything}
        if dataset and not everything:
            kwargs["dataset"] = dataset
        if self._user is not None:
            kwargs["user"] = self._user
        return await cognee.forget(**kwargs)

    async def _do_improve(self):
        """Bridge session data into permanent graph via V2 improve()."""
        import cognee
        kwargs: Dict[str, Any] = {
            "dataset": self._dataset,
            "session_ids": [self._session_cognee_id],
            "run_in_background": False,
        }
        if self._user is not None:
            kwargs["user"] = self._user
        return await cognee.improve(**kwargs)

    # -- Prefetch (background recall) ---------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Cognee Knowledge Graph\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                results = self._bridge.run(
                    self._do_recall(query, None, min(self._top_k, 5), "session")
                )
                if results:
                    lines = []
                    for r in results:
                        if isinstance(r, dict):
                            source = r.get("_source", "")
                            text = r.get("answer", r.get("text", r.get("content", str(r))))
                            prefix = f"[{source}] " if source else ""
                            lines.append(f"- {prefix}{text[:200]}")
                        else:
                            lines.append(f"- {str(r)[:200]}")
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines[:5])
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Cognee prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="cognee-prefetch"
        )
        self._prefetch_thread.start()

    # -- Turn sync (session cache write) ------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Add the conversation turn to Cognee's session cache (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                turn_text = f"User: {user_content}\nAssistant: {assistant_content}"
                self._bridge.run(self._do_remember_session(turn_text))
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Cognee session sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="cognee-sync"
        )
        self._sync_thread.start()

    # -- Tool schemas and dispatch ------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Cognee temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        if tool_name == "cognee_recall":
            return self._handle_recall(args)
        elif tool_name == "cognee_remember":
            return self._handle_remember(args)
        elif tool_name == "cognee_forget":
            return self._handle_forget(args)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_recall(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Missing required parameter: query"})
        search_type = args.get("search_type")
        top_k = min(int(args.get("top_k", self._top_k)), 20)
        scope = args.get("scope", "auto")

        try:
            results = self._bridge.run(self._do_recall(query, search_type, top_k, scope))
            self._record_success()
            if not results:
                return json.dumps({"result": "No relevant results found."})
            items = []
            for r in results:
                if isinstance(r, dict):
                    item = {
                        "text": r.get("answer", r.get("text", r.get("content", str(r)))),
                        "source": r.get("_source", "unknown"),
                    }
                    if r.get("score") is not None:
                        item["score"] = r["score"]
                    items.append(item)
                else:
                    items.append({"text": str(r), "source": "unknown"})
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            self._record_failure()
            return json.dumps({"error": f"Recall failed: {e}"})

    def _handle_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "Missing required parameter: content"})
        dataset = args.get("dataset", self._dataset)

        try:
            result = self._bridge.run(
                self._do_remember_permanent(content, dataset),
                timeout=120,  # permanent remember runs full pipeline
            )
            self._record_success()
            status = getattr(result, "status", "completed")
            elapsed = getattr(result, "elapsed_seconds", None)
            msg = f"Content stored permanently in knowledge graph (status: {status})."
            if elapsed is not None:
                msg += f" Took {elapsed:.1f}s."
            return json.dumps({"result": msg, "status": status})
        except Exception as e:
            self._record_failure()
            return json.dumps({"error": f"Remember failed: {e}"})

    def _handle_forget(self, args: dict) -> str:
        dataset = args.get("dataset")
        everything = args.get("everything", False)

        if not dataset and not everything:
            return json.dumps({"error": "Specify 'dataset' name or set 'everything' to true."})

        try:
            result = self._bridge.run(self._do_forget(dataset, everything))
            self._record_success()
            return json.dumps({"result": "Data deleted.", "details": result})
        except Exception as e:
            self._record_failure()
            return json.dumps({"error": f"Forget failed: {e}"})

    # -- Session lifecycle hooks --------------------------------------------

    def on_session_end(self, messages: List[Dict]) -> None:
        """Bridge session data into permanent graph via improve()."""
        # Wait for any pending sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        if not self._improve_on_end:
            return

        try:
            self._bridge.run(self._do_improve(), timeout=120)
            self._record_success()
            logger.info(
                "Cognee session-to-graph improve completed for session %s",
                self._session_cognee_id,
            )
        except Exception as e:
            self._record_failure()
            logger.warning("Cognee improve on session end failed: %s", e)

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        # Disconnect from remote if connected
        if self._remote_mode:
            try:
                self._bridge.run(self._do_disconnect(), timeout=5)
            except Exception:
                pass
        self._bridge.shutdown()


def register(ctx) -> None:
    """Register Cognee as a memory provider plugin."""
    ctx.register_memory_provider(CogneeMemoryProvider())
