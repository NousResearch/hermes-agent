"""Mem0 memory plugin — MemoryProvider interface.

Server-side fact extraction and semantic search via the Mem0 Platform API (cloud), a
self-hosted Mem0 server (MEM0_HOST, HTTP), or OSS Memory. Secrets live in $HERMES_HOME/.env
(MEM0_API_KEY, MEM0_HOST); settings in $HERMES_HOME/mem0.json via `hermes memory setup`:
mode ("platform"|"oss"), host, user_id (canonical id across gateways; unset → gateway-native
id), agent_id. MEM0_* env vars remain a fallback.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after _BREAKER_THRESHOLD consecutive failures, pause API
# calls for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS, _PREFETCH_WAIT_SECS = 5, 120, 3
_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")
# Placeholder user_id. initialize() treats it as "no operator-configured user_id"
# so legacy mem0.json files written by the wizard don't override gateway-native ids.
_DEFAULT_USER_ID = "hermes-user"


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    err_str = str(exc).lower()
    return type(exc).__name__ in _CLIENT_ERROR_TYPES or any(s in err_str for s in ("404", "not found", "valid uuid"))


def _read_mem0_json(config_path: Path) -> dict:
    """Best-effort read of mem0.json; missing/corrupt file -> {}."""
    if config_path.exists():
        with suppress(Exception):
            return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _load_config() -> dict:
    """Env vars provide defaults; $HERMES_HOME/mem0.json overrides individual keys.
    Layering avoids a silent failure when the JSON file exists but lacks fields
    like ``api_key`` that the user set in ``.env``."""
    from hermes_constants import get_hermes_home
    config = {"mode": os.environ.get("MEM0_MODE", "platform"), "api_key": get_secret("MEM0_API_KEY", ""), "host": os.environ.get("MEM0_HOST", ""), "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"), "oss": {}}
    if os.environ.get("MEM0_USER_ID"):  # only when explicitly configured, so initialize() can fall back to the gateway-native id
        config["user_id"] = os.environ["MEM0_USER_ID"]
    file_cfg = _read_mem0_json(get_hermes_home() / "mem0.json")
    config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
    return config


def _schema(name: str, description: str, properties: dict[str, tuple[str, str]], required: list[str]) -> dict:
    props = {k: {"type": t, "description": d} for k, (t, d) in properties.items()}
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS = [
    _schema("mem0_search", "Search the user's memories by meaning; returns facts ranked by relevance. Use this before answering any question that may depend on what you know about the user (preferences, facts, history, people, projects, past decisions). For multi-part or multi-hop questions, call it several times — vary the wording and run follow-up searches on what earlier results reveal; one search is rarely enough.",
            {"query": ("string", "What to search for."), "top_k": ("integer", "Max results (default: 10, max: 50)."), "rerank": ("boolean", "Rerank results for relevance (default: false, platform mode only).")}, ["query"]),
    _schema("mem0_add", "Store a durable fact about the user, verbatim (no LLM extraction). Call this the moment the user states a lasting preference, correction, decision, or personal detail worth recalling on future turns — don't wait to be asked to remember. Skip transient chit-chat and facts you've already stored.",
            {"content": ("string", "The fact to store.")}, ["content"]),
    _schema("mem0_update", "Replace the text of an existing memory by its ID (take the ID from a mem0_search result). Use when a stored fact has changed or was wrong — correct it in place instead of adding a duplicate.",
            {"memory_id": ("string", "Memory UUID to update."), "text": ("string", "New text content.")}, ["memory_id", "text"]),
    _schema("mem0_delete", "Delete a memory by its ID (take the ID from a mem0_search result). Use when a stored fact is obsolete or the user asks you to forget it; prefer mem0_update if the fact merely changed.",
            {"memory_id": ("string", "Memory UUID to delete.")}, ["memory_id"]),
]

_PROMPT_BODY = (
    "You have persistent memory of this user from past conversations. You should call mem0_search before answering anything that could depend on prior context (the user's preferences, facts, history, people, projects, or earlier decisions) — do not rely on the chat window alone, and do not assume you have no memory.\n"
    "For multi-part or multi-hop questions, run several searches with different wording/angles and follow-up searches on what the first results surface; one search is rarely enough. Keep searching until you have every fact the question needs before you answer.\n"
    "Tools: mem0_search to find memories, mem0_add to store facts, mem0_update and mem0_delete to manage by ID."
)


class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search (platform, self-hosted or OSS)."""

    def __init__(self):
        self._config = self._backend = self._sync_thread = self._prefetch_thread = None
        self._mode, self._api_key, self._host, self._user_id, self._agent_id = "platform", "", "", _DEFAULT_USER_ID, "hermes"
        self._rerank_default, self._channel = False, "cli"  # channel = gateway name (cli/telegram/discord/...)
        self._prefetch_query = self._prefetch_result = ""
        self._prefetch_done = self._atexit_registered = False
        self._consecutive_failures, self._breaker_open_until = 0, 0.0  # circuit breaker state
        self._breaker_lock, self._sync_lock, self._prefetch_lock = threading.Lock(), threading.Lock(), threading.Lock()

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        if cfg.get("mode", "platform") == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        return bool(cfg.get("api_key") or cfg.get("host"))  # platform needs a key; self-hosted a host (key optional with AUTH_DISABLED)

    def save_config(self, values, hermes_home):
        """Merge-write config to $HERMES_HOME/mem0.json."""
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "mem0.json"
        atomic_json_write(config_path, {**_read_mem0_json(config_path), **values}, mode=0o600)

    def get_config_schema(self):
        api_key_required = _load_config().get("mode", "platform") != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _oss_hint(self, template: str, default: str = "vector store") -> str:
        """OSS-only hint; ``{vs}`` is the configured vector-store provider. "" in other modes."""
        return template.format(vs=self._config.get("oss", {}).get("vector_store", {}).get("provider", default)) if self._mode == "oss" else ""

    def _create_backend(self):
        # Lazy-install the mem0 SDK before the backend imports it (honors security.allow_lazy_installs);
        # on failure the backend import raises the canonical error, captured below.
        with suppress(Exception):
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        try:
            from . import _backend
            if self._mode == "oss":
                return _backend.OSSBackend(self._config.get("oss", {}))
            return _backend.SelfHostedBackend(self._api_key, self._host) if self._host else _backend.PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """True while the breaker is tripped; an expired cooldown resets the failure count."""
        with self._breaker_lock:
            if self._consecutive_failures >= _BREAKER_THRESHOLD and time.monotonic() < self._breaker_open_until:
                return True
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._consecutive_failures = 0
            return False

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if any(s in str(exc).lower() for s in ("connection", "refused", "timeout")):
            msg += self._oss_hint(" (check that {vs} is running)")
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures = count = self._consecutive_failures + 1
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if count >= _BREAKER_THRESHOLD:
            hint = self._oss_hint(" Check that your {vs} vector store is running and reachable.", "unknown")
            logger.warning("Mem0 circuit breaker tripped after %d consecutive failures. Pausing API calls for %ds.%s", count, _BREAKER_COOLDOWN_SECS, hint)

    def _try(self, call, log, msg: str):
        """Background-path wrapper: run ``call`` under the breaker; on error log ``msg`` and return None."""
        try:
            result = call()
        except Exception as e:
            self._record_failure()
            log(msg, e)
            return None
        self._record_success()
        return result

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = cfg = _load_config()
        self._mode, self._api_key, self._host, self._agent_id = cfg.get("mode", "platform"), cfg.get("api_key", ""), cfg.get("host", ""), cfg.get("agent_id", "hermes")
        # user_id precedence: operator-configured (env/mem0.json) > gateway-native id (kwargs) > _DEFAULT_USER_ID.
        # The literal placeholder counts as unset so wizard users still get gateway-native ids.
        configured = cfg.get("user_id")
        self._user_id = (None if configured == _DEFAULT_USER_ID else configured) or kwargs.get("user_id") or _DEFAULT_USER_ID
        # Persisted rerank preference: default for mem0_search when the model omits ``rerank``. Platform-only.
        _rr = cfg.get("rerank", False)
        self._rerank_default = _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        self._channel = kwargs.get("platform") or "cli"
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _search(self, query: str, top_k: int = 10, rerank: bool = False, backend=None) -> list:
        # Scoped to user_id only — by design — so recall surfaces memories from any gateway/agent under this
        # principal; writes attach agent_id and metadata.channel so narrower views remain possible at query time.
        return (backend or self._backend).search(query, filters={"user_id": self._user_id}, top_k=top_k, rerank=rerank)

    def _add(self, messages: list, infer: bool):
        metadata = {"channel": self._channel} if self._channel else {}
        return self._backend.add(messages, user_id=self._user_id, agent_id=self._agent_id, infer=infer, metadata=metadata)

    def system_prompt_block(self) -> str:
        # Mirror _create_backend precedence (oss > host > platform). Rerank is a Mem0 Platform feature only.
        mode_label = "OSS (self-hosted)" if self._mode == "oss" else "self-hosted (HTTP API)" if self._host else "platform (cloud API)"
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return f"# Mem0 Memory\nActive. Mode: {mode_label}. User: {self._user_id}.\n{_PROMPT_BODY}{rerank_note}"

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        """Pop the finished prefetch body for ``query`` (None if absent or still running)."""
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result, self._prefetch_result, self._prefetch_done = self._prefetch_result, "", False
            return result

    def _start_prefetch(self, query: str) -> None:
        backend = self._backend
        if not query or backend is None or self._is_breaker_open():
            return

        def _run():
            results = self._try(lambda: self._search(query, backend=backend), logger.debug, "Mem0 prefetch failed: %s")
            lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
            body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines) if lines else ""
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result, self._prefetch_done = body, True

        with self._prefetch_lock:
            # Same query already answered or still in flight: don't restart it.
            if self._prefetch_query == query and (self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive())):
                return
            self._prefetch_query, self._prefetch_result, self._prefetch_done = query, "", False
            self._prefetch_thread = t = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        if (cached := self._consume_prefetch_result(query)) is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        return self._consume_prefetch_result(query) or ""  # slow backend: skip injection; mem0_search remains the backstop

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        def _sync():
            if self._backend is not None:
                messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                self._try(lambda: self._add(messages, infer=True), logger.warning, "Mem0 sync failed: %s")

        with self._sync_lock:
            prev = self._sync_thread
            if prev and prev.is_alive():
                prev.join(timeout=5.0)
                if prev.is_alive():  # still busy after the wait: skip to avoid duplicate ingestion
                    return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    # -- tool handlers: (required params, error label, body, client-error policy) ---
    # Client errors (bad ID / not found) never trip the breaker, except for mem0_add
    # where they count as failures; update/delete answer them with "Memory not found".

    def _tool_search(self, args: dict) -> str:
        top_k = max(1, min(int(args.get("top_k", 10)), 50))
        rerank_raw = args.get("rerank", self._rerank_default)
        rerank = rerank_raw.lower() not in ("false", "0", "no") if isinstance(rerank_raw, str) else bool(rerank_raw)
        results = self._search(args["query"], top_k, rerank)
        if not results:
            return json.dumps({"result": "No relevant memories found."})
        items = [{"id": r.get("id"), "memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
        return json.dumps({"results": items, "count": len(items)})

    def _tool_add(self, args: dict) -> str:
        result = self._add([{"role": "user", "content": args["content"]}], infer=False)
        event_id = result.get("event_id") if isinstance(result, dict) else None
        # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
        msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
        return json.dumps({"result": msg, "event_id": event_id})

    _TOOL_HANDLERS = {
        "mem0_search": (("query",), "Search failed", _tool_search, "skip"),
        "mem0_add": (("content",), "Failed to store", _tool_add, "count"),
        "mem0_update": (("memory_id", "text"), "Update failed", lambda self, a: json.dumps(self._backend.update(a["memory_id"], a["text"])), "not_found"),
        "mem0_delete": (("memory_id",), "Delete failed", lambda self, a: json.dumps(self._backend.delete(a["memory_id"])), "not_found"),
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{self._oss_hint(' Check that {vs} is running and reachable.')}"})
        if self._is_breaker_open():
            return json.dumps({"error": f"Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically.{self._oss_hint(' Check that your {vs} is running.')}"})
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        required, label, body, on_client_error = self._TOOL_HANDLERS[tool_name]
        if missing := next((k for k in required if not args.get(k, "")), None):
            return tool_error(f"Missing required parameter: {missing}")
        try:
            result = body(self, args)
        except Exception as e:
            client = _is_client_error(e)
            if client and on_client_error == "not_found":
                return tool_error(f"Memory not found: {args['memory_id']}")
            if not client or on_client_error == "count":
                self._record_failure()
            return tool_error(self._format_error(label, e))
        self._record_success()
        return result

    def _shutdown_backend(self):
        with suppress(Exception):
            if self._backend:
                self._backend.close()
                self._backend = None

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}
# ---- END PLUGIN-COMPAT ----
