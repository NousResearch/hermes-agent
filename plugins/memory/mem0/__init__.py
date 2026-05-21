"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API. Also supports a
local shadow mode using the Mem0 Python library + local Qdrant storage.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
        # Local shadow defaults: candidate recall only, no automatic turn
        # extraction.  Durable authority writes can be mirrored via
        # on_memory_write() with infer=False.
        "shadow": True,
        "sync_turn": False,
        "local_path": "",
        "local_collection": "jarvis_local_mem0_shadow",
        "local_embedder_provider": "fastembed",
        "local_embedder_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "local_embedding_dims": 384,
        "local_llm_provider": "ollama",
        "local_llm_model": "qwen3.5:4b",
        "local_ollama_base_url": "http://localhost:11434",
        # Optional Qdrant server settings. When set, these replace embedded
        # Qdrant ``path`` storage so multiple Hermes processes can share the
        # same Mem0 vector store without local file locks.
        "local_qdrant_url": os.environ.get("MEM0_LOCAL_QDRANT_URL", ""),
        "local_qdrant_host": os.environ.get("MEM0_LOCAL_QDRANT_HOST", ""),
        "local_qdrant_port": os.environ.get("MEM0_LOCAL_QDRANT_PORT", ""),
        "local_qdrant_api_key": os.environ.get("MEM0_LOCAL_QDRANT_API_KEY", ""),
    }

    config_path = get_hermes_home() / "mem0.json"
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

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
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
    "name": "mem0_conclude",
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


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._mode = "platform"
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._shadow = True
        self._sync_turn_enabled = True
        self._hermes_home = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = str(cfg.get("mode") or "platform").lower()
        if mode == "local":
            try:
                import mem0  # noqa: F401
                return True
            except ImportError:
                return False
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
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
            {"key": "mode", "description": "Mem0 mode: platform or local", "default": "platform", "choices": ["platform", "local"]},
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
            {"key": "shadow", "description": "Treat Mem0 recall as candidate-only, not authority", "default": "true", "choices": ["true", "false"]},
        ]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _build_local_memory_config(self) -> dict:
        """Build Mem0's local ``Memory.from_config`` configuration."""
        from hermes_constants import get_hermes_home

        hermes_home = Path(self._hermes_home or get_hermes_home())
        base_path = self._config.get("local_path") or str(hermes_home / "local-mem0-shadow")
        base = Path(base_path).expanduser()
        dims = int(self._config.get("local_embedding_dims") or 384)
        ollama_base_url = self._config.get("local_ollama_base_url") or "http://localhost:11434"

        qdrant_config = {
            "collection_name": self._config.get("local_collection") or "jarvis_local_mem0_shadow",
            "embedding_model_dims": dims,
        }
        qdrant_url = str(self._config.get("local_qdrant_url") or "").strip()
        qdrant_host = str(self._config.get("local_qdrant_host") or "").strip()
        qdrant_port = self._config.get("local_qdrant_port")
        qdrant_api_key = str(self._config.get("local_qdrant_api_key") or "").strip()
        if qdrant_url:
            qdrant_config["url"] = qdrant_url
        elif qdrant_host and qdrant_port:
            qdrant_config["host"] = qdrant_host
            qdrant_config["port"] = int(qdrant_port)
        else:
            qdrant_config["path"] = str(base / "qdrant")
            qdrant_config["on_disk"] = True
        if qdrant_api_key:
            qdrant_config["api_key"] = qdrant_api_key

        return {
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
            "llm": {
                "provider": self._config.get("local_llm_provider") or "ollama",
                "config": {
                    "model": self._config.get("local_llm_model") or "qwen3.5:4b",
                    "ollama_base_url": ollama_base_url,
                    "temperature": 0.0,
                    "max_tokens": 1200,
                },
            },
            "embedder": {
                "provider": self._config.get("local_embedder_provider") or "fastembed",
                "config": {
                    "model": self._config.get("local_embedder_model") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "embedding_dims": dims,
                },
            },
            "history_db_path": str(base / "history.db"),
        }

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                if self._mode == "local":
                    from mem0 import Memory
                    self._client = Memory.from_config(self._build_local_memory_config())
                else:
                    from mem0 import MemoryClient
                    self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
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
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = str(self._config.get("mode") or "platform").lower()
        self._api_key = self._config.get("api_key", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._as_bool(self._config.get("rerank"), True)
        self._shadow = self._as_bool(self._config.get("shadow"), self._mode == "local")
        self._sync_turn_enabled = self._as_bool(self._config.get("sync_turn"), self._mode != "local")
        self._hermes_home = kwargs.get("hermes_home")

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        if self._shadow:
            return (
                "# Mem0 Candidate Recall\n"
                f"Active in {self._mode} shadow mode. User: {self._user_id}.\n"
                "Mem0 results are candidate recall only; verify against AGENTS, "
                "MEMORY, user profile, and skills before treating them as authority. "
                "Use mem0_search to find candidate memories, mem0_conclude to mirror "
                "explicit durable facts."
            )
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        if self._shadow:
            return f"## Mem0 Candidate Recall (non-authoritative; verify before use)\n{result}"
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=self._rerank,
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if not self._sync_turn_enabled:
            return
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Mirror explicit built-in memory writes into Mem0 as candidate recall.

        This is intentionally verbatim/infer=False so the authority memory layer
        remains the source of truth and Mem0 cannot independently invent or
        mutate durable facts.
        """
        if action == "remove" or not content:
            return
        if self._is_breaker_open():
            return
        try:
            client = self._get_client()
            meta = dict(metadata or {})
            meta.update({"authority_target": target, "memory_action": action})
            client.add(
                [{"role": "user", "content": content}],
                **self._write_filters(),
                infer=False,
                metadata=meta,
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.debug("Mem0 on_memory_write mirror failed: %s", e)

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
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
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
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
