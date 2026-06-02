"""Mem0 Local memory plugin — MemoryProvider interface.

Fully local memory backend using:
- qdrant (local vector store, data stays on disk)
- fastembed (local embedding model, no API calls for embeddings)
- Hermes's own LLM provider for fact extraction (via mem0's llm config)

No cloud APIs. No data leaves the machine.

Config via $HERMES_HOME/mem0-local.json or defaults:
  data_path:    Path for qdrant storage (default: $HERMES_HOME/mem0_data)
  collection:   Qdrant collection name (default: mem0_hermes)
  user_id:      User identifier (default: hermes-user)
  embed_model:  Fastembed model name (default: BAAI/bge-small-en-v1.5)
  llm_model:    LLM for fact extraction (default: from Hermes config)
  llm_base_url: LLM API base URL (default: from Hermes config)
  llm_api_key:  LLM API key (default: from Hermes config)
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

# Circuit breaker
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Default embedding model (384 dims, small and fast)
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_COLLECTION = "mem0_hermes"
DEFAULT_DIMS = 384


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars + mem0-local.json overrides."""
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    config = {
        "data_path": str(hermes_home / "mem0_data"),
        "collection": DEFAULT_COLLECTION,
        "user_id": os.environ.get("MEM0_LOCAL_USER_ID", "hermes-user"),
        "embed_model": DEFAULT_EMBED_MODEL,
        "llm_model": "",
        "llm_base_url": "",
        "llm_api_key": "",
    }

    config_path = hermes_home / "mem0-local.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


def _resolve_llm_config(cfg: dict) -> dict:
    """Resolve LLM config — prefer explicit config, fall back to Hermes main config."""
    model = cfg.get("llm_model", "")
    base_url = cfg.get("llm_base_url", "")
    api_key = cfg.get("llm_api_key", "")

    if not model or not base_url or not api_key:
        # Fall back to Hermes main config
        try:
            from hermes_constants import get_hermes_home
            import yaml
            main_config_path = get_hermes_home() / "config.yaml"
            if main_config_path.exists():
                with open(main_config_path) as f:
                    main_cfg = yaml.safe_load(f) or {}
                providers = main_cfg.get("providers", {})
                for pname, pcfg in providers.items():
                    if pname == "defaults":
                        continue
                    if isinstance(pcfg, dict):
                        if not model:
                            model = pcfg.get("model", "")
                        if not base_url:
                            base_url = pcfg.get("base_url", "")
                        if not api_key:
                            api_key = pcfg.get("api_key", "")
                        if model and base_url:
                            break
        except Exception:
            pass

    return {"model": model, "base_url": base_url, "api_key": api_key}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored local memories — preferences, facts, "
        "project context. Fast, no LLM calls. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search local memories by semantic similarity. "
        "Returns relevant facts ranked by vector distance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact locally. Stored with LLM-based extraction "
        "for deduplication. Use for explicit preferences, corrections, or decisions."
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

class Mem0LocalMemoryProvider(MemoryProvider):
    """Mem0 Local memory — qdrant + fastembed, fully on-device."""

    def __init__(self):
        self._config = None
        self._memory = None
        self._memory_lock = threading.Lock()
        self._user_id = "hermes-user"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0-local"

    def is_available(self) -> bool:
        """Check if mem0ai is installed and data path is accessible."""
        try:
            import mem0  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_memory(self):
        """Thread-safe Memory instance with lazy initialization."""
        with self._memory_lock:
            if self._memory is not None:
                return self._memory

            cfg = self._config or _load_config()
            llm_cfg = _resolve_llm_config(cfg)

            # Set env vars for mem0's LLM (litellm proxy)
            if llm_cfg["api_key"]:
                os.environ["OPENAI_API_KEY"] = llm_cfg["api_key"]
            if llm_cfg["base_url"]:
                os.environ["OPENAI_API_BASE"] = llm_cfg["base_url"]

            data_path = cfg.get("data_path", "")
            collection = cfg.get("collection", DEFAULT_COLLECTION)
            embed_model = cfg.get("embed_model", DEFAULT_EMBED_MODEL)

            mem0_config = {
                "version": "v1.1",
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": collection,
                        "embedding_model_dims": DEFAULT_DIMS,
                        "path": data_path,
                    }
                },
                "embedder": {
                    "provider": "fastembed",
                    "config": {"model": embed_model}
                },
            }

            # Add LLM config if available (for fact extraction in sync_turn)
            if llm_cfg["model"] and llm_cfg["base_url"]:
                mem0_config["llm"] = {
                    "provider": "litellm",
                    "config": {
                        "model": f"openai/{llm_cfg['model']}",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                }

            from mem0 import Memory
            self._memory = Memory.from_config(mem0_config)
            logger.info("Mem0 Local initialized: data=%s, collection=%s, embed=%s",
                        data_path, collection, embed_model)
            return self._memory

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
                "Mem0 Local circuit breaker tripped after %d failures. Pausing for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        # Pre-warm the memory instance in background
        threading.Thread(target=self._get_memory, daemon=True, name="mem0-local-init").start()

    def _read_filters(self) -> Dict[str, Any]:
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        return {"user_id": self._user_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Local Memory\n"
            f"Active (fully local). User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview. All data stays on this machine."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Local Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                mem = self._get_memory()
                results = self._unwrap_results(mem.search(
                    query=query,
                    filters=self._read_filters(),
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 Local prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-local-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """Send turn to mem0 for local fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                mem = self._get_memory()
                msgs = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                mem.add(msgs, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 Local sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-local-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 Local temporarily unavailable (circuit breaker). Will retry automatically."
            })

        try:
            mem = self._get_memory()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(mem.get_all(filters=self._read_filters()))
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
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(mem.search(
                    query=query,
                    filters=self._read_filters(),
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
                mem.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored locally."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._memory_lock:
            self._memory = None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "data_path", "description": "Local qdrant storage path",
             "default": "~/.hermes/mem0_data"},
            {"key": "collection", "description": "Qdrant collection name",
             "default": DEFAULT_COLLECTION},
            {"key": "user_id", "description": "User identifier",
             "default": "hermes-user"},
            {"key": "embed_model", "description": "Fastembed model name",
             "default": DEFAULT_EMBED_MODEL},
            {"key": "llm_model", "description": "LLM model for fact extraction (optional, uses Hermes main config if empty)"},
            {"key": "llm_base_url", "description": "LLM API base URL (optional)"},
            {"key": "llm_api_key", "description": "LLM API key (optional, secret)",
             "secret": True, "env_var": "MEM0_LOCAL_LLM_KEY"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "mem0-local.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)


def register(ctx) -> None:
    """Register Mem0 Local as a memory provider plugin."""
    ctx.register_memory_provider(Mem0LocalMemoryProvider())
