"""Mem0 OSS memory plugin — MemoryProvider interface.

Self-hosted memory with configurable LLM, embedder, and vector store
using the open-source mem0 Memory class.

Config via $HERMES_HOME/mem0_oss.json.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Provider defaults for LLM, embedder, vector store
_DEFAULT_LLM = {"provider": "openai", "config": {"model": "gpt-5.4", "temperature": 0.1}}
_DEFAULT_EMBEDDER = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
_DEFAULT_VECTOR_STORE = {"provider": "qdrant", "config": {"path": "/tmp/qdrant", "collection_name": "mem0"}}

# Custom extraction prompt — tells the LLM what to keep and what to skip.
# Without this, the default prompt stores casual chat ("Doing fine") as memories.
_FACT_EXTRACTION_PROMPT = """\
Extract only durable, meaningful facts from the conversation.

STORE: personal details, preferences, decisions, project context, technical choices,
corrections, names, roles, goals, constraints, deadlines.

SKIP: greetings, small talk, pleasantries, filler ("Doing fine", "sounds good",
"let me check"), meta-conversation about the chat itself, questions without answers,
acknowledgements, and anything that would be stale within minutes.

If nothing meaningful is present, return {{"facts": []}}.
"""

# Providers that do NOT require an API key
_LOCAL_LLM_PROVIDERS = {"ollama"}
_LOCAL_EMBEDDER_PROVIDERS = {"ollama"}

# LLM provider -> env var for API key
_LLM_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
}

_EXTRA_DEPS = {
    "ollama": ("ollama", "ollama"),
    "pgvector": ("psycopg2", "psycopg2-binary"),
    "milvus": ("pymilvus", "pymilvus"),
}

# Known embedding dimensions — used to auto-configure the vector store.
# Without this, qdrant defaults to 1536 (OpenAI) and mismatches with
# other embedders, causing shape errors at runtime.
_EMBEDDER_DIMS = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "ollama": {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    },
}


def _check_dependencies(config: dict) -> list[str]:
    """Return list of missing pip packages required by the configured providers."""
    missing = []
    # Check mem0ai itself
    try:
        import mem0  # noqa: F401
    except ImportError:
        missing.append("mem0ai")
        return missing  # No point checking further

    providers_used = set()
    for section in ("llm", "embedder", "vector_store"):
        prov = config.get(section, {}).get("provider", "")
        if prov:
            providers_used.add(prov)

    for prov in providers_used:
        if prov in _EXTRA_DEPS:
            import_name, pip_name = _EXTRA_DEPS[prov]
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pip_name)

    return missing


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from $HERMES_HOME/mem0_oss.json with defaults."""
    from hermes_constants import get_hermes_home

    config = {
        "llm": copy.deepcopy(_DEFAULT_LLM),
        "embedder": copy.deepcopy(_DEFAULT_EMBEDDER),
        "vector_store": copy.deepcopy(_DEFAULT_VECTOR_STORE),
        "user_id": "hermes-user",
        "agent_id": "hermes",
    }

    config_path = get_hermes_home() / "mem0_oss.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("llm", "embedder", "vector_store"):
                if key in file_cfg and isinstance(file_cfg[key], dict):
                    config[key] = file_cfg[key]
            for key in ("user_id", "agent_id"):
                if file_cfg.get(key):
                    config[key] = file_cfg[key]
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_oss_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_oss_search",
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
    "name": "mem0_oss_conclude",
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

class Mem0OSSMemoryProvider(MemoryProvider):
    """Mem0 OSS memory — self-hosted with configurable LLM, embedder, vector store."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0_oss"

    def is_available(self) -> bool:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "mem0_oss.json"
        if not config_path.exists():
            return False
        cfg = _load_config()
        # Check that required Python packages are installed
        missing = _check_dependencies(cfg)
        if missing:
            logger.warning(
                "Mem0 OSS missing dependencies: %s. Run: pip install %s",
                ", ".join(missing), " ".join(missing),
            )
            return False
        llm_provider = cfg.get("llm", {}).get("provider", "openai")
        if llm_provider in _LOCAL_LLM_PROVIDERS:
            return True
        env_var = _LLM_ENV_VARS.get(llm_provider, "")
        return bool(env_var and os.environ.get(env_var))

    def get_config_schema(self):
        return [
            {"key": "llm_provider", "description": "LLM provider", "default": "openai",
             "choices": ["openai", "ollama"]},
            {"key": "llm_model", "description": "LLM model name", "default": "gpt-5.4",
             "default_from": {"field": "llm_provider", "map": {
                 "openai": "gpt-5.4", "ollama": "llama3.1:8b"}}},
            {"key": "llm_api_key", "description": "OpenAI API key", "secret": True,
             "env_var": "OPENAI_API_KEY", "url": "https://platform.openai.com/api-keys",
             "when": {"llm_provider": "openai"}},
            {"key": "ollama_base_url", "description": "Ollama server URL", "default": "http://localhost:11434",
             "when": {"llm_provider": "ollama"}},
            {"key": "embedder_provider", "description": "Embedder provider", "default": "openai",
             "choices": ["openai", "ollama"]},
            {"key": "embedder_model", "description": "Embedding model name", "default": "text-embedding-3-small",
             "default_from": {"field": "embedder_provider", "map": {
                 "openai": "text-embedding-3-small", "ollama": "nomic-embed-text"}}},
            {"key": "vector_store_provider", "description": "Vector store", "default": "qdrant",
             "choices": ["qdrant", "pgvector", "milvus"]},
            {"key": "vector_store_path", "description": "Local storage path", "default": "/tmp/qdrant",
             "when": {"vector_store_provider": "qdrant"}},
            {"key": "vector_store_connection_string", "description": "PostgreSQL connection string",
             "when": {"vector_store_provider": "pgvector"}},
            {"key": "vector_store_url", "description": "Milvus server URL", "default": "http://localhost:19530",
             "when": {"vector_store_provider": "milvus"}},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
        ]

    def get_extra_dependencies(self, values: dict) -> list[str]:
        """Return pip packages needed based on the user's provider choices."""
        missing = []
        for key in ("llm_provider", "embedder_provider", "vector_store_provider"):
            prov = values.get(key, "")
            if prov in _EXTRA_DEPS:
                import_name, pip_name = _EXTRA_DEPS[prov]
                try:
                    __import__(import_name)
                except ImportError:
                    if pip_name not in missing:
                        missing.append(pip_name)
        return missing

    def save_config(self, values, hermes_home):
        """Restructure flat wizard values into nested mem0_oss.json."""
        from pathlib import Path

        llm_config = {"model": values.get("llm_model", "gpt-5.4"), "temperature": 0.1}
        llm_provider = values.get("llm_provider", "openai")
        if llm_provider == "ollama":
            llm_config["ollama_base_url"] = values.get("ollama_base_url", "http://localhost:11434")

        embedder_config = {"model": values.get("embedder_model", "text-embedding-3-small")}
        embedder_provider = values.get("embedder_provider", "openai")
        if embedder_provider == "ollama":
            embedder_config["ollama_base_url"] = values.get("ollama_base_url", "http://localhost:11434")

        vs_provider = values.get("vector_store_provider", "qdrant")
        vs_config = {}
        if vs_provider == "qdrant":
            vs_config["path"] = values.get("vector_store_path", "/tmp/qdrant")
            vs_config["collection_name"] = "mem0"
        elif vs_provider == "pgvector":
            vs_config["connection_string"] = values.get("vector_store_connection_string", "")
            vs_config["collection_name"] = "mem0"
        elif vs_provider == "milvus":
            vs_config["url"] = values.get("vector_store_url", "http://localhost:19530")
            vs_config["collection_name"] = "mem0"

        # Auto-set embedding_model_dims so the vector store creates
        # collections with the correct size for the chosen embedder.
        # Without this, qdrant defaults to 1536 and mismatches with
        # non-OpenAI embedders (e.g. Ollama nomic-embed-text = 768).
        embedder_model = values.get("embedder_model", "")
        provider_dims = _EMBEDDER_DIMS.get(embedder_provider, {})
        dims = provider_dims.get(embedder_model)
        if dims:
            vs_config["embedding_model_dims"] = dims

        config = {
            "llm": {"provider": llm_provider, "config": llm_config},
            "embedder": {"provider": embedder_provider, "config": embedder_config},
            "vector_store": {"provider": vs_provider, "config": vs_config},
            "user_id": values.get("user_id", "hermes-user"),
            "agent_id": values.get("agent_id", "hermes"),
        }

        config_path = Path(hermes_home) / "mem0_oss.json"
        config_path.write_text(json.dumps(config, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._user_id = self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            # Pre-check dependencies so we get a clear error instead of
            # mem0's interactive input() prompt which hangs in Hermes.
            missing = _check_dependencies(self._config)
            if missing:
                raise RuntimeError(
                    f"Missing packages for Mem0 OSS: {', '.join(missing)}. "
                    f"Run: pip install {' '.join(missing)}"
                )
            try:
                from mem0 import Memory
                mem0_config = {
                    "llm": self._config["llm"],
                    "embedder": self._config["embedder"],
                    "vector_store": self._config["vector_store"],
                    "custom_fact_extraction_prompt": _FACT_EXTRACTION_PROMPT,
                }
                self._client = Memory.from_config(mem0_config)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

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
                "Mem0 OSS circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _read_kwargs(self) -> dict:
        """Kwargs for search/get_all — scoped to user only."""
        return {"user_id": self._user_id}

    def _write_kwargs(self) -> dict:
        """Kwargs for add — scoped to user + agent."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize mem0 response — dict wraps in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory (Self-Hosted)\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_oss_search to find memories, mem0_oss_conclude to store facts, "
            "mem0_oss_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory (Self-Hosted)\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    **self._read_kwargs(),
                    rerank=True,
                    limit=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 OSS prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-oss-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to mem0 for fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_kwargs(), infer=True)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 OSS sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-oss-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 OSS temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return json.dumps({"error": str(e)})

        if tool_name == "mem0_oss_profile":
            try:
                memories = self._unwrap_results(
                    client.get_all(**self._read_kwargs(), limit=100)
                )
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Failed to fetch profile: {e}"})

        elif tool_name == "mem0_oss_search":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            rerank = args.get("rerank", False)
            limit = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    **self._read_kwargs(),
                    rerank=rerank,
                    limit=limit,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Search failed: {e}"})

        elif tool_name == "mem0_oss_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return json.dumps({"error": "Missing required parameter: conclusion"})
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_kwargs(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Failed to store: {e}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 OSS as a memory provider plugin."""
    ctx.register_memory_provider(Mem0OSSMemoryProvider())
