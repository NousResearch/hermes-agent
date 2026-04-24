"""Mem0 OSS (self-hosted) memory plugin — MemoryProvider interface.

LLM-powered fact extraction, semantic vector search, and automatic
deduplication using the open-source ``mem0ai`` library — no cloud API key
required.  All data is stored locally on disk.

Backend choices:
  Vector store: Qdrant (local path, no server) — default
  LLM / Embedder: resolved from ``auxiliary.mem0_oss`` in config.yaml, then
                  from ``MEM0_OSS_*`` env vars, then auto-detected.

Primary config — config.yaml (auxiliary.mem0_oss):
  provider   — Hermes provider name: "auto", "aws_bedrock", "bedrock",
               "openai", "openrouter", "ollama", "anthropic", or "custom".
               "auto" follows the standard Hermes auxiliary resolution chain.
  model      — LLM model id (provider-specific slug).  Empty = provider default.
  base_url   — OpenAI-compatible endpoint (forces provider="custom").
  api_key    — API key for that endpoint.  Falls back to MEM0_OSS_API_KEY.

Secondary config — environment variables:
  MEM0_OSS_VECTOR_STORE_PATH   — on-disk path for Qdrant (default: $HERMES_HOME/mem0_oss/qdrant)
  MEM0_OSS_HISTORY_DB_PATH     — SQLite history path  (default: $HERMES_HOME/mem0_oss/history.db)
  MEM0_OSS_COLLECTION          — Qdrant collection name (default: hermes)
  MEM0_OSS_USER_ID             — memory namespace (default: hermes-user)
  MEM0_OSS_LLM_PROVIDER        — override auxiliary.mem0_oss.provider
  MEM0_OSS_LLM_MODEL           — override auxiliary.mem0_oss.model
  MEM0_OSS_EMBEDDER_PROVIDER   — mem0 embedder provider (default: matches llm provider)
  MEM0_OSS_EMBEDDER_MODEL      — embedder model id
  MEM0_OSS_EMBEDDER_DIMS       — embedding dimensions (default: auto per provider)
  MEM0_OSS_TOP_K               — max results returned per search (default: 10)

Secret config:
  MEM0_OSS_API_KEY             — dedicated API key for mem0 LLM calls; takes
                                 precedence over auxiliary.mem0_oss.api_key.
                                 Falls back to the provider's standard env var
                                 (OPENAI_API_KEY, ANTHROPIC_API_KEY,
                                 OPENROUTER_API_KEY, etc.) resolved via the
                                 Hermes provider registry — so no extra key is
                                 needed when a main Hermes provider is already
                                 configured.
  (AWS Bedrock uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION.)

Optional $HERMES_HOME/mem0_oss.json for non-secret overrides:
  {
    "llm_provider": "aws_bedrock",
    "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "embedder_provider": "aws_bedrock",
    "embedder_model": "amazon.titan-embed-text-v2:0",
    "embedder_dims": 1024,
    "collection": "hermes",
    "user_id": "hermes-user",
    "top_k": 10
  }
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause for cooldown.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Qdrant embedded lock error substring — used to detect contention gracefully.
_QDRANT_LOCK_ERROR = "already accessed by another instance"

# Retry parameters for Qdrant lock contention in _get_memory().
# Two processes (WebUI + gateway) may briefly overlap; retry resolves it.
# Prefetch + sync operations hold the lock during an LLM call (~1-3s),
# so we retry for up to 15s total with jitter to avoid thundering herd.
_LOCK_RETRY_ATTEMPTS = 10   # total attempts
_LOCK_RETRY_DELAY_S = 0.8   # base seconds between retries (with jitter, up to ~0.4s extra)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_aux_config() -> dict:
    """Read auxiliary.mem0_oss from config.yaml, with fallback to auxiliary.default.

    Keys not present in auxiliary.mem0_oss are inherited from auxiliary.default
    (if set) so that a single default auxiliary provider covers all aux tasks.
    Returns {} on any failure.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    if not isinstance(aux, dict):
        return {}
    default = aux.get("default", {}) or {}
    task = aux.get("mem0_oss", {}) or {}
    # task-specific keys win; default fills in anything not set
    merged = {**default, **task}
    return merged


def _resolve_auto_credentials(aux_provider: str, aux_model: str,
                               aux_base_url: str, aux_api_key: str):
    """When no specific provider is set, fall through to the default auxiliary chain.

    Mirrors the Hermes auxiliary auto-detection priority order so that users
    with a main provider configured (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, …)
    don't need to also set MEM0_OSS_API_KEY.

    If an explicit provider is already configured (aux_provider is non-empty and
    not "auto"), this function is a no-op and returns the inputs unchanged.

    Returns (hermes_provider, model, base_url, api_key) — all strings, never None.
    """
    # Only kick in when no explicit provider was configured
    if aux_provider and aux_provider.lower() not in ("", "auto"):
        return aux_provider, aux_model, aux_base_url, aux_api_key

    # If task-level config.yaml has a specific auxiliary.mem0_oss entry with a
    # provider set, _resolve_task_provider_model returns that; otherwise "auto".
    # Rather than creating a full OpenAI client we probe env vars directly in
    # the same priority order the auxiliary auto-detect chain uses.
    try:
        from agent.auxiliary_client import _resolve_task_provider_model
        h_provider, h_model, h_base_url, h_api_key, _api_mode = (
            _resolve_task_provider_model("mem0_oss")
        )
        # If the task config actually resolved a specific non-auto provider,
        # use that directly (covers auxiliary.mem0_oss.provider = "openrouter" etc.)
        if h_provider and h_provider != "auto":
            resolved_provider = h_provider
            resolved_model = aux_model or h_model or ""
            resolved_base_url = aux_base_url or h_base_url or ""
            resolved_api_key = aux_api_key or h_api_key or ""
            # Still try to fill missing key from provider registry
            if not resolved_api_key and resolved_provider not in (
                    "aws_bedrock", "bedrock", "aws", "ollama", "lmstudio"):
                try:
                    from hermes_cli.auth import resolve_api_key_provider_credentials
                    creds = resolve_api_key_provider_credentials(resolved_provider)
                    resolved_api_key = str(creds.get("api_key", "") or "").strip()
                    if not resolved_base_url:
                        resolved_base_url = str(creds.get("base_url", "") or "").strip()
                except Exception:
                    pass
            return resolved_provider, resolved_model, resolved_base_url, resolved_api_key
    except Exception:
        pass

    # Full auto-detect: first try to mirror the main runtime provider so that
    # mem0 uses the same provider as the rest of Hermes.  Fall back to env-var
    # probe only when the main provider isn't usable for aux tasks.
    try:
        from agent.auxiliary_client import _read_main_provider
        main_provider = (_read_main_provider() or "").strip().lower()
        if main_provider in ("bedrock", "aws_bedrock", "aws"):
            return "aws_bedrock", aux_model, aux_base_url, aux_api_key
        if main_provider == "anthropic":
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if anthropic_key:
                return "anthropic", aux_model, aux_base_url, aux_api_key or anthropic_key
        if main_provider == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if openai_key:
                base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
                return "openai", aux_model, aux_base_url or base_url, aux_api_key or openai_key
        if main_provider == "openrouter":
            openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if openrouter_key:
                base_url = os.environ.get("OPENROUTER_BASE_URL",
                                           "https://openrouter.ai/api/v1").strip()
                return "openrouter", aux_model, aux_base_url or base_url, aux_api_key or openrouter_key
    except Exception:
        pass

    # Fallback env-var probe (no main provider available)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        base_url = os.environ.get("OPENROUTER_BASE_URL",
                                   "https://openrouter.ai/api/v1").strip()
        return "openrouter", aux_model, aux_base_url or base_url, aux_api_key or openrouter_key

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return "anthropic", aux_model, aux_base_url, aux_api_key or anthropic_key

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        return "openai", aux_model, aux_base_url or base_url, aux_api_key or openai_key

    # Bedrock: no API key needed, boto3 reads from env/profile automatically
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        return "aws_bedrock", aux_model, aux_base_url, aux_api_key

    # Nothing found — return "auto" and let _load_config fall back to aws_bedrock default
    return aux_provider or "auto", aux_model, aux_base_url, aux_api_key


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0_oss.json overrides.

    Priority for LLM provider/model/api_key (highest → lowest):
      1. MEM0_OSS_LLM_PROVIDER / MEM0_OSS_LLM_MODEL env vars
      2. auxiliary.mem0_oss.provider / .model in config.yaml
      3. Default auxiliary chain (auto-detect from Hermes config) — uses the
         provider's standard env var (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, …)
         so MEM0_OSS_API_KEY is not required when a main provider is configured.
      4. Defaults (aws_bedrock)

    Priority for API key:
      1. MEM0_OSS_API_KEY env var
      2. auxiliary.mem0_oss.api_key in config.yaml
      3. Provider standard env var (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.)
         resolved via the Hermes provider registry.

    Environment variables are the base; the JSON file (if present) overrides
    individual keys.  Neither source is required — sensible defaults apply.
    """
    hermes_home = get_hermes_home()
    qdrant_path = str(hermes_home / "mem0_oss" / "qdrant")
    history_path = str(hermes_home / "mem0_oss" / "history.db")

    aux = _get_aux_config()
    aux_provider = str(aux.get("provider", "") or "").strip()
    aux_model = str(aux.get("model", "") or "").strip()
    aux_base_url = str(aux.get("base_url", "") or "").strip()
    aux_api_key = str(aux.get("api_key", "") or "").strip()

    # MEM0_OSS_API_KEY is the dedicated key; falls back to aux config key, then
    # to the provider's standard env var via _resolve_auto_credentials below.
    explicit_api_key = (
        os.environ.get("MEM0_OSS_API_KEY", "").strip()
        or aux_api_key
    )
    # base_url: env var wins, then aux config
    explicit_base_url = (
        os.environ.get("MEM0_OSS_OPENAI_BASE_URL", "").strip()
        or aux_base_url
    )

    # When no specific provider is configured, fall through to the default
    # auxiliary chain so we inherit the user's main Hermes provider + key.
    auto_provider, auto_model, auto_base_url, auto_api_key = _resolve_auto_credentials(
        aux_provider, aux_model, explicit_base_url, explicit_api_key
    )

    resolved_api_key = explicit_api_key or auto_api_key
    resolved_base_url = explicit_base_url or auto_base_url

    # LLM provider: env > aux config > auto-detected > default
    default_llm_provider = aux_provider or auto_provider or "openai"
    llm_provider = os.environ.get("MEM0_OSS_LLM_PROVIDER", default_llm_provider).strip()
    # Normalise Hermes provider aliases → mem0 provider keys
    llm_provider = _normalise_provider(llm_provider)

    # LLM model: env > aux config > auto-detected > per-provider default
    default_llm_model = aux_model or auto_model or _default_model_for(llm_provider)
    llm_model = os.environ.get("MEM0_OSS_LLM_MODEL", default_llm_model).strip()

    # Embedder defaults mirror the LLM provider
    default_emb_provider = _default_embedder_provider(llm_provider)
    default_emb_model = _default_embedder_model(default_emb_provider)
    default_emb_dims = _default_embedder_dims(default_emb_provider)

    config: dict = {
        "vector_store_path": os.environ.get("MEM0_OSS_VECTOR_STORE_PATH", qdrant_path),
        "history_db_path": os.environ.get("MEM0_OSS_HISTORY_DB_PATH", history_path),
        "collection": os.environ.get("MEM0_OSS_COLLECTION", "hermes"),
        "user_id": os.environ.get("MEM0_OSS_USER_ID", "hermes-user"),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedder_provider": _normalise_provider(
            os.environ.get("MEM0_OSS_EMBEDDER_PROVIDER", default_emb_provider)
        ),
        "embedder_model": os.environ.get("MEM0_OSS_EMBEDDER_MODEL", default_emb_model),
        "embedder_dims": int(os.environ.get("MEM0_OSS_EMBEDDER_DIMS", str(default_emb_dims))),
        "top_k": int(os.environ.get("MEM0_OSS_TOP_K", "10")),
        # Resolved credentials / endpoint
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        # Legacy key kept for backwards compat with tests and mem0_oss.json
        "openai_api_key": resolved_api_key,
        "openai_base_url": resolved_base_url,
    }

    config_path = hermes_home / "mem0_oss.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception as exc:
            logger.warning("mem0_oss: failed to read config file %s: %s", config_path, exc)

    return config


# ---------------------------------------------------------------------------
# Provider normalisation helpers
# ---------------------------------------------------------------------------

# Maps Hermes provider names / aliases → mem0 LLM provider keys
_HERMES_TO_MEM0_PROVIDER: dict = {
    "bedrock": "aws_bedrock",
    "aws": "aws_bedrock",
    "aws_bedrock": "aws_bedrock",
    "openai": "openai",
    "openrouter": "openai",   # mem0 uses OpenAI adapter with OR base URL
    "anthropic": "anthropic",
    "ollama": "ollama",
    "lmstudio": "lmstudio",
    "custom": "openai",       # custom base_url → OpenAI-compatible adapter
    "auto": "aws_bedrock",    # resolved later in is_available(); placeholder
}

_PROVIDER_DEFAULTS: dict = {
    "aws_bedrock":  ("us.anthropic.claude-haiku-4-5-20251001-v1:0",
                     "aws_bedrock", "amazon.titan-embed-text-v2:0", 1024),
    # --- ordering note: openai is the last-resort default (most widely available) ---
    "openai":       ("gpt-4o-mini",    "openai", "text-embedding-3-small", 1536),
    "anthropic":    ("claude-haiku-4-5-20251001", "openai", "text-embedding-3-small", 1536),
    "ollama":       ("llama3.1",        "ollama", "nomic-embed-text", 768),
    "lmstudio":     ("llama-3.2-1b-instruct", "openai", "text-embedding-nomic-embed-text-v1.5", 768),
    "openrouter":   ("openai/gpt-4o-mini", "openai", "text-embedding-3-small", 1536),
}


def _normalise_provider(p: str) -> str:
    p = (p or "").strip().lower()
    return _HERMES_TO_MEM0_PROVIDER.get(p, p) or "openai"


def _default_model_for(mem0_provider: str) -> str:
    return _PROVIDER_DEFAULTS.get(mem0_provider, _PROVIDER_DEFAULTS["openai"])[0]


def _default_embedder_provider(mem0_provider: str) -> str:
    return _PROVIDER_DEFAULTS.get(mem0_provider, _PROVIDER_DEFAULTS["openai"])[1]


def _default_embedder_model(mem0_emb_provider: str) -> str:
    for _llm_p, (_, emb_p, emb_m, _) in _PROVIDER_DEFAULTS.items():
        if emb_p == mem0_emb_provider:
            return emb_m
    return "text-embedding-3-small"


def _default_embedder_dims(mem0_emb_provider: str) -> int:
    for _llm_p, (_, emb_p, _, emb_d) in _PROVIDER_DEFAULTS.items():
        if emb_p == mem0_emb_provider:
            return emb_d
    return 1536


def _build_mem0_config(cfg: dict) -> dict:
    """Build a mem0 MemoryConfig-compatible dict from our flattened config.

    Translates Hermes/mem0 provider names into the provider-specific config
    structures that mem0ai expects, including credentials and base URLs.
    """
    llm_provider = cfg["llm_provider"]
    llm_model = cfg["llm_model"]
    embedder_provider = cfg["embedder_provider"]
    embedder_model = cfg["embedder_model"]
    embedder_dims = cfg["embedder_dims"]
    api_key = cfg.get("api_key") or cfg.get("openai_api_key") or ""
    base_url = cfg.get("base_url") or cfg.get("openai_base_url") or ""

    llm_cfg = _build_llm_cfg(llm_provider, llm_model, api_key, base_url)
    emb_cfg = _build_embedder_cfg(embedder_provider, embedder_model, embedder_dims, api_key, base_url)

    vs_cfg = {
        "collection_name": cfg["collection"],
        "path": cfg["vector_store_path"],
        "embedding_model_dims": embedder_dims,
        "on_disk": True,
    }

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": vs_cfg,
        },
        "llm": {
            "provider": llm_provider,
            "config": llm_cfg,
        },
        "embedder": {
            "provider": embedder_provider,
            "config": emb_cfg,
        },
        "history_db_path": cfg["history_db_path"],
        "version": "v1.1",
    }


def _build_llm_cfg(provider: str, model: str, api_key: str, base_url: str) -> dict:
    """Build the provider-specific LLM config dict for mem0ai."""
    cfg: dict = {"model": model}

    if provider == "aws_bedrock":
        # Bedrock reads creds from env vars automatically; we don't pass them
        # explicitly unless they're set (boto3 picks them up from the environment).
        pass

    elif provider in ("openai", "anthropic", "lmstudio"):
        if api_key:
            cfg["api_key"] = api_key
        if base_url and provider == "openai":
            cfg["openai_base_url"] = base_url

    elif provider == "ollama":
        # Ollama uses openai_base_url pointing at the local server
        cfg["openai_base_url"] = base_url or "http://localhost:11434"

    # openrouter is handled as openai with OR base URL — normalised upstream,
    # so if it reaches here with provider=="openai" it already has base_url set.

    return cfg


def _build_embedder_cfg(provider: str, model: str, dims: int,
                         api_key: str, base_url: str) -> dict:
    """Build the provider-specific embedder config dict for mem0ai."""
    cfg: dict = {"model": model}

    if provider == "aws_bedrock":
        cfg["embedding_dims"] = dims

    elif provider in ("openai",):
        cfg["embedding_dims"] = dims
        if api_key:
            cfg["api_key"] = api_key
        if base_url:
            cfg["openai_base_url"] = base_url

    elif provider == "ollama":
        cfg["embedding_dims"] = dims
        cfg["ollama_base_url"] = base_url or "http://localhost:11434"

    elif provider == "lmstudio":
        cfg["embedding_dims"] = dims
        if api_key:
            cfg["api_key"] = api_key

    return cfg


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mem0_oss_search",
    "description": (
        "Search long-term memory using semantic similarity. Returns facts and context "
        "ranked by relevance.  Use this when you need information from past sessions "
        "that is not already in the current conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {
                "type": "integer",
                "description": "Max results (default: 10, max: 50).",
            },
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "mem0_oss_add",
    "description": (
        "Store a fact, preference, or piece of context to long-term memory. "
        "mem0 deduplicates automatically — safe to call for any important detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
        },
        "required": ["content"],
    },
}


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class Mem0OSSMemoryProvider(MemoryProvider):
    """Self-hosted mem0 memory provider backed by a local Qdrant vector store.

    No cloud account required — all data stays on disk.  Uses AWS Bedrock
    (or OpenAI / Ollama) for LLM fact-extraction and embedding.
    """

    def __init__(self):
        # Config / identity
        self._cfg: dict = {}
        self._user_id: str = "hermes-user"
        self._top_k: int = 10
        self._session_id: str = ""
        self._agent_context: str = "primary"
        # Circuit-breaker state (lock-protected)
        self._lock = threading.Lock()
        self._fail_count: int = 0
        self._last_fail_ts: float = 0.0
        # Background thread state
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_result: str = ""

    # -- MemoryProvider identity --------------------------------------------

    @property
    def name(self) -> str:
        return "mem0_oss"

    # -- Availability -------------------------------------------------------

    def is_available(self) -> bool:
        """True if mem0ai is installed and at least one LLM backend is usable.

        We only check imports and credentials — no network calls here.
        """
        try:
            import mem0  # noqa: F401
        except ImportError:
            return False

        cfg = _load_config()
        llm_provider = cfg.get("llm_provider", "openai")

        if llm_provider == "aws_bedrock":
            if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
                return True
            try:
                from agent.bedrock_adapter import has_aws_credentials
                return has_aws_credentials()
            except Exception:
                return False
        if llm_provider == "anthropic":
            return bool(
                cfg.get("api_key")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
        if llm_provider == "openai":
            return bool(
                cfg.get("api_key")
                or cfg.get("openai_api_key")
                or os.environ.get("OPENAI_API_KEY")
            )
        if llm_provider in ("ollama", "lmstudio"):
            return True  # local, always assumed available
        # Generic / custom base_url: trust the user's config
        return True

    # -- Lifecycle ----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Build the mem0 Memory instance for this session."""
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")
        self._cfg = _load_config()
        self._user_id = self._cfg["user_id"]
        self._top_k = self._cfg["top_k"]
        # Reset circuit-breaker and prefetch state for this session.
        # (Lock is created in __init__ and reused across sessions.)
        with self._lock:
            self._fail_count = 0
            self._last_fail_ts = 0.0
        self._prefetch_result = ""
        import pathlib
        pathlib.Path(self._cfg["vector_store_path"]).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._cfg["history_db_path"]).parent.mkdir(parents=True, exist_ok=True)

    def _get_memory(self) -> Any:
        """Create a fresh mem0 Memory instance for each call.

        We intentionally do NOT cache the instance.  The embedded Qdrant store
        uses a portalocker (fcntl) exclusive lock that is held for the lifetime
        of the client object.  When both the WebUI and the gateway run on the
        same host they compete for this lock.

        We retry up to _LOCK_RETRY_ATTEMPTS times with _LOCK_RETRY_DELAY_S
        seconds between attempts so that brief overlaps (e.g. a concurrent
        prefetch in another process) are automatically resolved.
        """
        import time as _time

        last_exc: Optional[Exception] = None
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                from mem0 import Memory
                from mem0.configs.base import MemoryConfig

                mem0_dict = _build_mem0_config(self._cfg)
                mem_cfg = MemoryConfig(**{
                    "vector_store": mem0_dict["vector_store"],
                    "llm": mem0_dict["llm"],
                    "embedder": mem0_dict["embedder"],
                    "history_db_path": mem0_dict["history_db_path"],
                    "version": mem0_dict["version"],
                })
                return Memory(config=mem_cfg)
            except Exception as exc:
                last_exc = exc
                if _QDRANT_LOCK_ERROR in str(exc):
                    if attempt < _LOCK_RETRY_ATTEMPTS - 1:
                        import random as _random
                        jitter = _random.uniform(0, _LOCK_RETRY_DELAY_S * 0.5)
                        delay = _LOCK_RETRY_DELAY_S + jitter
                        logger.debug(
                            "mem0_oss: Qdrant lock busy (attempt %d/%d), retrying in %.2fs",
                            attempt + 1, _LOCK_RETRY_ATTEMPTS, delay,
                        )
                        _time.sleep(delay)
                        continue
                    # Last attempt also a lock error — fall through to raise below
                else:
                    # Non-lock error — fail fast, no retry
                    logger.error("mem0_oss: failed to initialize Memory: %s", exc)
                    raise
        logger.warning(
            "mem0_oss: Qdrant lock still held after %d attempts — giving up: %s",
            _LOCK_RETRY_ATTEMPTS, last_exc,
        )
        raise last_exc  # type: ignore[misc]

    # -- Circuit breaker helpers -------------------------------------------

    def _is_tripped(self) -> bool:
        with self._lock:
            if self._fail_count < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() - self._last_fail_ts >= _BREAKER_COOLDOWN_SECS:
                self._fail_count = 0
                return False
            return True

    def _record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            self._last_fail_ts = time.monotonic()

    def _record_success(self) -> None:
        with self._lock:
            self._fail_count = 0

    # -- System prompt block -----------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "## Mem0 OSS Memory (self-hosted)\n"
            "You have access to long-term memory stored locally via mem0.\n"
            "- Use `mem0_oss_search` to recall relevant facts before answering.\n"
            "- Use `mem0_oss_add` to store important new facts, preferences, or context.\n"
            "- Facts are extracted and deduplicated automatically on each turn.\n"
            "- Search is semantic — natural-language queries work well.\n"
        )

    # -- Prefetch (background recall before each turn) ---------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Start a background thread to recall context for the upcoming turn."""
        if self._is_tripped():
            return

        self._prefetch_result = ""
        self._prefetch_thread = threading.Thread(
            target=self._do_prefetch,
            args=(query,),
            daemon=True,
            name="mem0-oss-prefetch",
        )
        self._prefetch_thread.start()

    def _do_prefetch(self, query: str) -> None:
        try:
            mem = self._get_memory()
            results = mem.search(
                query=query[:500],
                top_k=self._top_k,
                filters={"user_id": self._user_id},
            )
            del mem  # release Qdrant lock ASAP — before any further processing
            memories = _extract_results(results)
            if memories:
                lines = "\n".join(f"- {m}" for m in memories)
                self._prefetch_result = f"Mem0 OSS Memory:\n{lines}"
            self._record_success()
        except Exception as exc:
            if _QDRANT_LOCK_ERROR in str(exc):
                logger.debug("mem0_oss: prefetch skipped — Qdrant lock held by another process")
                return  # not a real failure; don't trip the circuit breaker
            self._record_failure()
            logger.debug("mem0_oss: prefetch error: %s", exc)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched results (join background thread first)."""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=15.0)
            self._prefetch_thread = None
        return self._prefetch_result

    # -- Sync turn (auto-extract after each turn) --------------------------

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Spawn a background thread to extract and store facts from the turn."""
        if self._agent_context != "primary":
            return
        if self._is_tripped():
            return

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        self._sync_thread = threading.Thread(
            target=self._do_sync,
            args=(messages,),
            daemon=True,
            name="mem0-oss-sync",
        )
        self._sync_thread.start()

    def _do_sync(self, messages: List[dict]) -> None:
        try:
            mem = self._get_memory()
            mem.add(messages=messages, user_id=self._user_id, infer=True)
            del mem  # release Qdrant lock ASAP
            self._record_success()
        except Exception as exc:
            if _QDRANT_LOCK_ERROR in str(exc):
                logger.debug("mem0_oss: sync_turn skipped — Qdrant lock held by another process")
                return  # not a real failure; don't trip the circuit breaker
            self._record_failure()
            logger.debug("mem0_oss: sync_turn error: %s", exc)

    # -- Tool schemas & dispatch -------------------------------------------

    def get_tool_schemas(self) -> List[dict]:
        return [SEARCH_SCHEMA, ADD_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mem0_oss_search":
            return self._handle_search(args)
        if tool_name == "mem0_oss_add":
            return self._handle_add(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("mem0_oss_search requires 'query'")

        top_k = min(int(args.get("top_k", self._top_k)), 50)

        try:
            mem = self._get_memory()
            results = mem.search(
                query=query,
                top_k=top_k,
                filters={"user_id": self._user_id},
            )
            del mem  # release Qdrant lock ASAP
            memories = _extract_results(results)
            self._record_success()
            if not memories:
                return json.dumps({"result": "No relevant memories found."})
            return json.dumps({"result": "\n".join(f"- {m}" for m in memories)})
        except Exception as exc:
            if _QDRANT_LOCK_ERROR in str(exc):
                self._record_failure()  # already handled by retry in _get_memory, but track it
                logger.warning("mem0_oss: Qdrant lock held by another process — search skipped")
                return json.dumps({"result": "Memory temporarily unavailable (storage locked by another process)."})
            self._record_failure()
            logger.error("mem0_oss: search error: %s", exc)
            return tool_error(f"mem0_oss_search failed: {exc}")

    def _handle_add(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("mem0_oss_add requires 'content'")

        try:
            mem = self._get_memory()
            mem.add(
                messages=[{"role": "user", "content": content}],
                user_id=self._user_id,
                infer=True,
            )
            del mem  # release Qdrant lock ASAP
            self._record_success()
            return json.dumps({"result": "Memory stored successfully."})
        except Exception as exc:
            if _QDRANT_LOCK_ERROR in str(exc):
                self._record_failure()
                logger.warning("mem0_oss: Qdrant lock held by another process — add skipped")
                return json.dumps({"result": "Memory temporarily unavailable (storage locked by another process)."})
            self._record_failure()
            logger.error("mem0_oss: add error: %s", exc)
            return tool_error(f"mem0_oss_add failed: {exc}")

    # -- Config schema (for setup wizard) ----------------------------------

    def get_config_schema(self) -> List[dict]:
        return [
            {
                "key": "llm_provider",
                "label": "LLM provider",
                "description": "mem0 LLM provider key (openai, aws_bedrock, ollama, ...)",
                "default": "openai",
                "env": "MEM0_OSS_LLM_PROVIDER",
                "required": False,
            },
            {
                "key": "llm_model",
                "label": "LLM model",
                "description": "Model id passed to the LLM provider",
                "default": "gpt-4o-mini",
                "env": "MEM0_OSS_LLM_MODEL",
                "required": False,
            },
            {
                "key": "embedder_provider",
                "label": "Embedder provider",
                "description": "mem0 embedder provider key (openai, aws_bedrock, ...)",
                "default": "openai",
                "env": "MEM0_OSS_EMBEDDER_PROVIDER",
                "required": False,
            },
            {
                "key": "embedder_model",
                "label": "Embedding model id",
                "description": "Embedding model id",
                "default": "text-embedding-3-small",
                "env": "MEM0_OSS_EMBEDDER_MODEL",
                "required": False,
            },
            {
                "key": "embedder_dims",
                "label": "Embedding dimensions",
                "description": "Dimensions of the embedding model (must match the model)",
                "default": 1024,
                "env": "MEM0_OSS_EMBEDDER_DIMS",
                "required": False,
            },
            {
                "key": "collection",
                "label": "Qdrant collection name",
                "description": "Name of the Qdrant collection storing memories",
                "default": "hermes",
                "env": "MEM0_OSS_COLLECTION",
                "required": False,
            },
            {
                "key": "user_id",
                "label": "User ID",
                "description": "Memory namespace / user identifier",
                "default": "hermes-user",
                "env": "MEM0_OSS_USER_ID",
                "required": False,
            },
            {
                "key": "top_k",
                "label": "Top-K results",
                "description": "Default number of memories returned per search",
                "default": 10,
                "env": "MEM0_OSS_TOP_K",
                "required": False,
            },
            {
                "key": "api_key",
                "label": "API key (mem0 LLM)",
                "description": (
                    "Dedicated API key for mem0 LLM/embedder calls.  "
                    "Takes precedence over auxiliary.mem0_oss.api_key in config.yaml "
                    "and over OPENAI_API_KEY / ANTHROPIC_API_KEY.  "
                    "Not needed for AWS Bedrock (uses AWS_ACCESS_KEY_ID)."
                ),
                "default": "",
                "env": "MEM0_OSS_API_KEY",
                "secret": True,
                "required": False,
            },
            {
                "key": "openai_api_key",
                "label": "API key (legacy alias)",
                "description": "Legacy alias for api_key — prefer MEM0_OSS_API_KEY.",
                "default": "",
                "env": "MEM0_OSS_OPENAI_API_KEY",
                "secret": True,
                "required": False,
            },
            {
                "key": "base_url",
                "label": "OpenAI-compatible base URL",
                "description": (
                    "Custom LLM endpoint (e.g. http://localhost:11434/v1 for Ollama, "
                    "or an OpenRouter-compatible URL).  Also settable via "
                    "auxiliary.mem0_oss.base_url in config.yaml."
                ),
                "default": "",
                "env": "MEM0_OSS_OPENAI_BASE_URL",
                "required": False,
            },
        ]

    def save_config(self, values: dict, hermes_home) -> None:
        """Write non-secret config to $HERMES_HOME/mem0_oss.json.

        Merges ``values`` into any existing file so that only the supplied keys
        are overwritten.  Secret keys (api_key, openai_api_key) should be stored
        in ``.env`` instead; this method stores them only if explicitly passed.
        """
        import json
        from pathlib import Path

        config_path = Path(hermes_home) / "mem0_oss.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- Shutdown ----------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory tool writes into mem0 store.

        Called by the framework whenever the agent uses the builtin memory tool,
        so writes go to mem0 automatically without the agent needing to call
        mem0_oss_add explicitly.
        """
        if action != "add" or not (content or "").strip():
            return

        def _write():
            try:
                mem = self._get_memory()
                mem.add(
                    messages=[{"role": "user", "content": content.strip()}],
                    user_id=self._user_id,
                    infer=False,
                    metadata={"source": "hermes_memory_tool", "target": target},
                )
            except Exception as e:
                if _QDRANT_LOCK_ERROR in str(e):
                    logger.debug("mem0_oss on_memory_write skipped — Qdrant lock held by another process")
                    return
                logger.debug("mem0_oss on_memory_write failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="mem0-oss-memwrite")
        t.start()

    def shutdown(self) -> None:
        """Wait for any in-flight background threads."""
        for thread in (self._sync_thread, self._prefetch_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Result extraction helper
# ---------------------------------------------------------------------------

def _extract_results(results: Any) -> List[str]:
    """Normalize mem0 search results (v1 list or v2 dict) to plain strings."""
    if isinstance(results, dict) and "results" in results:
        items = results["results"]
    elif isinstance(results, list):
        items = results
    else:
        return []

    memories = []
    for item in items:
        if isinstance(item, dict):
            mem = item.get("memory") or item.get("text") or ""
        else:
            mem = str(item)
        if mem:
            memories.append(mem)
    return memories


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_memory_provider(Mem0OSSMemoryProvider())
