"""Hindsight memory plugin — MemoryProvider with knowledge graph, entity resolution
and multi-strategy retrieval; cloud (API key), local_external, or local_embedded.

Config: $HERMES_HOME/hindsight/config.json (profile-scoped), else ~/.hindsight/
config.json (legacy, shared), else env: HINDSIGHT_API_KEY / BANK_ID / BUDGET /
API_URL / MODE / TIMEOUT / IDLE_TIMEOUT / RETAIN_TAGS / RETAIN_OBSERVATION_SCOPES /
RETAIN_SOURCE / RETAIN_USER_PREFIX / RETAIN_ASSISTANT_PREFIX, and
HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT (config.json port_health_grace_timeout).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus
from agent.secret_scope import get_secret
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from tools.registry import tool_error

from .embedded import (
    _RETRIABLE_CONNECTION_MARKERS, _build_embedded_profile_env,
    _check_local_runtime, _embedded_llm_api_key, _embedded_profile_env_path,
    _export_port_health_grace_timeout, _load_simple_env, _local_runtime_hint, _materialize_embedded_profile_env,
)
from .settings import (
    _DEFAULT_API_URL, _DEFAULT_IDLE_TIMEOUT, _DEFAULT_LOCAL_URL, _DEFAULT_RETAIN_SOURCE,
    _DEFAULT_TIMEOUT, _HINDSIGHT_GLYPH, _MIN_CLIENT_VERSION, _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
    _PROVIDER_DEFAULT_MODELS, _VALID_BUDGETS, _daemon_llm_provider,
    _contains_term, _is_live_status_query, _is_simple_recall_query,
    _normalize_observation_scopes, _normalize_retain_tags, _parse_int_setting,
    _resolve_bank_id_template,
)

logger = logging.getLogger(__name__)

_LOCAL_MODES = {"local", "local_embedded"}
_RETAIN_CONTEXT_DEFAULT = "conversation between Hermes Agent and the User"


def _ensure_client_dependency() -> None:
    """Lazily install the Hindsight client (``tools.lazy_deps``) before importing it."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("memory.hindsight", prompt=False)
    except ImportError:
        pass
    except Exception as exc:
        raise ImportError(str(exc)) from exc


def _cloud_api_key(config: dict) -> str:
    return config.get("apiKey") or config.get("api_key") or get_secret("HINDSIGHT_API_KEY", "")


def _maybe_upgrade_client() -> None:
    """Auto-upgrade an outdated hindsight-client via the environment-aware lazy_deps
    installer (sealed hosted venvs redirect to the durable target)."""
    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version
        installed = pkg_version("hindsight-client")
        if Version(installed) < Version(_MIN_CLIENT_VERSION):
            logger.warning("hindsight-client %s is outdated (need >=%s), attempting upgrade...",
                           installed, _MIN_CLIENT_VERSION)
            from tools.lazy_deps import install_specs
            outcome = install_specs([f"hindsight-client>={_MIN_CLIENT_VERSION}"], timeout=120)
            if outcome.ok:
                logger.info("hindsight-client upgraded to >=%s", _MIN_CLIENT_VERSION)
            elif outcome.blocked:
                logger.warning("Auto-upgrade unavailable: %s. Run: uv pip install 'hindsight-client>=%s'",
                               outcome.reason, _MIN_CLIENT_VERSION)
            else:
                logger.warning("Auto-upgrade failed: %s. Run: uv pip install 'hindsight-client>=%s'",
                               (outcome.stderr or "").strip() or "install error", _MIN_CLIENT_VERSION)
    except Exception:
        pass  # packaging not available or other issue — proceed anyway


# update_mode='append' capability (Hindsight >= 0.5.0), cached per API URL per
# process so every provider on the same API shares one /version round trip.
_append_capability_cache: Dict[str, bool] = {}
_append_capability_lock = threading.Lock()


def _fetch_hindsight_api_version(api_url: str, api_key: str | None = None,
                                 timeout: float = 5.0) -> str | None:
    """GET ``<api_url>/version`` -> version string, or None on any failure (= legacy API)."""
    import urllib.request
    if not api_url:
        return None
    url = api_url.rstrip("/") + "/version"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("Hindsight /version probe failed for %s: %s", url, exc)
        return None
    version = (data.get("version") or data.get("api_version")) if isinstance(data, dict) else None
    return str(version) if version else None


def _check_api_supports_update_mode_append(api_url: str, api_key: str | None = None) -> bool:
    """Cached ``update_mode='append'`` check for *api_url*. False on any probe failure
    (safe default: per-process document_id, no update_mode = resume-overwrite fix intact).

    Probes once per URL per process. See #6654.
    """
    if not api_url:
        return False
    with _append_capability_lock:
        if api_url in _append_capability_cache:
            return _append_capability_cache[api_url]
    version = _fetch_hindsight_api_version(api_url, api_key)
    try:  # missing/invalid version -> unsupported
        from packaging.version import Version
        supported = bool(version) and Version(version) >= Version(_MIN_VERSION_FOR_UPDATE_MODE_APPEND)
    except Exception:
        supported = False
    with _append_capability_lock:
        # A concurrent probe may have filled the cache meanwhile; its answer wins.
        supported = _append_capability_cache.setdefault(api_url, supported)
    if supported:
        logger.debug("Hindsight API %s version %s supports update_mode='append'", api_url, version)
    else:
        logger.warning("Hindsight API at %s reports version %r, older than %s. "
                       "Falling back to per-process document_id — retains across "
                       "processes/sessions create separate documents instead of "
                       "appending to a session-scoped one. Upgrade Hindsight to "
                       "%s+ to enable update_mode='append' deduplication.",
                       api_url, version, _MIN_VERSION_FOR_UPDATE_MODE_APPEND, _MIN_VERSION_FOR_UPDATE_MODE_APPEND)
    return supported


# One long-lived event loop per process for Hindsight async calls; ephemeral
# loops would leak aiohttp sessions.
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# Pushed to the per-provider retain queue to wake the writer for a clean exit.
_WRITER_SENTINEL = object()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()), daemon=True, name="hindsight-loop",
        )
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = _DEFAULT_TIMEOUT):
    """Schedule *coro* on the shared loop and block until done."""
    from agent.async_utils import safe_schedule_threadsafe
    future = safe_schedule_threadsafe(coro, _get_loop())
    if future is None:
        raise RuntimeError("Hindsight loop unavailable")
    return future.result(timeout=timeout)


def _context_thread(target, name: str) -> threading.Thread:
    """Daemon thread running *target* in a snapshot of the spawner's contextvars.
    Threads start with an EMPTY Context; under multiplex_profiles get_secret fails
    closed without the profile's secret scope + HERMES_HOME override. (The shared
    loop needs no wrap: run_coroutine_threadsafe inherits the submitter's context.)"""
    return threading.Thread(target=contextvars.copy_context().run, args=(target,), daemon=True, name=name)


RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Optional per-call tags to merge with configured default retain tags."},
            "occurred_at": {"type": "string", "description": (
                "When the remembered event actually happened, as an ISO-8601 date "
                "or datetime (e.g. '2026-08-20' or '2026-08-20T14:30:00+02:00'). "
                "Pass this whenever the memory references a specific event time "
                "('yesterday', 'last Tuesday', 'on March 3rd') so Hindsight can "
                "anchor it on the timeline. Omit for timeless facts/preferences."
            )},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {"type": "object", "required": ["query"],
                   "properties": {"query": {"type": "string", "description": "What to search for."}}},
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {"type": "object", "required": ["query"],
                   "properties": {"query": {"type": "string", "description": "The question to reflect on."}}},
}


def _load_config() -> dict:
    """$HERMES_HOME/hindsight/config.json (profile-scoped), else ~/.hindsight/config.json
    (legacy, shared), else environment variables."""
    for path in (get_hermes_home() / "hindsight" / "config.json", Path.home() / ".hindsight" / "config.json"):
        if path.exists():
            with contextlib.suppress(Exception):
                return json.loads(path.read_text(encoding="utf-8"))
    return {
        "mode": os.environ.get("HINDSIGHT_MODE", "cloud"),
        "apiKey": get_secret("HINDSIGHT_API_KEY", ""),
        "timeout": _parse_int_setting(os.environ.get("HINDSIGHT_TIMEOUT"), _DEFAULT_TIMEOUT),
        "idle_timeout": _parse_int_setting(os.environ.get("HINDSIGHT_IDLE_TIMEOUT"), _DEFAULT_IDLE_TIMEOUT),
        "retain_tags": os.environ.get("HINDSIGHT_RETAIN_TAGS", ""),
        "observation_scopes": os.environ.get("HINDSIGHT_RETAIN_OBSERVATION_SCOPES", ""),
        "retain_source": os.environ.get("HINDSIGHT_RETAIN_SOURCE", _DEFAULT_RETAIN_SOURCE),
        "retain_user_prefix": os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User"),
        "retain_assistant_prefix": os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant"),
        "banks": {"hermes": {"bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                             "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"), "enabled": True}},
    }


def _event_timestamp() -> str:
    """Configured Hermes event time with an explicit UTC offset."""
    event_time = _hermes_now()
    # hermes_time.now() is aware; guard a replacement clock emitting offset-less dates.
    if event_time.tzinfo is None or event_time.utcoffset() is None:
        event_time = event_time.astimezone()
    return event_time.isoformat(timespec="seconds")


def _mint_document_id(session_id: str) -> str:
    """Per-process document id: reusing session_id alone overwrote the document on
    /resume (the reloaded session's first retain replaced the stored content)."""
    return f"{session_id}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


# initialize() kwargs copied verbatim (str, stripped) onto ``self._<name>``.
_SESSION_KWARGS = (
    "platform", "user_id", "user_name", "chat_id", "chat_name", "chat_type",
    "thread_id", "agent_identity", "agent_workspace",
)
# Retain metadata keys, each stamped from the attribute of the same name when set.
_METADATA_ATTRS = (
    "session_id", "platform", "user_id", "user_name", "chat_id", "chat_name",
    "chat_type", "thread_id", "agent_identity",
)
_SYSTEM_PROMPT_TAILS = {
    "context": "Relevant memories are automatically injected into context.",
    "tools": ("Use hindsight_recall to search, hindsight_reflect for synthesis, "
              "hindsight_retain to store facts."),
    "hybrid": ("Relevant memories are automatically injected into context. "
               "Use hindsight_recall to search, hindsight_reflect for synthesis, "
               "hindsight_retain to store facts."),
}


class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    # Each server-side op status poll is a round trip — coarser than the 0.05s queue poll.
    _RETAIN_OP_POLL_INTERVAL_S = 0.5

    def backup_paths(self) -> List[str]:
        """Legacy shared config + embedded-mode profile env files live under ~/.hindsight."""
        with contextlib.suppress(Exception):
            return [str(Path.home() / ".hindsight")]
        return []

    def __init__(self):
        self._config = self._api_key = self._client = None
        self._api_url, self._llm_base_url, self._mode = _DEFAULT_API_URL, "", "cloud"
        self._timeout, self._idle_timeout = _DEFAULT_TIMEOUT, _DEFAULT_IDLE_TIMEOUT
        self._bank_id, self._budget, self._bank_id_template = "hermes", "mid", ""
        self._bank_mission, self._bank_retain_mission = "", None
        self._memory_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._prefetch_method = "recall"  # "recall" or "reflect"
        for name in _SESSION_KWARGS:
            setattr(self, f"_{name}", "")
        self._session_id = self._parent_session_id = self._document_id = ""
        self._status_callback: Optional[Callable[[str], None]] = None

        # Retain: single-writer model — sync_turn() enqueues, one writer thread
        # drains sequentially (ad-hoc threads raced interpreter shutdown:
        # "cannot schedule new futures" / "Unclosed client session").
        self._retain_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._sync_thread = None  # legacy alias external callers may join; points at the writer
        self._shutting_down = threading.Event()
        self._atexit_registered = False
        self._retain_tags: List[str] = []
        self._tags: list[str] | None = None
        self._retain_source = _DEFAULT_RETAIN_SOURCE
        self._retain_user_prefix, self._retain_assistant_prefix = "User", "Assistant"
        self._turn_counter = self._turn_index = 0
        self._session_turns: list[str] = []  # ALL turns for the session
        self._last_retained_turn_count = 0  # append-mode delta watermark
        # Server-side async retain ops still in flight: aretain_batch returns on
        # *acceptance*, not durability, so the prefetch gates on these via
        # get_operation_status (a drained local queue is not a read-after-write signal).
        self._pending_retain_ops: set[str] = set()
        self._pending_retain_ops_lock = threading.Lock()
        self._retain_ops_bank_id = ""
        self._apply_retain_policy({})

        # Recall: pending prefetch block + count, and the indicator state (recall_status()).
        self._prefetch_result, self._prefetch_count = "", 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._last_recall_returned, self._last_recall_count = False, 0
        self._apply_recall_settings({})

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in _LOCAL_MODES:
                return _check_local_runtime()[0]
            return mode == "local_external" or bool(
                _cloud_api_key(cfg) or cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", ""))
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        """Install hint for an unavailable local_embedded runtime (is_available() gates
        initialize() out, so the hint it would log never fires; agent_init shows this).

        ``is_available()`` returns False for local modes when the embedded runtime can't be imported, so
        ``initialize()`` — and the hint it would log — is never reached (#7718). Surface the install
        guidance here, where agent_init warns about an unavailable provider.
        """
        try:
            if _load_config().get("mode", "cloud") not in _LOCAL_MODES:
                return ""
        except Exception:
            return ""
        available, reason = _check_local_runtime()
        return "" if available else _local_runtime_hint(reason).strip()

    def save_config(self, values, hermes_home):
        """Merge *values* into $HERMES_HOME/hindsight/config.json."""
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if config_path.exists():
            with contextlib.suppress(Exception):
                existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard — installs only the deps needed for the selected mode."""
        from .setup import run_setup
        run_setup(self, hermes_home, config)

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Connection mode", "default": "cloud", "choices": ["cloud", "local_embedded", "local_external"]},
            # Cloud mode
            {"key": "api_url", "description": "Hindsight Cloud API URL", "default": _DEFAULT_API_URL, "when": {"mode": "cloud"}},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://ui.hindsight.vectorize.io", "when": {"mode": "cloud"}},
            # Local external mode
            {"key": "api_url", "description": "Hindsight API URL", "default": _DEFAULT_LOCAL_URL, "when": {"mode": "local_external"}},
            {"key": "api_key", "description": "API key (optional)", "secret": True, "env_var": "HINDSIGHT_API_KEY", "when": {"mode": "local_external"}},
            # Local embedded mode
            {"key": "llm_provider", "description": "LLM provider", "default": "openai", "choices": ["openai", "anthropic", "gemini", "groq", "openrouter", "minimax", "ollama", "lmstudio", "openai_compatible"], "when": {"mode": "local_embedded"}},
            {"key": "llm_base_url", "description": "Endpoint URL (e.g. http://192.168.1.10:8080/v1)", "default": "", "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"}},
            {"key": "llm_api_key", "description": "LLM API key (optional for openai_compatible)", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY", "when": {"mode": "local_embedded"}},
            {"key": "llm_model", "description": "LLM model", "default": "gpt-4o-mini", "default_from": {"field": "llm_provider", "map": _PROVIDER_DEFAULT_MODELS}, "when": {"mode": "local_embedded"}},
            {"key": "bank_id", "description": "Memory bank name (static fallback when bank_id_template is unset)", "default": "hermes"},
            {"key": "bank_id_template", "description": "Optional template to derive bank_id dynamically. Placeholders: {profile}, {workspace}, {platform}, {user}, {session}. Example: hermes-{profile}", "default": ""},
            {"key": "bank_mission", "description": "Mission/purpose description for the memory bank"},
            {"key": "bank_retain_mission", "description": "Custom extraction prompt for memory retention"},
            {"key": "recall_budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "memory_mode", "description": "Memory integration mode", "default": "hybrid", "choices": ["hybrid", "context", "tools"]},
            {"key": "recall_prefetch_method", "description": "Auto-recall method", "default": "recall", "choices": ["recall", "reflect"]},
            {"key": "retain_tags", "description": "Default tags applied to retained memories (comma-separated)", "default": ""},
            {"key": "observation_scopes", "description": "How observations are scoped during consolidation: 'combined' (default — one pass over all tags), 'per_tag' (one isolated observation per tag), 'all_combinations' (every tag subset — expensive), or a JSON list of tag-lists for explicit custom scopes. Empty uses Hindsight's 'combined' default.", "default": ""},
            {"key": "retain_source", "description": "Metadata source value attached to retained memories (identifies the client that stored them)", "default": _DEFAULT_RETAIN_SOURCE},
            {"key": "retain_user_prefix", "description": "Label used before user turns in retained transcripts", "default": "User"},
            {"key": "retain_assistant_prefix", "description": "Label used before assistant turns in retained transcripts", "default": "Assistant"},
            {"key": "recall_tags", "description": "Tags to filter when searching memories (comma-separated)", "default": ""},
            {"key": "recall_tags_match", "description": "Tag matching mode for recall", "default": "any", "choices": ["any", "all", "any_strict", "all_strict"]},
            {"key": "recall_types", "description": "Fact types to surface on recall — applies to both auto-recall and the hindsight_recall tool (comma-separated or list). Defaults to observation-only — observations are Hindsight's consolidated, deduplicated, evidence-grounded knowledge layer; raw world/experience facts are the supporting evidence observations already summarize. Set to e.g. 'observation,world,experience' to also include raw facts.", "default": "observation"},
            {"key": "auto_recall", "description": "Automatically recall memories before each turn", "default": True},
            {"key": "recall_sync", "description": "Recall synchronously against the current message before each turn (higher relevance, adds recall latency to the turn). Default off: recall runs in the background and is injected on the next turn.", "default": False},
            {"key": "recall_indicator", "description": "Show a '👁️ Hindsight — recalled N memories' status line when auto-recall injects memory (turn off for customer-facing agents)", "default": True},
            {"key": "retain_indicator", "description": "Show a '👁️ Hindsight — saving to memory…' status line when a turn is saved to memory (turn off for customer-facing agents)", "default": True},
            {"key": "auto_retain", "description": "Automatically retain conversation turns", "default": True},
            {"key": "retain_every_n_turns", "description": "Retain every N turns (1 = every turn)", "default": 1},
            {"key": "retain_async","description": "Process retain asynchronously on the Hindsight server", "default": True},
            {"key": "prefetch_waits_for_retain", "description": "Have the background next-turn prefetch wait for the just-completed retain to become recall-visible on the server (local queue drain + async operation completion) before recalling, so recall includes the just-completed turn (runs off the reply path, adds no response latency)", "default": True},
            {"key": "prefetch_retain_drain_timeout", "description": "Max seconds the background prefetch waits for the retain to become recall-visible (queue drain + server-side completion) before recalling anyway", "default": 10.0},
            {"key": "retain_context", "description": "Context label for retained memories", "default": "conversation between Hermes Agent and the User"},
            {"key": "recall_max_tokens", "description": "Maximum tokens for recall results", "default": 4096},
            {"key": "recall_max_results", "description": "Ranked auto-recall result cap (0 keeps all)", "default": 0},
            {"key": "recall_live_status_bypass", "description": "Opt-in English/German phrase heuristic to skip live-status auto-recall", "default": False},
            {"key": "recall_simple_budget", "description": "Optional budget for short auto-recall queries", "default": "", "choices": ["", "low", "mid", "high"]},
            {"key": "recall_simple_max_words", "description": "Short-query word limit (0 disables)", "default": 0},
            {"key": "recall_document_tags", "description": "Base tags for optional document filtering", "default": []},
            {"key": "recall_document_terms", "description": "Phrases enabling document filtering", "default": []},
            {"key": "recall_document_tag_routes", "description": "Additional document tag to phrase-list mapping", "default": {}},
            {"key": "recall_document_types", "description": "Fact types for document-filtered auto-recall", "default": ["world", "observation"]},
            {"key": "recall_max_input_chars", "description": "Maximum input query length for auto-recall", "default": 800},
            {"key": "recall_prompt_preamble", "description": "Custom preamble for recalled memories in context"},
            {"key": "timeout", "description": "API request timeout in seconds", "default": _DEFAULT_TIMEOUT},
            {"key": "idle_timeout", "description": "Embedded daemon idle timeout in seconds (0 disables auto-shutdown)", "default": _DEFAULT_IDLE_TIMEOUT, "when": {"mode": "local_embedded"}},
            {"key": "port_health_grace_timeout", "description": "Seconds to wait for a slow daemon /health before treating it as stale (raise on busy/low-resource hosts; blank uses the 30s default)", "default": "", "when": {"mode": "local_embedded"}},
        ]

    # -- client -------------------------------------------------------------

    def _int_setting(self, key: str, env_var: str, default: int, env_default=None) -> int:
        """Config value if set (explicit 0 preserved), else env var, else default."""
        value = self._config.get(key)
        return _parse_int_setting(os.environ.get(env_var, env_default) if value is None else value, default)

    def _new_embedded_client(self):
        available, reason = _check_local_runtime()
        if not available:
            raise RuntimeError("Hindsight local runtime is unavailable" + (f": {reason}" if reason else ""))
        _ensure_client_dependency()
        from hindsight import HindsightEmbedded
        HindsightEmbedded.__del__ = lambda self: None
        cfg = self._config
        llm_provider = _daemon_llm_provider(cfg.get("llm_provider", ""))
        logger.debug("Creating HindsightEmbedded client (profile=%s, provider=%s)",
                     cfg.get("profile", "hermes"), llm_provider)
        self._idle_timeout = self._int_setting(
            "idle_timeout", "HINDSIGHT_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT, env_default=self._idle_timeout,
        )
        kwargs = dict(profile=cfg.get("profile", "hermes"), llm_provider=llm_provider,
                      llm_api_key=_embedded_llm_api_key(cfg), llm_model=cfg.get("llm_model", ""),
                      idle_timeout=self._idle_timeout)
        if self._llm_base_url:
            kwargs["llm_base_url"] = self._llm_base_url
        return HindsightEmbedded(**kwargs)

    def _new_cloud_client(self):
        _ensure_client_dependency()
        from hindsight_client import Hindsight
        kwargs = {"base_url": self._api_url, "timeout": float(self._timeout or _DEFAULT_TIMEOUT)}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        logger.debug("Creating Hindsight cloud client (url=%s, has_key=%s, timeout=%s)",
                     self._api_url, bool(self._api_key), kwargs["timeout"])
        return Hindsight(**kwargs)

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            self._client = self._new_embedded_client() if self._mode == "local_embedded" else self._new_cloud_client()
        return self._client

    def _run_sync(self, coro):
        """Schedule *coro* on the shared loop using the configured timeout."""
        return _run_sync(coro, timeout=self._timeout)

    def _run_hindsight_operation(self, operation):
        """Run an async client operation; for local_embedded, a stale-daemon
        connection failure recreates the client and retries once."""
        try:
            return self._run_sync(operation(self._get_client()))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}".lower()
            if self._mode != "local_embedded" or not any(m in text for m in _RETRIABLE_CONNECTION_MARKERS):
                raise
            logger.info("Hindsight embedded daemon appears unreachable; recreating client and retrying once: %s", exc)
            self._client = None
            self._client = client = self._get_client()
            return self._run_sync(operation(client))

    # -- retain writer thread + server-side visibility -------------------------

    def _ensure_writer(self) -> None:
        """Lazy-start the single retain-writer thread (tools-only providers never pay for it)."""
        if (thread := self._writer_thread) is not None and thread.is_alive():
            return
        # A previous writer may have exited after shutdown(); allow the fresh one to drain.
        self._shutting_down.clear()
        thread = _context_thread(self._writer_loop, "hindsight-writer")
        self._writer_thread = self._sync_thread = thread
        thread.start()

    def _register_atexit(self) -> None:
        """Idempotent atexit drain: a CLI exit that skips MemoryManager.shutdown_all()
        must not race interpreter teardown."""
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self._atexit_shutdown)

    def _writer_loop(self) -> None:
        """Drain the retain queue serially until the sentinel. A failing job can't
        kill the writer; task_done() always fires so queue.join() works."""
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                job()
            except Exception as exc:
                logger.warning("Hindsight retain failed: %s", exc, exc_info=True)
            finally:
                self._retain_queue.task_done()

    def _atexit_shutdown(self) -> None:
        try:
            if not self._shutting_down.is_set():
                self.shutdown()
        except Exception as exc:
            logger.debug("Hindsight atexit shutdown failed: %s", exc)

    def _track_retain_ops(self, retain_response, bank_id: str) -> None:
        """Record the async ``operation_id``/``operation_ids`` of an aretain_batch reply
        (pending until recall-visible). No id (older API / sync completion) leaves
        only the local queue drain as a signal."""
        raw_ids = [getattr(retain_response, "operation_id", None), *(getattr(retain_response, "operation_ids", None) or [])]
        if ids := [str(op) for op in raw_ids if op]:
            self._retain_ops_bank_id = bank_id
            with self._pending_retain_ops_lock:
                self._pending_retain_ops.update(ids)

    def _is_retain_op_complete(self, bank_id: str, op_id: str) -> bool:
        """True when a server-side retain op is done or gone (completed ops are evicted,
        so 404 = no longer pending). Transient errors -> False, caller keeps waiting."""
        from hindsight_client_api.exceptions import NotFoundException

        try:
            resp = self._run_hindsight_operation(
                lambda client: client.operations.get_operation_status(bank_id=bank_id, operation_id=op_id)
            )
        except NotFoundException:
            return True
        except Exception as exc:
            logger.debug("Prefetch: operation status check failed for %s: %s", op_id, exc)
            return False
        return str(getattr(resp, "status", "") or "").lower() in {"completed", "failed"}

    def _wait_for_retains_drained(self, timeout: float) -> bool:
        """Block up to *timeout* s for the last retain to become recall-visible
        (prefetch thread only, never the reply path). Two barriers on one budget:
        (1) the writer queue drains — polls ``unfinished_tasks`` rather than
        ``queue.join()`` so a wedged write can't hang the prefetch; (2) the
        server-side async ops complete (async retain returns on acceptance, not
        durability). False on timeout/shutdown."""
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        expired = lambda: deadline is not None and time.monotonic() >= deadline  # noqa: E731
        while self._retain_queue.unfinished_tasks > 0:
            if self._shutting_down.is_set():
                return False
            if expired():
                logger.debug("Prefetch: retain drain timed out after %.1fs (%d pending)",
                             timeout, self._retain_queue.unfinished_tasks)
                return False
            time.sleep(0.05)
        return self._wait_for_server_retain_ops(expired, timeout)

    def _wait_for_server_retain_ops(self, _expired: Callable[[], bool], timeout: float) -> bool:
        """Poll tracked async retain ops until complete or *_expired()* (deadline
        predicate). Ops still pending at the deadline are DROPPED: keeping them
        would let a permanently failing status endpoint burn the full timeout on
        EVERY later prefetch (a per-turn latency penalty via prefetch()'s bounded
        join). Trades a possibly-stale recall for liveness; WARNING once per prefetch."""
        while True:
            with self._pending_retain_ops_lock:
                bank_id = self._retain_ops_bank_id or self._bank_id
                pending = list(self._pending_retain_ops)
            if not pending:
                return True
            if self._shutting_down.is_set():
                return False
            done: set[str] = set()
            for op_id in pending:
                if self._shutting_down.is_set():
                    return False
                if _expired():
                    break
                if self._is_retain_op_complete(bank_id, op_id):
                    done.add(op_id)
            with self._pending_retain_ops_lock:
                self._pending_retain_ops.difference_update(done)
                if not self._pending_retain_ops:
                    return True
                dropped = len(self._pending_retain_ops) if _expired() else 0
                if dropped:
                    self._pending_retain_ops.clear()
            if dropped:
                logger.warning("Prefetch: server retain visibility timed out after %.1fs; "
                               "dropping %d unresolved op(s) so later prefetches stay "
                               "bounded (recall may miss the just-completed turn)", timeout, dropped)
                return False
            time.sleep(self._RETAIN_OP_POLL_INTERVAL_S)

    # -- retain target -----------------------------------------------------------

    def _resolve_retain_target(self, fallback_document_id: str) -> tuple[str, str | None]:
        """(document_id, update_mode) from live API capability: >= 0.5.0 reuses the
        stable session-scoped id with ``update_mode='append'``; older APIs get
        *fallback_document_id* (per-process unique) and no update_mode — the only
        way the resume-overwrite fix works there. The /version probe targets the
        embedded client's dynamic per-profile port when running, else api_url.

        On Hindsight ≥ 0.5.0 the API supports ``update_mode='append'``, which lets us reuse a stable
        session-scoped ``document_id`` across process lifecycles without overwriting prior turns. On older
        APIs we fall back to *fallback_document_id* (the per-process unique ``f"{session_id}-{start_ts}"``
        minted at initialize / switch time) and don't pass ``update_mode`` at all — that's the only way the
        resume-overwrite fix (#6654) keeps working on legacy servers.
        """
        url = getattr(self._client, "url", None) if self._mode == "local_embedded" else None
        probe_url = str(url) if url else (self._api_url or "")
        if self._session_id and _check_api_supports_update_mode_append(probe_url, self._api_key):
            return self._session_id, "append"
        return fallback_document_id, None

    # -- lifecycle ---------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "").strip()
        self._parent_session_id = str(kwargs.get("parent_session_id", "") or "").strip()
        # Status channel for the retain indicator (recall reports via recall_status()).
        if callable(kwargs.get("status_callback")):
            self._status_callback = kwargs["status_callback"]
        # session_id stays in tags so processes for one session remain filterable together.
        self._document_id = _mint_document_id(self._session_id)
        _maybe_upgrade_client()

        self._config = cfg = _load_config()
        for name in _SESSION_KWARGS:
            setattr(self, f"_{name}", str(kwargs.get(name) or "").strip())
        self._turn_index = self._last_retained_turn_count = 0
        self._session_turns = []
        self._mode = cfg.get("mode", "cloud")
        self._timeout = self._int_setting("timeout", "HINDSIGHT_TIMEOUT", _DEFAULT_TIMEOUT)
        self._idle_timeout = self._int_setting("idle_timeout", "HINDSIGHT_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT)
        if self._mode == "local":  # legacy alias
            self._mode = "local_embedded"
        if self._mode == "local_embedded":
            # Must precede the daemon_embed_manager import, which reads it at import time.
            _export_port_health_grace_timeout(cfg)
            available, reason = _check_local_runtime()
            if not available:
                logger.warning("Hindsight local mode disabled because its runtime could not be imported: %s.%s",
                               reason, _local_runtime_hint(reason))
                self._mode = "disabled"
                return
        self._apply_connection_settings(cfg)
        self._apply_retain_settings(cfg)
        self._apply_recall_settings(cfg)

        client_version = "unknown"
        with contextlib.suppress(Exception):
            from importlib.metadata import version as pkg_version
            client_version = pkg_version("hindsight-client")
        logger.info("Hindsight initialized: mode=%s, api_url=%s, bank=%s, budget=%s, memory_mode=%s, prefetch_method=%s, client=%s",
                    self._mode, self._api_url, self._bank_id, self._budget, self._memory_mode, self._prefetch_method, client_version)
        if self._bank_id_template:
            logger.debug("Hindsight bank resolved from template %r: profile=%s workspace=%s platform=%s user=%s -> bank=%s",
                         self._bank_id_template, self._agent_identity, self._agent_workspace,
                         self._platform, self._user_id, self._bank_id)
        logger.debug("Hindsight config: auto_retain=%s, auto_recall=%s, retain_every_n=%d, "
                     "retain_async=%s, retain_context=%s, recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, recall_tags=%s",
                     self._auto_retain, self._auto_recall, self._retain_every_n_turns,
                     self._retain_async, self._retain_context, self._recall_max_tokens, self._recall_max_input_chars,
                     self._tags, self._recall_tags)

        if self._mode == "local_embedded":
            self._start_embedded_daemon()

    def _apply_connection_settings(self, cfg: dict) -> None:
        """Endpoint, bank and mode selectors from *cfg* (env fallbacks where documented)."""
        self._api_key = _cloud_api_key(cfg)
        default_url = _DEFAULT_LOCAL_URL if self._mode in {"local_embedded", "local_external"} else _DEFAULT_API_URL
        self._api_url = cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        self._llm_base_url = cfg.get("llm_base_url", "")

        banks = cfg_get(cfg, "banks", "hermes", default={})
        self._bank_id_template = cfg.get("bank_id_template", "") or ""
        self._bank_id = _resolve_bank_id_template(
            self._bank_id_template,
            fallback=cfg.get("bank_id") or banks.get("bankId", "hermes"),
            profile=self._agent_identity, workspace=self._agent_workspace,
            platform=self._platform, user=self._user_id, session=self._session_id,
        )
        budget = cfg.get("recall_budget") or cfg.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"
        memory_mode = cfg.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in _SYSTEM_PROMPT_TAILS else "hybrid"
        prefetch_method = cfg.get("recall_prefetch_method") or cfg.get("prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in {"recall", "reflect"} else "recall"
        self._bank_mission = cfg.get("bank_mission", "")
        self._bank_retain_mission = cfg.get("bank_retain_mission") or None

    def _apply_retain_settings(self, cfg: dict) -> None:
        def _cfg_or_env(key: str, env_var: str, default: str = "") -> Any:
            return cfg.get(key) or os.environ.get(env_var, default)

        self._retain_tags = _normalize_retain_tags(_cfg_or_env("retain_tags", "HINDSIGHT_RETAIN_TAGS"))
        self._tags = self._retain_tags or None
        self._observation_scopes = _normalize_observation_scopes(
            _cfg_or_env("observation_scopes", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES"))
        self._retain_source = str(_cfg_or_env("retain_source", "HINDSIGHT_RETAIN_SOURCE", _DEFAULT_RETAIN_SOURCE)).strip()
        self._retain_user_prefix = str(_cfg_or_env("retain_user_prefix", "HINDSIGHT_RETAIN_USER_PREFIX", "User")).strip() or "User"
        self._retain_assistant_prefix = (
            str(_cfg_or_env("retain_assistant_prefix", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant")).strip()
            or "Assistant"
        )
        self._apply_retain_policy(cfg)

    def _apply_retain_policy(self, cfg: dict) -> None:
        """Pure-config retain knobs (no env/secret reads; ``{}`` yields the defaults)."""
        self._auto_retain = cfg.get("auto_retain", True)
        self._retain_every_n_turns = max(1, int(cfg.get("retain_every_n_turns", 1)))
        self._retain_context = cfg.get("retain_context", _RETAIN_CONTEXT_DEFAULT)
        self._retain_async = cfg.get("retain_async", True)
        # On by default so the user SEES memory working whether or not the model
        # mentions it; off switch for customer-facing agents (recall_indicator too).
        self._retain_indicator = bool(cfg.get("retain_indicator", True))
        # The next turn's warm prefetch could read BEFORE an async retain is
        # recall-visible; when True it first waits (bounded, off the reply path)
        # for the queue to drain AND the server-side op(s) to complete.
        self._prefetch_waits_for_retain = cfg.get("prefetch_waits_for_retain", True)
        self._prefetch_retain_drain_timeout = float(cfg.get("prefetch_retain_drain_timeout", 10.0))

    def _apply_recall_settings(self, cfg: dict) -> None:
        """Recall knobs are pure config too (``{}`` yields the defaults)."""
        self._recall_tags = cfg.get("recall_tags") or None
        self._recall_tags_match = cfg.get("recall_tags_match", "any")
        self._auto_recall = cfg.get("auto_recall", True)
        self._recall_sync = bool(cfg.get("recall_sync", False))
        self._recall_max_tokens = int(cfg.get("recall_max_tokens", 4096))
        self._recall_max_results = max(0, _parse_int_setting(cfg.get("recall_max_results"), 0))
        self._recall_live_status_bypass = cfg.get("recall_live_status_bypass") is True
        simple_budget = cfg.get("recall_simple_budget", "")
        self._recall_simple_budget = simple_budget if isinstance(simple_budget, str) and simple_budget in _VALID_BUDGETS else ""
        self._recall_simple_max_words = max(0, _parse_int_setting(cfg.get("recall_simple_max_words"), 0))
        self._recall_document_tags = _normalize_retain_tags(cfg.get("recall_document_tags"))
        self._recall_document_terms = _normalize_retain_tags(cfg.get("recall_document_terms"))
        routes = cfg.get("recall_document_tag_routes", {})
        self._recall_document_tag_routes = {
            tag.strip(): _normalize_retain_tags(terms) for tag, terms in routes.items()
            if isinstance(tag, str) and tag.strip()
        } if isinstance(routes, dict) else {}
        types = _normalize_retain_tags(cfg.get("recall_document_types", ["world", "observation"]))
        self._recall_document_types = [t for t in types if t in ("world", "observation", "experience")] or ["world", "observation"]
        self._recall_max_input_chars = int(cfg.get("recall_max_input_chars", 800))
        # None -> observation-only (Hindsight's consolidated, deduplicated layer; raw
        # world/experience facts re-ship the evidence they summarize and burn the
        # recall_max_tokens budget); a comma-separated string is accepted for parity
        # with recall_tags; an explicit list broadens or disables the filter.
        configured_types = cfg.get("recall_types")
        if isinstance(configured_types, str):
            self._recall_types = [t.strip() for t in configured_types.split(",") if t.strip()]
        else:
            self._recall_types = list([] if configured_types is None else configured_types) or ["observation"]
        self._recall_prompt_preamble = cfg.get("recall_prompt_preamble", "")
        self._recall_indicator = bool(cfg.get("recall_indicator", True))

    def _start_embedded_daemon(self) -> None:
        """Start the embedded daemon on a background thread (Rich output -> log file)."""
        # PostgreSQL's initdb refuses root; without this guard the start thread
        # retries forever, reloading embedding models (~958MB RAM, ~33% CPU)
        # with no user-visible error.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            msg = ("Hindsight local_embedded mode cannot run as root "
                   "(PostgreSQL initdb refuses root). Skipping the embedded "
                   "memory daemon. Run Hermes as a non-root user, or switch "
                   "to cloud / local_external mode via 'hermes memory setup'.")
            logger.warning(msg)
            # Also print: otherwise the user would only see Hermes get sluggish.
            with contextlib.suppress(Exception):
                # Surface to the terminal too — a daemon that never starts would otherwise fail silently and
                # the user would only see Hermes get sluggish. (issue #13125)
                print(f"  ⚠ {msg}", file=sys.stderr, flush=True)
            self._mode = "disabled"
            return
        _context_thread(self._daemon_start_worker, "hindsight-daemon-start").start()

    def _daemon_start_worker(self) -> None:
        import traceback
        log_path = get_hermes_home() / "logs" / "hindsight-embed.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _log(text: str) -> None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text)

        try:
            # Rich console -> our log file (redirecting global fds would capture other threads).
            import hindsight_embed.daemon_embed_manager as dem
            from rich.console import Console
            dem.console = Console(file=open(log_path, "a", encoding="utf-8"), force_terminal=False)

            client = self._get_client()
            profile = self._config.get("profile", "hermes")
            # Profile .env out of sync with config -> rewrite and restart a running daemon.
            if _load_simple_env(_embedded_profile_env_path(self._config)) != _build_embedded_profile_env(self._config):
                _materialize_embedded_profile_env(self._config)
                if client._manager.is_running(profile):
                    _log("\n=== Config changed, restarting daemon ===\n")
                    client._manager.stop(profile)
            client._ensure_started()
            _log("\n=== Daemon started successfully ===\n")
        except Exception as e:
            _log(f"\n=== Daemon startup failed: {e} ===\n" + traceback.format_exc())

    def system_prompt_block(self) -> str:
        mode = self._memory_mode if self._memory_mode in _SYSTEM_PROMPT_TAILS else "hybrid"
        label = "" if mode == "hybrid" else f" ({mode} mode)"
        return f"# Hindsight Memory\nActive{label}. Bank: {self._bank_id}, budget: {self._budget}.\n{_SYSTEM_PROMPT_TAILS[mode]}"

    # -- recall ------------------------------------------------------------------

    def _recall_disabled(self) -> bool:
        """Guards shared by the async and synchronous recall paths."""
        why = ("tools-only mode" if self._memory_mode == "tools" else "auto_recall disabled" if not self._auto_recall
               else "shutting down" if self._shutting_down.is_set() else None)
        if why:
            logger.debug("Prefetch: skipped (%s)", why)
        return why is not None

    def _recall(self, query: str, **overrides) -> list:
        kwargs: dict = {"bank_id": self._bank_id, "query": query, "budget": self._budget, "max_tokens": self._recall_max_tokens}
        if self._recall_tags:
            kwargs.update(tags=self._recall_tags, tags_match=self._recall_tags_match)
        if self._recall_types:
            kwargs["types"] = self._recall_types
        kwargs.update(overrides)
        resp = self._run_hindsight_operation(lambda client: client.arecall(**kwargs))
        return resp.results or []

    def _reflect(self, query: str, *, budget: str | None = None) -> str | None:
        resp = self._run_hindsight_operation(
            lambda client: client.areflect(bank_id=self._bank_id, query=query, budget=budget or self._budget)
        )
        return resp.text

    def _do_recall(self, query: str) -> tuple[str, int]:
        """One recall/reflect for *query* (background prefetch and ``recall_sync`` paths)
        -> (text, memory count); the count is 0 for reflect (synthesis) and on error."""
        if self._recall_live_status_bypass and _is_live_status_query(query):
            logger.debug("Recall: skipped by opt-in live-status phrase heuristic")
            return "", 0
        # Classify before truncation: a long/complex question must not become simple
        # merely because its context-injection query is clipped.
        overrides = {}
        if self._recall_simple_budget and _is_simple_recall_query(query, self._recall_simple_max_words):
            overrides["budget"] = self._recall_simple_budget
        document_query = bool(self._recall_document_tags and self._recall_document_terms
                              and _contains_term(query, self._recall_document_terms))
        if self._recall_max_input_chars:
            query = query[:self._recall_max_input_chars]
        try:
            if self._prefetch_method == "reflect":
                logger.debug("Recall: calling reflect (bank=%s, query_len=%d)", self._bank_id, len(query))
                return self._reflect(query, **overrides) or "", 0
            if document_query:
                tags = list(self._recall_document_tags)
                for tag, terms in self._recall_document_tag_routes.items():
                    if _contains_term(query, terms) and tag not in tags:
                        tags.append(tag)
                overrides.update(tags=tags, tags_match="all", types=self._recall_document_types)
            logger.debug("Recall: calling recall (bank=%s, query_len=%d, budget=%s)",
                         self._bank_id, len(query), self._budget)
            results = self._recall(query, **overrides)
            if self._recall_max_results:
                results = results[:self._recall_max_results]
            logger.debug("Recall: returned %d results", len(results))
            return "\n".join(f"- {r.text}" for r in results if r.text), len(results)
        except Exception as e:
            logger.debug("Hindsight recall failed: %s", e, exc_info=True)
            return "", 0

    def _finish_prefetch(self, result: str, count: int) -> str:
        """Record indicator state (cleared on empty turns, never a stale count); format the block."""
        self._last_recall_returned, self._last_recall_count = bool(result), count if result else 0
        if not result:
            logger.debug("Prefetch: no results available")
            return ""
        logger.debug("Prefetch: returning %d chars of context", len(result))
        header = self._recall_prompt_preamble or (
            "# Hindsight Memory (persistent cross-session context)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def _join_prefetch(self, timeout: float, *, log: bool = False) -> None:
        if not (self._prefetch_thread and self._prefetch_thread.is_alive()):
            return
        if log:
            logger.debug("Prefetch: waiting for background thread to complete")
        self._prefetch_thread.join(timeout=timeout)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Opt-in: recall synchronously against the *current* message so the
        # injected memories match this turn's query, not the previous turn's.
        # See NousResearch/hermes-agent#5820.
        if self._recall_sync:
            return self._finish_prefetch(*(("", 0) if self._recall_disabled() else self._do_recall(query)))
        # Default: the background worker's result for the previous turn (capped join).
        self._join_prefetch(3.0, log=True)
        with self._prefetch_lock:
            result, count = self._prefetch_result, self._prefetch_count
            self._prefetch_result, self._prefetch_count = "", 0
        return self._finish_prefetch(result, count)

    def recall_status(self) -> Optional[RecallStatus]:
        """Count injected by the last prefetch; None if nothing injected or ``recall_indicator=false``."""
        if not self._recall_indicator or not self._last_recall_returned:
            return None
        return RecallStatus(provider_label="Hindsight", count=self._last_recall_count, glyph=_HINDSIGHT_GLYPH)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Sync mode recalls live each turn — nothing to prime in the background.
        if self._recall_sync or self._recall_disabled():
            return

        def _run():
            # Wait (bounded, off the reply path) for the just-completed turn's
            # retain to be recall-visible so the warmed context includes it.
            if self._prefetch_waits_for_retain:
                self._wait_for_retains_drained(self._prefetch_retain_drain_timeout)
            text, count = self._do_recall(query)
            if text:
                with self._prefetch_lock:
                    self._prefetch_result, self._prefetch_count = text, count

        self._prefetch_thread = _context_thread(_run, "hindsight-prefetch")
        self._prefetch_thread.start()

    # -- retain ------------------------------------------------------------------

    def _build_turn_messages(self, user_content: str, assistant_content: str) -> List[Dict[str, str]]:
        now = _event_timestamp()  # one turn -> both messages share the event timestamp
        return [{"role": role, "content": f"{prefix}: {content}", "timestamp": now} for role, prefix, content in
                (("user", self._retain_user_prefix, user_content), ("assistant", self._retain_assistant_prefix, assistant_content))]

    def _build_metadata(self, *, message_count: int, turn_index: int) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            # UTC write/audit time (event time lives on the item timestamp).
            "retained_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "message_count": str(message_count),
            "turn_index": str(turn_index),
        }
        if self._retain_source:
            metadata["source"] = self._retain_source
        metadata.update({name: value for name in _METADATA_ATTRS if (value := getattr(self, f"_{name}"))})
        return metadata

    def _build_retain_kwargs(self, content: str, *, context: str | None = None,
                             metadata: Dict[str, str] | None = None, tags: List[str] | None = None,
                             occurred_at: str | None = None, update_mode: str | None = None) -> Dict[str, Any]:
        """Build one aretain_batch item. The server resolves occurred_start/end (incl.
        relative phrases in content) from the item timestamp: explicit occurred_at
        wins, else the configured event clock."""
        item: Dict[str, Any] = {
            # See #93568.
            "content": content,
            "metadata": metadata or self._build_metadata(message_count=1, turn_index=self._turn_index),
            "timestamp": (occurred_at or "").strip() or _event_timestamp(),
        }
        merged_tags = _normalize_retain_tags(list(self._retain_tags) + _normalize_retain_tags(tags))
        item.update({k: v for k, v in (("context", context), ("update_mode", update_mode)) if v is not None})
        item.update({k: v for k, v in (("tags", merged_tags), ("observation_scopes", self._observation_scopes)) if v})
        return item

    def _retain_batch(self, item: dict, *, bank_id: str, document_id: str | None = None,
                      retain_async: bool | None = None):
        """Dispatch one item via aretain_batch (bank_id/document_id/retain_async are
        call-level args, never item keys)."""
        kwargs: Dict[str, Any] = {"bank_id": bank_id, "items": [item], "document_id": document_id, "retain_async": retain_async}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self._run_hindsight_operation(lambda client: client.aretain_batch(**kwargs))

    def _make_turn_retain_job(self, turns: list[str], *, document_id: str, update_mode: str | None,
                              label: str, track_ops: bool = True) -> Callable[[], None]:
        """Writer job shipping *turns* as one document. Inputs are snapshotted NOW: the
        writer runs after later sync_turn() calls mutate _session_turns/_turn_index/_session_id."""
        content = "[" + ",".join(turns) + "]"
        metadata = self._build_metadata(message_count=len(turns) * 2, turn_index=self._turn_index)
        lineage = (("session", self._session_id), ("parent", self._parent_session_id))
        tags = [f"{kind}:{sid}" for kind, sid in lineage if sid] or None
        bank_id, retain_async, retain_context = self._bank_id, self._retain_async, self._retain_context

        def _job() -> None:
            item = self._build_retain_kwargs(content, context=retain_context, metadata=metadata,
                                             tags=tags, update_mode=update_mode)
            logger.debug("Hindsight %s: bank=%s, doc=%s, mode=%s, async=%s, content_len=%d, num_turns=%d",
                         label, bank_id, document_id, update_mode, retain_async, len(content), len(turns))
            resp = self._retain_batch(item, bank_id=bank_id, document_id=document_id, retain_async=retain_async)
            # Async retains are only *accepted* here; track the op id(s) so the
            # next-turn prefetch can wait for true server-side completion.
            if retain_async and track_ops:
                self._track_retain_ops(resp, bank_id)
            logger.debug("Hindsight %s succeeded", label)

        return _job

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Enqueue a retain for the current turn (non-blocking; writer thread). Dropped
        once shutdown() fired so post-exit retains never reach aiohttp during teardown."""
        why = "auto_retain disabled" if not self._auto_retain else "shutting down" if self._shutting_down.is_set() else None
        if why:
            logger.debug("sync_turn: skipped (%s)", why)
            return
        if session_id:
            self._session_id = str(session_id).strip()

        self._session_turns.append(json.dumps(self._build_turn_messages(user_content, assistant_content), ensure_ascii=False))
        self._turn_counter = self._turn_index = self._turn_counter + 1
        if remainder := self._turn_counter % self._retain_every_n_turns:
            logger.debug("sync_turn: buffered turn %d (will retain at turn %d)",
                         self._turn_counter, self._turn_counter + (self._retain_every_n_turns - remainder))
            return

        document_id, update_mode = self._resolve_retain_target(self._document_id)
        # Append-capable APIs get only the delta since the last retain; legacy /
        # overwrite APIs need the whole session because each retain replaces the document.
        start = self._last_retained_turn_count if update_mode == "append" else 0
        turns_to_retain = self._session_turns[start:]
        if not turns_to_retain:
            logger.debug("sync_turn: skipped append retain; no new turns since last retain")
            return
        logger.debug("sync_turn: retaining %d/%d turns, payload %d chars",
                     len(turns_to_retain), len(self._session_turns), sum(len(t) for t in turns_to_retain))

        job = self._make_turn_retain_job(turns_to_retain, document_id=document_id,
                                         update_mode=update_mode, label="retain")
        # Indicator fires only past every skip/buffer gate: solely on turns that persist.
        # Model-independent status line; no-op without retain_indicator/status channel.
        if self._retain_indicator and self._status_callback is not None:
            try:
                self._status_callback(f"{_HINDSIGHT_GLYPH} Hindsight — saving to memory…")
            except Exception:
                logger.debug("Retain indicator emit failed (non-fatal)", exc_info=True)
        self._enqueue_retain(job)
        # Advance the watermark only after the delta is queued so a later retain
        # doesn't re-ship turns already handed to the writer.
        if update_mode == "append":
            self._last_retained_turn_count = len(self._session_turns)

    def _enqueue_retain(self, job: Callable[[], None]) -> None:
        """Hand *job* to the (lazily started) writer and arm the atexit drain."""
        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(job)

    # -- tools -------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [] if self._memory_mode == "context" else [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    def _tool_retain(self, args: dict) -> str:
        content, context = args["content"], args.get("context")
        item = self._build_retain_kwargs(content, context=context, tags=args.get("tags"),
                                         occurred_at=args.get("occurred_at"))
        logger.debug("Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                     self._bank_id, len(content), context)
        self._retain_batch(item, bank_id=self._bank_id)
        logger.debug("Tool hindsight_retain: success")
        return "Memory stored successfully."

    def _tool_recall(self, args: dict) -> str:
        query = args["query"]
        logger.debug("Tool hindsight_recall: bank=%s, query_len=%d, budget=%s",
                     self._bank_id, len(query), self._budget)
        results = self._recall(query)
        logger.debug("Tool hindsight_recall: %d results", len(results))
        return "\n".join(f"{i}. {r.text}" for i, r in enumerate(results, 1)) or "No relevant memories found."

    def _tool_reflect(self, args: dict) -> str:
        query = args["query"]
        logger.debug("Tool hindsight_reflect: bank=%s, query_len=%d, budget=%s",
                     self._bank_id, len(query), self._budget)
        text = self._reflect(query) or ""
        logger.debug("Tool hindsight_reflect: response_len=%d", len(text))
        return text or "No relevant memories found."

    # tool name -> (required arg, handler, user-facing failure prefix)
    _TOOL_HANDLERS = {
        "hindsight_retain": ("content", _tool_retain, "Failed to store memory"),
        "hindsight_recall": ("query", _tool_recall, "Failed to search memory"),
        "hindsight_reflect": ("query", _tool_reflect, "Failed to reflect"),
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        required, handler, failure = self._TOOL_HANDLERS[tool_name]
        if not args.get(required, ""):
            return tool_error(f"Missing required parameter: {required}")
        try:
            return json.dumps({"result": handler(self, args)})
        except Exception as e:
            logger.warning("%s failed: %s", tool_name, e, exc_info=True)
            return tool_error(f"{failure}: {e}")

    # -- session lifecycle -------------------------------------------------------

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, **kwargs) -> None:
        """Rotate per-session state (/resume, /branch, /reset, /new, compression) so
        writes don't land in the previous session's document. Always: flush buffered
        turns under the OLD ids first (``retain_every_n_turns > 1`` would silently
        lose them), join the in-flight prefetch and drop its result (no stale recall
        for the new session), then set ``_session_id``, mint a fresh ``_document_id``
        and clear the batch buffers. ``reset`` is accepted but unneeded: buffer
        clearing is correct for every switch.

        Without this hook, initialize()-cached state (``_session_id``, ``_document_id``, ``_session_turns``,
        ``_turn_counter``) would keep pointing at the previous session and writes would land in the wrong
        document. See hermes-agent#6672.
        Always update ``_session_id`` so metadata and tags on subsequent retains reflect the active session.
        Always clear the accumulated batch buffers (``_session_turns``, ``_turn_counter``, ``_turn_index``)
        — even for /resume and /branch, the new session's batching must start from zero so an in-flight
        retain doesn't flush under the wrong ``_document_id``. See #1303.
        """
        new_id = str(new_session_id or "").strip()
        if not new_id:
            return

        # 1. Flush buffered turns under the OLD identifiers, resolved BEFORE the
        # rotation (legacy: per-process unique; >=0.5.0: session-scoped + append).
        if self._session_turns:
            old_document_id, old_update_mode = self._resolve_retain_target(self._document_id)
            job = self._make_turn_retain_job(list(self._session_turns), document_id=old_document_id,
                                             update_mode=old_update_mode, label="flush-on-switch",
                                             track_ops=False)

            def _flush():
                try:
                    job()
                except Exception as e:
                    logger.warning("Hindsight flush-on-switch failed: %s", e, exc_info=True)
            # Same writer queue as sync_turn: FIFO behind queued old-session retains,
            # no two threads racing aretain_batch on one document, shutdown drain intact.
            if not self._shutting_down.is_set():
                self._enqueue_retain(_flush)

        # 2. Drain the old session's in-flight prefetch and drop its result.
        self._join_prefetch(3.0)
        with self._prefetch_lock:
            self._prefetch_result = ""

        # 3. Rotate to the new session.
        if parent_session_id:
            self._parent_session_id = str(parent_session_id).strip()
        self._session_id, self._document_id = new_id, _mint_document_id(new_id)
        self._session_turns = []
        self._turn_counter = self._turn_index = self._last_retained_turn_count = 0
        logger.debug("Hindsight on_session_switch: new_session=%s parent=%s reset=%s doc=%s",
                     self._session_id, self._parent_session_id, reset, self._document_id)

    def _close_client(self) -> None:
        if self._mode != "local_embedded":
            self._run_sync(self._client.aclose())
            return
        # HindsightEmbedded.close() closes its sync client from this thread ("attached
        # to a different loop" before aiohttp releases the session): aclose the inner
        # client on the shared loop first, then let the wrapper clean up bookkeeping.
        inner_client = getattr(self._client, "_client", None)
        if inner_client is not None and hasattr(inner_client, "aclose"):
            _run_sync(inner_client.aclose())
            with contextlib.suppress(Exception):
                self._client._client = None
        with contextlib.suppress(RuntimeError):
            self._client.close()

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: stopping writer + waiting for background threads")
        # Stop accepting retain jobs first so late sync_turn() calls are dropped.
        self._shutting_down.set()
        # The writer finishes in-flight work then exits on the sentinel; the
        # bounded join keeps shutdown predictable even if the daemon is wedged.
        if (writer := self._writer_thread) is not None and writer.is_alive():
            self._retain_queue.put(_WRITER_SENTINEL)
            writer.join(timeout=10.0)
            if writer.is_alive():
                logger.warning("Hindsight writer did not stop within 10s; abandoning %d pending retain(s)",
                               self._retain_queue.qsize())
        self._join_prefetch(5.0)
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._close_client()
            self._client = None
        # The module-global loop is intentionally NOT stopped: it's shared by every
        # provider in the process (one per gateway chat session); stopping it would
        # strand siblings' aiohttp sessions ("Unclosed client session"). Daemon
        # thread, reclaimed at process exit.


# The module-global background event loop (_loop / _loop_thread) is intentionally NOT stopped here. It is
# shared across every HindsightMemoryProvider instance in the process — the plugin loader creates a new
# provider per AIAgent, and the gateway creates one AIAgent per concurrent chat session. Stopping the loop
# from one provider's shutdown() strands the aiohttp ClientSession + TCPConnector owned by every sibling
# provider on a dead loop, which surfaces as the "Unclosed client session" / "Unclosed connector" warnings
# reported in #11923. The loop runs on a daemon thread and is reclaimed on process exit; per-session cleanup
# happens via self._client.aclose() above.
def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import dataclass  # noqa: F401,E402
import importlib  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
