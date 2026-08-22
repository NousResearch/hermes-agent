"""Models.dev registry integration — primary database for providers and models.

Fetches from https://models.dev/api.json — a community-maintained database
of 4000+ models across 109+ providers.  Provides:

- **Provider metadata**: name, base URL, env vars, documentation link
- **Model metadata**: context window, max output, cost/M tokens, capabilities
  (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
  open-weights flag, family grouping, deprecation status

Data resolution order:
  1. In-memory cache (fresh, or stale served immediately while a single
     background daemon thread refreshes)
  2. Disk cache (~/.hermes/models_dev_cache.json — any age; stale data is
     served rather than blocking callers on the network)
  3. Network fetch (https://models.dev/api.json) — only when no cache
     exists at all; failed refreshes back off for 5 minutes process-wide

Network hardening:

- **ETag conditional GET**: network refreshes send ``If-None-Match``
  with the last-known ETag whenever a servable registry is held (memory,
  hydrated from disk on cold force-refresh). A 304 Not Modified response
  is a no-op — the existing cache is re-confirmed fresh without
  re-downloading the full registry (≈2 MB). The ETag is persisted
  atomically alongside the cache file.
- **No-network-on-hot-paths invariant**: resolution, picker, and resume
  paths NEVER perform network I/O. ``allow_network=False`` is threaded
  through every query function, and hot-path callers (vision routing,
  image routing, cost guard, context-length lookup) pass it explicitly.
- **Corrupt-cache rejection**: a disk cache that fails to parse, is not a
  dict, or is empty is ignored with a warning rather than served as
  ``{}`` and silently breaking provider/model resolution.
- **Mirror URL override**: ``models_dev.url`` in config.yaml lets
  deployments point at a mirror (e.g. a self-hosted copy) without code
  changes.

Other modules should import the dataclasses and query functions from here
rather than parsing the raw JSON themselves.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_json_write

import requests

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 4 * 3600  # 4 hours — ETag conditional GET makes refresh cheap
_MODELS_DEV_RETRY_DELAY = 300  # 5 minutes after a failed refresh

# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0
_models_dev_retry_after: float = 0
_models_dev_fetch_lock = threading.Lock()
_models_dev_refresh_lock = threading.Lock()
_models_dev_refresh_in_flight = False


# ---------------------------------------------------------------------------
# Dataclasses — rich metadata for providers and models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Full metadata for a single model from models.dev."""

    id: str
    name: str
    family: str
    provider_id: str        # models.dev provider ID (e.g. "anthropic")

    # Capabilities
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False       # supports image/file attachments (vision)
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False

    # Modalities
    input_modalities: Tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: Tuple[str, ...] = ()

    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None

    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None

    # Metadata
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""          # "alpha", "beta", "deprecated", or ""
    interleaved: Any = False  # True or {"field": "reasoning_content"}

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_cost(self) -> str:
        """Human-readable cost string, e.g. '$3.00/M in, $15.00/M out'."""
        if not self.has_cost_data():
            return "unknown"
        parts = [f"${self.cost_input:.2f}/M in", f"${self.cost_output:.2f}/M out"]
        if self.cost_cache_read is not None:
            parts.append(f"cache read ${self.cost_cache_read:.2f}/M")
        return ", ".join(parts)

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        caps = []
        if self.reasoning:
            caps.append("reasoning")
        if self.tool_call:
            caps.append("tools")
        if self.supports_vision():
            caps.append("vision")
        if self.supports_pdf():
            caps.append("PDF")
        if self.supports_audio_input():
            caps.append("audio")
        if self.structured_output:
            caps.append("structured output")
        if self.open_weights:
            caps.append("open weights")
        return ", ".join(caps) if caps else "basic"


@dataclass
class ProviderInfo:
    """Full metadata for a provider from models.dev."""

    id: str                         # models.dev provider ID
    name: str                       # display name
    env: Tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


# ---------------------------------------------------------------------------
# Provider ID mapping: Hermes ↔ models.dev
# ---------------------------------------------------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter",
    "novita": "novita-ai",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "zai": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "stepfun": "stepfun-ai",
    "stepfun-plan": "stepfun-ai-step-plan",
    "kimi-coding-cn": "kimi-for-coding",
    "minimax": "minimax",
    "minimax-oauth": "minimax",
    "minimax-cn": "minimax-cn",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "qwen-oauth": "alibaba",
    "copilot": "github-copilot",
    "ai-gateway": "vercel",
    "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "gemini": "google",
    "google": "google",
    "xai": "xai",
    # xAI OAuth is an authentication/transport path for the same xAI model
    # catalog, so model metadata should resolve through the xAI provider.
    "xai-oauth": "xai",
    "xiaomi": "xiaomi",
    "nvidia": "nvidia",
    # Meta Model API (Muse Spark family, api.meta.ai). models.dev keys these
    # under the "meta" provider id; Hermes' provider is "meta-ai" (and the
    # api.meta.ai host reverse-maps to "meta-ai"), so without both aliases the
    # context/pricing lookup misses and muse-spark-* falls back to the generic
    # 256K default instead of its true 1M window.
    "meta-ai": "meta",
    "meta": "meta",
    "groq": "groq",
    "mistral": "mistral",
    "togetherai": "togetherai",
    "perplexity": "perplexity",
    "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}

# Reverse mapping: models.dev id → Hermes ids (built lazily; many-to-one,
# e.g. both "meta" and "meta-ai" may map to the same models.dev id).
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, List[str]]] = None


def _models_dev_to_hermes_ids(mdev_id: str) -> List[str]:
    """Return the Hermes provider ids that map to *mdev_id* (may be [])."""
    global _MODELS_DEV_TO_PROVIDER
    if _MODELS_DEV_TO_PROVIDER is None:
        reverse: Dict[str, List[str]] = {}
        for hermes_id, mapped in PROVIDER_TO_MODELS_DEV.items():
            reverse.setdefault(mapped, []).append(hermes_id)
        _MODELS_DEV_TO_PROVIDER = reverse
    return _MODELS_DEV_TO_PROVIDER.get(mdev_id, [])


# StepFun ships distinct China (`.com`) and international (`.ai`) endpoints,
# and models.dev tracks them as separate provider catalogs. The default
# PROVIDER_TO_MODELS_DEV entries above point at the international IDs; when the
# resolved endpoint is China we swap to the China IDs so catalog/capability
# metadata stays endpoint-correct. (The China Step Plan catalog additionally
# carries `step-router-v1`, absent from the international one.)
_STEPFUN_CN_MODELS_DEV: Dict[str, str] = {
    "stepfun": "stepfun",
    "stepfun-plan": "stepfun-step-plan",
}
# Per-id base-url override env var — the region toggle writes the chosen host
# here at setup time, so it's the source of truth when no base_url is passed.
_STEPFUN_BASE_URL_ENV: Dict[str, str] = {
    "stepfun": "STEPFUN_BASE_URL",
    "stepfun-plan": "STEPFUN_STEP_PLAN_BASE_URL",
}


def _is_china_stepfun(base_url: Optional[str]) -> bool:
    """True when a StepFun base URL points at the China (`.com`) endpoint."""
    return "api.stepfun.com" in (base_url or "").strip().lower()


def _read_base_url_override(env_var: str) -> Optional[str]:
    """Read a base-url override from ~/.hermes/.env or the environment."""
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(env_var)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(env_var)


def _resolve_models_dev_id(
    provider: str, base_url: Optional[str] = None
) -> Optional[str]:
    """Resolve a Hermes provider ID to its region-correct models.dev ID.

    For StepFun (whose China `.com` and international `.ai` endpoints have
    separate models.dev catalogs) the region is derived from *base_url* when
    given, else from the provider's base-url override env var. Everything else
    is a straight PROVIDER_TO_MODELS_DEV lookup.
    """
    if provider in _STEPFUN_CN_MODELS_DEV:
        url = base_url
        if not url:
            env_var = _STEPFUN_BASE_URL_ENV.get(provider)
            if env_var:
                url = _read_base_url_override(env_var)
        if _is_china_stepfun(url):
            return _STEPFUN_CN_MODELS_DEV[provider]
    return PROVIDER_TO_MODELS_DEV.get(provider)



def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.json"


def _get_etag_path() -> Path:
    """Return path to the ETag sidecar file for conditional GET."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.etag"


def _load_etag() -> str:
    """Load the last-known ETag from disk, or empty string if missing."""
    try:
        etag_path = _get_etag_path()
        if etag_path.exists():
            return etag_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug("Failed to load models.dev ETag: %s", e)
    return ""


def _save_etag(etag: str) -> None:
    """Persist an ETag to the sidecar file atomically."""
    try:
        from utils import atomic_write_text

        etag_path = _get_etag_path()
        etag_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(etag_path, etag)
    except Exception as e:
        logger.debug("Failed to save models.dev ETag: %s", e)


def _clear_etag() -> None:
    """Delete the ETag sidecar so the next fetch is unconditional.

    Called when the cached registry the ETag vouches for is gone or
    unusable — sending If-None-Match without a servable cache invites a
    304 that would leave the process with no data at all.
    """
    try:
        _get_etag_path().unlink(missing_ok=True)
    except Exception as e:
        logger.debug("Failed to clear models.dev ETag: %s", e)


def _get_models_dev_url() -> str:
    """Resolve the models.dev API URL, honoring a config.yaml override.

    The ``models_dev.url`` config key lets deployments point at a mirror
    (e.g. a self-hosted copy behind a corporate proxy) without code changes.
    Falls back to the default public URL when unset or empty.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        cfg = load_config_readonly()
        url = cfg_get(cfg, "models_dev", "url", default="")
        if isinstance(url, str) and url.strip():
            return url.strip()
    except Exception:
        pass
    # Fall back to the module global (not the constant) so existing
    # code/tests that patch MODELS_DEV_URL keep working.
    return MODELS_DEV_URL


def _validate_registry(data: Any) -> bool:
    """Return True if *data* is a non-empty dict suitable for serving."""
    return isinstance(data, dict) and len(data) > 0


def _load_disk_cache() -> Dict[str, Any]:
    """Load models.dev data from disk cache.

    A corrupt cache (invalid JSON, not a dict, or empty) is rejected with
    a warning so it doesn't silently masquerade as ``{}`` and break
    provider/model resolution for every caller.
    """
    try:
        cache_path = _get_cache_path()
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if not _validate_registry(data):
                logger.warning(
                    "models.dev disk cache is corrupt or empty; "
                    "quarantining (will refetch from network)"
                )
                _quarantine_corrupt_cache(cache_path)
                return {}
            return data
    except Exception as e:
        logger.warning(
            "Failed to load models.dev disk cache; quarantining: %s", e
        )
        try:
            _quarantine_corrupt_cache(_get_cache_path())
        except Exception:
            pass
    return {}


def _quarantine_corrupt_cache(cache_path: Path) -> None:
    """Move a rejected cache aside and drop its ETag sidecar.

    Renaming (rather than leaving the file in place) makes the rejection
    a one-time event: without it, every hot-path call that finds the
    in-memory cache empty re-reads and re-parses the corrupt file and
    re-emits the warning until a network fetch succeeds. The sidecar is
    cleared because it vouches for a registry we no longer hold — a 304
    against a missing cache would leave the process with no data at all.
    """
    try:
        cache_path.rename(cache_path.with_suffix(".json.corrupt"))
    except Exception as e:
        logger.debug("Could not quarantine corrupt models.dev cache: %s", e)
    _clear_etag()


def _disk_cache_age_seconds() -> Optional[float]:
    """Return age (in seconds) of the disk cache file, or None if missing.

    Used by ``fetch_models_dev`` to short-circuit the network probe when
    a recent on-disk cache exists. Errors (missing file, permission
    denied, weird filesystem) all return None — callers fall through
    to the network fetch path.
    """
    try:
        cache_path = _get_cache_path()
        if not cache_path.exists():
            return None
        mtime = cache_path.stat().st_mtime
        age = time.time() - mtime
        # Negative age means the file's mtime is in the future (clock skew
        # or system clock reset). Treat as "unknown freshness" → fall
        # through to network so we don't serve potentially-bad data
        # forever.
        if age < 0:
            return None
        return age
    except Exception as e:
        logger.debug("Failed to stat models.dev disk cache: %s", e)
        return None


def _save_disk_cache(data: Dict[str, Any], etag: str = "") -> None:
    """Save models.dev data to disk cache atomically.

    Also persists the ETag sidecar when *etag* is non-empty so the next
    refresh can issue a conditional GET.
    """
    try:
        cache_path = _get_cache_path()
        atomic_json_write(cache_path, data, indent=None, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save models.dev disk cache: %s", e)
    if etag:
        _save_etag(etag)


class _NotModified(Exception):
    """Server returned 304 Not Modified — existing cache is still valid."""


def _fetch_models_dev_from_network(
    *, conditional: bool = False
) -> Tuple[Dict[str, Any], str]:
    """Fetch the live models.dev registry.

    ``conditional`` enables ETag conditional GET (``If-None-Match`` with
    the sidecar's ETag). Callers must pass True ONLY while holding
    ``_models_dev_fetch_lock`` AND holding a servable registry the 304
    can re-confirm — a conditional request without one invites a 304
    that leaves the process with no data at all (previously a permanent
    empty-registry loop when the sidecar outlived a corrupt cache file).
    A 304 raises ``_NotModified`` so the caller can re-confirm the
    existing cache's freshness without re-downloading the full payload.

    Returns ``(registry, etag)``; the etag is empty when the server sent
    none. The caller persists it together with the cache body
    (``_commit_registry``) so the sidecar can never get ahead of the data
    it vouches for. Raises on network errors and on an empty/invalid
    registry payload.
    """
    url = _get_models_dev_url()
    headers: Dict[str, str] = {}
    if conditional:
        etag = _load_etag()
        if etag:
            headers["If-None-Match"] = etag

    # Tuple (connect, read): a flat timeout=15 let a blackholed connect
    # stall the first-turn critical path for the full 15 s. 5 s connect
    # fails fast on unreachable hosts; 10 s read still tolerates a slow
    # registry response (matches the OpenRouter fetch convention in
    # agent/model_metadata.py).
    response = requests.get(url, headers=headers, timeout=(5, 10))

    if response.status_code == 304:
        raise _NotModified()

    response.raise_for_status()
    data = response.json()
    if not _validate_registry(data):
        raise ValueError("models.dev returned an empty or invalid registry")

    return data, response.headers.get("ETag", "")


def _mark_stale_cache_grace() -> None:
    """Give stale cache data a short in-memory grace before retrying refresh.

    Only ever moves the timestamp forward: if a background refresh completed
    between the caller's staleness check and this call, the fresh timestamp
    is preserved instead of being rewound to a 5-minute grace.
    """
    global _models_dev_cache_time
    grace_time = time.time() - _MODELS_DEV_CACHE_TTL + _MODELS_DEV_RETRY_DELAY
    if grace_time > _models_dev_cache_time:
        _models_dev_cache_time = grace_time


def _commit_registry(data: Dict[str, Any], *, etag: str = "", where: str) -> None:
    """Persist a freshly fetched registry: disk + in-mem + clear backoff.

    Callers must hold ``_models_dev_fetch_lock`` so a failing refresh on one
    path can never stomp the state a succeeding refresh on the other path
    just committed (e.g. a failing background worker re-arming the backoff
    immediately after a successful ``force_refresh``).
    """
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    _save_disk_cache(data, etag)
    _models_dev_cache = data
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug(
        "Refreshed models.dev registry (%s): %d providers, %d total models",
        where,
        len(data),
        sum(len(p.get("models", {})) for p in data.values() if isinstance(p, dict)),
    )


def _confirm_cache_not_modified(*, where: str) -> None:
    """Re-confirm the existing cache as fresh after a 304 Not Modified.

    Callers must hold ``_models_dev_fetch_lock``. Clears the backoff and
    resets the in-memory cache timestamp so the next caller hits the fast
    path. The disk cache itself is not rewritten — its contents are
    unchanged, only its freshness marker is advanced.
    """
    global _models_dev_cache_time, _models_dev_retry_after
    if not _models_dev_cache:
        # Pathological: a 304 arrived but we hold no registry. Should be
        # unreachable now that conditional GETs require a servable cache
        # (see _fetch_models_dev_from_network); kept as defense in depth
        # because this state previously caused a permanent empty-registry
        # loop. Drop the sidecar so the next attempt is unconditional and
        # arm the normal failure backoff instead of marking {} "fresh".
        _clear_etag()
        _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
        logger.warning(
            "models.dev returned 304 but no cached registry is held (%s); "
            "cleared ETag sidecar, will refetch unconditionally",
            where,
        )
        return
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug(
        "models.dev registry unchanged (304 Not Modified, %s); "
        "cache re-confirmed fresh",
        where,
    )


def _note_refresh_failure(exc: Exception, *, where: str) -> None:
    """Record a failed refresh: arm the process-wide 5-minute backoff.

    Callers must hold ``_models_dev_fetch_lock`` (see ``_commit_registry``).
    """
    global _models_dev_retry_after
    _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
    logger.debug(
        "models.dev refresh failed (%s); retry suppressed for %ds: %s",
        where,
        _MODELS_DEV_RETRY_DELAY,
        exc,
    )


def _background_refresh_models_dev() -> None:
    """Best-effort refresh after serving stale cache data."""
    global _models_dev_refresh_in_flight
    try:
        # Fetch INSIDE the lock: symmetric with the foreground path, so
        # conditional-GET inputs (memory cache + etag sidecar) can't be
        # mutated mid-fetch by a concurrent force_refresh, and the two
        # paths can't double-download concurrently. Hot-path callers are
        # unaffected — they return stale data without touching this lock.
        with _models_dev_fetch_lock:
            data, etag = _fetch_models_dev_from_network(
                conditional=bool(_models_dev_cache)
            )
            _commit_registry(data, etag=etag, where="background")
    except _NotModified:
        with _models_dev_fetch_lock:
            _confirm_cache_not_modified(where="background")
    except Exception as e:
        with _models_dev_fetch_lock:
            _note_refresh_failure(e, where="background")
    finally:
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False


def _start_background_refresh_models_dev() -> None:
    """Start one daemon refresh worker if none is already running.

    Honors the process-wide failure backoff: after a failed refresh,
    no new background worker is spawned until ``_models_dev_retry_after``.
    """
    global _models_dev_refresh_in_flight
    if time.time() < _models_dev_retry_after:
        return
    with _models_dev_refresh_lock:
        if _models_dev_refresh_in_flight:
            return
        _models_dev_refresh_in_flight = True
    thread = threading.Thread(
        target=_background_refresh_models_dev,
        name="models-dev-refresh",
        daemon=True,
    )
    try:
        thread.start()
    except Exception as e:
        # Thread/fd exhaustion: clear the flag so refresh isn't disabled
        # for the rest of the process lifetime. Callers still get stale data.
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False
        logger.debug("Failed to start models.dev refresh thread: %s", e)


def fetch_models_dev(
    force_refresh: bool = False, *, allow_network: bool = True
) -> Dict[str, Any]:
    """Fetch models.dev registry. Cache hierarchy: in-mem → disk → network.

    Returns the full registry dict keyed by provider ID, or empty dict on failure.

    Network requests use ETag conditional GET when a cached ETag exists
    AND a servable registry is held (on a cold ``force_refresh`` the
    memory cache is hydrated from disk first). A 304 Not Modified
    response re-confirms the existing cache's freshness without
    re-downloading the full (~2 MB) registry.

    Cache hierarchy (when ``force_refresh=False``):
      1. Fresh in-memory cache → return immediately.
      2. Stale in-memory cache → return immediately and refresh in a single
         background daemon thread. Callers never block on the network while
         any cache exists; ``models.dev`` only changes when providers add
         new models, so stale data is preferable to a foreground timeout.
      3. Disk cache file (any age) → load, populate in-mem, return
         immediately. Stale disk caches trigger the same background refresh.
         A corrupt or empty disk cache is rejected with a warning.
      4. No cache at all → singleflight foreground network fetch. On
         success, save to disk + in-mem and return.
      5. Any failed refresh (foreground or background) suppresses further
         automatic refreshes for 5 minutes process-wide.

    When ``force_refresh=True`` (used by ``hermes config refresh``, the
    \"refresh model catalog\" code path), cache fast paths and the failure
    backoff are bypassed; the function hits the network and only falls back
    to cached data if the call fails. When ``allow_network=False``, any
    memory or disk cache is returned regardless of age and no request is
    made — used by latency-sensitive paths (gateway route-identity checks,
    vision routing, context-length lookup) that must never wait on the
    network.
    """
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after

    if not allow_network:
        if _models_dev_cache:
            return _models_dev_cache
        disk_data = _load_disk_cache()
        if disk_data:
            _models_dev_cache = disk_data
            disk_age = _disk_cache_age_seconds()
            _models_dev_cache_time = (
                time.time() - disk_age if disk_age is not None else 0
            )
        return _models_dev_cache

    # Stage 1: fresh in-memory cache wins. This is the hot path on
    # long-lived processes — no I/O, no system calls.
    if (
        not force_refresh
        and _models_dev_cache
        and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL
    ):
        return _models_dev_cache

    # Stage 2: stale in-memory cache is still better than blocking provider
    # resolution on a foreground network timeout. Refresh it in the background.
    if not force_refresh and _models_dev_cache:
        _mark_stale_cache_grace()
        _start_background_refresh_models_dev()
        logger.debug(
            "Using stale in-memory models.dev cache; refreshing in background"
        )
        return _models_dev_cache

    # Stage 3: disk cache short-circuits the network call.
    # Only kicks in on cold-start processes (in-mem cache is empty) and only
    # when the user hasn't asked for a forced refresh. A stale disk cache is
    # deliberately usable: provider/model resolution should not hang just
    # because models.dev is unreachable.
    if not force_refresh:
        disk_age = _disk_cache_age_seconds()
        if disk_age is not None:
            disk_data = _load_disk_cache()
            if disk_data:
                _models_dev_cache = disk_data
                if disk_age < _MODELS_DEV_CACHE_TTL:
                    # Anchor in-mem TTL to the disk file's age so we don't
                    # extend an already-aging cache by another full hour.
                    _models_dev_cache_time = time.time() - disk_age
                    logger.debug(
                        "Loaded models.dev from fresh disk cache "
                        "(%d providers, age=%.0fs)", len(disk_data), disk_age,
                    )
                else:
                    _mark_stale_cache_grace()
                    _start_background_refresh_models_dev()
                    logger.debug(
                        "Using stale models.dev disk cache (age=%.0fs); "
                        "refreshing in background",
                        disk_age,
                    )
                return _models_dev_cache

    # Failed automatic refreshes are process-wide. Avoid making every caller
    # retry the same unreachable endpoint while no usable cache exists.
    if not force_refresh and time.time() < _models_dev_retry_after:
        return _models_dev_cache

    # Stage 4: singleflight foreground network fetch — only reached when no
    # memory or disk cache exists (or on force_refresh). Recheck state after
    # acquiring the lock because another caller may have refreshed or
    # established backoff while we waited.
    with _models_dev_fetch_lock:
        now = time.time()
        if not force_refresh:
            if _models_dev_cache:
                return _models_dev_cache
            if now < _models_dev_retry_after:
                return _models_dev_cache

        # Cold force_refresh (fresh CLI process): stages 1-3 were skipped,
        # so the memory cache may be empty even though a servable disk
        # cache + ETag sidecar exist. Hydrate first so the conditional GET
        # fires (a 304 then re-confirms the disk data instead of
        # re-downloading the full ~2 MB registry).
        if force_refresh and not _models_dev_cache:
            disk = _load_disk_cache()
            if disk:
                _models_dev_cache = disk
                _models_dev_cache_time = 0  # servable but not fresh

        try:
            data, etag = _fetch_models_dev_from_network(
                conditional=bool(_models_dev_cache)
            )
            _commit_registry(data, etag=etag, where="foreground")
            return data
        except _NotModified:
            # Server confirmed our cache is still valid. Re-confirm freshness
            # without re-downloading the full registry.
            _confirm_cache_not_modified(where="foreground")
            return _models_dev_cache
        except Exception as e:
            _note_refresh_failure(e, where="foreground")

        # Stage 5: network failed — return any stale memory/disk cache. Cache
        # freshness remains expired; the retry-after timestamp controls when
        # the next automatic request is allowed.
        if not _models_dev_cache:
            _models_dev_cache = _load_disk_cache()
            _models_dev_cache_time = 0
            if _models_dev_cache:
                logger.debug(
                    "Loaded stale models.dev disk cache (%d providers)",
                    len(_models_dev_cache),
                )

        return _models_dev_cache


def lookup_models_dev_context(
    provider: str, model: str, base_url: Optional[str] = None, *, allow_network: bool = False
) -> Optional[int]:
    """Look up context_length for a provider+model combo in models.dev.

    Returns the context window in tokens, or None if not found.
    Handles case-insensitive matching and filters out context=0 entries.
    *base_url* routes region-split providers (e.g. StepFun) to the
    endpoint-correct catalog.

    An EXPLICIT ``model_overrides`` config entry for this provider+model
    wins over the catalog value; ``_default`` entries fill the gap only
    when the catalog has no answer — the supported self-unblock path for
    models with wrong or missing context in models.dev (#84482).

    ``allow_network`` defaults to False — context-length lookup is a
    hot path (called during every conversation turn) and must never block
    on the network. Pass True only from explicit refresh flows.
    """
    # Explicit config override — checked before catalog so it always wins.
    override_ctx = _override_context_window(provider, model)
    if override_ctx is not None:
        return override_ctx

    mdev_provider_id = _resolve_models_dev_id(provider, base_url)
    if not mdev_provider_id:
        return _default_override_context(provider)

    # NOTE: keep the zero-argument call on the allow_network path. Dozens
    # of test sites monkeypatch fetch_models_dev with zero-arg lambdas;
    # passing the kwarg unconditionally breaks them all (TypeError).
    data = (
        fetch_models_dev()
        if allow_network
        else fetch_models_dev(allow_network=False)
    )
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return _default_override_context(provider)

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return _default_override_context(provider)

    # Exact match
    entry = models.get(model)
    if entry:
        ctx = _extract_context(entry)
        if ctx:
            return ctx

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower:
            ctx = _extract_context(mdata)
            if ctx:
                return ctx

    # Suffix-aware fallback: some providers (e.g. ollama-cloud) store
    # model IDs with :cloud / -cloud suffixes in models.dev while the
    # live API returns bare names.  Without this, kimi-k2.6 misses the
    # kimi-k2.6:cloud entry and falls through to stale OpenRouter metadata
    # reporting 32768 — tripping the 64k minimum-context guard.
    # The suffix-stripping in fetch_ollama_cloud_models() handles the
    # model-picker UX; this handles the context-length lookup path.
    for suffix in (":cloud", "-cloud"):
        suffixed_key = model + suffix
        entry = models.get(suffixed_key)
        if entry:
            ctx = _extract_context(entry)
            if ctx:
                return ctx
        # Also try case-insensitive
        suffixed_lower = model_lower + suffix
        for mid, mdata in models.items():
            if mid.lower() == suffixed_lower:
                ctx = _extract_context(mdata)
                if ctx:
                    return ctx

    # Catalog miss — a _default override may fill the gap (#84482).
    return _default_override_context(provider)


def _default_override_context(provider: str) -> Optional[int]:
    """Fill-gap context from a ``_default`` override, for catalog misses."""
    default = _default_model_override(provider)
    if default is None:
        return None
    return _override_int(default, "context_window")


def _extract_context(entry: Dict[str, Any]) -> Optional[int]:
    """Extract context_length from a models.dev model entry.

    Returns None for invalid/zero values (some audio/image models have context=0).
    """
    if not isinstance(entry, dict):
        return None
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    ctx = limit.get("context")
    if isinstance(ctx, (int, float)) and ctx > 0:
        return int(ctx)
    return None


# ---------------------------------------------------------------------------
# Model capability metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelCapabilities:
    """Structured capability metadata for a model from models.dev."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    context_window: int = 200000
    max_output_tokens: int = 8192
    model_family: str = ""


# --------------------------------------------------------------------------- #
# Per-model metadata overrides (config.yaml → model_overrides)               #
# --------------------------------------------------------------------------- #
#
# Canonical override schema (the ONLY key space consumers accept):
#   context_window, max_output_tokens, supports_tools, supports_vision,
#   supports_reasoning, model_family
#
# Resolution semantics:
#   1. ``model_overrides.<provider>.<model_id>`` — explicit override. Always
#      wins over the catalog for the fields it sets (partial patch).
#   2. ``model_overrides.<provider>._default`` / ``model_overrides._default``
#      — FILL-GAP defaults. They apply ONLY to models the catalog does not
#      know (the #8731/#84482 self-unblock path for custom/local/new
#      models) and never displace catalog data for known models. A
#      ``_default: {context_window: 128000}`` therefore cannot clamp every
#      catalog-known model of a provider.
#
# Provider keys accept the Hermes provider id (as used elsewhere in
# config.yaml) or the models.dev provider id. Model ids match exactly,
# then case-insensitively (mirroring catalog lookup).

_OVERRIDE_WARNED_KEYS: set = set()


def _load_model_overrides() -> Dict[str, Any]:
    """Load the ``model_overrides`` config section.

    No local memoization on purpose: ``load_config_readonly()`` is already
    (mtime, size)-cached upstream (a hit is ~one stat, no deepcopy, no
    parse), and an ``id(cfg)``-keyed layer here can serve stale overrides
    after a config reload when CPython reuses the freed dict's address.
    Returns empty dict on any failure.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        raw = cfg_get(load_config_readonly(), "model_overrides", default={})
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _provider_override_section(provider: str) -> Optional[Dict[str, Any]]:
    """Return the override section for *provider*, or None.

    Accepts either the Hermes provider id or the models.dev provider id as
    the config key, so ``copilot`` and ``github-copilot`` both work
    regardless of which id space a caller passes in.
    """
    overrides = _load_model_overrides()
    if not overrides:
        return None
    provider_key = (provider or "").strip()
    if not provider_key:
        return None

    candidates = [provider_key]
    mapped = PROVIDER_TO_MODELS_DEV.get(provider_key)
    if mapped and mapped != provider_key:
        candidates.append(mapped)
    # Reverse: caller passed a models.dev id, config keyed by Hermes id.
    for hermes_id in _models_dev_to_hermes_ids(provider_key):
        if hermes_id != provider_key:
            candidates.append(hermes_id)

    for key in candidates:
        section = overrides.get(key)
        if isinstance(section, dict):
            return section
    return None


def _explicit_model_override(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Return the explicit per-provider+model override dict, or None.

    Model ids match exactly first, then case-insensitively (skipping the
    ``_default`` sentinel), mirroring catalog lookup behavior.
    """
    model_key = (model or "").strip()
    if not model_key:
        return None
    section = _provider_override_section(provider)
    if section is None:
        return None

    entry = section.get(model_key)
    if isinstance(entry, dict):
        return entry

    model_lower = model_key.lower()
    for mid, mdata in section.items():
        if mid == "_default":
            continue
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata
    return None


def _default_model_override(provider: str) -> Optional[Dict[str, Any]]:
    """Return the fill-gap ``_default`` override for *provider*, or None.

    Checks the per-provider ``_default`` first, then the global one. Only
    consulted for models the catalog does not know — see the block comment.
    """
    section = _provider_override_section(provider)
    if section is not None:
        default = section.get("_default")
        if isinstance(default, dict):
            return default
    overrides = _load_model_overrides()
    global_default = overrides.get("_default")
    if isinstance(global_default, dict):
        return global_default
    return None


def _override_for(
    provider: str, model: str, *, catalog_hit: bool
) -> Optional[Dict[str, Any]]:
    """Select the override dict for a lookup, honoring fill-gap semantics.

    Explicit per-provider+model overrides always apply. ``_default``
    entries apply only when the catalog has no entry for the model.
    """
    explicit = _explicit_model_override(provider, model)
    if explicit is not None:
        return explicit
    if catalog_hit:
        return None
    return _default_model_override(provider)


def _override_int(override: Dict[str, Any], key: str) -> Optional[int]:
    """Coerce an override field to a positive int, warning once on garbage."""
    raw = override.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    warn_key = (key, repr(raw))
    if warn_key not in _OVERRIDE_WARNED_KEYS:
        _OVERRIDE_WARNED_KEYS.add(warn_key)
        logger.warning(
            "model_overrides: ignoring invalid %s value %r "
            "(expected a positive integer)", key, raw,
        )
    return None


def _override_context_window(provider: str, model: str) -> Optional[int]:
    """Return the EXPLICITLY overridden context_window, or None.

    Explicit-only on purpose: this runs early in the resolution chain
    (agent/model_metadata.py step 0b, before custom_providers and live
    probes), where a ``_default`` must not preempt more specific sources.
    Fill-gap defaults are applied later by ``lookup_models_dev_context``
    once the catalog has actually missed.
    """
    ov = _explicit_model_override(provider, model)
    if ov is None:
        return None
    return _override_int(ov, "context_window")


def _override_to_catalog_shape(
    override: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Translate canonical override keys into a models.dev-shaped patch.

    ``get_model_info``/``_parse_model_info`` consume the raw catalog shape
    (``limit.context``, ``tool_call``, ...). All override consumers accept
    ONE canonical schema (the documented ``context_window``/``supports_*``
    keys), so this boundary translates rather than forcing users to know
    the internal catalog shape.

    Returns ``(patch, vision)`` — vision is returned out-of-band (not as
    a key in the patch) because it maps onto the catalog's
    ``modalities.input`` list rather than a scalar field.
    """
    patch: Dict[str, Any] = {}
    limit: Dict[str, Any] = {}
    ctx = _override_int(override, "context_window")
    if ctx is not None:
        limit["context"] = ctx
    out = _override_int(override, "max_output_tokens")
    if out is not None:
        limit["output"] = out
    if limit:
        patch["limit"] = limit
    if "supports_tools" in override:
        patch["tool_call"] = bool(override["supports_tools"])
    if "supports_reasoning" in override:
        patch["reasoning"] = bool(override["supports_reasoning"])
    vision: Optional[bool] = None
    if "supports_vision" in override:
        vision = bool(override["supports_vision"])
        patch["attachment"] = vision
    if "model_family" in override:
        patch["family"] = str(override["model_family"] or "")
    return patch, vision


def _merge_catalog_entry_with_override(
    raw: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Patch a catalog entry with a canonical-schema override.

    Sub-dicts (``limit``, ``modalities``) are merged, not clobbered — an
    override setting only ``context_window`` must not wipe the catalog's
    ``limit.output``.
    """
    shaped, vision_override = _override_to_catalog_shape(override)
    merged = dict(raw)
    limit_patch = shaped.pop("limit", None)
    if limit_patch:
        base_limit = raw.get("limit")
        base_limit = dict(base_limit) if isinstance(base_limit, dict) else {}
        base_limit.update(limit_patch)
        merged["limit"] = base_limit
    if vision_override is not None:
        base_mods = raw.get("modalities")
        base_mods = dict(base_mods) if isinstance(base_mods, dict) else {}
        input_mods = base_mods.get("input")
        input_mods = list(input_mods) if isinstance(input_mods, list) else []
        if vision_override and "image" not in input_mods:
            input_mods.append("image")
        elif not vision_override and "image" in input_mods:
            input_mods.remove("image")
        base_mods["input"] = input_mods
        merged["modalities"] = base_mods
    merged.update(shaped)
    return merged


def _get_provider_models(
    provider: str, base_url: Optional[str] = None, *, allow_network: bool = False
) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict from models.dev.

    Returns the models dict or None if the provider is unknown or has no data.
    *base_url* routes region-split providers to the endpoint-correct catalog.

    ``allow_network`` defaults to False — this is called from hot paths
    (vision routing, image routing, capability checks) and must never block.
    """
    mdev_provider_id = _resolve_models_dev_id(provider, base_url)
    if not mdev_provider_id:
        return None

    # NOTE: keep the zero-argument call on the allow_network path. Dozens
    # of test sites monkeypatch fetch_models_dev with zero-arg lambdas;
    # passing the kwarg unconditionally breaks them all (TypeError).
    data = (
        fetch_models_dev()
        if allow_network
        else fetch_models_dev(allow_network=False)
    )
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    return models


def _find_model_entry(models: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Find a model entry: exact, case-insensitive, then suffix fallback.

    The ``:cloud``/``-cloud`` suffix fallback mirrors
    ``lookup_models_dev_context`` so "is this model in the catalog" means
    the same thing to every consumer — important for ``model_overrides``
    fill-gap ``_default`` semantics, where a suffix-keyed catalog model
    (e.g. ``kimi-k2.6:cloud``) must count as KNOWN and keep its catalog
    metadata rather than being displaced by a ``_default``.
    """
    # Exact match
    entry = models.get(model)
    if isinstance(entry, dict):
        return entry

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata

    # Suffix-aware fallback (e.g. ollama-cloud stores kimi-k2.6:cloud
    # while the live API returns the bare name).
    for suffix in (":cloud", "-cloud"):
        entry = models.get(model + suffix)
        if isinstance(entry, dict):
            return entry
        suffixed_lower = model_lower + suffix
        for mid, mdata in models.items():
            if mid.lower() == suffixed_lower and isinstance(mdata, dict):
                return mdata

    return None


def get_model_capabilities(
    provider: str, model: str, base_url: Optional[str] = None, *, allow_network: bool = False
) -> Optional[ModelCapabilities]:
    """Look up full capability metadata from models.dev cache.

    Uses the existing fetch_models_dev() and PROVIDER_TO_MODELS_DEV mapping.
    Returns None if model not found. *base_url* routes region-split providers
    to the endpoint-correct catalog.

    EXPLICIT ``model_overrides`` entries (per-provider+model) win over
    catalog values for the fields they set. ``_default`` entries fill the
    gap only for models the catalog does not know — the supported
    self-unblock path for custom/local models (#8731) and for models with
    wrong metadata in models.dev (#84482). An override may set any subset
    of fields; unspecified fields fall through to the catalog value (or
    sensible defaults when the model is absent from the catalog).

    ``allow_network`` defaults to False — capability lookup is a hot path
    (vision routing, image routing) and must never block on the network.

    Extracts from model entry fields:
      - reasoning  (bool)  → supports_reasoning
      - tool_call  (bool)  → supports_tools
      - attachment (bool)  → supports_vision
      - limit.context (int) → context_window
      - limit.output  (int) → max_output_tokens
      - family     (str)   → model_family
    """
    models = _get_provider_models(provider, base_url, allow_network=allow_network)
    entry = _find_model_entry(models, model) if models is not None else None

    # Select the override AFTER the catalog lookup: explicit overrides
    # always apply; _default entries only fill gaps for catalog misses.
    override = _override_for(provider, model, catalog_hit=entry is not None)

    # If no catalog entry and no override, we can't resolve capabilities.
    if entry is None and override is None:
        return None

    # Start from catalog entry (if found), else use defaults.
    if entry is not None:
        supports_tools = bool(entry.get("tool_call", False))
        # Vision: prefer explicit `modalities.input` when models.dev provides it.
        # The older `attachment` flag can be stale or too broad for image routing;
        # fall back to it only when the input modalities are absent/invalid.
        input_mods = entry.get("modalities", {})
        if isinstance(input_mods, dict):
            input_mods = input_mods.get("input")
        else:
            input_mods = None
        if isinstance(input_mods, list):
            supports_vision = "image" in input_mods
        else:
            supports_vision = bool(entry.get("attachment", False))
        supports_reasoning = bool(entry.get("reasoning", False))

        limit = entry.get("limit", {})
        if not isinstance(limit, dict):
            limit = {}

        ctx = limit.get("context")
        context_window = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 200000

        out = limit.get("output")
        max_output_tokens = int(out) if isinstance(out, (int, float)) and out > 0 else 8192

        model_family = entry.get("family", "") or ""
    else:
        # Unknown model — derive sensible defaults. The override will
        # patch whichever fields it specifies; the rest stay at defaults
        # that are safe for agentic use (tools on, vision/reasoning off).
        supports_tools = True
        supports_vision = False
        supports_reasoning = False
        context_window = 200000
        max_output_tokens = 8192
        model_family = ""

    # Apply override patches (each field is optional in the override dict).
    if override is not None:
        if "supports_tools" in override:
            supports_tools = bool(override["supports_tools"])
        if "supports_vision" in override:
            supports_vision = bool(override["supports_vision"])
        if "supports_reasoning" in override:
            supports_reasoning = bool(override["supports_reasoning"])
        ctx_ov = _override_int(override, "context_window")
        if ctx_ov is not None:
            context_window = ctx_ov
        out_ov = _override_int(override, "max_output_tokens")
        if out_ov is not None:
            max_output_tokens = out_ov
        if "model_family" in override:
            model_family = str(override["model_family"] or "")

    return ModelCapabilities(
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        model_family=model_family,
    )


def list_provider_models(
    provider: str, base_url: Optional[str] = None, *, allow_network: bool = True
) -> List[str]:
    """Return all model IDs for a provider from models.dev.

    Returns an empty list if the provider is unknown or has no data.
    *base_url* routes region-split providers to the endpoint-correct catalog.

    ``allow_network`` defaults to True — this is called from the model
    picker (``hermes model``), which is an interactive user-facing flow
    where a fresh catalog is worth a short network wait.
    """
    from hermes_cli.models import normalize_provider
    provider = normalize_provider(provider) or provider

    models = _get_provider_models(provider, base_url, allow_network=allow_network)
    if models is None:
        return []
    return [
        mid for mid in models.keys()
        if not _should_hide_from_provider_catalog(provider, mid)
    ]


# Patterns that indicate non-agentic or noise models (TTS, embedding,
# dated preview snapshots, live/streaming-only, image-only).
import re
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|"
    r"-image\b|-image-preview\b|-customtools\b",
    re.IGNORECASE,
)

# Google's live Gemini catalogs currently include a mix of stale slugs and
# Gemma models whose TPM quotas are too small for normal Hermes agent traffic.
# Keep capability metadata available for direct/manual use, but hide these from
# the Gemini model catalogs we surface in setup and model selection.
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under
    # agent-style traffic despite advertising large context windows.
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "gemma-4-26b-a4b-it",
    "gemma-3-1b",
    "gemma-3-1b-it",
    "gemma-3-2b",
    "gemma-3-2b-it",
    "gemma-3-4b",
    "gemma-3-4b-it",
    "gemma-3-12b",
    "gemma-3-12b-it",
    "gemma-3-27b",
    "gemma-3-27b-it",
    # Stale/retired Google slugs that still surface through models.dev-backed
    # Gemini selection but 404 on the current Google endpoints.
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    provider_lower = (provider or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    if provider_lower in {"gemini", "google"} and model_lower in _GOOGLE_HIDDEN_MODELS:
        return True
    return False


def list_agentic_models(
    provider: str, base_url: Optional[str] = None, *, allow_network: bool = True
) -> List[str]:
    """Return model IDs suitable for agentic use from models.dev.

    Filters for tool_call=True and excludes noise (TTS, embedding,
    dated preview snapshots, live/streaming, image-only models).
    Returns an empty list on any failure. *base_url* routes region-split
    providers to the endpoint-correct catalog.

    ``allow_network`` defaults to True — like ``list_provider_models``,
    this is called from interactive model selection flows.
    """
    models = _get_provider_models(provider, base_url, allow_network=allow_network)
    if models is None:
        return []

    result = []
    for mid, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if _should_hide_from_provider_catalog(provider, mid):
            continue
        if not entry.get("tool_call", False):
            continue
        if _NOISE_PATTERNS.search(mid):
            continue
        result.append(mid)
    return result



# ---------------------------------------------------------------------------
# Rich dataclass constructors — parse raw models.dev JSON into dataclasses
# ---------------------------------------------------------------------------

def _parse_model_info(model_id: str, raw: Dict[str, Any], provider_id: str) -> ModelInfo:
    """Convert a raw models.dev model entry dict into a ModelInfo dataclass."""
    limit = raw.get("limit") or {}
    if not isinstance(limit, dict):
        limit = {}

    cost = raw.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}

    modalities = raw.get("modalities") or {}
    if not isinstance(modalities, dict):
        modalities = {}

    input_mods = modalities.get("input") or []
    output_mods = modalities.get("output") or []

    ctx = limit.get("context")
    ctx_int = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 0
    out = limit.get("output")
    out_int = int(out) if isinstance(out, (int, float)) and out > 0 else 0
    inp = limit.get("input")
    inp_int = int(inp) if isinstance(inp, (int, float)) and inp > 0 else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        reasoning=bool(raw.get("reasoning", False)),
        tool_call=bool(raw.get("tool_call", False)),
        attachment=bool(raw.get("attachment", False)),
        temperature=bool(raw.get("temperature", False)),
        structured_output=bool(raw.get("structured_output", False)),
        open_weights=bool(raw.get("open_weights", False)),
        input_modalities=tuple(input_mods) if isinstance(input_mods, list) else (),
        output_modalities=tuple(output_mods) if isinstance(output_mods, list) else (),
        context_window=ctx_int,
        max_output=out_int,
        max_input=inp_int,
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=float(cost["cache_read"]) if "cache_read" in cost and cost["cache_read"] is not None else None,
        cost_cache_write=float(cost["cache_write"]) if "cache_write" in cost and cost["cache_write"] is not None else None,
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def _parse_provider_info(provider_id: str, raw: Dict[str, Any]) -> ProviderInfo:
    """Convert a raw models.dev provider entry dict into a ProviderInfo."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


# ---------------------------------------------------------------------------
# Provider-level queries
# ---------------------------------------------------------------------------

def get_provider_info(
    provider_id: str, base_url: Optional[str] = None, *, allow_network: bool = True
) -> Optional[ProviderInfo]:
    """Get full provider metadata from models.dev.

    Accepts either a Hermes provider ID (e.g. "kilocode") or a models.dev
    ID (e.g. "kilo").  Returns None if the provider is not in the catalog.
    *base_url* routes region-split providers to the endpoint-correct catalog.

    ``allow_network`` defaults to True — the primary caller is
    ``resolve_provider_full`` during interactive setup, where a fresh
    catalog is worth a short network wait. Hot-path callers should pass
    ``allow_network=False``.
    """
    # Resolve Hermes ID → models.dev ID (region-aware for split providers)
    mdev_id = _resolve_models_dev_id(provider_id, base_url) or provider_id

    # NOTE: keep the zero-argument call on the default path. Dozens of test
    # sites monkeypatch fetch_models_dev with zero-arg lambdas; passing the
    # kwarg unconditionally would break them all (they raise TypeError).
    data = (
        fetch_models_dev()
        if allow_network
        else fetch_models_dev(allow_network=False)
    )
    raw = data.get(mdev_id)
    if not isinstance(raw, dict):
        return None

    return _parse_provider_info(mdev_id, raw)


# ---------------------------------------------------------------------------
# Model-level queries (rich ModelInfo)
# ---------------------------------------------------------------------------

def get_model_info(
    provider_id: str, model_id: str, base_url: Optional[str] = None, *, allow_network: bool = False
) -> Optional[ModelInfo]:
    """Get full model metadata from models.dev.

    Accepts Hermes or models.dev provider ID.  Tries exact match then
    case-insensitive fallback.  Returns None if not found. *base_url* routes
    region-split providers to the endpoint-correct catalog.

    ``model_overrides`` entries use the SAME canonical schema as every
    other consumer (``context_window``, ``max_output_tokens``,
    ``supports_*``, ``model_family``) — they are translated into the
    catalog shape at this boundary, and sub-dicts (``limit``,
    ``modalities``) are merged rather than clobbered. EXPLICIT entries
    patch known catalog models; ``_default`` entries fill the gap only
    for models the catalog does not know (#8731, #84482).

    ``allow_network`` defaults to False — model info lookup is a hot path
    (cost guard, inventory) and must never block on the network.
    """
    mdev_id = _resolve_models_dev_id(provider_id, base_url) or provider_id

    def _from_override_alone() -> Optional[ModelInfo]:
        override = _override_for(provider_id, model_id, catalog_hit=False)
        if override is None:
            return None
        # Seed the same safe defaults get_model_capabilities uses for
        # unknown models (200K context, tools on) so the two
        # unknown-model paths agree; the override patches its fields on
        # top.
        base = {
            "limit": {"context": 200000, "output": 8192},
            "tool_call": True,
        }
        shaped = _merge_catalog_entry_with_override(base, override)
        return _parse_model_info(model_id, shaped, mdev_id)

    # NOTE: keep the zero-argument call on the allow_network path. Dozens
    # of test sites monkeypatch fetch_models_dev with zero-arg lambdas;
    # passing the kwarg unconditionally breaks them all (TypeError).
    data = (
        fetch_models_dev()
        if allow_network
        else fetch_models_dev(allow_network=False)
    )
    pdata = data.get(mdev_id)
    if not isinstance(pdata, dict):
        return _from_override_alone()

    models = pdata.get("models", {})
    if not isinstance(models, dict):
        return _from_override_alone()

    def _with_override(mid: str, raw: Dict[str, Any]) -> ModelInfo:
        override = _override_for(provider_id, model_id, catalog_hit=True)
        if override is not None:
            merged = _merge_catalog_entry_with_override(raw, override)
            return _parse_model_info(mid, merged, mdev_id)
        return _parse_model_info(mid, raw, mdev_id)

    # Exact match
    raw = models.get(model_id)
    if isinstance(raw, dict):
        return _with_override(model_id, raw)

    # Case-insensitive fallback
    model_lower = model_id.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return _with_override(mid, mdata)

    # Model not in catalog — an override (explicit or _default) may still
    # provide the metadata.
    return _from_override_alone()
