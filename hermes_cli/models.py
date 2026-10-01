"""Application catalogue acquisition and scoped observation caches.

Provider identity, static catalogue interpretation, pricing and selection
semantics are owned by providers/ and models/. This application module acquires
configuration, credentials and network observations for those domains.
"""

from __future__ import annotations

import contextvars
import copy
import gzip
import json
import logging
import os
import re
import sys
import threading
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from models import normalize_model_id
from models.catalog_projection import (project_openrouter_catalog, project_ai_gateway_catalog,
    _merge_unique, _model_dedup_key, _drop_delisted_opencode_models, merge_profile_models)
from models.catalog_chat import without_generation_models as _chat_catalog_rows
from hermes_constants import hermes_home_key
from application_deepinfra_catalog import deepinfra_model_ids
from models.catalog_github import fetch_github_model_catalog as _fetch_github_model_catalog

from providers import (
    copilot_request_headers,
    get_provider_profile,
    is_aggregator,
    list_providers,
    normalize_provider as _normalize_provider,
)
from providers import normalize_route_base_url
from hermes_cli.urllib_security import open_credentialed_url
from hermes_cli.version_info import get_version_info
from models import catalog_static
from models.metadata.reasoning import (
    _OPENROUTER_CATALOG_URL,
    _seed_reasoning_caps,
    configure_reasoning_metadata_sources,
    is_astra_model,
)
from hermes_cli import models_local
from hermes_cli.models_local import (
    _OLLAMA_LOCAL_MODELS_CACHE,
    _OLLAMA_LOCAL_MODELS_CACHE_TTL,
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE,
    _OLLAMA_LOCAL_PROBE_REACHABLE,
    _ollama_local_catalog,
    _ollama_probe_cache_key,
    _root_for_ollama_native_api,
    fetch_ollama_cloud_models)

logger = logging.getLogger(__name__)

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{get_version_info().base_version}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"

def _urlopen_model_catalog_request(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """Open catalog requests without forwarding headers across origins."""
    return open_credentialed_url(req, timeout=timeout, ssl_context=ssl_context)


def _reasoning_catalog_request(req: urllib.request.Request, *, timeout: float):
    """Late-bound guarded opener for the lower metadata source seam."""
    return _urlopen_model_catalog_request(req, timeout=timeout)


configure_reasoning_metadata_sources(request=_reasoning_catalog_request, user_agent=_HERMES_USER_AGENT, nous_url=None)


def _get_json(
    url: str, *, timeout: float, headers: Optional[dict[str, str]] = None, opener=None, **open_kwargs: Any
) -> Any:
    """GET ``url`` and parse the JSON body. ``opener`` defaults to the catalog opener (resolved at
    call time so monkeypatching ``_urlopen_model_catalog_request`` still applies). Raises on failure."""
    req = urllib.request.Request(url, headers=headers or {})
    with (opener or _urlopen_model_catalog_request)(req, timeout=timeout, **open_kwargs) as resp:
        body = resp.read()
        if req.get_header("Accept-encoding") == "gzip" and resp.headers.get("Content-Encoding", "").lower() == "gzip":
            body = gzip.decompress(body)
        return json.loads(body.decode())


def _read_json_cache(path: Path, *, errors=Exception) -> Optional[dict]:
    """Load a JSON-object cache file; None when missing, unreadable, or not a dict."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except errors:
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Atomically persist a cache file (creating parents). Raises on failure — callers decide
    whether a failed cache write is worth logging."""
    from utils import atomic_json_write
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    atomic_json_write(path, data, **dump_kwargs)




def _custom_provider_ssl_context(base_url: str):
    """Use the same trust decision for urllib catalogs and HTTPX metadata/chat."""
    from agent.model_metadata_http import resolve_verify

    verify = resolve_verify(base_url)
    if verify is True:
        return None
    if verify is False:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return verify


# Process-lifetime picker lists refreshed from the live catalogs (see fetch_*_models).
_openrouter_catalog_cache: list[tuple[str, str]] | None = None

# The in-memory ``_openrouter_catalog_cache`` is per-process, so without a disk cache every cold
# picker open re-downloads the full ~686KB /api/v1/models catalog. The *curated* result
# (post-filter) is persisted under the same TTL the catalog manifest uses, so both layers go
# stale together.


def _openrouter_catalog_disk_ttl() -> float:
    """Same TTL as the canonical manifest cache used by this active profile."""
    from hermes_cli.catalog_context import catalog_runtime_context
    from models.catalog_runtime import refresh_interval_seconds

    settings, _cache_path = catalog_runtime_context()
    return refresh_interval_seconds(settings)


def _openrouter_catalog_disk_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "openrouter_curated_catalog.json"


def _read_openrouter_catalog_disk(*, allow_stale: bool = False) -> list[tuple[str, str]] | None:
    """Fresh curated catalog from disk, or None (missing, corrupt, expired, or empty).

    ``allow_stale`` ignores the TTL — the cache-only read path prefers a stale copy over a live GET."""
    obj = _read_json_cache(_openrouter_catalog_disk_path())
    if obj is None:
        return None
    try:
        if not allow_stale and time.time() - float(obj.get("fetched_at", 0)) > _openrouter_catalog_disk_ttl():
            return None
    except (TypeError, ValueError):
        return None
    items = obj.get("curated")
    if not isinstance(items, list):
        return None
    out = [(str(it[0]), str(it[1])) for it in items if isinstance(it, (list, tuple)) and len(it) == 2]
    return out or None


def _write_openrouter_catalog_disk(curated: list[tuple[str, str]]) -> None:
    try:
        _write_json_cache(
            _openrouter_catalog_disk_path(),
            {"fetched_at": time.time(), "curated": [list(c) for c in curated]})
    except Exception as exc:
        logger.debug("openrouter curated catalog disk write failed: %s", exc)
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


# ---------------------------------------------------------------------------
# Nous Portal free-model helpers — the Portal models endpoint is the source of truth for what is
# offered (free or paid); we surface it as-is, no local allowlist filtering.
# ---------------------------------------------------------------------------


def _union_with_portal_recommendations(
    tier_key: str, curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str,
    *, force_refresh: bool, synthesize_free_pricing: bool,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Append the Portal's ``<tier_key>`` recommendations missing from ``curated_ids``.

    Curated models show first, Portal-only picks follow. Failures (network, parse, missing field)
    silently return the inputs unchanged — never block the picker on a Portal-side hiccup.
    """
    try:
        from application_nous_recommendations import fetch_recommended_models

        payload = fetch_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        payload = None
    block = payload.get(tier_key) if isinstance(payload, dict) else None
    entries = block if isinstance(block, list) else []
    portal_ids = [name for entry in entries if (name := _extract_model_name(entry))]
    if not portal_ids:
        return (list(curated_ids), dict(pricing))

    augmented_pricing = dict(pricing)
    if synthesize_free_pricing:
        for mid in portal_ids:
            augmented_pricing.setdefault(mid, {"prompt": "0", "completion": "0"})
    seen = set(curated_ids)
    return (list(curated_ids) + [mid for mid in portal_ids if mid not in seen], augmented_pricing)


def union_with_portal_free_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list + pricing plus the Portal's ``freeRecommendedModels``; Portal-only free picks get a
    synthetic $0 pricing entry so tier partitioning sees them as free."""
    return _union_with_portal_recommendations(
        "freeRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=True)


def union_with_portal_paid_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list plus the Portal's ``paidRecommendedModels``; ``pricing`` is deliberately left untouched."""
    return _union_with_portal_recommendations(
        "paidRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=False)


# Free-tier detection cache, per profile — short so an account upgrade shows within minutes.
_FREE_TIER_CACHE_TTL: int = 180  # seconds
_free_tier_cache: dict[str, tuple[bool, float]] = {}  # profile key -> (result, timestamp)


def get_cached_nous_free_tier() -> Optional[bool]:
    """This profile's live cached entitlement, or ``None`` if unknown/expired."""
    cached = _free_tier_cache.get(hermes_home_key())
    if cached is None or time.monotonic() - cached[1] >= _FREE_TIER_CACHE_TTL:
        return None
    return cached[0]


def check_nous_free_tier(*, force_fresh: bool = False, cached_only: bool = False) -> bool:
    """True only when the Nous Portal user is KNOWN to be free-tier (unknown/error → False so this
    never blocks users). Cached ``_FREE_TIER_CACHE_TTL`` seconds so an upgrade shows within minutes.
    ``cached_only`` returns the live cached answer or the fail-open ``False`` without contacting Portal."""
    now = time.monotonic()
    profile_key = hermes_home_key()
    if not force_fresh:
        cached_result = get_cached_nous_free_tier()
        if cached_result is not None:
            return cached_result
    if cached_only:
        return False
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        result = get_nous_portal_account_info(force_fresh=force_fresh).is_free_tier
    except Exception:
        result = False  # default to paid on error — don't block users
    _free_tier_cache[profile_key] = (result, now)
    return result


def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    model_name = entry.get("modelName") if isinstance(entry, dict) else None
    return model_name.strip() if isinstance(model_name, str) and model_name.strip() else None


def _fetch_live_catalog_index(url: str, timeout: float, opener) -> Optional[tuple[list, dict[str, dict[str, Any]]]]:
    """GET an OpenAI-style ``/models`` listing → ``(raw data array, {id: item})``, or None when the
    endpoint is unreachable or the payload has no ``data`` list."""
    try:
        payload = _get_json(url, timeout=timeout, headers={"Accept": "application/json"}, opener=opener)
    except Exception:
        return None
    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return None
    live_by_id = {
        mid: item for item in live_items if isinstance(item, dict) and (mid := str(item.get("id") or "").strip())
    }
    return live_items, live_by_id


def fetch_openrouter_models(
    timeout: float = 8.0, *, force_refresh: bool = False, cache_only: bool = False) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible.

    ``cache_only`` never opens a socket: memory, then disk (stale accepted), then the in-repo snapshot."""
    # The curated list is filtered from this profile's manifest (``model_catalog.*`` config, its
    # ``<home>/cache`` copy), so a routed profile keeps its own slot instead of the module one.
    from hermes_cli.models_profile_cache import profile_slot_get, profile_slot_set
    _me = sys.modules[__name__]
    cached = profile_slot_get(_me, "_openrouter_catalog_cache")

    if cached is not None and not force_refresh:
        return list(cached)

    # Cold process: serve from the persisted disk cache when fresh so the
    # picker doesn't re-download the full ~686KB catalog on every open.
    if not force_refresh:
        disk = _read_openrouter_catalog_disk(allow_stale=cache_only)
        if disk:
            if not cache_only:  # a stale copy is served, never memoized as fresh
                profile_slot_set(_me, "_openrouter_catalog_cache", disk)
            return list(disk)
    if cache_only:
        return list(catalog_static.OPENROUTER_MODELS)

    # Remote catalog manifest first, in-repo snapshot when unreachable; the live /v1/models filter
    # (tool support, free pricing) is applied on top either way.
    try:
        from hermes_cli.catalog_context import catalog_runtime_context
        from models.catalog_runtime import curated_openrouter

        settings, cache_path = catalog_runtime_context()
        remote = curated_openrouter(
            settings, cache_path, user_agent=_HERMES_USER_AGENT
        )
    except Exception:
        remote = ()
    fallback = list(remote) if remote else list(catalog_static.OPENROUTER_MODELS)

    live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    if live is None:
        return list(cached or fallback)
    live_items, live_by_id = live

    # Free warm-up for the reasoning-capability cache: same payload the caps fetch would pull.
    _seed_reasoning_caps(_OPENROUTER_CATALOG_URL, live_items)

    from application_model_selection_defaults import preferred_silent_default_model
    curated = project_openrouter_catalog(fallback, live_by_id, preferred_silent_default_model("openrouter"))
    if not curated:
        return list(cached or fallback)
    profile_slot_set(_me, "_openrouter_catalog_cache", curated)
    _write_openrouter_catalog_disk(curated)
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_nous_model_ids() -> list[str]:
    """Curated Nous Portal model ids: the remote catalog manifest, else the in-repo
    ``catalog_static._PROVIDER_MODELS["nous"]`` snapshot. Always a list."""
    try:
        from hermes_cli.catalog_context import catalog_runtime_context
        from models.catalog_runtime import curated_ids

        settings, cache_path = catalog_runtime_context()
        remote = curated_ids(
            settings, cache_path, "nous", user_agent=_HERMES_USER_AGENT
        )
    except Exception:
        remote = ()
    return list(remote or catalog_static._PROVIDER_MODELS.get("nous", []))


def fetch_ai_gateway_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = list(catalog_static.VERCEL_AI_GATEWAY_MODELS)
    live = _fetch_live_catalog_index(f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models", timeout, _urlopen_model_catalog_request)
    if live is None:
        return list(_ai_gateway_catalog_cache or fallback)
    _, live_by_id = live

    curated = project_ai_gateway_catalog(fallback, live_by_id)
    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)
    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]


# ---------------------------------------------------------------------------
# Provider identity: ``provider:model`` parsing, auto-detection, labels
# ---------------------------------------------------------------------------

def _known_provider_names() -> set[str]:
    """Provider IDs and aliases currently valid left of ``provider:model``."""
    names = {"custom"}
    for profile in list_providers():
        names.add(str(profile.name or "").strip().lower())
        names.update(str(alias or "").strip().lower() for alias in profile.aliases)
    names.discard("")
    return names


_CONFIG_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError)


def _configured_custom_provider_ids() -> set[str]:
    """Return routable custom-provider IDs configured by the user."""
    ids = {"custom"}
    try:
        from hermes_cli.config import load_config
        from providers import custom_provider_slug

        config = load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict):
            ids.update(custom_provider_slug(str(entry.get("name") or key), str(key))
                       for key, entry in providers.items() if isinstance(entry, dict))
        legacy = config.get("custom_providers", [])
        if isinstance(legacy, list):
            ids.update(
                custom_provider_slug(str(entry.get("name") or "")) for entry in legacy if isinstance(entry, dict))
    except _CONFIG_ERRORS:
        pass
    return ids


from application_provider_listing import list_available_providers


def _get_custom_base_url() -> str:
    """The custom endpoint ``model.base_url`` from config.yaml."""
    return str(_get_model_config_dict().get("base_url", "")).strip()


def _get_provider_config_dict(provider: str) -> dict[str, Any]:
    """Return config.yaml providers.<provider>, or an empty dict."""
    key = str(provider or "").strip()
    if not key:
        return {}
    try:
        from hermes_cli.config import load_config
        providers_cfg = load_config().get("providers", {})
        if isinstance(providers_cfg, dict):
            entry = providers_cfg.get(key) or providers_cfg.get(key.lower())
            if isinstance(entry, dict):
                return entry
    except _CONFIG_ERRORS:
        pass
    return {}


def _get_model_config_dict() -> dict[str, Any]:
    """Return the main model config mapping, or an empty dict."""
    try:
        from hermes_cli.config import load_config
        model_cfg = load_config().get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``catalog_static._PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = _normalize_provider(provider or "openrouter")
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)

    # Try live API first (Codex, Nous, etc. all support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = catalog_static._PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _configured_provider_ids() -> set[str]:
    """Provider ids (incl. ``custom:*``) from the user's ``providers:`` config block; empty when config
    is unreadable (callers fall through to built-in catalogs)."""
    try:
        from hermes_cli.config import load_config

        providers = (load_config() or {}).get("providers")
        if not isinstance(providers, dict):
            return set()
        return {key for pid in providers if (key := str(pid).strip().lower())}
    except Exception:
        return set()






def _first_exchangeable_copilot_token(raw_tokens) -> str:
    """Exchange stored GitHub tokens in order; the first that validates AND exchanges wins (every
    entry is tried so a later valid token survives an earlier malformed one)."""
    from hermes_cli.copilot_auth import exchange_copilot_token, validate_copilot_token

    for raw in raw_tokens:
        raw = str(raw or "").strip()
        if not raw or not validate_copilot_token(raw)[0]:
            continue
        try:
            api_token = exchange_copilot_token(raw)[0]  # (api_token, expires_at, base_url)
        except Exception:
            continue
        if api_token:
            return api_token
    return ""


def _copilot_cli_config_tokens() -> list[str]:
    """``copilotTokens`` from the GitHub Copilot CLI's own plaintext store (JSONC — strip
    ``//``-comment lines), written by ``copilot login`` on hosts without an OS keychain."""
    cli_config = os.path.expanduser("~/.copilot/config.json")
    if not os.path.isfile(cli_config):
        return []
    with open(cli_config, "r", encoding="utf-8-sig", errors="ignore") as fh:
        raw_text = "\n".join(
            line for line in fh.read().splitlines() if not line.lstrip().startswith("//"))
    data = json.loads(raw_text) if raw_text.strip() else {}
    tokens = data.get("copilotTokens")
    return list(tokens.values()) if isinstance(tokens, dict) else []


def _resolve_copilot_catalog_api_key() -> str:
    """Best-effort GitHub token for the Copilot catalog: env vars / ``gh auth token`` via
    ``resolve_api_key_provider_credentials``, then ``auth.json`` ``credential_pool.copilot[]``, then
    ``~/.copilot/config.json`` ``copilotTokens`` (the ACP CLI's own store). Without the latter two,
    keyless users see the picker fall back to the stale curated list on a silent 401."""
    def _pool_token() -> str:
        from hermes_cli.auth import read_credential_pool

        return _first_exchangeable_copilot_token(
            entry.get("access_token") for entry in read_credential_pool("copilot") if isinstance(entry, dict))

    sources = (
        lambda: _api_key_credentials("copilot")[0],
        _pool_token,
        lambda: _first_exchangeable_copilot_token(_copilot_cli_config_tokens()),
    )
    for source in sources:
        try:
            token = source()
        except Exception:
            continue
        if token:
            return token
    return ""




def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """models.dev entries first (their order), then curated-only extras, case-insensitively deduped
    while preserving curated casing. Curated unchanged when models.dev is unreachable/empty."""
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []
    if not mdev:
        return list(curated)
    return _merge_unique(_merge_unique([], mdev), curated)


def _openai_discovery_base_url(provider: str) -> str:
    """OpenAI endpoint for model discovery, mirroring runtime precedence so discovery probes the SAME
    endpoint inference uses: ``$OPENAI_BASE_URL`` → config ``model.base_url`` (when the configured
    provider matches) → the canonical default."""
    env_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_raw:
        return env_raw
    try:
        model_cfg = _get_model_config_dict()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        same_provider = _normalize_provider(provider) == _normalize_provider(cfg_provider)
        if cfg_provider in ("openai", "openai-api") and same_provider:
            cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_url:
                return cfg_url
    except Exception:
        pass
    return "https://api.openai.com/v1"


def _codex_catalog(normalized: str, force_refresh: bool) -> list[str]:
    from hermes_cli.codex_models import get_codex_model_ids

    # Live OAuth token so the picker matches what ChatGPT lists for this account; hardcoded
    # catalog without a token / when unreachable. Read-only (#68004): a picker never imports,
    # refreshes or persists a credential, so an expired stored token means the hardcoded catalog
    # until the runtime lease refreshes it.
    # The token and the host it is routed to come from the same resolution (#121486): a pooled
    # gateway key is only ever sent to that gateway, never to the chatgpt.com default.
    base_url = None
    try:
        from hermes_cli.auth import _codex_access_token_is_expiring, resolve_codex_runtime_credentials

        creds = resolve_codex_runtime_credentials(read_only=True)
        access_token, base_url = creds.get("api_key"), creds.get("base_url")
        if _codex_access_token_is_expiring(access_token, 0):
            access_token = None
    except Exception:
        access_token = None
    return get_codex_model_ids(access_token=access_token, base_url=base_url)


_COPILOT_ACP_SESSION_MEMO_TTL = 300.0  # 5 min; SWR disk cache handles the rest
_COPILOT_ACP_SESSION_FAIL_TTL = 30.0  # failed probes re-probe quickly so a fresh CLI login is picked up
_copilot_acp_session_memo: Optional[tuple[float, float, Optional[list[str]]]] = None  # (at, ttl, models)


def _copilot_acp_session_models(force_refresh: bool) -> Optional[list[str]]:
    """Enabled models from a signed-in ``copilot --acp`` session, memoized for a few minutes —
    successes AND failures. Model-switch validation (``models_validate._static_catalog``) reads
    this uncached on every ``/model`` switch, and each miss is a CLI spawn + handshake (up to the
    probe timeout), so without the memo every switch paid a subprocess. A failed probe is
    memoized much more briefly so a user who signs in to the CLI right after a miss is picked up
    on the next switch (or immediately via ``/model --refresh``, which clears this memo)."""
    global _copilot_acp_session_memo
    now = time.monotonic()
    memo = _copilot_acp_session_memo
    if not force_refresh and memo is not None and now - memo[0] < memo[1]:
        return memo[2]
    from providers import get_provider_profile

    try:
        live = get_provider_profile("copilot-acp").fetch_models() or None
    except Exception:
        logger.debug("copilot-acp session model discovery failed", exc_info=True)
        live = None
    _copilot_acp_session_memo = (now, _COPILOT_ACP_SESSION_MEMO_TTL if live else _COPILOT_ACP_SESSION_FAIL_TTL, live)
    return live


class CuratedFallbackModels(list[str]):
    """A curated list served because the provider's live catalog was unavailable. The disk cache
    treats it as a placeholder, never as the account's real catalog (#107391)."""


def _copilot_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    if normalized == "copilot-acp" and (live := _copilot_acp_session_models(force_refresh)):
        return live
    try:
        live = _fetch_github_models(_resolve_copilot_catalog_api_key())
        if live:
            return live
    except Exception:
        pass
    return CuratedFallbackModels(catalog_static._PROVIDER_MODELS.get("copilot", []))


def _nous_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
            if live:
                return live
    except Exception:
        pass
    # Live failed / no creds: the docs-hosted manifest — NOT the in-repo snapshot — so newly added
    # Portal models still surface without a Hermes release.
    return get_curated_nous_model_ids() or None


def _api_key_credentials(normalized: str) -> tuple[str, str]:
    """``(api_key, base_url)`` from ``resolve_api_key_provider_credentials``; empty strings on any miss."""
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials(normalized)
        return str(creds.get("api_key") or "").strip(), str(creds.get("base_url") or "").strip()
    except Exception:
        return "", ""


def _api_key_provider_live(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    """Live /v1/models for a simple api-key provider (stepfun, gmi); None on any miss."""
    api_key, base_url = _api_key_credentials(normalized)
    if not (api_key and base_url):
        return None
    try:
        return fetch_api_models(api_key, base_url) or None
    except Exception:
        return None


def _anthropic_catalog(normalized: str, force_refresh: bool) -> list[str]:
    model_cfg = _get_model_config_dict()
    cfg_base_url = cfg_api_key = ""
    if _normalize_provider(str(model_cfg.get("provider", "") or "")) == "anthropic":
        cfg_base_url = str(model_cfg.get("base_url", "") or "").strip()
        cfg_api_key = str(model_cfg.get("api_key", "") or "").strip()
    live = _fetch_anthropic_models(base_url=cfg_base_url or None, api_key=cfg_api_key or None)
    curated = list(catalog_static._PROVIDER_MODELS.get("anthropic", []))
    if not live:
        return curated
    # The live /v1/models dump lags newly-routed curated aliases (reachable before enumerated):
    # curated first, then live-only extras, so a fresh curated model never disappears.
    return live if cfg_base_url else _merge_unique(curated, live)


def _openai_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base = _openai_discovery_base_url(normalized)
    # Custom OpenAI-compatible endpoints serve a small curated catalog — use it verbatim. Official
    # OpenAI hosts (canonical and data-residency regional) return 120+ embeddings/whisper/tts/…
    # entries, so intersect with the curated agentic catalog so ``/model`` matches ``hermes model``.
    # Model not in live /v1/models — check the curated catalog before rejecting. Providers may omit models
    # from their live listing that are still valid (stale cache, partial rollout, gated previews). Use the
    # pure-catalog helper (no extra live fetch) so we only accept models Hermes actually ships. (#46850)
    # Their /v1/models listing is access-scoped and authoritative — a model absent from it is one this key
    # CANNOT serve, so the curated soft-accept would manufacture a selection that 400s at first use. Custom
    # OpenAI-compatible proxies keep the fallback (incomplete listings are common there).
    from providers.routing import is_official_openai_host

    try:
        live = fetch_api_models(api_key, base)
    except Exception:
        live = None
    if not live:
        return None
    if not is_official_openai_host(base):
        return live
    live_lower = {m.lower() for m in live}
    curated = list(catalog_static._PROVIDER_MODELS.get(normalized, []))
    # Curated order, only models the account has access to; an account serving none of them (rare)
    # falls back to curated so the picker still offers sane defaults.
    discovered = [m for m in curated if m.lower() in live_lower]
    # Astra is intentionally absent from offline/static catalogs: the official API's
    # account-scoped /models response is the only source that may advertise it.
    discovered.extend(m for m in live if is_astra_model(m))
    return discovered or curated or live


def _custom_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    base_url = _get_custom_base_url()
    if not base_url:
        return None
    model_cfg = _get_model_config_dict()
    # Try common API key env vars for custom endpoints.
    api_key = (
        str(model_cfg.get("api_key", "") or "").strip()
        or os.getenv("CUSTOM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY", ""))
    from providers.routing import endpoint_api_mode
    api_mode = endpoint_api_mode(base_url)
    return fetch_api_models(api_key, base_url, api_mode=api_mode) or None


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids.
    try:
        from agent.bedrock_adapter import bedrock_model_ids_or_none

        return bedrock_model_ids_or_none()
    except Exception:
        return None


def _azure_foundry_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    """Live ``GET <base>/models`` of the configured Azure Foundry resource (#27989).

    Deployments are per-resource, so the static catalog is intentionally empty and the plugin
    profile ships ``base_url=""`` — which is why the generic profile fetch never fires. Resolve
    through the runtime resolver so the picker targets the same resource inference hits
    (``model.base_url`` / ``AZURE_FOUNDRY_BASE_URL``) with the same credential: an API key string,
    or the Entra ID token-provider callable that ``azure_detect`` already accepts. Anthropic-style
    ``/anthropic`` routes serve no ``/models``; the probe never raises, so any miss keeps ``[]``.
    """
    try:
        from hermes_cli.azure_detect import _probe_openai_models
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(requested_provider=normalized, model_cfg=_get_model_config_dict())
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        credential = runtime.get("api_key")
        if not (base_url and credential):
            return None
        ok, ids = _probe_openai_models(base_url, credential)
        return ids if ok and ids else None
    except Exception:
        return None


# Per-provider catalog sources tried before the generic profile fetch. A fetcher returning None
# falls through to the profile/curated path; a list is returned as-is (even empty).
_PROVIDER_CATALOG_FETCHERS: dict[str, Any] = {
    "openrouter": lambda normalized, force_refresh: model_ids(force_refresh=force_refresh),
    "openai-codex": _codex_catalog,
    "copilot": _copilot_catalog,
    "copilot-acp": _copilot_catalog,
    "nous": _nous_catalog,
    "stepfun": _api_key_provider_live,
    "gmi": _api_key_provider_live,
    "anthropic": _anthropic_catalog,
    "ai-gateway": lambda normalized, force_refresh: _fetch_ai_gateway_models() or None,
    # DeepInfra's generic /models mixes chat, image, video, speech and embedding models; the tagged
    # catalog helper is the only safe source for the chat picker, including its empty/failure result.
    "deepinfra": lambda normalized, force_refresh: deepinfra_model_ids("chat", force_refresh=force_refresh),
    "ollama-cloud": lambda normalized, force_refresh: fetch_ollama_cloud_models(force_refresh=force_refresh) or None,
    "openai": _openai_catalog,
    "openai-api": _openai_catalog,
    "custom": _custom_catalog,
    "bedrock": _bedrock_catalog,
    "azure-foundry": _azure_foundry_catalog}


# ``-free`` slugs the relay still LISTS but no longer serves: the Go-only twin (``ox-alpha-free``)
# and the promo it delisted without removing from ``/models`` (``deepseek-v4-flash-free``). The
# live-first keyed Zen/Go pickers filter through this so a stale live listing can never route
# into a 400/403 (#111749).


def _profile_live_catalog(normalized: str) -> Optional[list[str]]:
    """Generic live fetch for any provider registered in providers/ with ``auth_type="api_key"``.

    Live results are merged with the curated list so models the live endpoint omits still appear:
    curated-first by default so the newest curated models lead when the live API lags;
    ``catalog_static._LIVE_FIRST_PICKER_PROVIDERS`` (OpenCode Zen/Go, authoritative live API) live-first so stale
    curated entries stop polluting the top. Plugin providers without a static entry use the
    profile's ``fallback_models`` as the curated list (Fireworks lists an image model first).
    """
    from providers import get_provider_profile

    profile = get_provider_profile(normalized)
    if not profile:
        return None
    # external_process providers (ACP agent CLIs) have no api_key/base_url credentials: the
    # profile's fetch_models drives its own subprocess (kwargs are ignored per the base contract).
    # Every non-api-key profile falls back to its own fallback_models (OAuth plugins have no
    # static catalog_static._PROVIDER_MODELS row), exactly as api_key plugins do below.
    if profile.auth_type == "external_process":
        try:
            live = profile.fetch_models()
        except Exception as exc:  # a failed subprocess launch degrades to the curated list, like api_key below
            logger.debug("external_process catalog fetch failed for %s: %s", normalized, exc)
            live = None
        # Same merge as setup (`_model_flow_plugin_provider`) so /model, the Desktop picker and
        # `hermes model` offer one list: live ids plus any pinned id the probe omitted.
        return merge_profile_catalog(normalized, profile, list(live) if live else None)
    if not (profile.auth_type == "api_key" and profile.base_url):
        return list(profile.fallback_models) or None
    api_key, base_url = _api_key_credentials(normalized)
    return probe_profile_catalog(normalized, profile, api_key, base_url or profile.base_url or None)


def probe_profile_catalog(normalized: str, profile, api_key: Optional[str], base_url: Optional[str]) -> Optional[list[str]]:
    """``profile.fetch_models`` gated on a key (no key → no doomed probe) and merged with the curated
    list; a raising catalog override degrades like a None return — fallback_models, not an empty picker."""
    live = None
    if api_key:
        try:
            live = profile.fetch_models(api_key=api_key, base_url=base_url)
        except Exception:
            live = None
    return merge_profile_catalog(normalized, profile, live)


def merge_profile_catalog(normalized: str, profile, live: Optional[list[str]]) -> Optional[list[str]]:
    """Combine a profile's live catalog with its curated list the way the ``/model`` picker does, so
    first-time setup (``model_setup_flows._api_key_provider_model_list``) offers the same rows the
    picker will later show. Empty live → ``fallback_models`` (None when the profile has none)."""
    if not live:
        rows = CuratedFallbackModels(profile.fallback_models) if profile.fallback_models else None
    else:
        curated = list(catalog_static._PROVIDER_MODELS.get(normalized, [])) or list(profile.fallback_models or ())
        if not curated:
            rows = live
        else:
            rows = merge_profile_models(normalized, live, curated)
    return _drop_delisted_opencode_models(normalized, rows)






def _configured_relay_base_url(provider: str) -> str:
    """``model.base_url`` when it points the *configured* provider at a relay/proxy, else "".

    Discovery must probe the same endpoint inference uses (#121387): when ``model.base_url``
    differs from the provider's own endpoint, the vendor's canonical host is NOT the catalog to list.
    Mirrors the ``$OPENAI_BASE_URL`` -> ``model.base_url`` -> canonical precedence of
    ``_openai_discovery_base_url`` for every built-in provider, not just OpenAI.
    """
    try:
        model_cfg = _get_model_config_dict()
    except Exception:
        return ""
    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    if not cfg_provider or not provider:
        return ""
    try:
        normalized = _normalize_provider(provider)
        if normalized != _normalize_provider(cfg_provider):
            return ""
    except Exception:
        return ""
    base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return ""
    # A base_url equal to the provider's own endpoint is not a relay (setup persists canonical
    # URLs too): keep native discovery, which OAuth providers such as Codex need because the
    # generic relay probe only speaks api_key. Profiles cover providers PROVIDER_REGISTRY lacks
    # (OpenRouter).
    try:
        from providers import get_provider_profile

        canonical = getattr(get_provider_profile(normalized), "base_url", "") or ""
    except Exception:
        return base_url  # lookup failed: stay a relay, never widening where credentials go
    if canonical and normalize_route_base_url(base_url) == normalize_route_base_url(canonical):
        return ""
    return base_url


def _relay_model_catalog(normalized: str, relay: str) -> Optional[list[str]]:
    """Live catalog probed at a configured ``model.base_url`` relay, or None to fall through.

    Returns only the relay's live ids (no curated merge): a relay user must see the relay's
    catalog, and a failed/empty probe degrades to the canonical fetchers untouched.
    """
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(normalized)
        if profile is None or getattr(profile, "auth_type", "") != "api_key":
            return None
        api_key, _ = _api_key_credentials(normalized)
        live = profile.fetch_models(api_key=api_key, base_url=relay)
        return [str(m) for m in (live or []) if m] or None
    except Exception:
        return None


# Canonical fetchers that already resolve `model.base_url` themselves for the configured
# provider AND degrade to their curated list when that relay fails — `_anthropic_catalog`,
# `_custom_catalog`, `_openai_catalog` (via `_openai_discovery_base_url`) and the simple
# api-key fetchers (via `resolve_api_key_provider_credentials`). They already satisfy the
# "no vendor egress when a relay is configured" invariant, so intercepting them would only
# override correct, better-merged behaviour. Everything else is vendor-pinned (#121387).
_RELAY_AWARE_CATALOG_FETCHERS = frozenset(
    {"anthropic", "custom", "openai", "openai-api", "stepfun", "gmi"}
)


def _static_catalog(normalized: str, fetcher: Any) -> list[str]:
    """The local, no-egress catalog tail: curated static list (+ models.dev merge where preferred).

    Shared by the normal path's final fallback and by the configured-relay degrade path, which
    must never reach a live vendor fetcher (#121387).
    """
    # Merge static curated list with live API results so models that the live endpoint omits (stale cache,
    # partial rollout) still appear in the picker. Single providers (kimi, zai) use curated-first (commit
    # 658ac1d86) to surface newest models even when live API lags (#46309). OpenCode Zen / Go are different:
    # their live API is the authoritative catalog, so they merge live-first — live entries lead and stale
    # curated entries no longer pollute the top of the picker. (#49129) Plugin providers with no static
    # catalog_static._PROVIDER_MODELS entry fall back to the profile's curated fallback_models so their agentic picks lead
    # the picker instead of whatever the live catalog happens to return first (e.g. Fireworks lists an image
    # model, flux-*, ahead of its chat models).
    # A provider with a live fetcher that declined is serving a placeholder; one without any live
    # source is serving its authoritative catalog.
    curated_static = (CuratedFallbackModels if fetcher is not None else list)(catalog_static._PROVIDER_MODELS.get(normalized, []))
    if normalized not in catalog_static._MODELS_DEV_PREFERRED:
        return _chat_catalog_rows(_drop_delisted_opencode_models(normalized, curated_static))
    # models.dev keeps listing retired Zen ids too: filter after the merge, not before.
    merged = _drop_delisted_opencode_models(normalized, _merge_with_models_dev(normalized, curated_static))
    return _chat_catalog_rows(catalog_static._xai_finalize_catalog(merged) if normalized in {"xai", "xai-oauth"} else merged)


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Best known model catalog for a provider: per-provider live fetchers, then the generic profile
    fetch, then the static list (merged with models.dev for ``catalog_static._MODELS_DEV_PREFERRED`` providers)."""
    requested = str(provider or "").strip().lower()
    if requested == "ollama":
        return _ollama_local_catalog(force_refresh)

    normalized = _normalize_provider(provider)
    # A configured `model.base_url` relay is TERMINAL for live catalog egress: the picker must
    # list what the configured endpoint serves and must never touch the vendor host (#121387).
    # A failed or empty probe degrades to the local curated list — falling through to the
    # canonical fetchers would send the provider credential to exactly the host the user
    # deliberately routed away from, recreating the bug on the failure path.
    relay = _configured_relay_base_url(provider or "")
    if relay and normalized not in _RELAY_AWARE_CATALOG_FETCHERS:
        relayed = _relay_model_catalog(normalized, relay)
        if relayed:
            return _chat_catalog_rows(relayed)
        return _static_catalog(normalized, _PROVIDER_CATALOG_FETCHERS.get(normalized))
    fetcher = _PROVIDER_CATALOG_FETCHERS.get(normalized)
    if fetcher is not None:
        models = fetcher(normalized, force_refresh)
        if models is not None:
            return _chat_catalog_rows(models)
    try:
        models = _profile_live_catalog(normalized)
    except Exception:
        models = None
    if models is not None:
        return _chat_catalog_rows(models)

    return _static_catalog(normalized, fetcher)


# ---------------------------------------------------------------------------
# Disk cache for provider_model_ids() — keeps /model picker fast (otherwise every open re-fetches
# every authed provider's /v1/models). One JSON file at $HERMES_HOME/provider_models_cache.json;
# entries keyed by credential fingerprint (rotate OPENAI_API_KEY → entry invalidates); 1h TTL;
# only NON-EMPTY results are cached so a transient failure is never pinned; any read/write error
# degrades silently to a live fetch.
# ---------------------------------------------------------------------------

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h
# Stale-while-revalidate window: an expired same-credentials entry is served IMMEDIATELY while a
# daemon thread refreshes the disk cache; beyond this bound the caller blocks on a live fetch.
# Catalogs change on release timescales, so hour-old data beats stalling every picker surface.
_PROVIDER_MODELS_STALE_SERVE_MAX = 7 * 24 * 3600  # 7d
# A curated fallback row is a placeholder for an outage, not a catalog: re-probe soon and never
# serve it through the stale window.
_PROVIDER_MODELS_FALLBACK_TTL = 60

# Cache keys with a background SWR refresh in flight — dedupes concurrent refreshes.
_swr_refresh_inflight: set = set()
_swr_refresh_lock = threading.Lock()


def _cache_entry(fp: str, models: list[str], at: Optional[float] = None) -> dict:
    """One provider row of the disk cache: credential fingerprint, write time, model ids."""
    return {"fp": fp, "at": time.time() if at is None else at, "models": list(models)}


def _live_result_entry(fp: str, live: list[str], existing: Any, at: Optional[float] = None) -> Optional[dict]:
    """Row to store for a ``provider_model_ids`` result, or ``None`` to keep *existing*: a curated
    fallback never replaces the account's real catalog for the same credentials, and when it is
    stored it is flagged so it expires on the short fallback TTL."""
    if not isinstance(live, CuratedFallbackModels):
        return _cache_entry(fp, live, at)
    if _cache_entry_valid(existing, fp) and not existing.get("fallback"):
        return None
    return {**_cache_entry(fp, live, at), "fallback": True}


def _ollama_native_probe_reachable() -> bool:
    """Whether the configured local Ollama root answered the native ``/api/tags`` probe (an empty
    catalog from a reachable server is authoritative; a failed probe is not)."""
    base_url = models_local._get_ollama_base_url()
    headers = models_local._get_ollama_native_headers(base_url) or None
    probe_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers)
    return _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True


def _spawn_swr_refresh(cache_key: str, refresh_fn=None) -> None:
    """Fire-and-forget daemon refresh of *cache_key*'s cache entry, at most one in flight per key.
    Failures are swallowed — the stale entry stays served until a later refresh succeeds.
    ``refresh_fn`` (no-args → fresh entry dict or None) lets ``custom:<base_url>`` keys from
    :func:`cached_fetch_api_models` reuse the same inflight-dedupe scaffolding."""
    # Under a routed profile the inflight key includes the home: the same provider slug names a
    # different disk cache and credential set per profile, so one profile's refresh must not
    # suppress another's. Unscoped keeps the bare key (tests inspect the set by slug).
    from hermes_constants import get_hermes_home_override, hermes_home_key
    inflight_key = cache_key if get_hermes_home_override() is None else (hermes_home_key(), cache_key)
    with _swr_refresh_lock:
        if inflight_key in _swr_refresh_inflight:
            return
        _swr_refresh_inflight.add(inflight_key)

    def _default_refresh():
        live = provider_model_ids(cache_key, force_refresh=True)
        if live or (cache_key == "ollama" and _ollama_native_probe_reachable()):
            fp = _credential_fingerprint(cache_key)
            return _live_result_entry(fp, live or [], _load_provider_models_cache().get(cache_key))
        return None

    def _refresh() -> None:
        try:
            entry = (refresh_fn or _default_refresh)()
            if entry:
                # Under the write lock: the GUI read path spawns one of these per stale provider, so
                # the plain load-modify-save would let concurrent warms drop each other's rows.
                with _cache_write_lock:
                    _store_cache_entry(cache_key, entry)
        except Exception:
            logger.debug("SWR refresh failed for %s", cache_key, exc_info=True)
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.discard(inflight_key)

    # copy_context: the refresh must read the calling profile's credentials and write ITS disk cache.
    ctx = contextvars.copy_context()
    threading.Thread(target=lambda: ctx.run(_refresh), daemon=True, name=f"model-cache-swr-{cache_key}").start()


def _provider_models_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Short hash of the credentials ``provider_model_ids(provider)`` would see right now.

    API-key providers include their configured values and credential-file mtimes. Codex uses the
    stable principal selected by its read-only resolver: routine token and pool-state writes must
    not discard an account-scoped catalog, while a real account switch must invalidate it.
    """
    import hashlib

    parts: list[str] = []
    try:
        from hermes_cli.provider_auth import get_provider_config
        pcfg = get_provider_config(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={os.environ.get(bev, '')}")
    except Exception:
        pass

    # External-process providers discover models through the launched program, so the command /
    # argv env overrides identify the catalog the way an API key identifies an HTTP catalog.
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        if profile is not None and profile.auth_type == "external_process":
            for ev in (*profile.process_command_env_vars, profile.process_args_env_var):
                if ev:
                    parts.append(f"{ev}={os.environ.get(ev, '')}")
    except Exception:
        pass

    # config.yaml's model.base_url changes the endpoint discovery probes (data-residency hosts)
    # without touching any env var, so it must change the fingerprint too.
    if provider in ("openai", "openai-api"):
        try:
            parts.append(f"effective_base={_openai_discovery_base_url(provider)}")
        except Exception:
            pass

    # Azure Foundry deployments are per-resource and the wizard writes only model.base_url, so a
    # resource switch under the same key must not serve the previous resource's catalog (#27989).
    if provider == "azure-foundry":
        try:
            from hermes_cli.runtime_provider import _config_base_url_for_provider
            parts.append(f"effective_base={_config_base_url_for_provider(_get_model_config_dict(), 'azure-foundry')}")
        except Exception:
            pass

    if provider == "ollama":
        provider_cfg = _get_provider_config_dict("ollama")
        key_env = provider_cfg.get("key_env") or provider_cfg.get("api_key_env") or ""
        model_cfg = _get_model_config_dict()
        parts += [
            f"OLLAMA_HOST={os.environ.get('OLLAMA_HOST', '')}",
            "providers.ollama.base_url="
            f"{provider_cfg.get('base_url', '') or provider_cfg.get('api', '') or provider_cfg.get('url', '')}",
            f"providers.ollama.api_key={provider_cfg.get('api_key', '')}",
            f"providers.ollama.key_env={key_env}",
        ]
        if key_env:
            parts.append(f"{key_env}={os.environ.get(str(key_env), '')}")
        parts += [
            f"model.provider={model_cfg.get('provider', '')}|model.base_url={model_cfg.get('base_url', '')}",
            "providers.ollama.extra_headers="
            + json.dumps(provider_cfg.get("extra_headers", {}), sort_keys=True, default=str),
        ]

    def _mtime_part(label: str, path) -> None:
        try:
            parts.append(f"{label}@{os.stat(path).st_mtime_ns}")
        except FileNotFoundError:
            parts.append(f"{label}@missing")
        except Exception:
            pass

    if provider == "openai-codex":
        from hermes_cli.codex_models import codex_catalog_credential_identity

        parts.append(f"codex_identity={codex_catalog_credential_identity()}")
    else:
        try:
            from hermes_constants import get_hermes_home
            for rel in ("auth.json", "credentials.json"):
                _mtime_part(rel, get_hermes_home() / rel)
        except Exception:
            pass
        for rel in ("~/.codex/auth.json", "~/.claude/.credentials.json",
                    "~/.config/github-copilot/hosts.json", "~/.minimax/credentials.json"):
            path = os.path.expanduser(rel)
            _mtime_part(path, path)

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b, not sha256: fingerprint only (collisions = a harmless cache miss), and CodeQL's
    # weak-sensitive-data-hashing rule flags sha256 over env vars named *API_KEY*/*TOKEN*.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        return _read_json_cache(_provider_models_cache_path()) or {}
    except Exception:
        return {}


_cache_write_lock = threading.Lock()


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        _write_json_cache(_provider_models_cache_path(), data, indent=None)
    except Exception:
        pass


def _store_cache_entry(cache_key: str, entry: dict, cache: Optional[dict] = None) -> None:
    """Write one row into the disk cache (reloading the latest state unless ``cache`` is given)."""
    if cache is None:
        cache = _load_provider_models_cache()
    cache[cache_key] = entry
    _save_provider_models_cache(cache)


def update_provider_cache_entry(provider: str, models: list[str]) -> None:
    """Thread-safe single-entry update for parallel prefetch workers: load-modify-save under a lock
    so concurrent fetches don't clobber each other's rows. Best-effort, silent on any error."""
    try:
        normalized = _normalize_provider(provider) or (provider or "")
        if not normalized or not models:
            return
        fp = _credential_fingerprint(normalized)
        with _cache_write_lock:
            _store_cache_entry(normalized, _cache_entry(fp, models))
    except Exception:
        pass


def _normalized_cache_slug(provider: Optional[str]) -> str:
    """``ollama`` stays a raw slug (its alias would canonicalize to ``custom``); everything else normalizes."""
    requested = str(provider or "").strip().lower()
    return requested if requested == "ollama" else (_normalize_provider(provider or "openrouter") or (provider or ""))


def _model_requires_account_discovery(provider: Optional[str], model: str) -> bool:
    """Astra names cannot confer API/OAuth entitlement through picker state."""
    return _normalized_cache_slug(provider) in {"openai", "openai-api", "openai-codex"} and is_astra_model(model)


def cached_provider_model_ids(
    provider: Optional[str], *, force_refresh: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL, non_blocking: bool = False) -> list[str]:
    """Disk-cached :func:`provider_model_ids`: fresh cache hit, else live fetch persisting a non-empty
    result. Always returns a list.

    ``non_blocking`` marks the GUI read path (``model.options``): it NEVER waits on a provider probe.
    A same-credentials row of any age is served as-is and a daemon thread warms the next open; a
    cold/mismatched row returns ``[]`` so the caller keeps its curated list. One degraded provider
    (hanging or timing-out ``/v1/models``) therefore delays nothing but itself (#114215)."""
    normalized = _normalized_cache_slug(provider)
    if not normalized:
        return []
    is_ollama = normalized == "ollama"

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    if not force_refresh:
        tier = _disk_serve_tier(entry, fp, now, is_ollama=is_ollama, ttl_seconds=ttl_seconds)
        if tier is not None:
            if tier == "stale":
                _spawn_swr_refresh(normalized)
            return _chat_catalog_rows(list(entry["models"]))

    if non_blocking and not force_refresh:
        # Read path: never touch the network in the caller's thread. A same-credentials row past the
        # SWR window is still served (hour-old catalog beats an empty picker) while a daemon thread
        # warms the next open; a cold row returns [] and the caller falls back to its curated list.
        _spawn_swr_refresh(normalized)
        if _cache_entry_valid(entry, fp, allow_empty=is_ollama):
            return _chat_catalog_rows([
                model for model in entry["models"]
                if not _model_requires_account_discovery(normalized, model)])
        return []

    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        fresh = _live_result_entry(fp, live, entry, now)
        if fresh is None:
            # The live fetch degraded to the curated list; the account's real catalog is on disk.
            return _chat_catalog_rows([model for model in entry["models"] if not _model_requires_account_discovery(normalized, model)])
        _store_cache_entry(normalized, fresh, cache)
        return _chat_catalog_rows(list(live))

    if is_ollama:
        if _ollama_native_probe_reachable():
            # A reachable empty native catalog is authoritative; do not resurrect a stale disk catalog.
            _store_cache_entry(normalized, _cache_entry(fp, [], now), cache)
            return []
        # A failed/non-native probe is not authoritative: keep a stale catalog rather than blanking
        # the picker during a transient outage.
        same_creds = isinstance(entry, dict) and entry.get("fp") == fp
        if same_creds and isinstance(entry.get("models"), list) and entry["models"]:
            return _chat_catalog_rows(list(entry["models"]))
        return []
    # Live returned nothing: a stale same-fingerprint entry beats an empty result — minus account-gated
    # models, which only a successful discovery may advertise (the entry itself is untouched, so the
    # next successful fetch restores them).
    if _cache_entry_valid(entry, fp):
        return _chat_catalog_rows([model for model in entry["models"] if not _model_requires_account_discovery(normalized, model)])
    return []


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop one provider's cache entry, or wipe the whole cache (``provider=None``). Used by
    ``/model --refresh`` and ``hermes model --refresh``."""
    try:
        # Native Ollama tags are keyed by root URL, not provider slug — a targeted refresh can't
        # identify the root from the name alone, so clear this small in-process cache every time.
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
        # A fresh copilot-acp CLI login must be visible to the next /model switch (this helper is
        # what ``--refresh`` runs): don't let the 5-min session memo (or its failure memo) serve
        # a stale signed-out probe past an explicit refresh.
        global _copilot_acp_session_memo
        _copilot_acp_session_memo = None
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        normalized = _normalized_cache_slug(provider)
        if normalized in cache:
            del cache[normalized]
            _save_provider_models_cache(cache)
    except Exception:
        pass


def _resolve_anthropic_pool_catalog_credentials() -> tuple[str, str]:
    """Read-only API-key pool credential for model discovery (``resolve_anthropic_token()`` ignores
    ``api_key`` pool entries — its runtime contract is OAuth-oriented)."""
    try:
        from agent.credential_pool import AUTH_TYPE_API_KEY
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if not isinstance(entry, dict) or entry.get("auth_type") != AUTH_TYPE_API_KEY:
                continue
            token = str(entry.get("access_token") or "").strip()
            if token:
                return token, str(entry.get("base_url") or entry.get("inference_base_url") or "").strip()
    except Exception:
        pass
    return "", ""


def _fetch_anthropic_models(
    timeout: float = 5.0, *, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[list[str]]:
    """Application-owned credential resolution; provider plugin owns catalogue pagination."""
    try:
        from agent.anthropic_credentials import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    resolved_base_url = base_url
    token = (api_key or "").strip() or resolve_anthropic_token()
    if not token:
        # Never pair a pool credential with a caller-supplied endpoint.
        token, resolved_base_url = _resolve_anthropic_pool_catalog_credentials()
    if not token:
        return None

    try:
        profile = get_provider_profile("anthropic")
        if profile is None:
            return None
        fetch = getattr(profile, "fetch_catalog_models", None)
        if callable(fetch):
            models = fetch(
                api_key=token, base_url=resolved_base_url, timeout=timeout,
                oauth=_is_oauth_token(token), request_json=_get_json,
            )
        elif not _is_oauth_token(token):
            models = profile.fetch_models(api_key=token, base_url=resolved_base_url, timeout=timeout)
        else:
            return None
        if models is None:
            return None
        # Preserve the existing CLI picker presentation ordering.
        return sorted(models, key=lambda m: ("opus" not in m, "sonnet" not in m, "haiku" not in m, m))
    except Exception as exc:
        logger.debug("Failed to fetch Anthropic models: %s", exc)
        return None


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    return (base_url or "").strip().rstrip("/").lower().startswith(
        (COPILOT_BASE_URL, "https://models.github.ai/inference", "https://models.inference.ai.azure.com")
    )


def _fetch_github_models(api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[str]]:
    catalog = _fetch_github_model_catalog(api_key=api_key, timeout=timeout)
    return [item.get("id", "") for item in catalog if item.get("id")] if catalog else None


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None, api_key: Optional[str] = None) -> set[str]:
    if catalog is None and api_key:
        catalog = _fetch_github_model_catalog(api_key=api_key)
    return {mid for item in (catalog or []) if (mid := str(item.get("id") or "").strip())}


def normalize_copilot_model_id(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""
    return normalize_model_id(
        "copilot",
        raw,
        known_ids=_copilot_catalog_ids(catalog=catalog, api_key=api_key),
    )


# Negative cache: monotonic timestamp of the last timed-out probe, keyed
# by ``host:port`` so both URL candidates (``/v1`` + root) share one entry.
# Without this, an unreachable endpoint (TCP blackhole — SYN draws no reply,
# so every attempt burns its full connect timeout) makes every picker open /
# chat turn re-pay the timeout per candidate, and the sequential stalls stack
# past 10s while the Desktop sits on a spinner with no error (#81123). Short
# TTL collapses the burst but still picks up recovery without a restart.
# Mirrors _deepinfra_catalog_neg_cache.
_probe_neg_cache: dict[str, float] = {}
_PROBE_NEG_TTL = 60.0  # seconds


def _probe_neg_key(base_url: str) -> Optional[str]:
    """``host:port`` for *base_url* (both URL candidates share one entry), or None without a host."""
    from utils import base_url_origin

    _, host, port = base_url_origin(base_url)
    return f"{host}:{port}" if host else None


def _probe_result(
    models, probed_url, resolved_base_url, suggested_base_url=None, used_fallback=False
) -> dict[str, Any]:
    return {
        "models": models,
        "probed_url": probed_url,
        "resolved_base_url": resolved_base_url,
        "suggested_base_url": suggested_base_url,
        "used_fallback": used_fallback}


def probe_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, request_headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics (``base`` then ``base±/v1``).
    ``anthropic_messages`` mode sends ``x-api-key`` + ``anthropic-version`` instead of a bearer; the
    ``data[].id`` response shape is identical. ``models`` is None when no candidate answered."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return _probe_result(None, None, "")
    if _is_github_models_base_url(normalized):
        models = _fetch_github_models(api_key=api_key, timeout=timeout)
        return _probe_result(models, COPILOT_MODELS_URL, COPILOT_BASE_URL)

    alternate_base = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    _neg_key = _probe_neg_key(normalized)
    if _neg_key is not None:
        _neg_seen = _probe_neg_cache.get(_neg_key)
        if _neg_seen is not None and (time.monotonic() - _neg_seen) < _PROBE_NEG_TTL:
            return _probe_result(
                None, normalized.rstrip("/") + "/models", normalized,
                alternate_base if alternate_base != normalized else None)
    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    if urllib.parse.urlparse(normalized).hostname == "generativelanguage.googleapis.com":
        headers["X-Goog-Api-Client"] = f"hermes-agent/{get_version_info().base_version}"
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(COPILOT_BASE_URL):
        headers.update(copilot_request_headers())
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain secrets: merge last so endpoint config wins; never log.
        from hermes_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    # Only thread ssl_context when a per-provider TLS override applies; public endpoints keep the
    # original 2-arg call so existing call-seam mocks stay valid.
    _open_kwargs: dict[str, Any] = {}
    _ssl_context = _custom_provider_ssl_context(normalized)
    if _ssl_context is not None:
        _open_kwargs["ssl_context"] = _ssl_context
    all_timed_out = True
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        try:
            data = _get_json(url, timeout=timeout, headers=headers, **_open_kwargs)
        except Exception as exc:
            # TLS, authentication and parsing failures must not hide corrected settings.
            cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            all_timed_out = all_timed_out and isinstance(cause, TimeoutError)
            continue
        if _neg_key is not None:
            _probe_neg_cache.pop(_neg_key, None)
        from models.catalog_chat import note_catalog_item

        probed = []
        for item in data.get("data", []):
            if isinstance(item, dict) and note_catalog_item(item):
                continue
            probed.append(item.get("id", ""))
        return _probe_result(
            probed, url, candidate_base.rstrip("/"),
            alternate_base if alternate_base != candidate_base else normalized, is_fallback)

    if _neg_key is not None and all_timed_out:
        _probe_neg_cache[_neg_key] = time.monotonic()
    return _probe_result(
        None, tried[0] if tried else normalized.rstrip("/") + "/models", normalized,
        alternate_base if alternate_base != normalized else None)


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": _HERMES_USER_AGENT}
    try:
        url = base_url.rstrip("/") + "/models"
        data = _get_json(url, timeout=timeout, headers=headers)
        return [
            m["id"] for m in data.get("data", [])
            if m.get("id") and m.get("type") == "language" and "tool-use" in (m.get("tags") or [])]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint."""
    result = probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode, request_headers=headers)
    return result.get("models")


def _custom_endpoint_fingerprint(
    api_key: Any, api_mode: Optional[str], headers: Optional[dict[str, str]]) -> str:
    """Custom endpoints have no canonical provider-config slug, so hash exactly what callers pass to
    :func:`fetch_api_models`: a rotated ``api_key``, changed ``api_mode`` or edited ``extra_headers``
    each bust the cache entry. blake2b for the same CodeQL rationale as ``_credential_fingerprint``."""
    import hashlib

    from agent.command_token_source import CommandTokenSource
    identity = api_key.cache_identity if isinstance(api_key, CommandTokenSource) else api_key
    blob = "|".join((identity or "", api_mode or "", json.dumps(headers or {}, sort_keys=True)))
    return hashlib.blake2b(blob.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _cache_entry_valid(
    entry: Any, fp: str, *, allow_empty: bool = False) -> "TypeGuard[dict[str, Any]]":
    """Well-formed cache row for fingerprint *fp*. Requires a numeric ``at`` so corrupt disk state
    degrades to a cache miss instead of raising; empty model lists are valid only when the caller
    opts into an authoritative empty catalog."""
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]))
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool))


def _disk_serve_tier(entry: Any, fp: str, now: float, *, is_ollama: bool,
                     ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL) -> Optional[str]:
    """How :func:`cached_provider_model_ids` serves *entry* without the network.

    ``"fresh"`` inside the row's TTL (a curated fallback row only for
    ``_PROVIDER_MODELS_FALLBACK_TTL``), ``"stale"`` for a non-empty, non-fallback row inside
    ``_PROVIDER_MODELS_STALE_SERVE_MAX`` (served while an SWR thread revalidates), else ``None``:
    the call would block on a live fetch. Empty native catalogs are authoritative only inside the
    short native TTL, never through the stale window."""
    if is_ollama:
        ttl_seconds = min(ttl_seconds, _OLLAMA_LOCAL_MODELS_CACHE_TTL)
    if not _cache_entry_valid(entry, fp, allow_empty=is_ollama):
        return None
    age = now - entry["at"]
    if age < (_PROVIDER_MODELS_FALLBACK_TTL if entry.get("fallback") else ttl_seconds):
        return "fresh"
    if entry["models"] and not entry.get("fallback") and age < _PROVIDER_MODELS_STALE_SERVE_MAX:
        return "stale"
    return None


def cached_fetch_api_models(
    api_key: Any, base_url: Optional[str], *, timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
    force_refresh: bool = False, cache_only: bool = False,
    fetch_models=None,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL) -> Optional[list[str]]:
    """Disk-cached :func:`fetch_api_models` for custom endpoints. ``cache_only`` callers (GUI picker
    opens that must not block on a stopped local endpoint) still get a warm catalog instead of
    collapsing to the config-declared subset. ``fetch_models`` supplies native-aware discovery
    without minting a command token before cache admission."""
    from application_provider_discovery import _NativePickerModelList

    def _catalog(entry):
        rows = (_NativePickerModelList if entry.get("native_catalog") else list)(entry["models"])
        return _chat_catalog_rows(rows)

    def _entry(live, at=None):
        return {**_cache_entry(fp, live, at), "native_catalog": isinstance(live, _NativePickerModelList)}

    def _live():
        if fetch_models is not None:
            return fetch_models()
        from agent.command_token_source import materialize_probe_api_key
        return fetch_api_models(materialize_probe_api_key(api_key), base_url, timeout=timeout, api_mode=api_mode, headers=headers)

    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    if not normalized_url:  # nothing to key the cache on
        return None if cache_only else _chat_catalog_rows(_live())

    # Key on URL AND credential fingerprint: N ``custom_providers`` rows can share one proxy URL
    # with distinct keys (#106184). A URL-only key let the last probe overwrite its siblings'
    # slot, so every other same-URL row failed the fingerprint check, got an empty catalog and
    # vanished from the no-probe pickers.
    fp = _custom_endpoint_fingerprint(api_key, api_mode, headers)
    cache_key = f"custom:{normalized_url}#{fp}"
    cache = _load_provider_models_cache()
    entry = cache.get(cache_key)
    now = time.time()
    native_row = isinstance(entry, dict) and entry.get("native_catalog") is True
    valid = not force_refresh and _cache_entry_valid(entry, fp, allow_empty=native_row)

    if valid:
        age = now - entry["at"]
        if age < ttl_seconds:
            return _catalog(entry)
        # An empty native catalog is authoritative only inside the TTL (as in
        # cached_provider_model_ids): never stale-serve it, or an Ollama that was model-less at
        # first open keeps an empty row for the whole stale window after models are pulled.
        if entry["models"] and age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve now, refresh off-thread for the next open. cache_only
            # opens (GUI pickers that must not block on a stopped local server) take the same
            # non-blocking refresh: without it a locally loaded model stayed invisible for the
            # whole 7-day stale window unless the user found "Refresh Models" (#71169 class).
            def _refresh_custom():
                live = _live()
                return _entry(live) if live or isinstance(live, _NativePickerModelList) else None

            _spawn_swr_refresh(cache_key, _refresh_custom)
            return _catalog(entry)

    if cache_only:
        return None

    live = _live()
    if live or isinstance(live, _NativePickerModelList):
        stored = _entry(live, now)
        _store_cache_entry(cache_key, stored, cache)
        return _catalog(stored)
    # Live returned nothing (offline, timeout, auth hiccup): a stale same-fingerprint entry beats it
    # (non-empty only: an empty native row is not worth resurrecting over the generic fallback).
    if _cache_entry_valid(entry, fp):
        return _catalog(entry)
    return _chat_catalog_rows(live)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import NamedTuple  # noqa: F401,E402
from difflib import get_close_matches  # noqa: F401,E402
import http.client  # noqa: F401,E402

def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Prefer the Portal's explicit ``paid_service_access.allowed`` entitlement
    decision.  Legacy payloads fall back to ``subscription.monthly_charge == 0``.
    Returns False when both signals are missing or unparseable.
    """
    paid_access = account_info.get("paid_service_access")
    if isinstance(paid_access, dict):
        allowed = paid_access.get("allowed")
        if isinstance(allowed, bool):
            return not allowed
        paid = paid_access.get("paid_access")
        if isinstance(paid, bool):
            return not paid

    sub = account_info.get("subscription")
    if not isinstance(sub, dict):
        return False
    charge = sub.get("monthly_charge")
    if charge is None:
        return False
    try:
        return float(charge) == 0
    except (TypeError, ValueError):
        return False

_PLUGIN_COMPAT_LAZY = {
    'LMStudioLoadResult': ('hermes_cli.models_local', 'LMStudioLoadResult'),
    'PROVIDER_GROUPS': ('application_provider_groups', 'PROVIDER_GROUPS'),
    'ProviderEntry': ('hermes_cli.provider_catalog', 'ProviderEntry'),
    'atomic_json_write': ('utils', 'atomic_json_write'),
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'compute_sale_discount': ('models.metadata.pricing', 'compute_sale_discount'),
    'ensure_lmstudio_model_loaded': ('hermes_cli.models_local', 'ensure_lmstudio_model_loaded'),
    'fetch_ai_gateway_pricing': ('application_model_pricing', 'fetch_ai_gateway_pricing'),
    'fetch_lmstudio_models': ('hermes_cli.models_local', 'fetch_lmstudio_models'),
    'fetch_models_with_pricing': ('application_model_pricing', 'fetch_models_with_pricing'),
    'fetch_ollama_local_models': ('hermes_cli.models_local', 'fetch_ollama_local_models'),
    'get_cached_nous_inference_base_url': ('application_model_pricing', 'get_cached_nous_inference_base_url'),
    'get_pricing_for_provider': ('application_model_pricing', 'get_pricing_for_provider'),
    'group_providers': ('application_provider_groups', 'group_providers'),
    'lmstudio_model_reasoning_options': ('models.metadata.local', 'lmstudio_model_reasoning_options'),
    'nous_policy_allowed_ids': ('application_model_pricing', 'nous_policy_allowed_ids'),
    'ollama_model_supports_thinking': ('models.metadata.local', 'ollama_model_supports_thinking'),
    'peek_cached_pricing': ('application_model_pricing', 'peek_cached_pricing'),
    'pricing_cache_scope': ('application_model_pricing', 'pricing_cache_scope'),
    'probe_lmstudio_models': ('hermes_cli.models_local', 'probe_lmstudio_models'),
    'probe_ollama_local_models': ('hermes_cli.models_local', 'probe_ollama_local_models'),
    'provider_group_for_slug': ('application_provider_groups', 'provider_group_for_slug'),
    'restrict_to_nous_policy': ('models.catalog_policy', 'restrict_to_nous_policy'),
    'should_use_ollama_native_catalog': ('hermes_cli.models_local', 'should_use_ollama_native_catalog'),
    'url_origin': ('hermes_cli.urllib_security', 'url_origin'),
    'validate_requested_model': ('hermes_cli.models_validate', 'validate_requested_model'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----


def detect_provider_for_model(model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    """Acquire detection facts and apply the canonical model-selection decision."""
    from hermes_cli.model_selection_facts import build_explicit_detection_facts
    from models.selection_detection import select_detected_model

    raw = str(model_name or "").strip()
    if not raw:
        return None
    selected = select_detected_model(raw, current_provider, build_explicit_detection_facts(raw, current_provider))
    if selected is None or (selected.provider == current_provider and selected.model == raw):
        return None
    return selected.provider, selected.model


def _resolve_provider_prefix(model_name: str) -> Optional[tuple[str, str]]:
    from models.catalog_detection import resolve_declared_provider_prefix
    return resolve_declared_provider_prefix(model_name, _configured_provider_ids())


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    from models.catalog_detection import find_openrouter_slug
    return find_openrouter_slug(model_name, model_ids())
