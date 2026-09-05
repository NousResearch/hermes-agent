"""Picker provider listing: credential discovery, curated/live model lists, row builders for
list_authenticated_providers / list_picker_providers, and the parallel cache prefetch.

Split out of ``hermes_cli/model_switch.py``; every moved name is re-imported there so
``hermes_cli.model_switch.<name>`` keeps resolving (and monkeypatching) as before."""

from __future__ import annotations

import logging
import http.client
import os
import time
import threading as _threading
from dataclasses import dataclass, field
from typing import Any, List, Optional
from hermes_cli.providers import custom_provider_aliases, custom_provider_slug, get_label
from utils import base_url_host_matches

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.model_switch")

# Aggregators whose full catalogs (70+ models) must stay visible: never capped by max_models.
_UNCAPPED_PICKER_PROVIDERS: frozenset[str] = frozenset({"opencode-zen", "opencode-go"})


def _save_discovered_models_to_config(
    api_url: str, model_ids: list[str], *, api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None) -> None:
    """Persist a successful ``/v1/models`` probe into the matching ``custom_providers`` entry.

    Matches by base_url (slash-normalised), api_mode and headers. A failed config write is
    swallowed — the picker still shows the live models for this session."""
    from hermes_cli.model_switch import _extra_headers_from_config
    if not api_url or not model_ids:
        return
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "")).strip()
            if entry_url.rstrip("/").lower() != norm_url or _entry_api_mode(entry) != api_mode:
                continue
            if headers is not None and _extra_headers_from_config(entry) != headers:
                continue
            if not _discovered_catalog_stale(entry, model_ids):
                continue
            entry["models"] = {model_id: {} for model_id in model_ids}
            entry["models_discovered"] = True
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


def _discovered_catalog_stale(entry: dict, model_ids: list[str]) -> bool:
    """Whether a live probe may overwrite ``entry["models"]``.

    A ``models`` mapping or list of dicts is user-curated per-model metadata — never replaced.
    A mapping Hermes itself discovered (entry flag or legacy in-mapping sentinel) is ours to
    refresh, but only when stale; a legacy-shape entry is always rewritten so the save migrates
    it to the clean entry-level flag."""
    existing = entry.get("models")
    legacy_discovered = isinstance(existing, dict) and existing.get("__discovered_model_catalog__") is True
    entry_discovered = entry.get("models_discovered") is True or legacy_discovered
    if isinstance(existing, dict):
        return entry_discovered and (legacy_discovered or list(existing) != model_ids)
    if isinstance(existing, list):
        return not any(isinstance(m, dict) for m in existing) and existing != model_ids
    return True


class _NativePickerModelList(list[str]):
    """A successful native catalog, including an authoritative empty one."""


def _fetch_picker_live_models(
    api_key: str, api_url: str, native_catalog_provider: str, preserve_native_models: bool,
    headers: dict[str, str] | None = None, timeout: float = 5.0,
    api_mode: str | None = None) -> list[str] | None:
    """Fetch picker models with native Ollama and cached generic discovery."""
    from hermes_cli.models import _get_ollama_native_headers, cached_fetch_api_models
    from hermes_cli.models_local import (
        _normalize_openai_base_url,
        fetch_ollama_local_models,
        should_use_ollama_native_catalog,
    )

    candidate_headers = _get_ollama_native_headers(api_url, api_key=api_key)

    def _drop(pred) -> None:
        for key in tuple(candidate_headers):
            if pred(key.lower()):
                del candidate_headers[key]

    caller_has_authorization = any(key.lower() == "authorization" for key in (headers or {}))
    if headers:
        lowered = {existing.lower() for existing in headers}
        _drop(lambda k: k in lowered)
        candidate_headers.update(headers)
    if api_key and not caller_has_authorization:
        _drop(lambda k: k == "authorization")
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(
        native_catalog_provider, api_url, headers=candidate_headers or None)
    resolved_headers = candidate_headers or None if use_native else headers

    if use_native:
        if preserve_native_models:
            return None
        native_models = fetch_ollama_local_models(api_url, timeout=timeout, headers=resolved_headers)
        if native_models is not None:
            return _NativePickerModelList(native_models)
        # A failed native probe is not authoritative: retry the cached generic catalog.
        api_url = _normalize_openai_base_url(api_url)
    generic_models = cached_fetch_api_models(
        api_key, api_url, timeout=timeout, headers=resolved_headers, api_mode=api_mode)
    return generic_models if generic_models or use_native else None


# Process-level guard: the prewarm thread is spawned at most once per process, otherwise a
# long-lived process (or repeated triggers) would leak one OS thread per call.
_picker_prewarm_done = _threading.Event()


def _credential_pool_is_usable(provider: str, *, raw_pool_present: bool = False) -> bool:
    """Whether *provider* has a credential that can be selected now.

    Legacy opaque ``auth.json`` pool values that do not deserialize into ``PooledCredential``
    stay visible (``raw_pool_present``); a real pool's availability is authoritative — an
    all-exhausted/dead pool is not authenticated."""
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider)
        if pool.has_credentials():
            return pool.has_available()
    except Exception:
        pass
    return raw_pool_present


def prewarm_picker_cache_async() -> Optional["_threading.Thread"]:
    """Warm ``provider_models_cache.json`` in a daemon thread by running the picker path once.

    The first ``/model`` open (or the first after the 1h TTL) otherwise blocks ~1-2s on serial
    live ``/v1/models`` fetches. Fire-and-forget, at most once per process, fully
    exception-isolated. Returns the thread (for tests) or None if already warmed."""
    from hermes_cli.model_switch import list_authenticated_providers
    if _picker_prewarm_done.is_set():
        return None
    _picker_prewarm_done.set()

    def _warm() -> None:
        try:
            from hermes_cli.inventory import load_picker_context
            ctx = load_picker_context()
            # The result is discarded; the warm disk cache is the point.
            list_authenticated_providers(
                current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
                current_model=ctx.current_model, user_providers=ctx.user_providers,
                custom_providers=ctx.custom_providers,
                excluded_providers=ctx.excluded_providers or [])
        except Exception:
            logger.debug("picker cache prewarm failed", exc_info=True)

    t = _threading.Thread(target=_warm, daemon=True, name="picker-cache-prewarm")
    t.start()
    return t


def _prefetch_provider_models_parallel(provider_slugs: list[str]) -> None:
    """Fetch stale/missing provider catalogs in parallel before the serial picker loop.

    On a cold cache the serial loop would block 1-8s per provider; after the prefetch the wait is
    the slowest single provider. Each worker re-persists through the thread-safe
    ``update_provider_cache_entry`` so concurrent writes cannot clobber each other."""
    from hermes_cli.models import (
        _PROVIDER_MODELS_CACHE_TTL, _credential_fingerprint, _load_provider_models_cache,
        cached_provider_model_ids, normalize_provider)

    # Read-only staleness check mirroring cached_provider_model_ids (which re-reads the cache
    # itself, so a concurrent change between check and fetch is harmless).
    now = time.time()
    stale_slugs: list[str] = []
    cache = _load_provider_models_cache()
    for slug in provider_slugs:
        normalized = normalize_provider(slug) or (slug or "")
        if not normalized:
            continue
        entry = cache.get(normalized)
        if (
            isinstance(entry, dict) and entry.get("fp") == _credential_fingerprint(normalized)
            and isinstance(entry.get("models"), list) and entry["models"]
            and now - float(entry.get("at", 0)) < _PROVIDER_MODELS_CACHE_TTL):
            continue
        stale_slugs.append(normalized)

    if not stale_slugs:
        return

    import concurrent.futures
    def _fetch_one(slug: str) -> None:
        try:
            models = cached_provider_model_ids(slug, force_refresh=True)
            # cached_provider_model_ids persists via a non-locked read-modify-write; re-persist
            # through the locked path so no write is lost under concurrency.
            if models:
                from hermes_cli.models import update_provider_cache_entry
                update_provider_cache_entry(slug, models)
        except Exception:
            pass  # best-effort; picker falls back to curated list

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(stale_slugs)), thread_name_prefix="model-cache-prefetch",
    ) as executor:
        list(executor.map(_fetch_one, stale_slugs))


def _any_env(env_vars, read_env=os.environ.get) -> bool:
    return any(read_env(ev) for ev in env_vars)


def _skip(seen: set, excluded: set, *keys: str) -> bool:
    """True when any of *keys* (lowercased) is already emitted or excluded."""
    lowered = [k.lower() for k in keys]
    return any(k in seen for k in lowered) or any(k in excluded for k in lowered)


def _iter_builtin_candidates(models_dev_data: dict, excluded: set, seen: set):
    """Yield ``(hermes_id, mdev_id, pconfig, env_vars)`` for section-1 rows.

    Skips vendor names that alias through an aggregator (bare "openai" -> "openrouter" would
    silently switch a user onto an endpoint they may have no key for), aliases of another canonical
    profile ("kimi" -> "kimi-coding"), non-api_key auth types (section 2 handles them) and
    unroutable providers. PROVIDER_REGISTRY env var names win over models.dev's."""
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY, is_runtime_provider_routable
    from hermes_cli.models import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import ALIASES
    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        alias_target = ALIASES.get(hermes_id)
        if alias_target and alias_target != hermes_id and alias_target in _AGGREGATOR_PROVIDERS:
            continue
        try:
            from providers import get_provider_profile
            prof = get_provider_profile(hermes_id)
            if prof is not None and prof.name != hermes_id:
                continue
        except Exception:
            pass
        if hermes_id.lower() in seen:
            continue
        if hermes_id.lower() in excluded or mdev_id.lower() in excluded:
            continue
        pdata = models_dev_data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue
        pconfig = PROVIDER_REGISTRY.get(hermes_id)
        if (pconfig and pconfig.auth_type != "api_key") or not is_runtime_provider_routable(hermes_id):
            continue
        env_vars = list(pconfig.api_key_env_vars) if pconfig and pconfig.api_key_env_vars else pdata.get("env", [])
        if isinstance(env_vars, list):
            yield hermes_id, mdev_id, pconfig, env_vars


def _auth_store_has_provider(*keys: str) -> bool:
    """True when ``auth.json`` has a ``providers`` entry under any of *keys*."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        providers_store = store.get("providers", {})
        return bool(store and any(k in providers_store for k in keys))
    except Exception as exc:
        logger.debug("Auth store check failed for %s: %s", keys[0] if keys else "", exc)
        return False


def _raw_pool_usable(hermes_id: str) -> bool:
    """Section-1 pool check: only consult the pool when auth.json lists a raw entry."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        if store and store.get("credential_pool", {}).get(hermes_id):
            return _credential_pool_is_usable(hermes_id, raw_pool_present=True)
    except Exception:
        pass
    return False


def _pool_usable(slug: str) -> bool:
    try:
        return _credential_pool_is_usable(slug)
    except Exception as exc:
        logger.debug("Credential pool check failed for %s: %s", slug, exc)
        return False


def _overlay_has_env_creds(pid: str, hermes_slug: str, overlay, read_env) -> bool:
    """Section-2 env/SDK credential check shared by the picker and the prefetch scan.

    Vertex authenticates via OAuth2 (service-account JSON / ADC), not an API key, so it gets its
    own probe; otherwise the provider is hidden from the picker even when fully configured."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    has_creds = False
    if overlay.auth_type == "vertex":
        try:
            from agent.vertex_adapter import has_vertex_credentials
            has_creds = has_vertex_credentials()
        except Exception as exc:
            logger.debug("Vertex credential check failed: %s", exc)
    elif overlay.extra_env_vars:
        has_creds = _any_env(overlay.extra_env_vars, read_env)
    if not has_creds and overlay.auth_type == "api_key":
        for key in (pid, hermes_slug):
            pcfg = PROVIDER_REGISTRY.get(key)
            if pcfg and pcfg.api_key_env_vars and _any_env(pcfg.api_key_env_vars, read_env):
                return True
    return has_creds


def _has_fast_aws_sdk_signal() -> bool:
    """True when explicit AWS auth config is present in the environment.

    Deliberately avoids botocore's full credential chain: picker discovery runs for non-Bedrock
    providers too, and botocore may probe EC2 IMDS (169.254.169.254) before giving up."""
    def _set(name: str) -> bool:
        return bool(os.environ.get(name, "").strip())
    return (
        _set("AWS_BEARER_TOKEN_BEDROCK")
        or (_set("AWS_ACCESS_KEY_ID") and _set("AWS_SECRET_ACCESS_KEY"))
        or any(_set(name) for name in (
            "AWS_PROFILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_WEB_IDENTITY_TOKEN_FILE")))


def _has_aws_sdk_creds_for_listing(slug: str, current_provider: str) -> bool:
    """AWS SDK credential check; the full boto3 chain is only consulted for the *current* provider."""
    if _has_fast_aws_sdk_signal():
        return True
    if str(slug or "").strip().lower() != str(current_provider or "").strip().lower():
        return False
    try:
        from agent.bedrock_adapter import has_aws_credentials
        return bool(has_aws_credentials())
    except Exception:
        return False


def _is_aws_sdk(pconfig) -> bool:
    return bool(pconfig) and getattr(pconfig, "auth_type", "") == "aws_sdk"


def _live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str, merge_models_dev: bool = True) -> list:
    """``cached_provider_model_ids`` (the SAME disk-cached list ``hermes model`` builds), falling
    back to the curated list (merged with models.dev for preferred providers) when live is empty."""
    from hermes_cli.models import _MODELS_DEV_PREFERRED, _merge_with_models_dev, cached_provider_model_ids
    model_ids = cached_provider_model_ids(slug)
    if not model_ids:
        model_ids = _first_curated(curated, fallback_keys or (slug,))
        if merge_models_dev and slug in _MODELS_DEV_PREFERRED:
            model_ids = _merge_with_models_dev(slug, model_ids)
    return model_ids


def _first_curated(curated: dict, keys) -> list:
    """First non-empty curated list under *keys*; the last key's (possibly empty) value otherwise."""
    model_ids: list = []
    for key in keys:
        model_ids = curated.get(key, [])
        if model_ids:
            break
    return model_ids


def _aws_live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str) -> list:
    """Bedrock: live discovery reflects the active region (eu.*, ap.*) rather than the static
    us.* list; any failure falls back to the curated list."""
    try:
        return _live_or_curated_ids(slug, curated, *fallback_keys, merge_models_dev=False) or []
    except Exception:
        return _first_curated(curated, fallback_keys or (slug,)) or []


def _nous_picker_model_ids(curated: dict, force_fresh_nous_tier: bool) -> list:
    """Nous serves a huge live catalog; the picker shows ONLY the curated agentic list, augmented
    with the Portal's free/paid recommendations (new models surface without a CLI release) and
    narrowed by org policy. Mirrors ``_model_flow_nous`` so GUI pickers match the CLI. A failed
    recommendation fetch still yields a policy-filtered curated list."""
    model_ids = curated.get("nous", [])
    try:
        from hermes_cli.models_pricing import get_pricing_for_provider
        from hermes_cli.models import (
            check_nous_free_tier,
            union_with_portal_free_recommendations,
            union_with_portal_paid_recommendations,
        )
        from hermes_cli.auth import get_provider_auth_state
        pricing = get_pricing_for_provider("nous") or {}
        try:
            portal = (get_provider_auth_state("nous") or {}).get("portal_base_url", "") or ""
        except Exception:
            portal = ""
        if check_nous_free_tier(force_fresh=force_fresh_nous_tier):
            model_ids, _ = union_with_portal_free_recommendations(model_ids, pricing, portal)
        else:
            model_ids, _ = union_with_portal_paid_recommendations(model_ids, pricing, portal)
    except Exception:
        pass
    try:
        from hermes_cli.models_pricing import nous_policy_allowed_ids, restrict_to_nous_policy
        model_ids = restrict_to_nous_policy(model_ids, nous_policy_allowed_ids(), rescue_empty=True)
    except Exception:
        pass
    return model_ids


def _cap_models(model_ids: list, max_models: int | None, slug: str = "") -> list:
    """Apply ``max_models``; aggregators in ``_UNCAPPED_PICKER_PROVIDERS`` show everything."""
    if slug in _UNCAPPED_PICKER_PROVIDERS or max_models is None:
        return model_ids
    return model_ids[:max_models]


def _absorb_entry_models(grp: dict, entry: dict, active_model: Any) -> None:
    """Fold one config entry's models into its group: the active selection first, then the declared
    ``models:`` ids. The active selection alone never suppresses discovery, and a dict-shaped
    ``models:`` is metadata rather than an allowlist (see ``_models_config_is_allowlist``), so only
    list/string shapes pin the row."""
    from hermes_cli.model_switch import _declared_model_ids, _entry_models_discovered, _models_config_is_allowlist
    _extend_unique(grp["models"], [active_model])
    models_field = entry.get("models")
    if _models_config_is_allowlist(models_field, _entry_models_discovered(entry)):
        grp["has_explicit_models"] = True
    _extend_unique(grp["models"], _declared_model_ids(models_field))


def _extend_unique(target: list, items) -> None:
    """Append each truthy item of *items* not already in *target* (order preserved)."""
    for item in items:
        if item and item not in target:
            target.append(item)


def _norm_url(url: Any) -> str:
    # Effective base URLs of every built-in row we emit (normalized lower+rstrip). Section 4 uses this to
    # hide ``custom_providers`` entries that point at the same endpoint as a built-in (e.g. a user-defined
    # "my-dashscope" on https://coding-intl.dashscope.aliyuncs.com/v1 collides with the built-in
    # alibaba-coding-plan row when DASHSCOPE_API_KEY is present). Fixes #16970.
    return str(url or "").strip().rstrip("/").lower()


def _entry_base_url(entry: dict, keys: tuple = ("base_url", "url", "api")) -> str:
    return next((entry.get(key, "") for key in keys if entry.get(key, "")), "")


def _entry_api_mode(entry: dict) -> str | None:
    return str(entry.get("api_mode") or entry.get("transport") or "").strip().lower() or None


def _entry_credentials(entry: dict, *key_env_keys: str) -> tuple[str, str, str]:
    """``(inline_api_key, key_env, identity)`` — identity is the inline key, else ``env:<VAR>``, else ""."""
    inline_api_key = str(entry.get("api_key", "") or "").strip()
    key_env = str(next((entry.get(k) for k in key_env_keys if entry.get(k)), "")).strip()
    return inline_api_key, key_env, inline_api_key or (f"env:{key_env}" if key_env else "")


def _discover_flag(entry: dict):
    """``discover_models`` (default True); ``"false"/"no"/"0"`` strings mean False."""
    discover = entry.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    return discover


def _display_prefix(name: str) -> str:
    """Text before the per-model separator Hermes's own writer uses ("—" / " - ")."""
    return next((name.split(sep)[0].strip() for sep in ("—", " - ") if sep in name), name)


def _group_display_name(display_name: str) -> str:
    """Section-3 row label: strip the per-model suffix and trailing version tokens ("Palantir
    Claude 4.7 Opus" -> "Palantir Claude") — cut at the first token containing a digit, only when
    >= 2 words remain (avoids over-trimming)."""
    grp_display = _display_prefix(display_name)
    toks = grp_display.split()
    cut_at = next((i for i, t in enumerate(toks) if any(c.isdigit() for c in t.strip(".,()"))), None)
    if cut_at is not None and cut_at >= 2:
        grp_display = " ".join(toks[:cut_at]).strip()
    return grp_display or display_name


def _discover_endpoint_models(
    api_key: str, api_url: str, native_catalog_provider: str, has_explicit_models: bool, *,
    headers: dict | None, api_mode: str | None, probe_live: bool, discovery_allowed: bool,
    for_picker: bool) -> tuple[list | None, bool]:
    """Return ``(models, native_catalog_empty)`` for a custom endpoint row.

    ``probe_live`` runs the native-aware picker fetch; otherwise, when discovery is allowed, a
    warm same-fingerprint cache entry still serves the full catalog with no round-trip.
    ``has_explicit_models`` gates the *probe* (a network-cost guard for keyless endpoints that
    declare a catalog), never the cache read — applying it to the read re-pins the endpoint to
    its declared subset. Returns ``(None, False)`` when nothing usable was found."""
    timeout = 1.5 if for_picker else 5.0
    if probe_live:
        try:
            live_models = _fetch_picker_live_models(
                api_key, api_url, native_catalog_provider, has_explicit_models,
                headers=headers, timeout=timeout, api_mode=api_mode)
            is_native = isinstance(live_models, _NativePickerModelList)
            if live_models is not None and (live_models or not has_explicit_models or is_native):
                return live_models, (is_native and not live_models)
        except Exception:
            pass
    elif discovery_allowed:
        try:
            from hermes_cli.models import cached_fetch_api_models
            cached_models = cached_fetch_api_models(
                api_key, api_url, cache_only=True, timeout=timeout, headers=headers, api_mode=api_mode,
            )
            if cached_models:
                return cached_models, False
        except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, http.client.HTTPException):
            pass
    return None, False


def _collect_authed_provider_slugs(
    models_dev_data: dict, curated: dict[str, list[str]], excluded: list[str]) -> list[str]:
    """Quick-scan which providers have credentials, without fetching model lists.

    Mirrors the credential checks of sections 1, 2 and 2b of :func:`list_authenticated_providers`
    but never calls ``cached_provider_model_ids``; feeds :func:`_prefetch_provider_models_parallel`.
    Env vars are read through the per-profile secret scope. AWS SDK providers are skipped
    (heavier detection)."""
    from hermes_cli.model_switch import _scoped_key_env
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import HERMES_OVERLAYS
    from hermes_cli.models import CANONICAL_PROVIDERS
    excluded_set = {str(p).strip().lower() for p in excluded if p}
    slugs: list[str] = []
    seen: set[str] = set()

    def _emit(slug: str, *keys: str) -> None:
        slugs.append(slug)
        seen.update(k.lower() for k in keys)

    for hermes_id, _mdev_id, _pconfig, env_vars in _iter_builtin_candidates(models_dev_data, excluded_set, seen):
        if _any_env(env_vars, _scoped_key_env) or _raw_pool_usable(hermes_id):
            _emit(hermes_id, hermes_id)

    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if _skip(seen, excluded_set, pid, hermes_slug) or overlay.auth_type == "aws_sdk":
            continue
        if (
            _overlay_has_env_creds(pid, hermes_slug, overlay, _scoped_key_env)
            or _auth_store_has_provider(pid, hermes_slug) or _pool_usable(hermes_slug)):
            _emit(hermes_slug, pid, hermes_slug)

    for cp in CANONICAL_PROVIDERS:
        if _skip(seen, excluded_set, cp.slug):
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = bool(
            cp_config and cp_config.api_key_env_vars and _any_env(cp_config.api_key_env_vars, _scoped_key_env))
        if has_creds or _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug):
            _emit(cp.slug, cp.slug)

    # Nous excluded: its picker branch builds from the curated list and never reads the
    # api_key-only cache entry a prefetch would write.
    return [s for s in slugs if s != "nous"]


@dataclass
class _PickerBuild:
    """State threaded through the ``list_authenticated_providers`` sections: 1 built-ins mapped to
    models.dev, 2 Hermes-only overlays, 2b canonical providers missed by 1/2, 3 ``providers:``
    entries + 3b the bare active custom endpoint, 4 ``custom_providers:`` entries. Row-builder
    imports of ``hermes_cli.auth/models`` stay lazy so tests can patch those modules."""
    current_provider: str
    current_base_url: str
    current_model: str
    max_models: int | None
    for_picker: bool
    force_fresh_nous_tier: bool
    probe_custom_providers: bool
    probe_current_custom_provider: bool
    refresh: bool
    excluded: set
    curated: dict
    results: list = field(default_factory=list)
    seen_slugs: set = field(default_factory=set)  # lowercase-normalized to catch case variants
    # Effective base URLs of every built-in row: section 4 hides ``custom_providers`` duplicates.
    builtin_endpoints: set = field(default_factory=set)
    # (display_name, base_url) pairs from section 3 so section 4 skips overlapping rows.
    section3_pairs: set = field(default_factory=set)

    @property
    def current_provider_norm(self) -> str:
        return self.current_provider.lower()

    @property
    def current_base_url_norm(self) -> str:
        return self.current_base_url.rstrip("/").lower()

    def can_probe_custom(self, *, row_is_current: bool) -> bool:
        return bool(self.probe_custom_providers or (self.probe_current_custom_provider and row_is_current))

    def record_builtin_endpoint(self, slug: str) -> None:
        """Prefer the live env override (e.g. DASHSCOPE_BASE_URL) over the static inference_base_url
        so dedup matches what a user typing that URL into custom_providers would actually hit."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(slug)
        if not pcfg:
            return
        url = os.environ.get(pcfg.base_url_env_var, "") if getattr(pcfg, "base_url_env_var", "") else ""
        normed = _norm_url(url or getattr(pcfg, "inference_base_url", "") or "")
        if normed:
            self.builtin_endpoints.add(normed)

    def add_builtin_row(
        self, slug: str, name: str, is_current: bool, model_ids: list, source: str, *, uncapped_ok: bool = True,
    ) -> None:
        self.results.append({
            "slug": slug, "name": name, "is_current": is_current, "is_user_defined": False,
            "models": _cap_models(model_ids, self.max_models, slug if uncapped_ok else ""),
            "total_models": len(model_ids), "source": source})
        self.seen_slugs.add(slug.lower())
        self.record_builtin_endpoint(slug)

    def add_endpoint_row(
        self, slug: str, name: str, api_url: str, models: list, is_current: bool, native_catalog_empty: bool,
        *, source: str = "user-config", shown: list | None = None) -> None:
        """Append a user-defined endpoint row (sections 3, 3b, 4)."""
        self.results.append({
            "slug": slug, "name": name, "is_current": is_current, "is_user_defined": True,
            "models": models if shown is None else shown, "total_models": len(models), "source": source,
            "api_url": api_url, "native_catalog_empty": native_catalog_empty})
        self.seen_slugs.add(slug.lower())

    def record_section3_pair(self, name: str, url_norm: str) -> bool:
        """Remember a (display_name, base_url) pair for section-4 dedup; False when either is blank."""
        pair = (str(name).strip().lower(), url_norm)
        if not (pair[0] and pair[1]):
            return False
        self.section3_pairs.add(pair)
        return True

    def endpoint_is_current(self, slug: str, aliases: set, url_norm: str, *, url_match_ok: bool = True) -> bool:
        """Row is current by slug/alias, or (bare ``custom`` provider) by matching base_url."""
        return (
            str(slug).strip().lower() == self.current_provider_norm
            or self.current_provider_norm in aliases
            or (
                self.current_provider_norm == "custom" and bool(self.current_base_url_norm)
                and url_norm == self.current_base_url_norm and url_match_ok))

    def discover_endpoint(
        self, api_key: str, api_url: str, native_provider: str, has_explicit_models: bool, *,
        headers: dict | None, api_mode: str | None, discovery_allowed: bool, is_current: bool,
    ) -> tuple[list | None, bool, bool]:
        """Probe policy shared by sections 3 and 4 (returns ``(models, native_empty, probed)``):
        with an api_key live /models is the source of truth (replaces the partial ``models:``
        subset); without one, an allowlist-shaped ``models:`` narrows a public endpoint and skips
        the probe. A dict-shaped ``models:`` is metadata, so still probe; pin with
        ``discover_models: false``."""
        probe_live = (
            discovery_allowed and (bool(api_key) or not has_explicit_models)
            and self.can_probe_custom(row_is_current=is_current))
        discovered, native_catalog_empty = _discover_endpoint_models(
            api_key, api_url, native_provider, has_explicit_models,
            headers=headers, api_mode=api_mode, probe_live=probe_live,
            discovery_allowed=discovery_allowed, for_picker=self.for_picker)
        return discovered, native_catalog_empty, probe_live


def _lap_builtin_rows(b: _PickerBuild, data: dict, user_providers: dict) -> None:
    """Section 1: models.dev-mapped providers with api_key auth."""
    from hermes_cli.model_switch import _declared_model_ids
    from agent.models_dev import get_provider_info
    for hermes_id, mdev_id, pconfig, env_vars in _iter_builtin_candidates(data, b.excluded, b.seen_slugs):
        if not (_any_env(env_vars) or _raw_pool_usable(hermes_id)):
            continue
        model_ids = _live_or_curated_ids(hermes_id, b.curated)
        # A providers.<built-in>.models block extends the discovered catalog; section 3 cannot
        # emit it later because this row owns the slug.
        configured = user_providers.get(hermes_id) if isinstance(user_providers, dict) else None
        configured_models = _declared_model_ids(configured.get("models")) if isinstance(configured, dict) else []
        model_ids = list(dict.fromkeys([*configured_models, *model_ids]))
        pinfo = get_provider_info(mdev_id)
        display_name = pconfig.name if pconfig and pconfig.name else (pinfo.name if pinfo else mdev_id)
        b.add_builtin_row(
            hermes_id, display_name, b.current_provider in (hermes_id, mdev_id), model_ids, "built-in")


def _overlay_has_creds(b: _PickerBuild, pid: str, hermes_slug: str, overlay) -> bool:
    """Section-2 credential ladder: env/SDK, external-process executable, auth store, pool,
    anthropic's external credential files."""
    if overlay.keyless:
        return True  # served anonymously (opencode-free)
    if overlay.auth_type == "aws_sdk":
        has_creds = _has_aws_sdk_creds_for_listing(hermes_slug, b.current_provider)
    else:
        has_creds = _overlay_has_env_creds(pid, hermes_slug, overlay, os.environ.get)
    # External-process providers (copilot-acp) hold no key/token/pool entry by design — the
    # spawned ACP subprocess brings its own auth. "Configured" means the executable resolves.
    # "Configured" means the executable resolves, which is exactly what get_auth_status() reports for them;
    # without this branch the has_creds filter below unconditionally hides the provider from every picker
    # (#63662).
    if not has_creds and overlay.auth_type == "external_process":
        try:
            from hermes_cli.auth import get_auth_status
            _ext_status = get_auth_status(hermes_slug) or {}
            has_creds = bool(_ext_status.get("logged_in") or _ext_status.get("configured"))
        except Exception as exc:
            logger.debug("External-process check failed for %s: %s", pid, exc)
    # Auth store / credential pool cover OAuth providers AND api_key providers that also support
    # OAuth (anthropic via Claude Code credential files).
    has_creds = has_creds or _auth_store_has_provider(pid, hermes_slug)
    if not has_creds:
        # Full auto-seeding pool check catches external stores (Codex CLI ~/.codex/auth.json)
        # not yet in auth.json.
        try:
            if _credential_pool_is_usable(hermes_slug):
                has_creds = True
            elif b.for_picker:
                # Show providers whose pool is entirely in cooldown: limits are per-model for
                # many providers, so another model may work.
                try:
                    from agent.credential_pool import load_pool
                    has_creds = load_pool(hermes_slug).has_credentials()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Credential pool check failed for %s: %s", hermes_slug, exc)
    if not has_creds and hermes_slug == "anthropic":
        # The pool gates anthropic behind is_provider_explicitly_configured() (aux tasks must not
        # consume Claude Code tokens); the picker is discovery-oriented, so read the files directly.
        try:
            from agent.anthropic_credentials import read_claude_code_credentials, read_hermes_oauth_credentials
            hermes_creds = read_hermes_oauth_credentials()
            cc_creds = read_claude_code_credentials()
            if (hermes_creds and hermes_creds.get("accessToken")) or (cc_creds and cc_creds.get("accessToken")):
                has_creds = True
        except Exception as exc:
            logger.debug("Anthropic external creds check failed: %s", exc)
    return has_creds


def _lap_overlay_rows(b: _PickerBuild, data: dict) -> None:
    """Section 2: Hermes-only providers (nous, openai-codex, copilot, opencode-go, ...)."""
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.providers import HERMES_OVERLAYS

    # HERMES_OVERLAYS keys may be models.dev IDs ("github-copilot") while config.yaml uses
    # Hermes IDs ("copilot").
    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if _skip(b.seen_slugs, b.excluded, pid, hermes_slug):
            continue
        if not _overlay_has_creds(b, pid, hermes_slug, overlay):
            continue
        if hermes_slug in {"openai-codex", "copilot", "copilot-acp"}:
            # Live OAuth-backed discovery so Pro-only Codex slugs not in the static catalog
            # appear; falls back to curated when unreachable.
            from hermes_cli.models import cached_provider_model_ids
            model_ids = cached_provider_model_ids(hermes_slug)
        elif overlay.auth_type == "aws_sdk":
            model_ids = _aws_live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        elif hermes_slug == "nous":
            model_ids = _nous_picker_model_ids(b.curated, b.force_fresh_nous_tier)
        else:
            model_ids = _live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        b.add_builtin_row(
            hermes_slug, get_label(hermes_slug), b.current_provider in (hermes_slug, pid), model_ids, "hermes")
        b.seen_slugs.add(pid.lower())


def _lap_canonical_rows(b: _PickerBuild) -> None:
    """Section 2b: CANONICAL_PROVIDERS missed by sections 1/2."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import CANONICAL_PROVIDERS
    for cp in CANONICAL_PROVIDERS:
        if _skip(b.seen_slugs, b.excluded, cp.slug):
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = False
        if cp_config and cp_config.api_key_env_vars:
            lit = {ev for ev in cp_config.api_key_env_vars if os.environ.get(ev)}
            has_creds = bool(lit)
            # A regional "-cn" twin lit only by key vars shared with its non-CN sibling is a
            # phantom row: hide it unless it is the current provider, and only when it has a
            # dedicated var of its own the user could set.
            sib = PROVIDER_REGISTRY.get(cp.slug[:-3]) if cp.slug.endswith("-cn") else None
            sib_vars = set(sib.api_key_env_vars) if sib else set()
            if lit and lit <= sib_vars < set(cp_config.api_key_env_vars) and cp.slug != b.current_provider:
                continue
        has_creds = has_creds or _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug) or (
            _is_aws_sdk(cp_config) and _has_aws_sdk_creds_for_listing(cp.slug, b.current_provider))
        if not has_creds:
            continue
        if _is_aws_sdk(cp_config):
            model_ids = _aws_live_or_curated_ids(cp.slug, b.curated)
        else:
            model_ids = _live_or_curated_ids(cp.slug, b.curated, merge_models_dev=False)
        b.add_builtin_row(
            cp.slug, cp.label, cp.slug == b.current_provider, model_ids, "canonical", uncapped_ok=False)


def _lap_user_provider_rows(b: _PickerBuild, user_providers: dict) -> None:
    """Section 3: ``providers:`` dict entries, grouped by (api_url, credential, api_mode,
    extra_headers) so keyed providers on one endpoint with the same wire protocol collapse into
    one row (two Palantir Claude entries -> one "Palantir Claude" row); a different
    key_env/api_mode/headers keeps distinct rows since the wire protocol or tenant differs."""
    from hermes_cli.model_switch import _extra_headers_from_config, _scoped_key_env
    from hermes_cli.config import coerce_provider_id, is_provider_enabled
    ep_groups: dict[tuple, dict] = {}
    for ep_name, ep_cfg in user_providers.items():
        if not isinstance(ep_cfg, dict) or not is_provider_enabled(ep_cfg) or ep_name.lower() in b.seen_slugs:
            continue
        display_name = coerce_provider_id(ep_cfg.get("name")) or ep_name
        api_url = _entry_base_url(ep_cfg, ("base_url", "api", "url"))
        inline_api_key, key_env, cred_identity = _entry_credentials(ep_cfg, "key_env", "api_key_env")
        headers = _extra_headers_from_config(ep_cfg)
        group_key = (_norm_url(api_url), cred_identity, _entry_api_mode(ep_cfg), tuple(sorted(headers.items())))

        if group_key not in ep_groups:
            # slug = first ep_name encountered; probe key from the first member (inline api_key,
            # else key_env through the per-profile secret scope).
            ep_groups[group_key] = {
                "slug": ep_name, "name": _group_display_name(display_name), "api_url": api_url, "models": [],
                "has_explicit_models": False, "api_key": inline_api_key or _scoped_key_env(key_env),
                "headers": headers, "api_mode": ep_cfg.get("api_mode"),
                "discovery_allowed": bool(api_url) and _discover_flag(ep_cfg), "raw_names": [], "aliases": set()}
        grp = ep_groups[group_key]
        # ``default_model`` is the legacy key; ``model`` matches custom_providers.
        _absorb_entry_models(grp, ep_cfg, ep_cfg.get("default_model", "") or ep_cfg.get("model", ""))
        grp["raw_names"].append(display_name)
        grp["aliases"].update(custom_provider_aliases(display_name, str(ep_name)))

    for grp in ep_groups.values():
        ep_name, display_name, api_url = grp["slug"], grp["name"], grp["api_url"]
        models_list = list(grp["models"])
        # Official OpenAI rows often have base_url but no models: dict — avoid a misleading zero count.
        if not models_list and base_url_host_matches(str(api_url).strip().lower(), "api.openai.com"):
            models_list = list(b.curated.get("openai") or [])

        ep_url_norm = _norm_url(api_url)
        ep_aliases = {str(alias).lower() for alias in grp["aliases"]}
        is_current = b.endpoint_is_current(ep_name, ep_aliases, ep_url_norm)
        discovered, native_catalog_empty, _ = b.discover_endpoint(
            grp["api_key"], api_url,
            ep_name if str(ep_name).strip().lower() in {"ollama", "custom:ollama"} else "custom",
            grp["has_explicit_models"], headers=grp["headers"] or None, api_mode=grp["api_mode"],
            discovery_allowed=grp["discovery_allowed"], is_current=is_current)
        if discovered is not None:
            models_list = discovered

        b.add_endpoint_row(ep_name, display_name, api_url, models_list, is_current, native_catalog_empty)
        b.seen_slugs.update(ep_aliases)
        # Record every raw member name so section 4 can match per-model custom_providers rows
        # even though the group label was collapsed.
        for raw_name in grp["raw_names"] or [display_name]:
            if b.record_section3_pair(raw_name, ep_url_norm):
                b.seen_slugs.add(custom_provider_slug(raw_name).lower())
        b.record_section3_pair(display_name, ep_url_norm)


def _lap_bare_custom_row(b: _PickerBuild, custom_providers: list | None) -> None:
    """Section 3b: ``model.provider: custom`` + ``model.base_url`` with no named
    providers:/custom_providers row — surface it so /model does not look like it ignored
    config.yaml."""
    if not (b.current_provider_norm == "custom" and b.current_base_url and "custom" not in b.seen_slugs):
        return
    if any(
        isinstance(cp, dict) and _norm_url(_entry_base_url(cp)) == _norm_url(b.current_base_url)
        for cp in (custom_providers or [])):
        return
    api_url = str(b.current_base_url).strip().rstrip("/")
    models = [b.current_model] if b.current_model else []
    native_catalog_empty = False
    try:
        discovered, native_catalog_empty = _discover_endpoint_models(
            "", api_url, "custom", False, headers=None, api_mode=None,
            probe_live=bool(b.refresh or b.probe_current_custom_provider), discovery_allowed=True,
            for_picker=b.for_picker)
        if discovered is not None:
            models = discovered
    except Exception:
        pass
    b.add_endpoint_row(
        "custom", "Custom endpoint", api_url, models, True, native_catalog_empty,
        source="model-config", shown=_cap_models(models, b.max_models))


def _lap_custom_provider_rows(b: _PickerBuild, custom_providers: list) -> None:
    """Section 4: ``custom_providers:`` entries (one model each) grouped into one row per
    (endpoint, credential identity, api_mode, extra_headers, display prefix). Four "Ollama — X"
    entries on one host become one "Ollama" row; distinct prefixes sharing a proxy URL keep
    their own rows."""
    from hermes_cli.model_switch import _extra_headers_from_config, _scoped_key_env
    from hermes_cli.config import coerce_provider_id
    groups: dict[tuple, dict] = {}
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        raw_name = coerce_provider_id(entry.get("name"))
        api_url = str(_entry_base_url(entry) or "").strip().rstrip("/")
        if not raw_name or not api_url:
            continue
        inline_api_key, key_env, cred_identity = _entry_credentials(entry, "key_env")
        api_key = inline_api_key or _scoped_key_env(key_env)
        api_mode = _entry_api_mode(entry)
        discover = _discover_flag(entry)
        entry_extra_headers = _extra_headers_from_config(entry)
        prefix = _display_prefix(raw_name)
        provider_key = str(entry.get("provider_key") or "").strip()
        group_key = (api_url, cred_identity, api_mode, tuple(sorted(entry_extra_headers.items())), prefix.lower())
        display_name = prefix or raw_name
        grp = groups.setdefault(group_key, {
            "slug": custom_provider_slug(display_name, provider_key), "name": display_name,
            "api_url": api_url, "api_key": "", "models": [], "has_explicit_models": False,
            "discover_models": True, "api_mode": api_mode, "extra_headers": entry_extra_headers,
            "aliases": set()})
        grp["api_key"] = grp["api_key"] or api_key  # first member with a key wins
        grp["discover_models"] = grp["discover_models"] and discover  # one opt-out pins the whole row
        grp["aliases"].update(custom_provider_aliases(raw_name, provider_key))
        # ``model:`` is only the active selection; every configured model lives under ``models:``.
        _absorb_entry_models(grp, entry, (entry.get("model") or "").strip())

    section4_slugs: set = set()
    current_url_group_count = sum(
        1 for grp in groups.values()
        if b.current_base_url_norm and _norm_url(grp["api_url"]) == b.current_base_url_norm)
    for grp in groups.values():
        api_url, api_key, slug = grp["api_url"], grp.get("api_key", ""), grp["slug"]
        # Slug claimed by a built-in/overlay/providers: row -> skip (don't shadow).
        if slug.lower() in b.seen_slugs and slug.lower() not in section4_slugs:
            continue
        # Two custom endpoints with the same cleaned name: suffix a counter so both stay visible.
        if slug.lower() in section4_slugs:
            base_slug, n = slug, 2
            while f"{base_slug}-{n}".lower() in b.seen_slugs:
                n += 1
            slug = f"{base_slug}-{n}"
            grp["slug"] = slug
        grp_url_norm = _norm_url(api_url)
        pair_key = (str(grp["name"]).strip().lower(), grp_url_norm)
        if pair_key[0] and pair_key[1] and pair_key in b.section3_pairs:
            continue
        # A built-in row already represents this endpoint (e.g. "my-dashscope" vs the
        # alibaba-coding-plan row): keep the built-in, hide the shadow.
        if grp_url_norm and grp_url_norm in b.builtin_endpoints:
            continue
        is_current = b.endpoint_is_current(
            slug, {str(alias).lower() for alias in grp["aliases"]}, grp_url_norm,
            url_match_ok=current_url_group_count == 1)
        discovered, native_catalog_empty, probe_live = b.discover_endpoint(
            api_key, api_url,
            "ollama" if "ollama" in {str(slug).strip().lower(), str(grp.get("name") or "").strip().lower()} else "custom",
            bool(grp.get("has_explicit_models")), headers=grp.get("extra_headers") or None,
            api_mode=grp.get("api_mode"), discovery_allowed=bool(api_url) and grp.get("discover_models", True),
            is_current=is_current)
        if discovered is not None:
            grp["models"] = discovered
            if probe_live:  # a successful live probe persists the catalog for no-probe surfaces
                try:
                    _save_discovered_models_to_config(
                        api_url, discovered, api_mode=grp.get("api_mode"), headers=grp.get("extra_headers") or None)
                except Exception:
                    pass
        b.add_endpoint_row(slug, grp["name"], grp["api_url"], grp["models"], is_current, native_catalog_empty)
        section4_slugs.add(slug.lower())


def _build_curated_lists(current_provider: str, current_base_url: str, current_model: str) -> dict[str, list[str]]:
    """Curated model lists keyed by hermes provider id, plus the dynamic ones (nous manifest,
    Ollama Cloud, LM Studio live probe)."""
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS, get_curated_nous_model_ids
    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)
    curated["openrouter"] = [mid for mid, _ in OPENROUTER_MODELS]
    # Remote manifest so new Portal models surface without a release; in-repo snapshot fallback.
    curated["nous"] = get_curated_nous_model_ids()
    if "ollama-cloud" not in curated:
        from hermes_cli.models import fetch_ollama_cloud_models
        curated["ollama-cloud"] = fetch_ollama_cloud_models()
    # LM Studio has no static catalog: probe its native endpoint live. Base URL precedence:
    # LM_BASE_URL > active config base_url (when current) > default. On auth rejection /
    # unreachable, fall back to the current model so the picker still shows something offline.
    is_current_lmstudio = current_provider.strip().lower() == "lmstudio"
    if "lmstudio" not in curated and (os.environ.get("LM_API_KEY") or os.environ.get("LM_BASE_URL") or is_current_lmstudio):
        from hermes_cli.models_local import fetch_lmstudio_models
        from hermes_cli.auth import AuthError
        lm_base = (
            os.environ.get("LM_BASE_URL")
            or (current_base_url if is_current_lmstudio and current_base_url else None)
            or "http://127.0.0.1:1234/v1")
        try:
            live = fetch_lmstudio_models(api_key=os.environ.get("LM_API_KEY", ""), base_url=lm_base, timeout=1.5)
        except AuthError:
            live = []
        if not live and is_current_lmstudio and current_model:
            live = [current_model]
        curated["lmstudio"] = live
    return curated


def list_authenticated_providers(
    current_provider: str = "", current_base_url: str = "", user_providers: dict = None,
    custom_providers: list | None = None, *, force_fresh_nous_tier: bool = False,
    max_models: int | None = None, current_model: str = "", refresh: bool = False,
    probe_custom_providers: bool = True, probe_current_custom_provider: bool = False,
    for_picker: bool = False, excluded_providers: list | None = None) -> List[dict]:
    """Detect which providers have credentials and list their curated (not full models.dev) models.

    Returns dicts with ``slug`` (the --provider value), ``name``, ``is_current``,
    ``is_user_defined``, ``models`` (up to max_models), ``total_models``, ``source``
    ("built-in", "hermes", "canonical", "user-config", "model-config").
    ``force_fresh_nous_tier`` bypasses the short Nous tier cache (account-sensitive flows only);
    ``refresh`` busts the model-id disk cache up front (explicit user action only);
    ``probe_custom_providers`` enables live ``/models`` discovery for saved custom endpoints (CLI
    true, GUI false); ``probe_current_custom_provider`` probes only the selected custom endpoint."""
    from agent.models_dev import fetch_models_dev
    from hermes_cli.config import coerce_provider_id, stringify_provider_map

    # Explicit refresh: drop every cached list so the calls below re-fetch live. A stale cache
    # can fall back to the curated static list when its live fetch fails, silently dropping
    # live-only models the user had seen.
    if refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
        except Exception:
            pass

    # PyYAML parses unquoted numeric names (`provider: 2070`) as int.
    # seen_slugs: set = set()  # lowercase-normalized to catch case variants (#9545)
    current_provider = coerce_provider_id(current_provider)
    current_base_url = str(current_base_url or "").strip()
    current_model = str(current_model or "").strip()
    user_providers = stringify_provider_map(user_providers)
    data = fetch_models_dev()

    # A single excluded entry like ``copilot`` hides the provider under every key it surfaces
    # as (hermes_id / mdev_id / canonical slug).
    b = _PickerBuild(
        current_provider=current_provider, current_base_url=current_base_url, current_model=current_model,
        max_models=max_models, for_picker=for_picker, force_fresh_nous_tier=force_fresh_nous_tier,
        probe_custom_providers=probe_custom_providers, probe_current_custom_provider=probe_current_custom_provider,
        refresh=refresh, excluded={str(p).strip().lower() for p in (excluded_providers or []) if p},
        curated=_build_curated_lists(current_provider, current_base_url, current_model))

    # Warm the disk cache in parallel before the serial section loops (otherwise 15-30s of live
    # round-trips on a cold cache). Skipped when refresh=True (serial path force-refreshes) and
    # for <=3 providers (serial is fast enough; avoids thread-pool overhead).
    prefetch_slugs = [] if refresh else _collect_authed_provider_slugs(data, b.curated, excluded_providers or [])
    if len(prefetch_slugs) > 3:
        try:
            _prefetch_provider_models_parallel(prefetch_slugs)
        except Exception:
            pass  # best-effort; serial path still works

    _lap_builtin_rows(b, data, user_providers)
    _lap_overlay_rows(b, data)
    _lap_canonical_rows(b)
    if user_providers and isinstance(user_providers, dict):
        _lap_user_provider_rows(b, user_providers)
    _lap_bare_custom_row(b, custom_providers)
    if custom_providers and isinstance(custom_providers, list):
        _lap_custom_provider_rows(b, custom_providers)
    return _finalize_picker_rows(b.results, user_providers, current_model)


def _inject_aliases_into_rows(results: list[dict]) -> None:
    """Surface configured model aliases on matching picker rows."""
    try:
        from hermes_cli.model_switch import DIRECT_ALIASES, _ensure_direct_aliases

        _ensure_direct_aliases()
        rows = {str(row.get("slug", "")).lower(): row for row in results}
        for alias_name, alias in DIRECT_ALIASES.items():
            row = rows.get(str(alias.provider or "").lower())
            model = str(alias.model or "").strip()
            if row is None or not model:
                continue
            models = row.get("models")
            if not isinstance(models, list):
                continue
            row.setdefault("model_aliases", {}).setdefault(model, alias_name)
            if any(model.lower() == str(item).lower() for item in models):
                continue
            models.insert(0, model)
            row["total_models"] = row.get("total_models", len(models) - 1) + 1
    except Exception:
        pass


def _finalize_picker_rows(results: list, user_providers, current_model: str) -> list:
    """Post-passes: drop ``providers.<name>.enabled: false`` rows, inject the current model, sort."""
    # The enabled post-filter covers built-in rows (sections 1-2) that bypass the per-section
    # gate; matched by slug and ``provider_id``.
    try:
        from hermes_cli.config import is_provider_enabled
        if isinstance(user_providers, dict):
            disabled = {
                str(name).strip().lower() for name, cfg in user_providers.items()
                if isinstance(cfg, dict) and not is_provider_enabled(cfg)}
            if disabled:
                results = [
                    r for r in results
                    if str(r.get("provider_id", "")).strip().lower() not in disabled
                    and str(r.get("slug", "")).strip().lower() not in disabled]
    except Exception:
        pass

    # A custom/uncurated model set via `/model <provider>/<name>` would be invisible in every
    # picker (main and MoA slot pickers read these rows); inject it at the front of the current
    # provider's row.
    if current_model:
        for row in results:
            if not row.get("is_current") or row.get("native_catalog_empty"):
                continue
            models = row.get("models") or []
            if current_model not in models:
                row["models"] = [current_model, *models]
                row["total_models"] = row.get("total_models", len(models)) + 1
            break

    _inject_aliases_into_rows(results)

    # Current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))
    return results


def _prepend_moa_picker_provider(providers: List[dict], current_provider: str = "") -> List[dict]:
    """Add the virtual MoA provider row used by interactive model pickers.

    ``list_authenticated_providers()`` only returns real/auth-backed providers; the CLI inventory
    adds MoA separately, so gateway pickers need the same virtual row here. Reuses the
    inventory's single row builder so the row shape stays defined in one place."""
    try:
        from hermes_cli.inventory import _moa_provider_row
        moa_row = _moa_provider_row(current_provider)
        if moa_row is None:
            return providers
        return [moa_row] + [p for p in providers if str(p.get("slug", "")).lower() != "moa"]
    except Exception:
        return providers


def list_picker_providers(
    current_provider: str = "", current_base_url: str = "", user_providers: dict = None,
    custom_providers: list | None = None, max_models: int | None = None, current_model: str = "",
    include_moa: bool = False, excluded_providers: list | None = None) -> List[dict]:
    """Interactive-picker variant of :func:`list_authenticated_providers`.

    OpenRouter's list is replaced with :func:`hermes_cli.models.fetch_openrouter_models` (curated
    snapshot filtered against the live catalog) and rows left with no models are dropped — except
    custom endpoints, where the user may supply their own model set through config."""
    from hermes_cli.model_switch import list_authenticated_providers
    from hermes_cli.models import fetch_openrouter_models
    providers = list_authenticated_providers(
        current_provider=current_provider, current_base_url=current_base_url,
        user_providers=user_providers, custom_providers=custom_providers, max_models=max_models,
        current_model=current_model, for_picker=True, excluded_providers=excluded_providers)
    if include_moa:
        providers = _prepend_moa_picker_provider(providers, current_provider=current_provider)

    filtered: List[dict] = []
    for p in providers:
        if str(p.get("slug", "")).lower() == "openrouter":
            try:
                live_ids = [mid for mid, _ in fetch_openrouter_models()]
            except Exception:
                live_ids = list(p.get("models", []))
            p = dict(p)
            p["models"] = live_ids[:max_models] if max_models is not None else live_ids
            p["total_models"] = len(live_ids)

        is_custom_endpoint = bool(p.get("is_user_defined")) and bool(p.get("api_url"))
        if p.get("models") or is_custom_endpoint:
            filtered.append(p)
    return filtered
