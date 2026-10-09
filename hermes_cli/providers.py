"""CLI-side provider resolution built on the canonical ``providers`` domain."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from providers import (
    ResolvedProvider as _ResolvedProvider,
    custom_provider_aliases as _custom_provider_aliases,
    custom_provider_slug as _custom_provider_slug,
    get_provider_profile as _get_provider_profile,
    normalize_provider as _normalize_provider,
)
from utils import base_url_hostname

logger = logging.getLogger(__name__)


# -- Transport → API mode mapping ---------------------------------------------

TRANSPORT_TO_API_MODE: Dict[str, str] = {
    "openai_chat": "chat_completions", "anthropic_messages": "anthropic_messages",
    "codex_responses": "codex_responses", "bedrock_converse": "bedrock_converse",
}


# -- Helper functions ---------------------------------------------------------

def _models_dev_info(canonical: str, allow_network: bool = True):
    """models.dev entry or None. Single-arg call on the default path: test sites monkeypatch
    ``get_provider_info`` with single-arg lambdas."""
    try:
        from agent.models_dev import get_provider_info as _mdev_provider
        return _mdev_provider(canonical) if allow_network else _mdev_provider(canonical, allow_network=False)
    except Exception:
        return None


def _profile_resolved_provider(name: str, mdev_info=None, *, source: str = "plugin-profile") -> Optional[_ResolvedProvider]:
    """Project a registered ProviderProfile into the canonical resolved-provider value."""
    try:
        profile = _get_provider_profile(name)
    except Exception:
        return None
    if profile is None:
        return None

    profile_env = tuple(profile.env_vars or ())
    base_url_env_var = str(profile.base_url_env_var or "").strip()

    env_vars = list(tuple(getattr(mdev_info, "env", ()) or ()))
    for value in profile_env:
        if value not in env_vars:
            env_vars.append(value)

    routing_aggregator = (
        profile.is_routing_aggregator
        if profile.is_routing_aggregator is not None
        else profile.is_aggregator
    )
    return _ResolvedProvider(
        id=profile.name,
        display_name=profile.display_name or getattr(mdev_info, "name", "") or profile.name or name,
        api_mode=(profile.api_mode or "chat_completions").strip(),
        auth_type=profile.auth_type or "api_key",
        env_vars=tuple(env_vars),
        base_url=(profile.base_url or getattr(mdev_info, "api", "") or "").strip(),
        base_url_env_var=base_url_env_var,
        is_aggregator=bool(profile.is_aggregator),
        is_routing_aggregator=bool(routing_aggregator),
        source=source,
    )


def _models_dev_resolved_provider(canonical: str, mdev_info) -> _ResolvedProvider:
    """Project a models.dev-only provider into the canonical resolved-provider value."""
    return _ResolvedProvider(
        id=canonical,
        display_name=mdev_info.name or canonical,
        api_mode="chat_completions",
        auth_type="api_key",
        env_vars=tuple(mdev_info.env or ()),
        base_url=mdev_info.api or "",
        source="models.dev",
    )


def get_provider(name: str, *, allow_network: bool = True) -> Optional[_ResolvedProvider]:
    """Resolve a built-in/profile provider without owning provider identity declarations."""
    canonical = _normalize_provider(name)
    mdev_info = _models_dev_info(canonical, allow_network)
    resolved = _profile_resolved_provider(
        canonical,
        mdev_info,
        source="models.dev" if mdev_info is not None else "plugin-profile",
    )
    if resolved is not None:
        # Match the historical resolver boundary: placeholder/runtime-minted profiles with no
        # endpoint do not preempt configured custom-provider resolution at this rung.
        if (
            mdev_info is not None
            or resolved.base_url
            or (
                resolved.auth_type == "api_key"
                and resolved.env_vars
                and resolved.base_url_env_var
            )
        ):
            return resolved
        return None
    if mdev_info is not None:
        return _models_dev_resolved_provider(canonical, mdev_info)
    return None


def _plugin_profile_pdef(name: str) -> Optional[_ResolvedProvider]:
    """Resolve a registered profile directly for the final full-resolution rung."""
    return _profile_resolved_provider(name)


# -- Provider from user config ------------------------------------------------

def _user_pdef(pid: str, name: str, base_url: str, key_env: str, transport: str = "openai_chat") -> _ResolvedProvider:
    """Canonical resolved-provider value shared by configured provider entry shapes."""
    api_mode = TRANSPORT_TO_API_MODE.get(transport, transport or "chat_completions")
    return _ResolvedProvider(
        id=pid,
        display_name=name,
        api_mode=api_mode,
        env_vars=(key_env,) if key_env else (),
        base_url=base_url,
        is_aggregator=False,
        is_routing_aggregator=False,
        auth_type="api_key",
        source="user-config",
    )


def resolve_user_provider(name: str, user_config: Dict[str, Any]) -> Optional[_ResolvedProvider]:
    """Resolve a configured provider by stored key or canonical custom identity."""
    if not isinstance(user_config, dict) or not user_config:
        return None
    requested = (name or "").strip().lower()
    if not requested:
        return None

    entry = user_config.get(name)
    provider_key = name
    if not isinstance(entry, dict):
        entry = user_config.get(requested)
        provider_key = requested
    if not isinstance(entry, dict):
        entry = None
        for stored_key, candidate in user_config.items():
            if not isinstance(candidate, dict):
                continue
            key = str(stored_key or "").strip()
            display_name = str(candidate.get("name") or key).strip()
            if requested in _custom_provider_aliases(display_name, key):
                provider_key, entry = key, candidate
                break
    if not isinstance(entry, dict):
        return None
    return _user_pdef(provider_key, entry.get("name", "") or provider_key,
                      entry.get("api", "") or entry.get("url", "") or entry.get("base_url", "") or "",
                      entry.get("key_env") or entry.get("api_key_env") or "",
                      entry.get("transport", "openai_chat") or "openai_chat")


def resolve_custom_provider(name: str, custom_providers: Optional[List[Dict[str, Any]]]) -> Optional[_ResolvedProvider]:
    """Resolve a provider from the user's config.yaml ``custom_providers`` list. A stored bare
    ``"custom"`` (corrupt state from a prior model-switch bug) falls back to the first valid entry
    so existing configs self-heal."""
    requested = (name or "").strip().lower()
    if not requested or not custom_providers or not isinstance(custom_providers, list):
        return None
    first_valid: Optional[_ResolvedProvider] = None
    # If the stored provider is the bare string "custom" (corrupt state from a prior model-switch bug), fall
    # back to the first custom provider entry so existing configs self-heal. (GH #17478)
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        display_name = (entry.get("name") or "").strip()
        api_url = (entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or "").strip()
        if not display_name or not api_url:
            continue
        provider_key = (entry.get("provider_key") or "").strip()
        pdef = _user_pdef(_custom_provider_slug(display_name, provider_key), display_name, api_url,
                          (entry.get("key_env") or "").strip())
        if first_valid is None:
            first_valid = pdef
        if requested in _custom_provider_aliases(display_name, provider_key):
            return pdef
    if requested == "custom" and first_valid:
        return first_valid
    return None


# The local llama.cpp runtime's provider id + aliases: ONE definition, shared by the resolver rung
# below and the picker's Local row (``hermes_cli/inventory.py``) — the two drifting apart is what
# made the row's own id unresolvable.
LLAMACPP_PROVIDER_ID = "llamacpp"
LLAMACPP_ALIASES: Tuple[str, ...] = (LLAMACPP_PROVIDER_ID, "llama.cpp", "llama-cpp")


def _has_staged_local_models() -> bool:
    """True when GGUFs are staged under the Hermes home's ``models/`` — the model the picker's Local
    row offers, which the runtime seam serves by booting/attaching a server on selection."""
    try:
        from hermes_cli.local_runtime.bootstrap import staged_model_ids
        return bool(staged_model_ids())
    except Exception:
        return False


def _llamacpp_pdef() -> Optional[_ResolvedProvider]:
    """The llamacpp aliases are a real provider whenever the managed server (or a detected external
    one) resolves — reachability is the credential — OR a model is staged for the runtime to serve.
    The picker's Local row is built from staged GGUFs and is deliberately offline-first (selection
    starts the server through the runtime seam), so requiring a live endpoint before admitting the id
    made that row offer a provider the resolver rejected ("Unknown provider 'llamacpp'"). Without
    this rung model-switch rejected the very provider the Local Models 'Use' flow writes to config."""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint
        endpoint = resolve_llamacpp_endpoint(config=load_config_readonly(), wait_for_boot_s=0)
    except Exception:
        endpoint = None
    if not endpoint and not _has_staged_local_models():
        return None
    return _ResolvedProvider(
        id=LLAMACPP_PROVIDER_ID,
        display_name="Local",
        api_mode="chat_completions",
        env_vars=(),
        base_url=(endpoint or {}).get("base_url", ""),
        source="local-runtime",
    )


def resolve_provider_full(name: str, user_providers: Optional[Dict[str, Any]] = None,
                          custom_providers: Optional[List[Dict[str, Any]]] = None) -> Optional[_ResolvedProvider]:
    """Full resolution chain: user ``providers.<raw name>`` -> canonical provider profile/models.dev
    -> user providers (canonical, then raw) -> ``custom_providers`` ->
    managed llamacpp -> models.dev directly. User-defined ``providers.<name>`` is tried FIRST on
    the raw (pre-alias) name: a configured ``providers.openai`` pointing at api.openai.com must not
    be hijacked by the legacy "openai" -> "openrouter" alias."""
    canonical = _normalize_provider(name)
    raw = name.strip().lower()
    if user_providers:
        user_pdef = resolve_user_provider(raw, user_providers)
        if user_pdef is not None:
            return user_pdef
    pdef = get_provider(canonical)
    if pdef is not None:
        if pdef.source == "plugin-profile" and user_providers:
            user_pdef = resolve_user_provider(pdef.id, user_providers)
            if user_pdef is not None:
                return user_pdef
        return pdef
    if user_providers:
        for candidate in (canonical, raw):
            user_pdef = resolve_user_provider(candidate, user_providers)
            if user_pdef is not None:
                return user_pdef
    custom_pdef = resolve_custom_provider(name, custom_providers)
    if custom_pdef is not None:
        return custom_pdef
    if raw in LLAMACPP_ALIASES:
        pdef = _llamacpp_pdef()
        if pdef is not None:
            return pdef
    try:
        mdev_info = _models_dev_info(canonical)
        if mdev_info is not None:
            return _models_dev_resolved_provider(canonical, mdev_info)
    except Exception:
        pass
    # Plugin profiles whose endpoint is minted at runtime (empty base_url, e.g. a token exchange
    # that also returns the host) are still real providers: /model --provider, the model picker
    # and `hermes model` must not reject them as unknown. Last rung, so every user-configured
    # entry above wins; the bare ``custom`` placeholder is excluded because model-switch completes
    # it from the current endpoint (see get_provider).
    pdef = _plugin_profile_pdef(canonical)
    return pdef if pdef is not None and pdef.id != "custom" else None