"""Resolve explicit session-create model routes through canonical model/provider domains."""

from __future__ import annotations

from models import parse_configured_provider_ref
from models.catalog_static import find_static_provider_model_id
from models.selection import ExplicitAlias, explicit_provider_hint, select_explicit_model
from providers import is_aggregator, is_routing_aggregator, normalize_provider
from providers.routing import InvocationRequest, resolve_invocation_route

from application_model_aliases import (
    alias_api_key,
    configured_provider_ids,
    model_aliases_from_config,
    provider_reference_context,
)


def _catalog_match(provider: str, raw_model: str) -> bool:
    """Whether *raw_model* is already a provider-native catalogue ID."""
    return find_static_provider_model_id(provider, raw_model) is not None


def _selected_ref(
    raw: str,
    current_provider: str,
    *,
    explicit_provider: str = "",
    direct_aliases: tuple[ExplicitAlias, ...] = (),
    known_provider_ids: tuple[str, ...] = (),
    named_custom_provider_ids: tuple[str, ...] = (),
):
    selection = select_explicit_model(
        raw,
        current_provider,
        explicit_provider=explicit_provider,
        direct_aliases=direct_aliases,
        known_provider_ids=known_provider_ids,
        named_custom_provider_ids=named_custom_provider_ids,
        block_provider_fallback=True,
    )
    return selection.selected.ref if selection.selected is not None else None


def resolve_launch_route(params: dict, config: dict) -> dict:
    """Resolve a session-create explicit model without delegating to CLI model semantics."""
    raw = params.get("model")
    if not isinstance(raw, str) or not raw.strip() or "base_url" in params or "api_key" in params:
        return params

    raw = raw.strip()
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    explicit_provider = str(params.get("provider") or "").strip()
    current_provider = explicit_provider or str(model_cfg.get("provider") or "").strip().lower()
    known_ids, named_custom_ids = provider_reference_context(config)

    aliases = model_aliases_from_config(config)
    direct = aliases.get(raw.lower())
    if direct is not None:
        ref = _selected_ref(
            raw,
            current_provider,
            explicit_provider=explicit_provider,
            direct_aliases=(ExplicitAlias(direct.name, direct.ref),),
            known_provider_ids=known_ids,
            named_custom_provider_ids=named_custom_ids,
        )
        if ref is None:
            return params
        requested_provider = explicit_provider or ("custom" if direct.base_url else ref.provider)
        route = resolve_invocation_route(
            InvocationRequest(provider=requested_provider, model=ref.model, base_url=direct.base_url)
        )
        resolved = dict(params, model=route.model)
        if requested_provider:
            resolved["provider"] = requested_provider
        if direct.base_url:
            resolved["base_url"] = direct.base_url
        if not explicit_provider and (key := alias_api_key(direct)):
            resolved["api_key"] = key
        return resolved

    # A caller-supplied provider already owns a plain model request. Only aliases above
    # contribute additional startup facts when --provider is explicit.
    if explicit_provider:
        return params

    qualified_provider = explicit_provider_hint(
        raw,
        known_provider_ids=known_ids,
        named_custom_provider_ids=named_custom_ids,
    )
    if qualified_provider:
        ref = _selected_ref(
            raw,
            current_provider,
            known_provider_ids=known_ids,
            named_custom_provider_ids=named_custom_ids,
        )
        if ref is None:
            return params
        route = resolve_invocation_route(InvocationRequest(provider=ref.provider, model=ref.model))
        return dict(params, model=route.model, provider=route.provider)

    # Routing aggregators own their native slash namespace before configured provider
    # prefixes are considered.
    if current_provider and is_routing_aggregator(current_provider) and _catalog_match(current_provider, raw):
        return params

    configured = configured_provider_ids(config)
    configured_ref = parse_configured_provider_ref(raw, configured)
    if configured_ref is None:
        return params

    prefix = raw.split("/", 1)[0].strip()
    canonical = normalize_provider(prefix)
    configured_set = set(configured)
    if prefix.lower() in configured_set:
        requested_provider = prefix
    elif canonical in configured_set:
        requested_provider = canonical
    else:
        return params
    if is_aggregator(canonical):
        return params

    ref = _selected_ref(
        configured_ref.model,
        current_provider,
        explicit_provider=requested_provider,
        known_provider_ids=known_ids,
        named_custom_provider_ids=named_custom_ids,
    )
    if ref is None:
        return params
    route = resolve_invocation_route(
        InvocationRequest(provider=requested_provider, model=ref.model)
    )
    # Preserve the configured request key for credential acquisition; route semantics
    # have already been resolved canonically by providers.routing.
    return dict(params, model=route.model, provider=requested_provider)
