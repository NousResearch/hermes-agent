"""Caller-owned facts for canonical application model selection."""

from __future__ import annotations

from typing import Any, Mapping

from models import ModelRef
from models.catalog_static import find_static_provider_model_id, static_provider_model_ids
from models.selection import ExplicitDetectionFacts, ExplicitProviderFacts
from providers import (
    custom_provider_aliases,
    custom_provider_slug,
    get_provider_profile,
    is_aggregator,
    list_providers,
    normalize_provider,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _declared_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, (list, tuple)):
        values = (
            item.get("id") or item.get("name") if isinstance(item, Mapping) else item
            for item in value
        )
    else:
        return ()
    return tuple(
        dict.fromkeys(_clean(item) for item in values if isinstance(item, str) and _clean(item))
    )


def configured_model_matches(raw: str, config: Mapping[str, Any]) -> tuple[ModelRef, ...]:
    wanted = raw.lower()
    refs: list[ModelRef] = []
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        for key, entry in providers.items():
            if not isinstance(entry, Mapping):
                continue
            hit = next(
                (
                    model
                    for field in ("models", "model", "default_model")
                    for model in _declared_ids(entry.get(field))
                    if model.lower() == wanted
                ),
                None,
            )
            if hit:
                refs.append(ModelRef(
                    custom_provider_slug(_clean(entry.get("name")) or _clean(key), _clean(key)),
                    hit,
                ))
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        for entry in legacy:
            if not isinstance(entry, Mapping) or not _clean(entry.get("name")):
                continue
            hit = next(
                (
                    model
                    for field in ("models", "model", "default_model")
                    for model in _declared_ids(entry.get(field))
                    if model.lower() == wanted
                ),
                None,
            )
            if hit:
                ref = ModelRef(
                    custom_provider_slug(_clean(entry.get("name")), _clean(entry.get("provider_key"))),
                    hit,
                )
                if ref not in refs:
                    refs.append(ref)
    return tuple(refs)


def _provider_fact(provider: str, raw: str, config: Mapping[str, Any]) -> ExplicitProviderFacts:
    canonical = normalize_provider(provider)
    profile = get_provider_profile(canonical)
    aliases = {canonical, _clean(provider).lower()}
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        for key, entry in providers.items():
            if not isinstance(entry, Mapping):
                continue
            entry_aliases = custom_provider_aliases(
                _clean(entry.get("name")) or _clean(key), _clean(key)
            )
            if canonical in entry_aliases or _clean(provider).lower() in entry_aliases:
                aliases.update(entry_aliases)
    static = tuple(static_provider_model_ids(canonical))
    fallback = tuple(getattr(profile, "fallback_models", ()) or ())
    exact = find_static_provider_model_id(canonical, raw)
    return ExplicitProviderFacts(
        provider=canonical,
        alias_models=tuple(dict.fromkeys((*static, *fallback))),
        catalog_models=(exact,) if exact else (),
        model_aliases=tuple(
            (str(key).strip().lower(), str(value).strip())
            for key, value in dict(getattr(profile, "model_aliases", {}) or {}).items()
            if str(key).strip() and str(value).strip()
        ),
        identity_aliases=tuple(sorted(alias for alias in aliases if alias)),
        aggregator=is_aggregator(canonical),
        normalization_ids=static,
    )


def provider_facts(
    providers: tuple[str, ...], raw: str, config: Mapping[str, Any]
) -> tuple[ExplicitProviderFacts, ...]:
    merged: dict[str, ExplicitProviderFacts] = {}
    for provider in providers:
        if provider:
            fact = _provider_fact(provider, raw, config)
            merged.setdefault(fact.provider, fact)
    return tuple(merged.values())


def static_detection(raw: str, current_provider: str) -> ExplicitDetectionFacts:
    candidates: list[ModelRef] = []
    for profile in list_providers():
        provider = _clean(profile.name)
        hit = find_static_provider_model_id(provider, raw)
        if hit:
            candidates.append(ModelRef(provider, hit))
    current_hit = find_static_provider_model_id(current_provider, raw) or ""
    return ExplicitDetectionFacts(
        current_catalog_model=current_hit,
        static_candidates=tuple(candidates),
        current_static_owns_model=bool(current_hit),
        eligible_providers=(normalize_provider(current_provider),),
    )


__all__ = ["configured_model_matches", "provider_facts", "static_detection"]
