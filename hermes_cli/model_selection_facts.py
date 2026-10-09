"""Caller-owned fact acquisition for canonical explicit model selection."""

from __future__ import annotations

from dataclasses import replace

from models import ModelRef
from models.selection import ExplicitDetectionFacts, select_detected_model
from models.catalog_static import static_provider_model_ids
from providers import list_providers, normalize_provider


def build_explicit_detection_facts(
    raw_model: str,
    current_provider: str,
) -> ExplicitDetectionFacts:
    """Acquire only the facts still needed by the canonical selector."""

    from hermes_cli.models import _find_openrouter_slug, _resolve_provider_prefix
    from models.catalog_detection import (
        _model_in_provider_catalog, _provider_keys, _static_catalog_matches,
        _resolve_static_model_alias, current_provider_owns_vendor,
    )
    from application_model_selection_defaults import select_provider_default, selected_model_id
    from hermes_cli.models_detect import current_provider_catalog_match, provider_has_credentials

    raw = str(raw_model or "").strip()
    current = str(current_provider or "").strip().lower()
    current_keys = _provider_keys(current)
    facts = ExplicitDetectionFacts(
        current_catalog_model=current_provider_catalog_match(raw, current) or "",
        current_provider_owns_model=current_provider_owns_vendor(raw, current),
        allow_first_guess=current in {"", "auto"},
    )
    if select_detected_model(raw, current, replace(facts, allow_first_guess=False)) is not None:
        return facts

    named = None
    named_provider = normalize_provider(raw.lower())
    known = {
        normalize_provider(str(profile.name or "").strip().lower())
        for profile in list_providers()
        if str(profile.name or "").strip()
    }
    if named_provider not in {"custom", "openrouter"} and named_provider in known:
        defaults = tuple(static_provider_model_ids(named_provider))
        if defaults and named_provider not in current_keys:
            named = ModelRef(
                named_provider,
                selected_model_id(select_provider_default(named_provider)) or defaults[0],
            )

    static = tuple(ModelRef(provider, model) for provider, model in _static_catalog_matches(raw, current))
    alias = _resolve_static_model_alias(raw.lower(), current_keys)
    if alias is not None:
        static = tuple(dict.fromkeys((ModelRef(*alias), *static)))
    candidate_providers = tuple(dict.fromkeys(ref.provider for ref in static))
    eligible = tuple(provider for provider in candidate_providers if provider_has_credentials(provider))
    facts = replace(
        facts,
        named_provider_candidate=named,
        static_candidates=static,
        current_static_owns_model=_model_in_provider_catalog(raw.lower(), current_keys),
        eligible_providers=eligible,
    )
    # The selector decides precedence; acquisition merely avoids unnecessary
    # network/configuration probes after it has an authoritative answer.
    if select_detected_model(raw, current, replace(facts, allow_first_guess=False)) is not None:
        return facts

    openrouter_slug = _find_openrouter_slug(raw)
    openrouter = ModelRef("openrouter", openrouter_slug) if openrouter_slug else None
    declared = _resolve_provider_prefix(raw)
    declared_ref = ModelRef(*declared) if declared else None
    later_providers = tuple(dict.fromkeys(
        ref.provider for ref in (openrouter, declared_ref)
        if ref is not None and ref.provider not in candidate_providers
    ))
    return replace(
        facts,
        openrouter_candidate=openrouter,
        declared_provider_candidate=declared_ref,
        eligible_providers=(*eligible, *(p for p in later_providers if provider_has_credentials(p))),
    )


__all__ = ["build_explicit_detection_facts"]
