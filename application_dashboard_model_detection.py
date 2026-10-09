"""Dashboard flat Settings model-field inference using canonical selection facts.

Only choose a foreign provider when one known route can serve the model AND
application credentials authorize it; never infer a paid provider from a vendor.
"""
from __future__ import annotations

from typing import Mapping, Any

from application_model_facts import configured_model_matches, static_detection
from models.catalog_static import static_provider_model_ids
from models.selection_detection import select_detected_model
from providers import is_aggregator, normalize_provider, vendor_for_model


def _authorized(provider: str) -> bool:
    # Credential availability remains Phase 6 application facts, not selection semantics.
    from hermes_cli.models_detect import provider_has_credentials
    return provider_has_credentials(provider)


def infer_dashboard_model_change(
    model: str, current_provider: str, config: Mapping[str, Any],
) -> tuple[str, str]:
    raw = str(model or "").strip()
    current = normalize_provider(current_provider)
    if not raw or current == "custom" or current.startswith("custom:"):
        return "", raw
    if "/" in raw and is_aggregator(current):
        return "", raw
    facts = static_detection(raw, current)
    if facts.current_catalog_model:
        return current, facts.current_catalog_model
    # Early-access first-party IDs may precede the curated catalogue. A native
    # provider with exactly one known vendor keeps its own vendor models.
    wanted_vendor = vendor_for_model(raw)
    native_vendors = {vendor_for_model(mid) for mid in static_provider_model_ids(current)}
    if wanted_vendor and native_vendors == {wanted_vendor} and not is_aggregator(current):
        return current, raw
    matches = configured_model_matches(raw, config)
    authorized_named = [ref for ref in matches if _authorized(ref.provider)]
    if len(authorized_named) == 1:
        return authorized_named[0].provider, authorized_named[0].model
    # Do not choose between two plausible foreign providers without a
    # user-entered provider; explicit assignment has no such ambiguity.
    eligible = {
        normalize_provider(ref.provider) for ref in facts.static_candidates
        if _authorized(ref.provider)
    }
    if len(eligible) == 1:
        ref = select_detected_model(
            raw, current, type(facts)(
                current_catalog_model=facts.current_catalog_model,
                static_candidates=facts.static_candidates,
                current_static_owns_model=facts.current_static_owns_model,
                eligible_providers=tuple(eligible),
            ),
        )
        if ref is not None:
            return ref.provider, ref.model
    if "/" in raw and _authorized("openrouter"):
        return "openrouter", raw
    return "", raw


__all__ = ["infer_dashboard_model_change"]
