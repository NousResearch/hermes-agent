"""Application-resolved Nous recommendation facts.

The app obtains Portal/account identity; the model domain owns the one cache,
the Nous plugin owns the public HTTP request, and the provider domain owns
pure auxiliary recommendation extraction.
"""
from __future__ import annotations

from typing import Any


def portal_base_url() -> str:
    """Use this profile's authenticated Portal endpoint without exposing credentials."""
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
        return str(state.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL).strip().rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def fetch_recommended_models(
    portal_url: str = "", timeout: float = 5.0, *, force_refresh: bool = False
) -> dict[str, Any]:
    from models.catalog_nous_recommendations import fetch_recommended_models as fetch

    return fetch(portal_url or portal_base_url(), timeout, force_refresh=force_refresh)


def auxiliary_model(*, vision: bool = False, force_refresh: bool = False) -> str:
    from hermes_cli.models import check_nous_free_tier
    from providers.nous_recommendations import recommended_aux_model

    payload = fetch_recommended_models(force_refresh=force_refresh)
    try:
        free_tier = check_nous_free_tier()
    except Exception:
        free_tier = False
    return recommended_aux_model(payload, vision=vision, free_tier=free_tier) or ""
