"""Runtime fact acquisition for canonical auxiliary-model selection."""

from __future__ import annotations

import logging

from models.selection import (
    select_auxiliary_fallback_model,
    select_auxiliary_model,
    selected_auxiliary_model_id,
)

logger = logging.getLogger(__name__)


def _nous_allowed_ids() -> set[str] | None:
    try:
        from application_model_pricing import nous_policy_allowed_ids
        return nous_policy_allowed_ids()
    except Exception:
        logger.debug("Nous auxiliary policy lookup unavailable", exc_info=True)
        return None


def _fast_catalog_ids(provider: str) -> tuple[str, ...]:
    """Fetch the provider catalog needed by fast-model policy; never choose here."""

    provider_id = str(provider or "").strip().lower()
    is_nous = provider_id == "nous"
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials
        from application_model_pricing import fetch_models_with_pricing
        from providers import get_provider_profile

        api_key = ""
        base_url = ""
        try:
            creds = resolve_api_key_provider_credentials(provider_id) or {}
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
        except Exception:
            logger.debug("No API-key credentials for %s catalog", provider_id, exc_info=True)

        if not api_key and is_nous:
            try:
                from application_model_pricing import _resolve_nous_pricing_credentials
                api_key, base_url = _resolve_nous_pricing_credentials()
            except Exception:
                logger.debug("No Nous credentials for auxiliary catalog", exc_info=True)

        profile = get_provider_profile(provider_id)
        if not base_url and profile is not None:
            base_url = str(profile.base_url or "")
        base_url = base_url.rstrip("/")
        if not base_url:
            return ()
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        kwargs = {}
        if is_nous:
            from application_model_pricing import _NOUS_CATALOG_TTL_SECONDS
            kwargs = {
                "include_sale_original": True,
                "cache_ttl_seconds": _NOUS_CATALOG_TTL_SECONDS,
            }
        catalog = fetch_models_with_pricing(
            api_key=api_key or None,
            base_url=base_url,
            timeout=3.0,
            **kwargs,
        ) or {}
        return tuple(str(model) for model in catalog)
    except Exception:
        logger.debug("Auxiliary catalog lookup failed for %s", provider_id, exc_info=True)
        return ()


def select_provider_auxiliary_model(
    provider: str,
    *,
    main_model: str = "",
    prefer_fast: bool = False,
) -> str:
    """Return the canonical provider-local auxiliary model from acquired facts."""

    from providers import get_provider_profile

    provider_id = str(provider or "").strip().lower()
    try:
        profile = get_provider_profile(provider_id)
    except Exception:
        profile = None

    resolved_aux = ""
    if prefer_fast and profile is not None:
        try:
            resolved_aux = str(profile.resolve_aux_model() or "").strip()
        except Exception:
            logger.debug("resolve_aux_model failed for %s", provider_id, exc_info=True)

    selection = select_auxiliary_model(
        provider_id,
        main_model=main_model,
        resolved_aux_model=resolved_aux,
        default_aux_model=str(getattr(profile, "default_aux_model", "") or ""),
        live_model_ids=_fast_catalog_ids(provider_id) if prefer_fast else (),
        prefer_fast=prefer_fast,
        allowed_model_ids=_nous_allowed_ids() if provider_id == "nous" else None,
    )
    return selected_auxiliary_model_id(selection)


def select_provider_auxiliary_fallback(
    provider: str,
    *,
    preferred_model: str = "",
    excluded_model: str = "",
) -> str:
    """Choose a fallback-lane model from caller recommendation + provider declaration."""

    from providers import get_provider_profile

    provider_id = str(provider or "").strip().lower()
    try:
        profile = get_provider_profile(provider_id)
    except Exception:
        profile = None
    selection = select_auxiliary_fallback_model(
        provider_id,
        preferred_model=preferred_model,
        fallback_model=str(getattr(profile, "fallback_aux_model", "") or ""),
        excluded_model=excluded_model,
        allowed_model_ids=_nous_allowed_ids() if provider_id == "nous" else None,
    )
    return selected_auxiliary_model_id(selection)


__all__ = [
    "select_provider_auxiliary_fallback",
    "select_provider_auxiliary_model",
]
