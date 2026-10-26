"""Profile-owned discovery for configured relay endpoints."""

def _relay_model_catalog(normalized: str, relay: str) -> list[str] | None:
    """Live catalog probed at a configured ``model.base_url`` relay, or None to fall through.

    Token Plan retains tier fallback on failure and empty success. Other providers
    return live ids or None for the caller's local static fallback; no vendor egress.
    """
    from hermes_cli import models
    from hermes_cli.models_catalog_static import _ALIBABA_TOKEN_PLAN_PROVIDERS

    try:
        from providers import get_provider_profile

        profile = get_provider_profile(normalized)
        if profile is None or getattr(profile, "auth_type", "") != "api_key":
            return None
        api_key, _ = models._api_key_credentials(normalized)
        if normalized in _ALIBABA_TOKEN_PLAN_PROVIDERS:
            return models.probe_profile_catalog(normalized, profile, api_key, relay)
        live = profile.fetch_models(api_key=api_key, base_url=relay)
        return [str(m) for m in (live or []) if m] or None
    except Exception:
        return None
