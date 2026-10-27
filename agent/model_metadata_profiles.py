"""Provider-owned context bounds consulted before the persistent cache."""

import logging


logger = logging.getLogger(__name__)


def resolve_authoritative_context_length(
    model: str,
    base_url: str,
    api_key: str,
    provider: str,
) -> int | None:
    from agent.model_metadata import _infer_provider_from_url
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    context = profile.get_model_context_length(model) if profile else None
    if type(context) is int and context > 0:
        return context

    effective_provider = provider
    if base_url and (not provider or provider in {"openrouter", "custom"}):
        effective_provider = _infer_provider_from_url(base_url) or provider
    profile = get_provider_profile(effective_provider) if effective_provider else None
    if profile is None or not profile.use_live_model_metadata:
        return None
    try:
        context = profile.live_context_length(model, api_key=api_key, base_url=base_url)
    except Exception:
        logger.debug("Provider live context lookup failed for %s", effective_provider)
        return None
    return context if type(context) is int and context > 0 else None
