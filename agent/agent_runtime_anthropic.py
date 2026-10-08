"""Anthropic Messages credential resolution for runtime provider switches."""


def resolve_switched_anthropic_credentials(agent, new_provider, api_key, base_url):
    """Retain the effective endpoint and never fall back to another provider's credentials."""
    from agent.anthropic_credentials import resolve_anthropic_token

    effective_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
    effective_key = api_key or agent.api_key or (
        resolve_anthropic_token(effective_base_url, model=getattr(agent, "model", None))
        if new_provider == "anthropic" else ""
    ) or ""
    return effective_key, effective_base_url
