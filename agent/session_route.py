"""Requested session route, distinct from the provider serving a fallback turn."""


def requested_session_runtime(agent) -> dict:
    """Return durable routing fields without promoting a turn-local fallback."""
    if getattr(agent, "_fallback_activated", False):
        primary = getattr(agent, "_primary_runtime", None)
        if isinstance(primary, dict) and primary.get("model"):
            return primary
    return {
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "base_url": getattr(agent, "base_url", None),
        "api_mode": getattr(agent, "api_mode", None),
        "reasoning_config": getattr(agent, "reasoning_config", None),
        "service_tier": getattr(agent, "service_tier", None),
    }
