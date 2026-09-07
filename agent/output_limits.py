"""Resolve route-local output ceilings without changing conversation state."""

from agent.models_dev import lookup_model_max_output_tokens


def request_max_tokens(agent, requested: int | None = None) -> int | None:
    """Combine the user's budget with the active model's configured ceiling."""
    provider = getattr(agent, "provider", "")
    if provider == "custom":
        identity = getattr(agent, "requested_provider", "") or provider
        if identity not in {"custom", "auto"}:
            provider = identity if identity.startswith("custom:") else f"custom:{identity}"
    ceiling = lookup_model_max_output_tokens(
        provider, agent.model, base_url=getattr(agent, "base_url", ""),
    )
    budget = requested if requested is not None else agent.max_tokens
    if ceiling is None:
        return budget
    return min(budget, ceiling) if budget is not None else ceiling
