"""Route-aware model normalization during agent initialization."""


def normalize_model_for_route(agent) -> None:
    """Preserve catalog IDs for aggregators and Bedrock Mantle."""
    from agent.anthropic_endpoints import _is_bedrock_mantle_endpoint
    from hermes_cli.model_normalize import (
        _AGGREGATOR_PROVIDERS, normalize_model_for_provider,
    )

    if (agent.provider not in _AGGREGATOR_PROVIDERS
            and not _is_bedrock_mantle_endpoint(agent.base_url)):
        agent.model = normalize_model_for_provider(agent.model, agent.provider)
