"""API-only fallback policy; never borrow another platform's fallback chain."""
from copy import deepcopy

from agent.error_classifier import FailoverReason

CAPACITY_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.upstream_rate_limit,
                              FailoverReason.overloaded})


def scoped_fallback_chain(config):
    """``gateway.api_server.fallback_providers``: None (unset) keeps the global chain; a list is
    the API server's own chain ([] = no fallback), each row exactly ``{provider, model}``."""
    chain = ((config.get("gateway") or {}).get("api_server") or {}).get("fallback_providers")
    if chain is not None and (not isinstance(chain, list) or any(
            not isinstance(row, dict) or set(row) != {"provider", "model"}
            or any(not isinstance(row[k], str) or not row[k].strip() for k in ("provider", "model"))
            for row in chain)):
        raise ValueError("gateway.api_server.fallback_providers requires provider/model entries")
    return chain


def base_runtime_kwargs(config, requested_provider, requested_model):
    """The gateway runtime, or with an API-scoped chain the explicitly requested provider's own:
    the global resolver may already have switched to a global fallback provider."""
    if requested_provider and scoped_fallback_chain(config) is not None:
        from gateway.platforms.api_server import _resolve_request_runtime_agent_kwargs
        return _resolve_request_runtime_agent_kwargs(requested_provider, requested_model)
    from gateway.run import _resolve_runtime_agent_kwargs
    return _resolve_runtime_agent_kwargs()


def constructor_fallback(config, locked):
    """Same chain as Telegram/Discord/Slack unless the API server scopes its own (or is locked)."""
    if locked or scoped_fallback_chain(config) is not None:
        return None
    from gateway.run import GatewayRunner
    return GatewayRunner._load_fallback_model()


def create_api_agent(factory, kwargs, config, locked=False, explicit_reasoning=None):
    """Construct the agent, then install the API-scoped chain (capacity failures only) and keep
    an explicit request reasoning setting across fallback switches."""
    chain = scoped_fallback_chain(config)
    if chain is not None or locked:
        kwargs = {**kwargs, "fallback_model": None}
    # Constructor auth failures must propagate, before installing capacity fallback.
    agent = factory(**kwargs)
    if chain is not None and not locked:
        from agent.agent_init_fallback import _init_fallback_chain
        _init_fallback_chain(agent, deepcopy(chain))
        agent._request_fallback_reasons = CAPACITY_REASONS
    if explicit_reasoning is not None:
        agent._request_reasoning_config = deepcopy(explicit_reasoning)
    return agent
