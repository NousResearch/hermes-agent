"""Request-scoped fallback constraints and reasoning precedence."""
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def fallback_reason_allowed(agent, reason):
    allowed = getattr(agent, "_request_fallback_reasons", None)
    return not isinstance(allowed, frozenset) or reason in allowed


def reresolve_fallback_reasoning_config(agent):
    """Explicit request reasoning survives the swap; otherwise per-model override > global
    reasoning_effort (YAML False = disabled) for the new fallback model (#21256). A config load
    failure keeps the current reasoning_config rather than killing the swap."""
    explicit = getattr(agent, "_request_reasoning_config", None)
    if isinstance(explicit, dict):
        agent.reasoning_config = deepcopy(explicit)
        logger.info("Fallback %s: keeping explicit request reasoning_config: %s", agent.model, agent.reasoning_config)
        return
    try:
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config
        agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, agent.reasoning_config)
    except Exception:
        logger.debug("Failed to resolve reasoning_config for fallback %s; keeping current", agent.model, exc_info=True)
