"""Reasoning policy when binding a fallback model."""
import logging

logger = logging.getLogger(__name__)


def reresolve_fallback_reasoning_config(agent) -> None:
    """Keep an ephemeral turn selection; otherwise use the fallback's native configured baseline."""
    from agent.turn_reasoning import _active_config
    selected = _active_config(agent)
    if selected is not None:
        agent.reasoning_config = selected
        return
    try:
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config
        agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, agent.reasoning_config)
    except Exception as exc:  # health: allow BLE001 -- profile read failure must not prevent fallback; preserve its cause
        logger.debug("Failed to resolve reasoning_config for fallback %s; keeping current", agent.model, exc_info=exc)
