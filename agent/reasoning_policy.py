"""Preserve scoped reasoning intent across model changes."""
import logging


def refresh_model_reasoning(agent) -> None:
    if getattr(agent, "_reasoning_config_fixed", False) is True:
        return
    from hermes_cli.config import load_config
    from hermes_constants import resolve_reasoning_config

    agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
    logging.getLogger("run_agent").info("Reasoning configuration resolved for %s: %s", agent.model, agent.reasoning_config)
