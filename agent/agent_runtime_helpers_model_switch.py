"""Resolve a model switch context and preserve LM Studio instance ownership."""

import logging

logger = logging.getLogger("agent.agent_runtime_helpers")


def resolve_switch_context_length(agent, snapshot):
    """Resolve the destination context length (LM Studio preload first); returns ``(custom_providers, effective_len)``."""
    from agent.agent_runtime_helpers import _restore_switch_snapshot

    custom_providers = None
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers, get_custom_provider_context_length, load_config
        )
        from agent.agent_init import config_context_length_for_runtime
        switch_cfg = load_config()
        custom_providers = get_compatible_custom_providers(switch_cfg)
        # The durable ``model.context_length`` pin is re-read from live config (never carried over
        # blindly, never simply dropped): the destination IS the configured default route -> keep the
        # ceiling; it is some other route -> the scoping inside returns None. Same precedence as
        # construction, where the pin outranks custom_providers metadata (#116467).
        intent = config_context_length_for_runtime(agent, switch_cfg)
        if intent is None:
            intent = get_custom_provider_context_length(
                model=agent.model, base_url=agent.base_url, custom_providers=custom_providers
            )
    except Exception:
        logger.debug("Could not read model switch context settings", exc_info=True)
        intent = None
    from agent.agent_init import set_config_context_length
    set_config_context_length(agent, intent)
    runtime_len = None
    if hasattr(agent, "_ensure_lmstudio_runtime_loaded"):
        try:
            from hermes_cli.models_local import _lmstudio_server_root
            same_lmstudio_endpoint = (
                agent.provider == snapshot.get("provider") == "lmstudio"
                and _lmstudio_server_root(agent.base_url) == _lmstudio_server_root(snapshot.get("base_url"))
            )
            if same_lmstudio_endpoint:
                runtime_len = agent._ensure_lmstudio_runtime_loaded(intent, previous_model=snapshot.get("model"))
            else:
                runtime_len = agent._ensure_lmstudio_runtime_loaded(intent)
        except Exception:
            _restore_switch_snapshot(agent, snapshot)
            raise
    if hasattr(agent, "_lmstudio_load_was_unverified") and agent._lmstudio_load_was_unverified(runtime_len):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length during model switch; continuing "
            "with configured context"
        )
    effective = intent
    if hasattr(agent, "_effective_lmstudio_context_length"):
        effective = agent._effective_lmstudio_context_length(intent, runtime_len)
    return custom_providers, effective

