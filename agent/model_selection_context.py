"""Context admission checks for a proposed model route, before it is committed."""

from types import SimpleNamespace


def _compression_context_pin(cfg):
    from hermes_cli.config import cfg_get
    value = cfg_get(cfg, "auxiliary", "compression", "context_length", default=None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def validate_resolved_context(agent, context_length, custom_providers, config_context_length):
    """Use startup's floor checks without mutating the live compressor or emitting notices."""
    from agent.agent_init import _enforce_minimum_context
    from agent.conversation_compression import check_compression_model_feasibility

    runtime = {name: getattr(agent, name, "") for name in
               ("model", "provider", "base_url", "api_key", "api_mode")}
    probe = SimpleNamespace(
        **runtime,
        context_compressor=SimpleNamespace(context_length=context_length, threshold_tokens=0),
        _config_context_length=config_context_length,
        _ollama_num_ctx=getattr(agent, "_ollama_num_ctx", None),
        compression_enabled=getattr(agent, "compression_enabled", False),
        _aux_compression_context_length_config=getattr(agent, "_aux_compression_context_length_config", None),
        _custom_providers=custom_providers,
        _current_main_runtime=lambda: runtime,
        _emit_diagnostic_status=lambda _message: None,
    )
    _enforce_minimum_context(probe)
    check_compression_model_feasibility(probe)


def validate_model_selection(result, custom_providers=None):
    """Validate sessionless selections too; a stale default-route pin must not admit another model."""
    from agent.agent_init import (
        config_context_length_for_runtime, _configure_ollama_num_ctx, _cfg_dict, _cfg_flag,
    )
    from agent.model_metadata import get_model_context_length
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly

    cfg = load_config_readonly()
    if custom_providers is None:
        custom_providers = get_compatible_custom_providers(cfg)
    candidate = SimpleNamespace(
        model=result.new_model, provider=result.target_provider, base_url=result.base_url,
        api_key=result.api_key, api_mode=result.api_mode, quiet_mode=True,
        compression_enabled=_cfg_flag(_cfg_dict(cfg, "compression"), "enabled", True),
        _aux_compression_context_length_config=_compression_context_pin(cfg),
    )
    pin = config_context_length_for_runtime(candidate, cfg)
    key = result.api_key if isinstance(result.api_key, str) else ""
    context = get_model_context_length(
        candidate.model, base_url=candidate.base_url, api_key=key, provider=candidate.provider,
        config_context_length=pin, custom_providers=custom_providers,
    )
    _configure_ollama_num_ctx(candidate, cfg.get("model", {}), pin)
    validate_resolved_context(candidate, context, custom_providers, pin)


def update_switch_compressor(agent, custom_providers, effective_context_length, snapshot):
    """Validate the destination before changing the working compressor; reject atomically."""
    from agent.agent_runtime_helpers import _restore_switch_snapshot
    from agent.agent_init import _enforce_minimum_context
    from agent.conversation_compression import check_compression_model_feasibility
    from agent.context_compressor import ContextCompressor
    from agent.model_metadata import get_model_context_length
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from copy import copy

    key = agent.api_key if isinstance(agent.api_key, str) else ""
    notices = []
    try:
        if custom_providers is None:
            custom_providers = get_compatible_custom_providers(load_config())
        candidate = copy(agent)
        compressor = agent.context_compressor
        candidate.context_compressor = (
            copy(compressor) if isinstance(compressor, ContextCompressor) else compressor.clone_for_agent()
        )
        candidate._custom_providers = custom_providers
        candidate.compression_enabled = getattr(agent, "compression_enabled", False)
        candidate._ollama_num_ctx = getattr(agent, "_ollama_num_ctx", None)
        candidate._emit_diagnostic_status = notices.append
        context = get_model_context_length(
            agent.model, base_url=agent.base_url, api_key=key, provider=agent.provider,
            config_context_length=effective_context_length, custom_providers=custom_providers,
        )
        candidate.context_compressor.update_model(
            model=agent.model, context_length=context, base_url=agent.base_url, api_key=agent.api_key,
            provider=agent.provider, api_mode=agent.api_mode,
        )
        _enforce_minimum_context(candidate)
        check_compression_model_feasibility(candidate)
    except Exception:
        _restore_switch_snapshot(agent, snapshot)
        raise
    agent.context_compressor.__dict__.update(candidate.context_compressor.__dict__)
    for name in ("_last_feasibility_notice", "_compression_warning"):
        if hasattr(candidate, name):
            setattr(agent, name, getattr(candidate, name))
    agent._compression_feasibility_checked = True
    for notice in notices:
        agent._emit_diagnostic_status(notice)
