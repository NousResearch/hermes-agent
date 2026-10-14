"""Stable ACP configuration, derived from the existing catalog and wire capabilities."""
from __future__ import annotations

from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption, SessionModelState


def reasoning_efforts(agent) -> tuple[str, ...]:
    """Unknown capability is not permission to advertise a generic effort ladder."""
    from agent.transports.codex import _codex_efforts_for_route, _is_openai_api_origin, _profile_declared_efforts
    from agent.model_metadata import _infer_provider_from_url

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip().lower()
    base_url = getattr(agent, "base_url", None)
    if provider == "openai-codex":
        return _codex_efforts_for_route(model, base_url, is_codex_backend=True)
    if _is_openai_api_origin(base_url) or provider == "openai":
        if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            return _codex_efforts_for_route(model, base_url)
        return ()
    if provider == "custom" or provider.startswith("custom:"):
        inferred = _infer_provider_from_url(str(base_url or ""))
        if not inferred or inferred == "custom":
            return ()
        provider = inferred
    return _profile_declared_efforts(provider, model, base_url) or ()


def reasoning_value(agent, efforts: tuple[str, ...]) -> str:
    from agent.reasoning_effort import clamp_effort

    config = getattr(agent, "reasoning_config", None)
    config = config if isinstance(config, dict) else {}
    if config.get("enabled") is False and "none" in efforts:
        return "none"
    return clamp_effort(config.get("effort") or "medium", efforts)


def build_config_options(state, models: SessionModelState, modes: dict) -> list[SessionConfigOptionSelect]:
    from acp_adapter.auth import detect_provider
    from acp_adapter.model_catalog import _semantic_provider
    from hermes_cli.config import load_config
    from hermes_cli.models import normalize_provider

    configured = load_config().get("model", {})
    if isinstance(configured, dict):
        default_model = str(configured.get("default", configured.get("name", "")) or "")
        provider = str(configured.get("provider") or "").strip().lower()
    else:
        default_model, provider = str(configured or ""), ""
    if not provider or provider == "auto":
        provider = detect_provider() or ""
    semantic_provider = _semantic_provider(provider, normalize_provider)
    default_choice = next((
        row.model_id for row in models.available_models
        if default_model and row.model_id.endswith(":" + default_model)
        and _semantic_provider(row.model_id[:-(len(default_model) + 1)], normalize_provider) == semantic_provider
    ), None)
    rows = list(models.available_models)
    rows.sort(key=lambda row: row.model_id != default_choice)
    options = [SessionConfigOptionSelect(
        id="model", name="Model", category="model", type="select",
        current_value=models.current_model_id,
        options=[SessionConfigSelectOption(
            value=row.model_id,
            name=(f"Configured Default — {row.name}" if row.model_id == default_choice else row.name),
            description=row.description,
        ) for row in rows],
    )]
    efforts = reasoning_efforts(state.agent)
    if efforts:
        options.append(SessionConfigOptionSelect(
            id="reasoning_effort", name="Thinking", category="thought_level", type="select",
            current_value=reasoning_value(state.agent, efforts),
            options=[SessionConfigSelectOption(value=effort, name=effort.title()) for effort in efforts],
        ))
    options.append(SessionConfigOptionSelect(
        id="edit_approval_policy", name="Edit approval", category="mode", type="select",
        current_value=modes.get(getattr(state, "mode", "default"), modes["default"])[0],
        options=[SessionConfigSelectOption(value=spec[0], name=spec[1], description=spec[2])
                 for spec in modes.values()],
    ))
    return options
