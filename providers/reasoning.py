"""Resolve settings for providers that opt into validated reasoning selection."""

import logging

from agent.reasoning_effort import clamp_effort
from providers import get_provider_profile


def reasoning_selection_efforts(provider: str, model: str) -> tuple[str, ...] | None:
    profile = get_provider_profile(provider)
    if profile is None or not profile.validate_reasoning_selection:
        return None
    return profile.supported_reasoning_efforts(model)


def resolve_provider_reasoning_config(
    provider: str, model: str, config: dict | None, *, explicit: bool = False
) -> dict | None:
    effort = "none" if config and config.get("enabled") is False else str((config or {}).get("effort") or "").strip().lower()
    if effort.startswith("budget:"):
        profile = get_provider_profile(provider)
        bounds = (profile.describe_models(model_ids=[model]).get(model, {}).get("reasoning_budget")
                  if profile else None)
        value = effort[7:]
        if bounds and value == "-1" and bounds.get("dynamic") is True:
            return config
        if (bounds and value.isascii() and value.isdigit()
                and bounds["min"] <= int(value) <= bounds["max"]):
            return config
        if explicit:
            raise ValueError(f"Unsupported thinking token budget for {provider}/{model}.")
        return {"enabled": True}
    supported = reasoning_selection_efforts(provider, model)
    if supported is None:
        return config
    if config is None:
        return {"enabled": True}  # provider default, distinct from inheritance
    effort = "none" if config.get("enabled") is False else str(config.get("effort") or "").strip().lower()
    if not effort or effort in supported:
        return config
    if explicit:
        if not supported:
            raise ValueError(
                f"No verified reasoning effort controls in Hermes for {provider}/{model}; choose auto."
            )
        raise ValueError(
            f"Unsupported reasoning effort {effort!r} for {provider}/{model}; choose auto"
            + (f" or {', '.join(supported)}" if supported else "") + "."
        )
    # Old presets keep the existing nearest-weaker mapping; report the actual
    # level instead of leaving e.g. medium in the UI while sending low.
    levels = tuple(level for level in supported if level != "none")
    resolved = clamp_effort(effort, levels) if effort != "none" and levels else None
    result = {"enabled": True, **({"effort": resolved} if resolved in levels else {})}
    logging.getLogger(__name__).warning(
        "Reasoning setting for %s/%s changed from %s to %s",
        provider, model, effort, result.get("effort", "auto"),
    )
    return result


def sync_primary_reasoning(agent) -> None:
    """Keep accepted edits when the primary runtime is restored after a fallback."""
    primary = getattr(agent, "_primary_runtime", None)
    if primary and (primary.get("provider"), primary.get("model")) == (agent.provider, agent.model):
        config = getattr(agent, "reasoning_config", None)
        primary["reasoning_config"] = dict(config) if config is not None else None
