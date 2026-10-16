"""SDK-v1 selectors projected from runtime or owner info; no independent model registry."""
from __future__ import annotations

import re

from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

from agent.fast_mode import parse_exact_service_tier, service_tier_word
from agent.reasoning_effort import effort_display_label, route_supported_efforts


_T3_RUNTIME_SUFFIX = re.compile(
    r"(?:^|\n)[ \t]*<runtime_info>[^<]*</runtime_info>"
    r"(?:\s*<pull_request_linking>[^<]*</pull_request_linking>)?\s*\Z"
)


def runtime_info(state):
    agent = state.agent
    info = {key: getattr(agent, key, None) for key in ("provider", "base_url", "api_mode")}
    config = getattr(agent, "reasoning_config", None)
    effort = ""
    if isinstance(config, dict):
        effort = "none" if config.get("enabled") is False else str(config.get("effort") or "")
    tier = getattr(agent, "service_tier", None)
    info["access_token"] = getattr(agent, "api_key", None)
    info.update(model=state.model, reasoning_effort=effort, service_tier=tier if isinstance(tier, str) else None)
    return info


def supported_reasoning(info):
    return route_supported_efforts(info.get("provider"), info.get("model"), info.get("api_mode"))


def fast_overrides(info, word):
    from hermes_cli.models import resolve_fast_mode_overrides
    tier = parse_exact_service_tier(word)
    if not tier:
        return {}
    if not info.get("provider") and not info.get("base_url"):
        return None
    if info.get("api_mode") == "codex_app_server":
        from agent.fast_mode import CODEX_TIER_WORDS, STATIC_TIERS
        if tier in STATIC_TIERS and tier not in CODEX_TIER_WORDS:
            return None
    if tier == "ultrafast":
        from agent.codex_service_tiers import ultrafast_offered
        if not ultrafast_offered(info.get("model"), provider=info.get("provider"),
                                base_url=info.get("base_url"), access_token=info.get("access_token")):
            return None
    return resolve_fast_mode_overrides(info.get("model"), provider=info.get("provider"),
                                       base_url=info.get("base_url"), tier=tier)


def parse_selection(config_id, value, info):
    """Canonical value or an actionable rejection, shared by native and text controls."""
    if config_id not in {"model", "reasoning", "fast"}:
        raise ValueError(f"Unsupported config option: {config_id}")
    if not isinstance(value, str):
        raise ValueError(f"{config_id} requires a string value")
    if "--global" in value.split():
        raise ValueError("ACP controls are session-scoped; change global defaults through Hermes config.")
    if config_id == "model":
        if not value.strip():
            raise ValueError("Model must not be empty")
        return value.strip()
    value = value.strip().lower()
    if config_id == "reasoning":
        if value not in supported_reasoning(info):
            raise ValueError(f"Unsupported reasoning effort for this route: {value}")
        return value
    word = service_tier_word(parse_exact_service_tier(value))
    if fast_overrides(info, word) is None:
        raise ValueError(f"Fast mode {word} is not available for the current model and route")
    return word


def _option(identifier, category, name, current, rows, description=None):
    choices = [SessionConfigSelectOption(value=value, name=label) for value, label in rows]
    if current not in {choice.value for choice in choices}:
        choices.append(SessionConfigSelectOption(value=current, name=f"{current or 'Provider default'} (current)"))
    return SessionConfigOptionSelect(id=identifier, category=category, name=name, type="select",
                                    current_value=current, options=choices, description=description)


def project_options(info, models):
    model_rows = [(row.model_id, row.name) for row in models.available_models] if models else []
    model_current = models.current_model_id if models else str(info.get("model") or "")
    effort = str(info.get("reasoning_effort") or "")
    reasoning_rows = [(level, effort_display_label(level, info.get("provider"), info.get("model"),
                                                  info.get("api_mode"))) for level in supported_reasoning(info)]
    if not effort:
        reasoning_rows.insert(0, ("", "Provider default"))
    elif effort not in {level for level, _ in reasoning_rows}:
        reasoning_rows.append((effort, effort_display_label(effort, info.get("provider"), info.get("model"),
                                                           info.get("api_mode"))))
    tier = service_tier_word(info.get("service_tier"))
    rows = [("normal", "Normal")]
    for word in ("fast", "ultrafast", "auto", "cold"):
        if fast_overrides(info, word) is not None:
            rows.append((word, f"{word.capitalize()} (paid)"))
    return [
        _option("model", "model", "Model", model_current, model_rows),
        _option("reasoning", "thought_level", "Reasoning", effort, reasoning_rows),
        _option("fast", "model_config", "Fast mode", tier, rows,
                "Session scope. Fast tiers may cost more; selecting a model never enables one."),
    ]


def text_selection(arg):
    # T3 appends this text block even to slash commands. Only controls consume it;
    # prompt content (including multiline /steer and /queue) stays intact.
    arg = _T3_RUNTIME_SUFFIX.sub("", arg)
    parts = arg.split()
    if "--global" in parts:
        raise ValueError("ACP controls are session-scoped; change global defaults through Hermes config.")
    return " ".join(part for part in parts if part != "--session")
