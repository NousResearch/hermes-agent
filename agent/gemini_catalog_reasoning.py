"""AI Studio generateContent controls from cached models.dev metadata.

Model names carry no policy. Vertex retains its separate transport policy.
Missing metadata leaves the API in charge; discovery never makes paid probes.
"""
from agent.models_dev import get_model_info


def describe_thinking_control(model: str) -> dict:
    name = model.strip().removeprefix("google/").removeprefix("models/")
    info = get_model_info("gemini", name, allow_network=False)
    result = {"reasoning_control": "unknown", "reasoning_efforts": [],
              "can_disable_reasoning": False, "fast": False}
    if info is None:
        return result
    result["reasoning"] = info.reasoning
    if not name.startswith("gemini-"):
        return result  # e.g. Gemma's toggle is not Gemini's thinkingConfig protocol
    if not info.reasoning:
        result["reasoning_control"] = "unsupported"
        return result
    if info.reasoning_options is None:
        return result
    efforts = []
    for option in info.reasoning_options:
        if option["type"] == "effort":
            efforts.extend(v for v in option["values"] if v not in (None, "default"))
        elif option["type"] == "toggle":
            efforts.append("none")
        elif option["type"] == "budget_tokens":
            # These bounds are for the Thinking-on numeric input, not a copy of
            # the raw API range. Zero is Off (requires a declared toggle), and
            # -1 is Dynamic; neither may bypass those separate controls here.
            # Preserve positive minima and never invent a missing bound.
            if "min" in option and "max" in option and option["max"] > 0:
                result["reasoning_budget"] = {"min": max(1, option["min"]), "max": option["max"], "dynamic": True}
                # -1 is generateContent's dynamic sentinel, not a model effort.
    result["reasoning_efforts"] = list(dict.fromkeys(efforts))
    result["can_disable_reasoning"] = "none" in efforts
    result["reasoning_control"] = "adjustable" if efforts or "reasoning_budget" in result else "default"
    return result


def supported_efforts(model: str | None) -> tuple[str, ...]:
    return tuple(describe_thinking_control(model or "")["reasoning_efforts"])


def build_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    from providers.reasoning import resolve_provider_reasoning_config
    if not isinstance(reasoning_config, dict):
        return None
    descriptor = describe_thinking_control(model)
    if descriptor["reasoning_control"] in ("unknown", "unsupported"):
        return None
    # The resolver shares the picker's declared Off capability. Inherited Off
    # on a mandatory-thinking model becomes an unset override before this branch;
    # explicit selections are rejected by the same resolver at the gateway.
    config = resolve_provider_reasoning_config("gemini", model, reasoning_config)
    if not config:
        return None
    if config.get("enabled") is False:
        return {"includeThoughts": False, "thinkingBudget": 0}
    effort = config.get("effort", "")
    if effort.startswith("budget:"):
        return {"includeThoughts": True, "thinkingBudget": int(effort[7:])}
    if effort:
        return {"includeThoughts": True, "thinkingLevel": effort}
    # Provider default means no level/budget override (in particular for Lite).
    return {"includeThoughts": True}
