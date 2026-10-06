"""Gemini parameter policy shared by main, auxiliary, and native transports."""
from __future__ import annotations
from collections.abc import Callable
from urllib.parse import urlparse

# GenerateContent/OpenAI-compat model support, verified against Google's thinking guide.
_THREE_LEVELS = ("low", "medium", "high")
_FOUR_LEVELS = ("minimal", *_THREE_LEVELS)
_LEVELS = {
    "gemini-3.1-pro-preview": _THREE_LEVELS,
    "gemini-3-flash-preview": _FOUR_LEVELS,
    "gemini-3.1-flash-lite": _FOUR_LEVELS,
    "gemini-3.5-flash-lite": _FOUR_LEVELS,
    "gemini-3.5-flash": _FOUR_LEVELS,
    "gemini-3.6-flash": _FOUR_LEVELS,
    "gemini-3.7-flash": _THREE_LEVELS,
    "gemini-3.8-flash": _THREE_LEVELS,
}
_SAMPLING = {"temperature", "top_p", "topP", "top_k", "topK"}
_BUDGETS = {"thinking_budget", "thinkingBudget"}
_CONTAINERS = {"extra_body", "google", "generationConfig", "generation_config", "thinkingConfig", "thinking_config"}


def is_gemini_model(model: str | None) -> bool:
    return (model or "").strip().lower().rsplit("/", 1)[-1].startswith("gemini-")


def is_verified_google_route(base_url: str | None) -> bool:
    url = urlparse(base_url or "")
    return url.hostname == "generativelanguage.googleapis.com" and url.path.rstrip("/") in {"/v1beta", "/v1beta/openai"}


def validate_thinking_transport(model: str, body: object, base_url: str | None) -> None:
    """Google-specific fields require the verified Google API surface."""
    if not is_gemini_model(model) or not isinstance(body, dict):
        return
    if {"thinking_config", "thinkingConfig"}.intersection(body) and not is_verified_google_route(base_url):
        raise ValueError("Gemini thinking_config is not verified for this endpoint; omit it for model defaults")
    for key in _CONTAINERS.intersection(body):
        validate_thinking_transport(model, body[key], base_url)


def thinking_level_for_reasoning(model: str, reasoning: dict | None) -> str | None:
    """Unknown/legacy models use defaults; none requests the lowest supported effort."""
    levels = _LEVELS.get((model or "").strip().lower().rsplit("/", 1)[-1])
    if not levels or not isinstance(reasoning, dict):
        return None
    effort = str(reasoning.get("effort") or "medium").lower()
    if reasoning.get("enabled") is False or effort == "none":
        return levels[0]
    if effort in {"ultra", "max", "xhigh"}:
        return "high"
    if effort not in {"minimal", "low", "medium", "high"}:
        raise ValueError(f"Invalid Gemini reasoning effort: {effort}")
    return effort if effort in levels else levels[0]


def validate_parameter_overrides(model: str, body: object) -> None:
    """Reject obsolete settings only in parameter containers, never tool/response schemas."""
    if not is_gemini_model(model) or not isinstance(body, dict):
        return
    forbidden = (_SAMPLING | _BUDGETS).intersection(body)
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and "max_tokens" in reasoning:
        forbidden.add("reasoning.max_tokens")
    if forbidden:
        raise ValueError(f"Gemini uses model-default sampling and thinking levels; remove {', '.join(sorted(forbidden))}")
    validate_parameter_overrides(model, reasoning)
    for key in _CONTAINERS.intersection(body):
        if key in {"thinking_config", "thinkingConfig"}:
            normalize_thinking_config(model, body[key])
        else:
            validate_parameter_overrides(model, body[key])


def normalize_thinking_config(model: str, config: object) -> dict | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("Gemini thinking configuration must be an object")
    unknown = config.keys() - {"thinkingLevel", "thinking_level", "includeThoughts", "include_thoughts"}
    if unknown:
        raise ValueError(f"Unsupported Gemini thinking configuration: {', '.join(sorted(unknown))}")
    validate_parameter_overrides(model, config)
    level = config.get("thinkingLevel", config.get("thinking_level"))
    include = config.get("includeThoughts", config.get("include_thoughts"))
    normalized = {}
    if level is not None:
        levels = _LEVELS.get((model or "").strip().lower().rsplit("/", 1)[-1], ())
        if not isinstance(level, str) or level.strip().lower() not in levels:
            raise ValueError(f"Gemini thinking level {level!r} is not verified for {model}; omit it for model defaults")
        normalized["thinkingLevel"] = level.strip().lower()
    if include is not None:
        if not isinstance(include, bool):
            raise ValueError("Gemini includeThoughts must be a boolean")
        normalized["includeThoughts"] = include
    return normalized or None


def validate_auxiliary_config(model: str, task: str | None, temperature: float | None, read_config: Callable) -> None:
    """Explicit task/preset settings must be migrated instead of silently ignored."""
    if not is_gemini_model(model):
        return
    validate_parameter_overrides(model, read_config(task) if task else {})
    if task in {"moa_reference", "moa_aggregator"} and temperature is not None:
        raise ValueError("Remove the MoA preset temperature for Gemini; use model-default sampling")


def validate_native_sampling(model: str, temperature: float | None, top_p: float | None) -> None:
    # The native API also serves Gemma; retain its pre-existing sampling controls.
    settings = {key: value for key, value in (("temperature", temperature), ("top_p", top_p)) if value is not None}
    validate_parameter_overrides(model, settings)


def finalize_kwargs(kwargs: dict, provider: str | None = None, base_url: str | None = None) -> dict:
    """Generic caller temperatures remain available to other models; Gemini uses defaults."""
    model = kwargs.get("model", "")
    if not is_gemini_model(model):
        return kwargs
    if not base_url and provider in {"gemini", "google", "google-gemini", "google-ai-studio"}:
        base_url = "https://generativelanguage.googleapis.com/v1beta"
    validate_parameter_overrides(model, kwargs.get("extra_body"))
    validate_thinking_transport(model, kwargs.get("extra_body"), base_url)
    result = {key: value for key, value in kwargs.items() if key not in _SAMPLING}
    effort = result.pop("reasoning_effort", None)
    if effort is not None and is_verified_google_route(base_url):
        level = thinking_level_for_reasoning(model, {"effort": effort})
        if level is not None:
            result["reasoning_effort"] = level
    extra = result.get("extra_body")
    if isinstance(extra, dict) and isinstance(extra.get("reasoning"), dict):
        # OpenRouter documents this mapping. Other routers use defaults until verified.
        level = thinking_level_for_reasoning(model, extra["reasoning"]) if provider == "openrouter" else None
        extra = dict(extra)
        if level is None:
            extra.pop("reasoning")
        else:
            hide = extra["reasoning"].get("enabled") is False or extra["reasoning"].get("exclude") is True
            extra["reasoning"] = {"effort": level, **({"exclude": True} if hide else {})}
        result["extra_body"] = extra
    return result
