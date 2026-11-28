"""Vendor-agnostic Gemini model-family rules (versioning, thinking config, sampling, output headroom).

Shared across core (OpenRouter, Nous Portal, Vertex AI, and custom endpoints serving Gemini models
with no Google provider plugin loaded) and the Google model-provider plugin. Keep import-light
(stdlib + ``agent.reasoning_effort`` only).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.reasoning_effort import (
    GEMINI4_THINKING_LEVELS,
    bare_gemini_model_id,
    clamp_effort,
)

# Published max output-token ceiling shared by every current Gemini text model; used
# for max_tokens=None because the native API's low internal default truncates output.
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 65535

_HIGH_EFFORTS = {"high", "xhigh", "max", "ultra"}

# (camelCase key, snake_case alias, accepted types, normalizer)
_THINKING_KEYS = (
    ("thinkingBudget", "thinking_budget", (int, float), int),
    ("includeThoughts", "include_thoughts", bool, lambda v: v),
    ("thinkingLevel", "thinking_level", str, lambda v: v.strip().lower()),
)


def gemini_version(model: str) -> tuple[int, int]:
    """(major, minor) of a Gemini model id: ``gemini-4-argon`` -> (4, 0), ``google/gemini-3.8-flash`` ->
    (3, 8); (0, 0) for non-Gemini and unversioned ids (``gemini-flash-latest``, ``gemini-exp-1206``)."""
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?", bare_gemini_model_id(model).lower())
    return (int(match.group(1)), int(match.group(2) or 0)) if match else (0, 0)


def gemini_requires_tool_call_ids(model: str) -> bool:
    """Gemini 3+ needs explicit functionCall/functionResponse ids so replayed parallel tool calls
    pair with their responses; 2.x rejects the field."""
    return gemini_version(model) >= (3, 0)


def gemini_uses_thinking_level(model: str) -> bool:
    """Gemini 3 and every later generation steer thinking with the ``thinkingLevel`` enum (``thinkingBudget``
    is deprecated there), as do the ``gemini-*-latest`` aliases, which Google hot-swaps to the newest
    release of each variant."""
    bare = bare_gemini_model_id(model).lower()
    return (bare.startswith("gemini-") and bare.endswith("-latest")) or gemini_version(bare) >= (3, 0)


def gemini_supports_multimodal_tool_responses(model: str) -> bool:
    """True for Gemini 3+/4.x and ``gemini-*-latest`` aliases that accept multimodal tool-result parts."""
    return gemini_requires_tool_call_ids(model) or gemini_uses_thinking_level(model)


def gemini_sampling_deprecated(model: str) -> bool:
    """Gemini 3.6+ (Gemini 4 Argon included) ignores ``temperature``/``top_p``/``top_k`` and Google
    slated them for removal; older families (2.x, 3.0-3.5) keep them."""
    return gemini_version(model) >= (3, 6)


def normalize_gemini_thinking_config(config: Any) -> Optional[dict[str, Any]]:
    """Normalize a camelCase or snake_case Gemini thinking config dict to native camelCase."""
    if not isinstance(config, dict):
        return None
    values = {key: config.get(key, config.get(alias)) for key, alias, _, _ in _THINKING_KEYS}
    normalized = {
        key: norm(values[key])
        for key, _, types, norm in _THINKING_KEYS
        if isinstance(values[key], types) and (values[key].strip() if isinstance(values[key], str) else True)
    }
    return normalized or None


def thinking_requests_output_headroom(thinking_config: Any) -> bool:
    """True when Gemini will spend output tokens on thinking: thought tokens bill against ``maxOutputTokens``,
    so a global 4096/16384 cap can be consumed entirely by high thinking (``finishReason=MAX_TOKENS``, no answer)."""
    normalized = normalize_gemini_thinking_config(thinking_config) or {}
    if normalized.get("includeThoughts") is False:
        # Hidden thoughts = reasoning disabled/minimised: never ask for headroom, or a small
        # caller budget (title generation's 64 tokens) balloons to the model ceiling. Gemini 3
        # still carries a "low" thinkingLevel here (no off-switch), which must not re-enable the
        # bump. (#91927)
        return False
    budget = normalized.get("thinkingBudget")
    has_level = "thinkingLevel" in normalized
    return bool(normalized) and not (isinstance(budget, int) and budget <= 0 and not has_level)


def effective_gemini_max_output_tokens(max_tokens: Optional[int], thinking_config: Any) -> int:
    """Native ``maxOutputTokens``: an omitted/invalid cap becomes the published ceiling (Gemini
    truncates on its low internal default); an explicit cap is raised to the ceiling when
    thinking is enabled so thoughts don't starve the answer."""
    try:
        requested = int(max_tokens)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0 or thinking_requests_output_headroom(thinking_config):
        return max(requested, GEMINI_DEFAULT_MAX_OUTPUT_TOKENS)
    return requested


def build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Translate Hermes/OpenRouter-style reasoning config to Gemini thinkingConfig."""
    if not isinstance(reasoning_config, dict):
        return None
    normalized_model = (model or "").strip().lower().removeprefix("google/")
    # Gemini-only; Gemma/PaLM on the same provider 400 on the field even as ``{"includeThoughts": False}``. (#17426)
    if not normalized_model.startswith("gemini"):
        return None
    effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if normalized_model.startswith("gemini-4"):
        # Gemini 4 always thinks: a disable request gets the floor level with thoughts hidden.
        disabled = reasoning_config.get("enabled") is False or effort == "none"
        level = "low" if disabled else clamp_effort(effort, GEMINI4_THINKING_LEVELS)
        return {"includeThoughts": not disabled, "thinkingLevel": level if level in GEMINI4_THINKING_LEVELS else "medium"}
    if reasoning_config.get("enabled") is False or effort == "none":
        config: dict[str, Any] = {"includeThoughts": False}
        if normalized_model.startswith("gemini-2.5-"):
            config["thinkingBudget"] = 0
        elif gemini_uses_thinking_level(normalized_model):
            config["thinkingLevel"] = "low"
        return config
    thinking_config: dict[str, Any] = {"includeThoughts": True}
    # Gemini 2.5 takes thinkingBudget; don't guess one from coarse effort levels.
    if normalized_model.startswith("gemini-2.5-"):
        return thinking_config
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
        effort = "medium"
    if gemini_uses_thinking_level(normalized_model):
        if "pro" in normalized_model:
            thinking_config["thinkingLevel"] = "high" if effort in _HIGH_EFFORTS else "low"
        else:
            thinking_config["thinkingLevel"] = (
                "low" if effort in {"minimal", "low"} else "high" if effort in _HIGH_EFFORTS else "medium"
            )
    return thinking_config


def snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Convert Gemini thinking config keys to the OpenAI-compat field names."""
    if not isinstance(config, dict) or not config:
        return None
    translated: dict[str, Any] = {}
    include, level, budget = config.get("includeThoughts"), config.get("thinkingLevel"), config.get("thinkingBudget")
    if isinstance(include, bool):
        translated["include_thoughts"] = include
    if isinstance(level, str) and level.strip():
        translated["thinking_level"] = level.strip().lower()
    if isinstance(budget, (int, float)):
        translated["thinking_budget"] = int(budget)
    return translated or None
