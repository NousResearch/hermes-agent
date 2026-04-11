"""Model capability pre-flight validation — advisory warnings only.

Pure pattern-matching lookups with NO API calls and NO network access.
Used to emit warnings when the user's request may be incompatible with
the selected model's known capabilities (e.g. sending tools to a model
that doesn't support function calling, or images to a non-vision model).

These are informational warnings only — they NEVER block, modify, or
strip anything from the request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelCapabilities:
    """Known capabilities for a model, resolved by pattern matching."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    supports_streaming: bool = True
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    source: str = "default"  # 'catalog', 'models_dev', 'runtime', 'default'


# ---------------------------------------------------------------------------
# Known capability patterns — ordered most-specific-first.
#
# Each entry is (regex_pattern, ModelCapabilities).
# The first match wins, so put specific models before broad family patterns.
# ---------------------------------------------------------------------------

KNOWN_CAPABILITIES: List[tuple[str, ModelCapabilities]] = [
    # -----------------------------------------------------------------------
    # Anthropic Claude
    # -----------------------------------------------------------------------
    # Claude 3.5 Sonnet/Haiku: tools + vision + reasoning (match before claude-3)
    (r"claude.*3[\.\-]5", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # Claude 3 Opus/Sonnet/Haiku (not 3.5): tools + vision, no reasoning
    (r"claude.*3(?![\.\-]5)", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Claude 2.x: tools, no vision, no reasoning
    (r"claude[\-\.\s]2(?:\.\d)?(?!\d)", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Claude 4.x+ / opus / sonnet / haiku (modern): full capabilities
    # This catches claude-sonnet-4, claude-opus-4-6, claude-haiku-4-5, etc.
    (r"claude", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------
    # o1, o3, o4-mini: reasoning models
    (r"(^|/)o[134]\b", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # GPT-5.x: full capabilities
    (r"gpt[\-\.]?5", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # GPT-4.1: tools + vision, reasoning via API
    (r"gpt[\-\.]?4[\.\-]1", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # GPT-4o / GPT-4o-mini: tools + vision
    (r"gpt[\-\.]?4o", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # GPT-4 (non-vision, non-turbo): tools only
    (r"gpt[\-\.]?4(?!o)", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # GPT-3.5: tools, no vision, no reasoning
    (r"gpt[\-\.]?3[\.\-]5", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Google Gemini
    # -----------------------------------------------------------------------
    # Gemini 2.5+: full capabilities including reasoning
    (r"gemini.*[23]\.[5-9]", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # Gemini 3.x: full capabilities
    (r"gemini.*3", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # Gemini 2.0: tools + vision
    (r"gemini.*2", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Gemini 1.5: tools + vision
    (r"gemini.*1[\.\-]5", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Gemini generic
    (r"gemini", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # DeepSeek
    # -----------------------------------------------------------------------
    # DeepSeek Reasoner: tools + reasoning, no vision
    (r"deepseek.*reason", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # DeepSeek V3+: tools, no vision, no reasoning
    (r"deepseek.*(v3|chat)", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # DeepSeek generic
    (r"deepseek", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Meta Llama
    # -----------------------------------------------------------------------
    # Llama 3.1+ / 3.2+ / 4+: tool calling support
    (r"llama.*([3-9]\.[1-9]|[4-9]\.)", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Llama 3.0: limited tool support
    (r"llama.*3", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Llama generic (older)
    (r"llama", ModelCapabilities(
        supports_tools=False, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Qwen
    # -----------------------------------------------------------------------
    # QwQ: reasoning model
    (r"qwq", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    # Qwen-VL: vision model
    (r"qwen.*vl", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Qwen 2.5+ / 3+: tools, no vision (unless VL variant)
    (r"qwen.*([23]\.[5-9]|[3-9]\.)", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Qwen generic
    (r"qwen", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Mistral
    # -----------------------------------------------------------------------
    # Pixtral: vision model
    (r"pixtral", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Mistral Large / Medium / Small: tools, no vision
    (r"mistral", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
    # Codestral
    (r"codestral", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # X.AI Grok
    # -----------------------------------------------------------------------
    (r"grok", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # MiniMax
    # -----------------------------------------------------------------------
    (r"minimax", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # GLM (Z.AI)
    # -----------------------------------------------------------------------
    (r"glm", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Kimi / Moonshot
    # -----------------------------------------------------------------------
    (r"kimi.*thinking", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    (r"kimi", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Nvidia Nemotron
    # -----------------------------------------------------------------------
    (r"nemotron", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Xiaomi MiMo
    # -----------------------------------------------------------------------
    (r"mimo", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # Arcee Trinity
    # -----------------------------------------------------------------------
    (r"trinity.*thinking", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=True,
        supports_streaming=True, source="catalog",
    )),
    (r"trinity", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),

    # -----------------------------------------------------------------------
    # StepFun
    # -----------------------------------------------------------------------
    (r"step[\-\.]", ModelCapabilities(
        supports_tools=True, supports_vision=False, supports_reasoning=False,
        supports_streaming=True, source="catalog",
    )),
]

# Compiled patterns for faster matching
_COMPILED_PATTERNS: List[tuple[re.Pattern, ModelCapabilities]] = [
    (re.compile(pattern, re.IGNORECASE), caps)
    for pattern, caps in KNOWN_CAPABILITIES
]


def _normalize_model_name(model: str) -> str:
    """Strip provider prefix (e.g. 'anthropic/claude-sonnet-4') for matching."""
    if "/" in model:
        # Keep the part after the last provider-like prefix
        parts = model.split("/")
        # If it looks like provider/model, use just the model part
        if len(parts) == 2:
            return parts[1]
        # For deeper paths, return from the second segment
        return "/".join(parts[1:])
    return model


def get_model_capabilities(model: str, provider: str = None) -> ModelCapabilities:
    """Look up known capabilities for a model by pattern matching.

    Pure lookup — no API calls, no network access.

    Args:
        model: Model identifier (e.g. 'claude-sonnet-4', 'anthropic/claude-sonnet-4.6')
        provider: Optional provider hint (unused currently, reserved for future use)

    Returns:
        ModelCapabilities with known values, or safe defaults for unknown models.
    """
    if not model:
        return ModelCapabilities(source="default")

    # Try matching against both the full model string and the normalized name
    candidates = [model]
    normalized = _normalize_model_name(model)
    if normalized != model:
        candidates.append(normalized)

    for candidate in candidates:
        for compiled_re, caps in _COMPILED_PATTERNS:
            if compiled_re.search(candidate):
                return ModelCapabilities(
                    supports_tools=caps.supports_tools,
                    supports_vision=caps.supports_vision,
                    supports_reasoning=caps.supports_reasoning,
                    supports_streaming=caps.supports_streaming,
                    max_output_tokens=caps.max_output_tokens,
                    context_window=caps.context_window,
                    source=caps.source,
                )

    # Safe defaults for unknown models — assume tools work, be conservative
    # about vision and reasoning
    return ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_reasoning=False,
        supports_streaming=True,
        source="default",
    )


def _messages_contain_images(messages: List[Dict[str, Any]]) -> bool:
    """Check if any message contains image content (image_url or base64 images)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        return True
                    # Also check for base64 image data in text
                    if part.get("type") == "image":
                        return True
    return False


def validate_preflight(
    model: str,
    provider: str,
    tools: Optional[list],
    messages: Optional[List[Dict[str, Any]]],
    reasoning_config: Optional[Dict[str, Any]],
) -> List[str]:
    """Validate model capabilities against the request — advisory warnings only.

    Returns a list of warning strings. Empty list means no issues detected.
    These are informational only — NEVER block, modify, or strip the request.

    Args:
        model: Model identifier
        provider: Provider identifier
        tools: List of tool schemas (may be None or empty)
        messages: Conversation messages (may be None or empty)
        reasoning_config: Reasoning configuration dict (may be None)

    Returns:
        List of warning strings (empty = all good)
    """
    warnings: List[str] = []
    caps = get_model_capabilities(model, provider)

    # Only warn for catalog-matched models where we have confidence.
    # For unknown models (source="default"), we don't know enough to warn.
    if caps.source == "default":
        return warnings

    # Check: tools sent to a model that doesn't support them
    if tools and not caps.supports_tools:
        warnings.append(
            f"Model '{model}' may not support tool/function calling. "
            f"{len(tools)} tool(s) were provided but may be ignored by the API."
        )

    # Check: images sent to a non-vision model
    if messages and not caps.supports_vision:
        if _messages_contain_images(messages):
            warnings.append(
                f"Model '{model}' may not support vision/image inputs. "
                f"Image content was detected in messages but may not be processed."
            )

    # Check: reasoning config for a non-reasoning model
    if reasoning_config and not caps.supports_reasoning:
        effort = reasoning_config.get("effort")
        if effort is not None and effort != "none":
            warnings.append(
                f"Model '{model}' may not support reasoning/thinking. "
                f"Reasoning effort '{effort}' was configured but may be ignored."
            )

    return warnings


# ---------------------------------------------------------------------------
# Model deprecation check via models.dev
# ---------------------------------------------------------------------------


def check_model_deprecation(model: str, provider: str = None) -> Optional[str]:
    """Check if a model is deprecated via models.dev.

    Returns a warning string if deprecated, None otherwise.
    This is a best-effort check — returns None if models.dev
    data is unavailable or model not found.
    """
    try:
        from agent.models_dev import get_model_info, get_model_info_any_provider

        info = None
        if provider:
            # Strip provider prefix from model if present
            bare_model = model.split('/')[-1] if '/' in model else model
            info = get_model_info(provider, bare_model)
        if not info:
            info = get_model_info_any_provider(model)

        if info and info.status == 'deprecated':
            return (f"⚠️ Model '{model}' is marked as deprecated. "
                    f"Consider switching to a newer model to avoid "
                    f"unexpected service disruptions.")
    except Exception:
        pass  # models.dev unavailable
    return None


# ---------------------------------------------------------------------------
# API key format validation
# ---------------------------------------------------------------------------

# Known API key prefix patterns per provider
_KEY_PREFIX_PATTERNS = {
    'openai': ('sk-',),
    'anthropic': ('sk-ant-',),
    'openrouter': ('sk-or-',),
    'deepseek': ('sk-',),
    'groq': ('gsk_',),
    'mistral': (None,),  # No known prefix pattern
}


def validate_api_key_format(provider: str, api_key: str) -> Optional[str]:
    """Check if API key matches expected format for the provider.

    Returns a warning string if the key looks wrong, None if OK or unknown.
    Advisory only — never blocks. Unknown providers return None.
    """
    if not api_key or not provider:
        return None

    provider_lower = provider.lower()
    prefixes = _KEY_PREFIX_PATTERNS.get(provider_lower)
    if not prefixes or prefixes == (None,):
        return None

    if not any(api_key.startswith(p) for p in prefixes if p):
        expected = ' or '.join(f"'{p}...'" for p in prefixes if p)
        return (f"⚠️ API key for {provider} doesn't match expected format "
                f"(expected {expected}). Verify the correct key is configured.")
    return None
