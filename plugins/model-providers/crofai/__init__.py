"""CrofAI provider profile.

CrofAI (https://crof.ai) is a multi-model inference gateway offering DeepSeek,
Kimi, GLM, Qwen, MiMo, and other models through a single OpenAI-compatible API.

Key quirks:
  - Models with ``custom_reasoning: true`` (deepseek-v4-*, mimo-*, kimi-k2.*,
    glm-5.1, etc.) support DeepSeek-style ``extra_body.thinking``.
  - Models set ``context_length`` to either 1M or 163K depending on backend.
  - Pricing is per-model with cache_prompt, prompt, and completion rates.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Model families that support CrofAI's custom reasoning (extra_body.thinking).
_CUSTOM_REASONING_MODELS = frozenset({
    "deepseek-v4-pro",
    "deepseek-v4-pro-precision",
    "deepseek-v4-flash",
    "mimo-v2.5-pro",
    "mimo-v2.5-pro-precision",
    "kimi-k2.6",
    "kimi-k2.6-precision",
    "kimi-k2.5",
    "kimi-k2.5-lightning",
    "glm-5.1",
    "glm-5.1-precision",
    "glm-5",
    "glm-4.7",
    "glm-4.7-flash",
    "greg",
    "minimax-m2.5",
})


def _model_supports_thinking(model: str | None) -> bool:
    """Check whether *model* is in the CrofAI custom-reasoning set."""
    m = (model or "").strip().lower()
    if not m:
        return False
    # Exact match first, then prefix match for model families
    if m in _CUSTOM_REASONING_MODELS:
        return True
    # Match by family prefix
    for known in _CUSTOM_REASONING_MODELS:
        if m.startswith(known):
            return True
    return False


class CrofAIProfile(ProviderProfile):
    """CrofAI — multi-model gateway with custom reasoning support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Mirror the DeepSeek wire shape for custom-reasoning models.

        For models with ``custom_reasoning=true``, pass ``extra_body.thinking``
        so the API returns reasoning content correctly and avoids the
        ``reasoning_content must be passed back`` trap on subsequent turns.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model):
            return extra_body, top_level

        # Determine enabled/disabled
        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        if not enabled:
            return extra_body, top_level

        # Effort mapping — pass low/medium/high through; xhigh/max → max
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max"}:
                top_level["reasoning_effort"] = "max"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort
            # omit → let server apply default (typically high)

        return extra_body, top_level


crofai = CrofAIProfile(
    name="crofai",
    aliases=(
        "crof",
        "crof.ai",
        "crof-ai",
    ),
    display_name="CrofAI",
    description="CrofAI — multi-model inference gateway",
    signup_url="https://crof.ai",
    env_vars=("CROFAI_API_KEY",),
    base_url="https://crof.ai/v1",
    fallback_models=(
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "kimi-k2.6",
        "kimi-k2.6-precision",
        "mimo-v2.5-pro",
        "glm-5.1",
        "glm-5",
        "gemma-4-31b-it",
        "qwen3.6-27b",
    ),
)

register_provider(crofai)
