"""Meta Model API (Muse Spark) provider profile.

Meta Model API is OpenAI-compatible and serves Muse Spark 1.1 with:

- Base URL: https://api.meta.ai/v1
- Auth: MODEL_API_KEY (alias META_API_KEY, META_MODEL_API_KEY), format LLM|...
- Context window: 1,048,576 tokens (no long-context premium)
- Input modalities: text, image, video, PDF; Output: text
- Pricing: $1.25 input / $0.15 cached input / $4.25 output per 1M tokens
- Features: parallel tool calling, structured output (response_format json_schema),
  prompt caching (automatic prefix, reports cached_tokens), search grounding
  (built-in web_search), file handling via file_id, token counting

Reasoning:
- Muse Spark is a reasoning model that always reasons internally.
- Control depth via top-level ``reasoning_effort`` on Chat Completions:
  minimal, low, medium, high, xhigh. On Responses API it's nested as
  reasoning.effort.
- ``none`` disables reasoning and Muse Spark does NOT support it: HTTP 400.
  Hermes never emits ``none`` — map disabled/``none`` → ``minimal``.
- ``max`` is not a Meta value (it is Hermes/Ollama shorthand). Map → ``xhigh``.
- Reasoning tokens count toward max_tokens / max_output_tokens and are billed.
- When omitted, model reasons at a model-determined level.

Ref: https://dev.meta.ai/docs/features/reasoning/
     https://dev.meta.ai/docs/getting-started/models/
     https://dev.meta.ai/docs/getting-started/pricing-rate-limits/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


@dataclass
class MetaAiProfile(ProviderProfile):
    """Meta Model API — vision + tool calling + structured output + reasoning.

    Implements Meta's reasoning_effort contract:

    - Chat Completions: top-level ``reasoning_effort``
    - Responses API: nested ``reasoning.effort`` (handled via extra_body fallback)
    - Valid values: minimal, low, medium, high, xhigh — NEVER none (400)
    - Hermes extras: ``max`` → xhigh

    When reasoning is disabled (``enabled: false``) or effort is ``none``,
    emit ``minimal`` so request stays valid at lowest thinking tier.
    Empty / missing effort omits field so model uses its default.

    Vision: Muse Spark accepts image, video and PDF in user messages.
    Hermes sets ``supports_vision=True`` so native image routing is used
    (user-role images, tool-result multipart) instead of text fallback.
    Per https://dev.meta.ai/docs/getting-started/models/ and
    https://dev.meta.ai/docs/features/image-understanding/
    """

    # Override dataclass default for vision — must be dataclass field, not
    # plain class attribute, so ProviderProfile dataclass machinery picks it up.
    supports_vision: bool = True

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit Meta-compatible reasoning_effort.

        The ``supports_reasoning`` flag from transport is intentionally ignored
        — this profile always handles reasoning when config is present.

        Returns (extra_body, top_level) where:
        - top_level: for Chat Completions (reasoning_effort)
        - extra_body: for Responses API fallback (reasoning.effort) if needed
          by a custom api_mode override.
        """
        top_level: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            # Meta rejects reasoning_effort=none with HTTP 400.
            # Map disabled → minimal (lowest valid tier).
            top_level["reasoning_effort"] = "minimal"
            # Also cover Responses API mode if user forces it via config.
            extra_body["reasoning"] = {"effort": "minimal"}
            return extra_body, top_level

        effort = (reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            # No explicit effort — let model decide (still reasons internally)
            return {}, {}

        # Normalize Hermes / generic values to Meta's valid set
        if effort == "none":
            # NEVER emit none — Meta returns HTTP 400 for Spark
            mapped = "minimal"
        elif effort in ("max", "xhigh"):
            mapped = "xhigh"
        elif effort in ("minimal", "low", "medium", "high"):
            mapped = effort
        else:
            # Unknown value — forward as-is to allow future values,
            # but guard against the only known invalid: none already handled.
            mapped = effort

        top_level["reasoning_effort"] = mapped
        # Defensive: if caller uses Responses api_mode, also expose nested form.
        # Hermes transport merges both; the non-applicable one is ignored.
        # Only emit extra_body when api_mode hints at Responses to avoid
        # polluting Chat Completions calls with both forms.
        api_mode = (ctx.get("api_mode") or "").strip().lower()
        if api_mode == "codex_responses" or ctx.get("is_responses_api"):
            extra_body["reasoning"] = {"effort": mapped}

        return extra_body, top_level


meta_ai = MetaAiProfile(
    name="meta-ai",
    aliases=("meta", "muse", "llama-api", "model-api"),
    display_name="Meta Model API",
    description="Meta Model API — Muse Spark (1M context, vision, tools, reasoning)",
    signup_url="https://dev.meta.ai/",
    env_vars=(
        "MODEL_API_KEY",
        "META_API_KEY",
        "META_MODEL_API_KEY",
        "META_BASE_URL",
    ),
    base_url="https://api.meta.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    supports_vision=True,
    # Muse Spark spends completion budget on hidden reasoning_tokens first;
    # low caps (e.g. 32–128) finish with empty content. Cap high like NVIDIA NIM.
    # See https://dev.meta.ai/docs/features/reasoning/ :
    # "Reasoning tokens count toward your output-token budget"
    default_max_tokens=16384,
    default_aux_model="muse-spark-1.1",
    fallback_models=("muse-spark-1.1",),
)

register_provider(meta_ai)
