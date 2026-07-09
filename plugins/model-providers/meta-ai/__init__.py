"""Meta Model API (Muse Spark) provider profile.

Meta's OpenAI-compatible ``/v1/chat/completions`` endpoint supports
top-level ``reasoning_effort`` with values ``minimal``, ``low``,
``medium``, ``high``, and ``xhigh``.

This profile maps Hermes's ``max`` → ``xhigh``.  When reasoning is
explicitly disabled (``enabled: false``) or effort is ``none``, the
profile emits ``reasoning_effort=minimal`` — Meta returns HTTP 400 for
``none``, so that value must never appear on the wire.  Empty / missing
effort omits the field so the model uses its default.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class MetaAiProfile(ProviderProfile):
    """Meta Model API — top-level reasoning_effort (never emit none)."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit top-level ``reasoning_effort`` for Meta Model API.

        The ``supports_reasoning`` flag passed by the transport is
        deliberately ignored — this profile always handles reasoning
        when ``reasoning_config`` is present.
        """
        top_level: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            if enabled is False:
                # Meta rejects reasoning_effort=none with HTTP 400.
                # Map disabled → minimal (lowest valid thinking tier).
                top_level["reasoning_effort"] = "minimal"
                return {}, top_level

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort:
                # No explicit effort — let the model decide
                return {}, {}
            if effort == "none":
                # NEVER emit none — Meta returns HTTP 400
                top_level["reasoning_effort"] = "minimal"
            elif effort in ("max", "xhigh"):
                top_level["reasoning_effort"] = "xhigh"
            elif effort in ("minimal", "low", "medium", "high"):
                top_level["reasoning_effort"] = effort
            else:
                # Unknown value — forward as-is, let the API decide
                top_level["reasoning_effort"] = effort

        return {}, top_level


meta_ai = MetaAiProfile(
    name="meta-ai",
    aliases=("meta", "muse", "llama-api", "model-api"),
    display_name="Meta Model API",
    description="Meta Model API — Muse Spark (OpenAI-compatible)",
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
    # Muse Spark spends completion budget on hidden reasoning_tokens first;
    # low caps (e.g. 32–128) finish with empty content. Cap high like NVIDIA NIM.
    default_max_tokens=16384,
    default_aux_model="muse-spark-1.1",
    fallback_models=("muse-spark-1.1",),
)

register_provider(meta_ai)
