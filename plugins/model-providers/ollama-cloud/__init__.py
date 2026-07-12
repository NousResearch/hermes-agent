"""Ollama Cloud provider profile.

Ollama Cloud's OpenAI-compatible ``/v1/chat/completions`` endpoint
supports top-level ``reasoning_effort`` with values ``none``, ``low``,
``medium``, ``high``, and ``max`` (the last being undocumented but
empirically confirmed for DeepSeek V4 — ``max`` produces ~2.5× more
thinking tokens than ``high``).

This profile maps Hermes's ``xhigh`` → ``max`` to unlock DeepSeek V4's
"Max thinking" tier through Ollama Cloud.  ``low`` / ``medium`` / ``high``
pass through unchanged.

When reasoning is explicitly disabled (``enabled: false`` or
``effort: "none"``), ``reasoning_effort`` is omitted entirely so the
model runs in non-thinking mode.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud — maps xhigh→max via top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit top-level ``reasoning_effort`` for Ollama Cloud.

        The ``supports_reasoning`` flag passed by the transport is
        deliberately ignored — this profile always handles reasoning
        when ``reasoning_config`` is present.
        """
        top_level: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            if enabled is False:
                return {}, {}  # omit → model runs without thinking

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort:
                # No explicit effort requested — let the model decide
                return {}, {}
            if effort == "none":
                return {}, {}  # explicit none → suppress thinking
            if effort in ("xhigh", "max", "ultra"):
                top_level["reasoning_effort"] = "max"
            elif effort in ("low", "medium", "high"):
                top_level["reasoning_effort"] = effort
            else:
                # Unknown value — forward as-is, let the API decide
                top_level["reasoning_effort"] = effort

        return {}, top_level


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
