"""Ollama Cloud provider profile.

Top-level ``reasoning_effort`` on /v1/chat/completions accepts none|low|medium|
high|max (``max`` is undocumented but real — ~2.5x more thinking tokens on
DeepSeek V4); Hermes' ``xhigh`` maps to ``max``.
"""

from typing import Any

from agent.reasoning_effort import OLLAMA_CLOUD_EFFORTS, OLLAMA_CLOUD_OVERRIDES, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud — maps xhigh→max via top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Gated on ``supports_reasoning`` (resolved from the model's /api/show
        ``thinking`` capability) so non-thinking models get no meaningless field."""
        if not supports_reasoning or not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {}

        if reasoning_config and isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            if enabled is False:
                # Ollama Cloud defaults to thinking ON, and ignores the
                # extra_body.thinking:{type:disabled} shape (verified live).
                # The ONLY way to actually suppress thinking on its
                # /v1/chat/completions endpoint is top-level
                # reasoning_effort:"none" — omitting the field leaves
                # thinking on.
                return {}, {"reasoning_effort": "none"}

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort:
                # No explicit effort requested — let the model decide
                # (Ollama Cloud's server default is thinking ON).
                return {}, {}
            if effort == "none":
                return {}, {"reasoning_effort": "none"}  # explicit off switch
            # Accepted set {none, low, medium, high, max} is declared in
            # agent.reasoning_effort ("minimal" is rejected with HTTP 400 →
            # clamps to low; xhigh rounds up to max). Bespoke levels outside
            # the ladder are omitted so the model applies its own default
            # rather than triggering a hard 400.
            from agent.reasoning_effort import (
                OLLAMA_CLOUD_EFFORTS,
                OLLAMA_CLOUD_OVERRIDES,
                clamp_effort,
            )

            clamped = clamp_effort(
                effort, OLLAMA_CLOUD_EFFORTS, OLLAMA_CLOUD_OVERRIDES
            )
            if clamped in OLLAMA_CLOUD_EFFORTS:
                top_level["reasoning_effort"] = clamped

        return {}, top_level


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud", aliases=("ollama_cloud",), default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",), base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
