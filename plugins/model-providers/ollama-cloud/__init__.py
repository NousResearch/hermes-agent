"""Ollama Cloud provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud provider with DeepSeek thinking-mode support.

    DeepSeek models accept a native ``{"thinking": {"type": "enabled"}}``
    extra_body parameter that surfaces chain-of-thought reasoning content.
    Without it the model still reasons internally but Hermes can't display
    or replay the reasoning track.
    """

    # Models known to support reasoning_effort=max (verified 2026-05-22).
    # ollama.com proxy rejects xhigh → 400; map to max for these.
    _MAX_EFFORT_MODELS = frozenset({
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "deepseek-v3.1:671b",
    })

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        model = str(context.get("model", "") or "")
        if "deepseek" in model.lower():
            extra["thinking"] = {"type": "enabled"}
        if session_id:
            extra["session_id"] = session_id
        return extra

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit reasoning_effort, translating xhigh→max for known models."""
        if not reasoning_config or not reasoning_config.get("enabled"):
            return {}, {}
        effort = reasoning_config.get("effort", "")
        if not effort:
            return {}, {}
        # ollama.com accepts: low, medium, high, max
        # Map xhigh → max for models that support it.
        if effort == "xhigh" and (model or "") in self._MAX_EFFORT_MODELS:
            effort = "max"
        if effort not in ("low", "medium", "high", "max"):
            return {}, {}
        return {}, {"reasoning_effort": effort}


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
