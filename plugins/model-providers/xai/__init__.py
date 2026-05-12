"""xAI (Grok) provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class XAIProviderProfile(ProviderProfile):
    """xAI/Grok request-time quirks."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Force highest Grok reasoning on OpenAI chat-completions wire.

        The Responses transport handles xAI separately via ``reasoning.effort``.
        This hook protects the alternate chat-completions path, where xAI accepts
        top-level ``reasoning_effort`` for the verified effort-capable Grok models.
        """
        model = str(context.get("model") or "").strip().lower()
        if not model:
            return {}, {}

        try:
            from agent.model_metadata import grok_supports_reasoning_effort
        except Exception:
            return {}, {}

        if grok_supports_reasoning_effort(model):
            return {}, {"reasoning_effort": "high"}
        return {}, {}


xai = XAIProviderProfile(
    name="xai",
    aliases=("grok", "x-ai", "x.ai"),
    api_mode="codex_responses",
    env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1",
    auth_type="api_key",
)

register_provider(xai)
