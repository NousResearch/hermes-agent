"""DeepSeek provider profile.

DeepSeek native Chat Completions API (api.deepseek.com/v1).
Thinking mode is enabled via ``extra_body={\"thinking\": {\"type\": \"enabled\"}}``.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class DeepSeekProfile(ProviderProfile):
    """DeepSeek native API — thinking mode via extra_body.thinking."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate Hermes reasoning_config to DeepSeek thinking parameter.

        DeepSeek API requires ``extra_body={\"thinking\": {\"type\": \"enabled\"}}``
        to return reasoning_content in chat completions responses.
        """
        extra_body: dict[str, Any] = {}
        if reasoning_config is not None and reasoning_config.get("enabled", True):
            extra_body["thinking"] = {"type": "enabled"}
        return extra_body, {}


deepseek = DeepSeekProfile(
    name="deepseek",
    aliases=("deepseek-chat",),
    env_vars=("DEEPSEEK_API_KEY",),
    display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API",
    signup_url="https://platform.deepseek.com/",
    fallback_models=(
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    base_url="https://api.deepseek.com/v1",
)

register_provider(deepseek)
