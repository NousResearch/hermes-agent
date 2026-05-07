"""OpenRouter provider profile."""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None


class OpenRouterProfile(ProviderProfile):
    """OpenRouter aggregator — provider preferences, reasoning config passthrough."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from public OpenRouter catalog — no auth required.

        Note: Tool-call capability filtering is applied by hermes_cli/models.py
        via fetch_openrouter_models() → _openrouter_model_supports_tools(), not
        here. The picker early-returns via the dedicated openrouter path before
        reaching this method, so filtering here would be unreachable.
        """
        global _CACHE  # noqa: PLW0603
        if _CACHE is not None:
            return _CACHE
        try:
            result = super().fetch_models(api_key=None, timeout=timeout)
            if result is not None:
                _CACHE = result
            return result
        except Exception as exc:
            logger.debug("fetch_models(openrouter): %s", exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """OpenRouter passes the full reasoning_config dict as extra_body.reasoning."""
        extra_body: dict[str, Any] = {}
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


openrouter = OpenRouterProfile(
    name="openrouter",
    aliases=("or",),
    env_vars=("OPENROUTER_API_KEY",),
    display_name="OpenRouter",
    description="OpenRouter — unified API for 200+ models",
    signup_url="https://openrouter.ai/keys",
    base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3-flash-preview",
        "qwen/qwen3-plus",
    ),
)

register_provider(openrouter)
