"""TrustedRouter.com provider profile."""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None


class TrustedRouterProfile(ProviderProfile):
    """TrustedRouter.com - end-to-end encrypted OpenRouter-compatible routing."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the TrustedRouter model catalog when credentials are available."""
        global _CACHE  # noqa: PLW0603
        if _CACHE is not None:
            return _CACHE
        try:
            result = super().fetch_models(api_key=api_key, timeout=timeout)
            if result is not None:
                _CACHE = result
            return result
        except Exception as exc:
            logger.debug("fetch_models(trustedrouter): %s", exc)
            return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """TrustedRouter accepts OpenRouter/OpenAI-compatible reasoning payloads."""
        if supports_reasoning:
            if reasoning_config is not None:
                return {"reasoning": dict(reasoning_config)}, {}
            return {"reasoning": {"enabled": True, "effort": "medium"}}, {}
        return {}, {}


trustedrouter = TrustedRouterProfile(
    name="trustedrouter",
    aliases=(
        "tr",
        "trusted-router",
        "trustedrouter.com",
        "quillrouter",
        "quill-router",
    ),
    env_vars=("TRUSTEDROUTER_API_KEY", "TRUSTEDROUTER_BASE_URL"),
    display_name="TrustedRouter.com",
    description="TrustedRouter.com - end-to-end encrypted OpenRouter-compatible LLM router",
    signup_url="https://trustedrouter.com/",
    base_url="https://api.quillrouter.com/v1",
    fallback_models=("trustedrouter/auto",),
)

register_provider(trustedrouter)
