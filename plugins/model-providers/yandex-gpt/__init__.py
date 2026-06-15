"""Yandex Cloud LLM (YandexGPT) provider profile.

Integrates native Yandex GPT models via the Yandex Cloud API.
Supports foundation models (Pro, Lite, Micro) with automatic model discovery.
"""

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class YandexGPTProfile(ProviderProfile):
    """Yandex Cloud LLM — OpenAI-compatible API with IAM token auth."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch available models from Yandex Cloud catalog.

        The public catalog endpoint (models.yandexcloud.net) does not require
        authentication. Falls back gracefully if the service is unavailable.
        """
        if not api_key:
            # No auth needed for public catalog; return well-known fallbacks
            return [
                "yandexgpt/latest",
                "yandexgpt/pro",
                "yandexgpt/lite",
                "yandexgpt/micro",
            ]

        try:
            req = urllib.request.Request(
                "https://models.yandexcloud.net/api/v1/models"
            )
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return [
                m["id"]
                for m in data.get("models", [])
                if isinstance(m, dict) and "id" in m
            ]
        except Exception as exc:
            logger.debug("fetch_models(yandex-gpt): %s", exc)
            # Fallback to well-known models on any error
            return [
                "yandexgpt/latest",
                "yandexgpt/pro",
                "yandexgpt/lite",
                "yandexgpt/micro",
            ]

    def prepare_messages(
        self, msgs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare messages for Yandex Cloud LLM API.

        Yandex Cloud uses OpenAI-compatible message format natively.
        No special preprocessing required.
        """
        return msgs

    def build_extra_body(self, **context: Any) -> dict[str, Any]:
        """Build Yandex-specific request body parameters.

        Yandex Cloud API accepts standard OpenAI parameters.
        Return empty dict to use defaults.
        """
        return {}


yandex_gpt = YandexGPTProfile(
    name="yandex-gpt",
    aliases=("yandex", "yandexgpt"),
    api_mode="openai-like",
    display_name="Yandex Cloud LLM",
    description="Yandex Cloud YandexGPT — foundation models and fine-tuned variants.",
    signup_url="https://console.cloud.yandex.com",
    env_vars=(
        "YANDEX_GPT_API_KEY",
        "YANDEX_GPT_FOLDER_ID",
    ),
    base_url="https://llm.api.cloud.yandex.net:443/foundationModels/v1",
    auth_type="api_key",
    default_aux_model="yandexgpt/lite",
)

register_provider(yandex_gpt)
