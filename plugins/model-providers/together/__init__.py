"""Together AI provider profile.

Together serves serverless and dedicated open models through an
OpenAI-compatible chat-completions endpoint. Model IDs use Together's native
``organization/model`` form, for example ``MiniMaxAI/MiniMax-M3``.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent


logger = logging.getLogger(__name__)


class TogetherProfile(ProviderProfile):
    """Provider profile with Together's bare-list catalog response."""

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Remove the optional tool-result name rejected by Together GPT-OSS."""
        prepared = list(messages)
        for index, message in enumerate(messages):
            if message.get("role") == "tool" and "name" in message:
                cleaned = dict(message)
                cleaned.pop("name", None)
                prepared[index] = cleaned
        return prepared

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        effective_base = (base_url or self.base_url).rstrip("/")
        if not effective_base:
            return None

        req = urllib.request.Request(f"{effective_base}/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("data", [])
        else:
            return None
        models: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            model_type = str(item.get("type") or "").strip().lower()
            # Omit non-chat catalog surfaces; private models may omit ``type``.
            if model_id and model_type in {"", "chat"}:
                models.append(str(model_id))
        return models


together = TogetherProfile(
    name="together",
    aliases=("together-ai", "togetherai"),
    display_name="Together AI",
    description="Together AI — OpenAI-compatible serverless inference",
    signup_url="https://api.together.ai/settings/api-keys",
    env_vars=("TOGETHER_API_KEY",),
    base_url="https://api.together.ai/v1",
    auth_type="api_key",
    default_aux_model="Qwen/Qwen3.5-9B",
    fallback_models=(
        "MiniMaxAI/MiniMax-M3",
        "moonshotai/Kimi-K2.7-Code",
        "Qwen/Qwen3.5-9B",
    ),
)

register_provider(together)
