"""Together AI provider profile."""

import json
import logging
from urllib.request import Request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent


class TogetherProfile(ProviderProfile):
    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        # Together's catalog also includes image, audio and embedding models.
        url = (base_url or self.base_url).rstrip("/") + "/models"
        headers = {"Accept": "application/json", "User-Agent": _profile_user_agent()}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            with open_credentialed_url(Request(url, headers=headers), timeout=timeout) as response:
                payload = json.loads(response.read())
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).debug("Together model catalog unavailable: %s", exc)
            return None
        items = payload if isinstance(payload, list) else payload.get("data", [])
        return [m["id"] for m in items if isinstance(m, dict) and m.get("id") and m.get("type") in (None, "", "chat")]


together = TogetherProfile(
    name="togetherai", aliases=("together", "together-ai"), display_name="Together AI",
    description="Together AI — OpenAI-compatible inference",
    signup_url="https://api.together.ai/settings/api-keys", env_vars=("TOGETHER_API_KEY",),
    base_url="https://api.together.ai/v1", auth_type="api_key",
    default_aux_model="Qwen/Qwen3.5-9B",
    fallback_models=("MiniMaxAI/MiniMax-M3", "Qwen/Qwen3.5-9B"),
)

register_provider(together)
