"""Fireworks AI provider profile.

Fireworks AI serves fast, production-grade inference for open and proprietary
models through an OpenAI-compatible chat-completions endpoint.
"""

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent


class FireworksProfile(ProviderProfile):
    """Fireworks provider with agent-compatible live model discovery."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        import json
        import urllib.request

        url = (self.models_url or "").strip()
        if not url:
            return None

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return None

        items = data if isinstance(data, list) else data.get("models", [])
        models = []
        for item in items:
            if not isinstance(item, dict) or item.get("public") is False:
                continue
            if not item.get("supportsTools"):
                continue
            name = item.get("name")
            if not name:
                continue
            model_name = name.rsplit("/", 1)[-1].lower()
            if (
                model_name.startswith(("flux-", "bge-"))
                or "embed" in model_name
                or model_name.endswith("-base")
                or "-base-v" in model_name
            ):
                continue
            models.append(name)
        return models


fireworks = FireworksProfile(
    name="fireworks",
    aliases=("fireworks-ai", "fw"),
    display_name="Fireworks AI",
    description="Fireworks AI — OpenAI-compatible direct model API",
    signup_url="https://app.fireworks.ai/settings/users/api-keys",
    env_vars=("FIREWORKS_API_KEY",),
    base_url="https://api.fireworks.ai/inference/v1",
    models_url="https://api.fireworks.ai/v1/accounts/fireworks/models",
    auth_type="api_key",
    default_aux_model="accounts/fireworks/models/glm-5p2",
    fallback_models=(
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
    ),
)

register_provider(fireworks)
