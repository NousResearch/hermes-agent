"""Prime Inference provider profile."""

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


class PrimeProfile(ProviderProfile):
    """Prime's public catalog includes decision and non-agentic models."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Offer tool-capable text models; custom proxies keep generic discovery."""
        caller_base = (base_url or "").strip().rstrip("/")
        if caller_base and caller_base != self.base_url.rstrip("/"):
            return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)

        from hermes_cli.urllib_security import open_credentialed_url

        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/models",
            headers={"Accept": "application/json", "User-Agent": _profile_user_agent()},
        )
        try:
            with open_credentialed_url(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            if not isinstance(data.get("data"), list):
                raise ValueError("Prime model catalog must contain a data list")
            models = []
            for model in data.get("data", []):
                modalities = (model.get("specs") or {}).get("modalities") or {}
                if (
                    model.get("id")
                    and "tools" in (model.get("supported_parameters") or ())
                    and "text" in (modalities.get("input") or ())
                    and "text" in (modalities.get("output") or ())
                ):
                    models.append(model["id"])
            return models
        except Exception as exc:
            logger.debug("fetch_models(prime): %s", exc, exc_info=True)
            return None


prime = PrimeProfile(
    name="prime",
    aliases=("prime-inference", "primeintellect", "pinference"),
    display_name="Prime Inference",
    description="Prime Intellect — OpenAI-compatible inference API",
    signup_url="https://app.primeintellect.ai/",
    env_vars=("PRIME_API_KEY",),
    base_url="https://api.pinference.ai/api/v1",
    auth_type="api_key",
    authoritative_model_catalog=True,
    default_aux_model="deepseek/deepseek-v4.1-flash",
    fallback_models=(
        "deepseek/deepseek-v4.1-flash",
        "z-ai/glm-5.3",
        "moonshotai/kimi-k3",
        "openai/gpt-5.4",
        "anthropic/claude-opus-5.5",
    ),
)

register_provider(prime)
