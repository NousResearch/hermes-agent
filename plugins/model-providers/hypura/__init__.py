"""Hypura provider profile.

Hypura is a storage-tier-aware LLM inference scheduler that serves an
Ollama-compatible API on localhost (default port 8080).
"""

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class HypuraProfile(ProviderProfile):
    """Hypura local inference server — Ollama-compatible API."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from local Hypura/Ollama /api/tags endpoint.

        Hypura exposes an Ollama-compatible /api/tags endpoint rather than
        the OpenAI-standard /v1/models, so we override the default behavior.
        """
        effective_base = (base_url or self.base_url).rstrip("/")
        url = f"{effective_base}/api/tags"

        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        from hermes_cli.urllib_security import open_credentialed_url

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            models_data = data if isinstance(data, list) else data.get("models", [])
            return [m["name"] for m in models_data if isinstance(m, dict) and "name" in m]
        except Exception as exc:
            logger.debug("fetch_models(hypura): %s", exc)
            return None


hypura = HypuraProfile(
    name="hypura",
    aliases=("hypura-local", "hypura_local"),
    default_aux_model="hypura",
    display_name="Hypura (Local)",
    description="Hypura local inference server — Ollama-compatible API",
    env_vars=(),  # No auth required
    base_url="http://localhost:8080",
    fallback_models=(
        "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
    ),
)

register_provider(hypura)