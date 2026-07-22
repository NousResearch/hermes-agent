"""Hypura provider profile.

Hypura is a storage-tier-aware LLM inference scheduler that serves an
Ollama-compatible API on localhost (default port 8080).
"""

import logging
from typing import Any

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
        """Fetch from local Hypura server's /api/tags endpoint."""
        try:
            result = super().fetch_models(
                api_key=None,  # Hypura doesn't require auth
                base_url=base_url or "http://localhost:8080",
                timeout=timeout,
            )
            return result
        except Exception as exc:
            logger.debug("fetch_models(hypura): %s", exc)
            return None


hypura = HypuraProfile(
    name="hypura",
    aliases=("hypura-local",),
    default_aux_model="hypura",
    env_vars=(),  # No auth required
    base_url="http://localhost:8080",
)

register_provider(hypura)