"""Jev provider profile (TypeSafe System One advisory model).

Jev is NOT a chat backend: its API is ``POST /v1/systemone``, not chat
completions, and there is no ``/v1/models`` catalog. This profile only makes
Hermes know Jev exists so ``hermes setup`` / ``auth`` / ``doctor`` / ``model``
auto-wire ``TYPESAFE_API_KEY``. The live consumer is the out-of-tree
typesafe-skill-router plugin. Never default a chat, aux, or compression
route to ``jev-latest``.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class JevProfile(ProviderProfile):
    """TypeSafe Jev — static catalog; no REST listing, no health probe."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return the static catalog without a network call (no /v1/models exists)."""
        return ["jev-latest"]


jev = JevProfile(
    name="jev", aliases=("typesafe", "typesafe-ai"),
    env_vars=("TYPESAFE_API_KEY",),
    base_url="https://api.typesafe.ai",
    display_name="Jev",
    description="Jev (TypeSafe System One advisory model; not a chat backend)",
    supports_health_check=False,  # no /models probe; System One only
    supports_model_listing=False,  # TypeSafe publishes no OpenAI model list
    supports_vision=False,
)

register_provider(jev)
