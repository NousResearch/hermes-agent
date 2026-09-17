"""TypeSafe Jev provider profile.

Jev is TypeSafe's System One decision model (``jev-latest``), not a chat-
completions backend. This profile exists so Hermes setup, ``hermes auth``,
``hermes doctor``, and the model picker know the provider and
``TYPESAFE_API_KEY``. The live consumer is the out-of-tree
``typesafe-skill-router`` plugin (POST ``/v1/systemone``). Do not use Jev as
the session chat model.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile


class JevProfile(ProviderProfile):
    """TypeSafe Jev — static catalog, no OpenAI ``/v1/models`` probe."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        # TypeSafe does not publish an OpenAI-compatible model catalog.
        # Returning the curated fallback avoids a failing /v1/models round-trip.
        return list(self.fallback_models)


jev = JevProfile(
    name="jev",
    aliases=("typesafe", "typesafe-ai"),
    display_name="TypeSafe (Jev)",
    description="TypeSafe Jev — System One typed decisions (not a chat-completions backend)",
    signup_url="https://console.typesafe.ai/settings/keys",
    env_vars=("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"),
    base_url="https://api.typesafe.ai/v1",
    auth_type="api_key",
    supports_health_check=False,
    fallback_models=("jev-latest",),
    default_aux_model="",
)

register_provider(jev)
