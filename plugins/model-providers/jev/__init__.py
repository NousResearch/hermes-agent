"""TypeSafe Jev provider profile.

Jev is TypeSafe's System One decision model (``jev-latest``), not a chat-
completions backend. This profile wires ``TYPESAFE_API_KEY`` into ``hermes
setup``, ``hermes auth``, and ``hermes doctor`` — the same judgment-only
posture as OMP's ``/login typesafe`` / ``TypeSafeJudge`` (no session chat
model). ``api_mode`` is ``systemone`` so the provider is excluded from
``hermes model`` / the session picker.

Consumers call ``POST /v1/systemone`` directly — e.g. the community
``typesafe-skill-router`` plugin or a copy-paste hook. Do not set
``model.provider: jev`` as the session chat model.
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
    api_mode="systemone",
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
