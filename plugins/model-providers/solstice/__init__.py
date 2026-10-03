"""Solstice provider profile (pre-release — hidden by default).

Solstice is a consumer subscription carried into Hermes. Inference goes to
``generativelanguage.googleapis.com`` on the ``:generateContentPerUserQuota``
endpoint, authenticated with the user's own OAuth token so usage maps to their
Solstice quota.

This profile is declarative metadata only — the transport lives in
``agent/solstice_adapter.py`` (per-user-quota endpoint + Bearer auth over
the native request builder) and the OAuth flow lives in
``hermes_cli/solstice.py`` (PKCE loopback → NAS-brokered code
exchange → direct inference).

Secret-launch mechanics:
- ``hidden=True`` keeps the provider out of the default discovery surfaces
  (``/model`` picker, setup wizard, ``hermes auth`` lists, doctor) until it is
  explicitly configured. With NAS owning the client config (discovered
  at login via ``/api/oauth/gemini-auth/config``), the provider is enabled by
  default and surfaced once a logged-in Nous user runs ``hermes auth add
  solstice``. The visibility gate defers to
  ``hermes_cli.solstice.solstice_enabled()`` (registered below).
- The provider still resolves by name via ``get_provider_profile()`` so an
  explicit ``model.provider: solstice`` in config.yaml works.
"""

from typing import Any

from providers import register_hidden_provider_gate, register_provider
from providers.base import ProviderProfile


# Inference host — the per-user-quota backend behind Solstice.  The
# standard request body (built by
# agent/gemini_native_adapter.build_gemini_request) is sent to the
# ``:generateContentPerUserQuota`` variant at this host.  Imported from
# hermes_cli/solstice.py so the host has a single source of truth.
from hermes_cli.solstice import (
    SOLSTICE_INFERENCE_BASE_URL as SOLSTICE_BASE_URL,
)

# Curated model list shown when live discovery is unavailable, verified
# present on the per-user-quota endpoint (2026-09). The endpoint serves a
# closed set, so the fallback lists only those confirmed to resolve.
#
# Order matters: the first entry is the default. lite is listed last as the
# reserve — the per-user quota is enforced PER MODEL, so when the flash/pro
# family is exhausted lite still answers.
SOLSTICE_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
)


class SolsticeProfile(ProviderProfile):
    """Solstice — per-user-quota OAuth provider."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """No extra_body: the transport client builds the request envelope."""
        return {}

    def get_max_tokens(self, model: str | None) -> int | None:
        # Solstice model output caps are enforced by the backend; do not
        # impose a Hermes-side default cap.
        return None


solstice_profile = SolsticeProfile(
    name="solstice",
    aliases=("solstice-oauth",),
    display_name="Solstice",
    description="Solstice (per-user-quota OAuth)",
    api_mode="chat_completions",
    auth_type="oauth_external",
    base_url=SOLSTICE_BASE_URL,
    env_vars=(),
    hidden=True,
    supports_health_check=False,
    fallback_models=SOLSTICE_FALLBACK_MODELS,
)

register_provider(solstice_profile)


def _solstice_gate() -> bool:
    """Enable predicate — defers to the single auth-side gate.

    Registered so ``list_providers()`` and ``solstice_enabled()`` read the
    SAME credential resolver, keeping the provider's visibility and its
    auth/runtime usability in lockstep (never surfaced-but-dead).
    """
    from hermes_cli.solstice import solstice_enabled

    return solstice_enabled()


register_hidden_provider_gate("solstice", _solstice_gate)
