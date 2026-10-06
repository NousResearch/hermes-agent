"""Solstice provider profile: a user's own per-user-quota Gemini subscription, signed in with OAuth.

Pre-release (``hidden=True``): no picker, setup list or accounts tab offers it until the user signs in
by name (``hermes auth add solstice``); after that it lists like any signed-in provider. Login and
token refresh go through the generic PKCE plugin flow with a Nous-portal broker as the token endpoint
(``auth.py``); inference goes straight to Google on the per-user-quota methods (``transport.py``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig, pkce_auth_handler, pkce_refresh_credential
from providers import register_provider
from providers.base import ProviderProfile

from .auth import broker_token_request, discover_client
from .transport import INFERENCE_BASE_URL, SolsticeClient

# Verified on the per-user-quota endpoint, which has no model listing a user token may read (its
# /models answers 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT). Quota is per model, so lite stays last as the
# reserve once the flash family is exhausted.
FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest")

# The broker owns client_id/scope (discovered per login); refresh needs neither, the broker adds them.
_BASE_CFG = OAuthPKCEConfig(
    client_id="", authorize_url="", token_url="", label="Solstice",
    redirect_path="/gemini-auth/callback", token_request=broker_token_request,
    # A refresh token is issued only for offline access, and Google omits it on a repeat consent.
    extra_authorize_params={"access_type": "offline", "prompt": "consent"},
)


def _auth_handler(action: str, args: Any) -> bool:
    cfg = _BASE_CFG
    if action == "add":
        client = discover_client()
        cfg = replace(_BASE_CFG, client_id=client["client_id"], authorize_url=client["authorize_url"],
                      scopes=tuple(client["scope"].split()))
    return pkce_auth_handler(cfg)(action, args)


class SolsticeProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        allowed = {"api_key", "base_url", "default_headers", "timeout", "http_client"}
        return SolsticeClient(**{k: v for k, v in client_kwargs.items() if k in allowed})

    def get_max_tokens(self, model: str | None) -> int | None:
        return None  # output caps are the backend's; a Hermes default would truncate long answers


register_provider(SolsticeProfile(
    name="solstice", aliases=("solstice-oauth",), display_name="Solstice",
    description="Solstice (your own per-user quota, OAuth sign-in)",
    auth_type="oauth_external", base_url=INFERENCE_BASE_URL, hidden=True,
    supports_health_check=False, supports_model_listing=False, supports_vision=True,
    fallback_models=FALLBACK_MODELS, default_aux_model="gemini-flash-lite-latest",
    auth_handler=_auth_handler, refresh_credential=pkce_refresh_credential(_BASE_CFG),
))
