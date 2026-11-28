"""KeycloakDashboardAuthProvider — Keycloak OpenID Connect dashboard auth.

A Keycloak-flavoured :class:`~plugins.dashboard_auth.self_hosted.SelfHostedOIDCProvider`:
endpoints come from realm discovery (``{issuer}/.well-known/openid-configuration``, e.g.
``https://idp.example/realms/myrealm``), tokens are verified against the discovered ``jwks_uri``
with ``iss``/``aud`` pinned, and the client is a **public** one — authorization code + PKCE (S256)
only, no client secret (register the Keycloak client with "Client authentication: Off"). PKCE
login, code/refresh grants, JWKS verification, claim mapping and RFC 7009 revocation are inherited
unchanged. This subclass adds the Keycloak-specific configuration surface, two optional
authorize-URL parameters, and one behavioural override: the **OAuth access token** is the session
credential (an ID token's audience is this dashboard client, so forwarding it to a resource server
was always wrong), verified for audience/client/principal rather than trusted on signature alone.

  ``kc_idp_hint`` — skip the realm's identity-provider chooser and go straight to a named
    external IdP (Keycloak's own parameter).
  ``prompt`` — ``login`` (force re-auth), ``consent``, or ``select_account``.

Config: ``dashboard.oauth.keycloak.{issuer,client_id,scopes,idp_hint,prompt,
id_token_leeway}`` or ``HERMES_DASHBOARD_KEYCLOAK_{ISSUER,CLIENT_ID}`` (env wins
when non-empty; ``scopes``/``idp_hint``/``prompt``/``id_token_leeway`` are config.yaml-only).
The session's ``provider`` is ``"keycloak"`` (``self.name``), so a Keycloak cookie is dispatched
back to this provider by the auth gate.
"""

from __future__ import annotations

import logging
import urllib.parse

from hermes_cli.dashboard_auth import LoginStart
from plugins.dashboard_auth._shared import (
    DEFAULT_TOKEN_LEEWAY_SECONDS, SkipRegistration, load_config_section, parse_leeway,
    register_provider, resolve_env_or_cfg)
from plugins.dashboard_auth.keycloak.access_token_grant import AccessTokenSessionGrant
from plugins.dashboard_auth.self_hosted import _DEFAULT_SCOPES, SelfHostedOIDCProvider

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-keycloak"

LAST_SKIP_REASON: str = ""

class KeycloakDashboardAuthProvider(AccessTokenSessionGrant, SelfHostedOIDCProvider):
    """Keycloak OIDC via authorization-code + PKCE (S256), with realm-specific extras.

    ``AccessTokenSessionGrant`` comes first so its ``_grant``/``_claims_for`` win the MRO: the
    session credential must be the OAuth access token a resource server accepts, not the ID token
    whose audience is this dashboard client. Everything else is inherited unchanged.
    """

    name = "keycloak"
    display_name = "Keycloak"

    def __init__(
        self, *, issuer: str, client_id: str, scopes: str = _DEFAULT_SCOPES,
        id_token_leeway: float = DEFAULT_TOKEN_LEEWAY_SECONDS, idp_hint: str = "", prompt: str = "",
    ) -> None:
        # No client_secret by design: public client, PKCE-only (see module docstring).
        super().__init__(issuer=issuer, client_id=client_id, scopes=scopes, id_token_leeway=id_token_leeway)
        # Optional Keycloak authorize-URL extras; empty string means "omit".
        self._idp_hint = (idp_hint or "").strip()
        self._prompt = (prompt or "").strip()

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        """PKCE authorize URL from realm discovery, plus ``kc_idp_hint``/``prompt`` when set."""
        base = super().start_login(redirect_uri=redirect_uri)
        extras = {k: v for k, v in (("kc_idp_hint", self._idp_hint), ("prompt", self._prompt)) if v}
        if not extras:
            return base
        parsed = urllib.parse.urlsplit(base.redirect_url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        present = {key for key, _ in query}
        query.extend((key, value) for key, value in extras.items() if key not in present)
        rebuilt = parsed._replace(query=urllib.parse.urlencode(query))
        return LoginStart(
            redirect_url=urllib.parse.urlunsplit(rebuilt), cookie_payload=base.cookie_payload)

# ---- Plugin entry point ----

def _load_config_oauth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "oauth", "keycloak")


def _settings() -> dict:
    """Resolve KeycloakDashboardAuthProvider kwargs; the skip reason names BOTH configuration surfaces."""
    kc_cfg = _load_config_oauth_section()

    def setting(env_name: str, cfg_key: str) -> str:
        return resolve_env_or_cfg(env_name, kc_cfg.get(cfg_key))

    issuer = setting("HERMES_DASHBOARD_KEYCLOAK_ISSUER", "issuer")
    client_id = setting("HERMES_DASHBOARD_KEYCLOAK_CLIENT_ID", "client_id")
    if not issuer or not client_id:
        raise SkipRegistration(
            "Keycloak dashboard auth is not configured. Set both an issuer (the realm URL, "
            "e.g. https://idp.example/realms/myrealm) and a client_id — either as env vars "
            "(HERMES_DASHBOARD_KEYCLOAK_ISSUER + HERMES_DASHBOARD_KEYCLOAK_CLIENT_ID) or under "
            "dashboard.oauth.keycloak.{issuer,client_id} in config.yaml — or pass "
            f"--insecure to skip the OAuth gate entirely. (issuer set: {bool(issuer)}; "
            f"client_id set: {bool(client_id)})")
    return {
        "issuer": issuer, "client_id": client_id,
        "scopes": setting("HERMES_DASHBOARD_KEYCLOAK_SCOPES", "scopes") or _DEFAULT_SCOPES,
        # Keycloak authorize-URL extras (config.yaml only; empty ⇒ omitted).
        "idp_hint": str(kc_cfg.get("idp_hint", "") or "").strip(),
        "prompt": str(kc_cfg.get("prompt", "") or "").strip(),
        # Clock-skew tolerance for ID-token exp/nbf/iat (config.yaml only; default 60s, 0 = strict).
        "id_token_leeway": parse_leeway(kc_cfg.get("id_token_leeway"))}


def register(ctx) -> None:
    """Register :class:`KeycloakDashboardAuthProvider` when issuer + client_id are set."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kw, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, KeycloakDashboardAuthProvider, _settings)
    if kw is not None:
        logger.info(
            "dashboard-auth-keycloak: registered provider "
            "(issuer=%s, client_id=%s, scopes=%r, idp_hint=%r, prompt=%r, id_token_leeway=%ss)",
            kw["issuer"], kw["client_id"], kw["scopes"],
            kw["idp_hint"], kw["prompt"], kw["id_token_leeway"])
