"""Keep the OAuth **access token** as the dashboard session credential, for an OIDC provider.

## Why this is a separate mixin and not four overrides in the provider

The stock ``SelfHostedOIDCProvider`` treats the ID token as the session credential: it stores the
ID token in ``Session.access_token`` and re-verifies it on every request. That is correct for a
dashboard that only needs identity, and wrong for this deployment: the same session token is
forwarded to a resource server (an MCP server), whose audience is *not* this dashboard client. An
ID token sent there is rejected — or, worse, accepted by a server that trusts the wrong audience.

Conversely, the OAuth access token *is* minted for the resource server. So the provider must keep
it and verify *it* as the session credential.

That needs exactly three behaviours, and they are the same three for any OIDC provider that wants
this — hence a mixin rather than a copy inside one plugin:

* ``_grant`` — the parent reads ``id_token`` out of the token response; this one reads
  ``access_token``, and still verifies the ID token once, for flow integrity only.
* ``_claims_for`` — the parent aliases it to ``_verify_id_token`` at class-creation time, so an
  override of ``_verify_id_token`` alone would never run (``verify_session`` calls
  ``self._claims_for``). This binds it to access-token verification.
* ``_reject_non_session_token`` — signature verification is not authorisation: Keycloak mints
  access tokens for *clients* as well as users, all signed by the same realm keys with the same
  ``iss``. Three claims separate a person's token from a machine one's, and each is checked here.

Mix it in **before** ``SelfHostedOIDCProvider`` so its methods win the MRO::

    class MyProvider(AccessTokenSessionGrant, SelfHostedOIDCProvider):
        ...
"""

from __future__ import annotations

from typing import Any

from hermes_cli.dashboard_auth import InvalidCodeError, Session
from plugins.dashboard_auth._shared import exchange_token, refresh_token_from, verify_jwt

#: ``typ`` values that identify an OAuth access token (``at+jwt`` is the newer JWT profile).
#: Keycloak writes this header; other IDPs may omit it, so absence is accepted.
_ACCESS_TOKEN_TYPES = frozenset({"bearer", "at+jwt"})

#: Keycloak names a service account's token subject ``service-account-<client_id>``. Such a token
#: is a machine credential every holder shares, so it must never open a person's session.
_SERVICE_ACCOUNT_PREFIX = "service-account-"


class AccessTokenSessionGrant:
    """OIDC ``_grant``/verification overrides that keep the OAuth access token, not the ID token.

    Relies on the provider supplying (all inherited from ``SelfHostedOIDCProvider``):
    ``_get_discovery()``, ``_token_endpoint_auth()``, ``_verify_id_token()``, ``_session()``,
    ``_get_jwks_client()``, ``_client_id``, ``_id_token_leeway``.
    """

    def _grant(
        self, data: dict[str, str], *, bad_request_exc: type[Exception],
        headers: dict[str, str] | None = None, previous_refresh_token: str = "",
    ) -> Session:
        """POST the discovered token endpoint and turn the response into a Session.

        The **access token** becomes the session credential (see the module docstring). The ID
        token is still verified once, as the OIDC end-to-end integrity check for the flow — a
        tampered or truncated code exchange cannot yield a valid ID token — but its claims are
        not kept: the access token carries the same realm user claims. Claim mapping then runs
        through the parent's ``_session``, whose first parameter is the session credential.
        """
        disco = self._get_discovery()
        extra_data, extra_headers = self._token_endpoint_auth(disco)
        access_token, payload = exchange_token(
            disco["token_endpoint"], {**data, **extra_data}, headers=extra_headers,
            bad_request_exc=bad_request_exc, idp="IDP", endpoint="OIDC token endpoint",
            token_key="access_token",
            missing_msg=(
                "OIDC token response missing access_token — the 'openid' scope must be "
                "configured and the client allowed to receive an access token."))
        if isinstance(payload.get("id_token"), str) and payload["id_token"]:
            self._verify_id_token(payload["id_token"])  # integrity only; see docstring
        claims = self._claims_for(access_token)
        # Prefer a freshly-issued RT, else keep the previous (some IDPs don't rotate).
        return self._session(
            access_token, refresh_token_from(payload, previous_refresh_token), claims)

    def _claims_for(self, access_token: str) -> dict[str, Any]:
        """Verify an access token and insist it may be a session credential for this client.

        Pinned by ``verify_jwt``: ``iss`` against the discovered issuer, ``aud`` against this
        client. Two things it cannot express — and which therefore live in
        :meth:`_reject_non_session_token` — are that the token must be an *access* token and must
        not be a shared machine credential.
        """
        claims = verify_jwt(
            access_token, self._get_jwks_client(), algorithms=["RS256"],
            audience=self._client_id, issuer=self._get_discovery()["issuer"],
            label="access token", leeway=self._id_token_leeway)
        self._reject_non_session_token(claims)
        return claims

    def _reject_non_session_token(self, claims: dict[str, Any]) -> None:
        """Refuse a verified token that must not open a dashboard session.

        Signature verification alone is not enough now that the session credential IS an access
        token: the IdP mints access tokens for clients as well as users, and every one of them is
        signed by the same realm keys with the same ``iss``. Three checks, each aimed at a real
        token:

        * ``typ`` — an ID token (``typ: ID``) is API-visible to any holder and must not be
          interchangeable with a session credential. Only access-token types are accepted; a token
          with no ``typ`` at all is accepted, because a realm may be configured without it.
        * ``azp`` — the client the token was issued *to* (OIDC §2). It keeps another client's
          token from being replayed here even when that client shares an audience entry.
        * ``preferred_username``/``sub`` — a ``service-account-<client_id>`` name marks a
          ``client_credentials`` token: a machine credential shared by every holder.
        """
        typ = claims.get("typ")
        if typ is not None and str(typ).strip().lower() not in _ACCESS_TOKEN_TYPES:
            raise InvalidCodeError(f"not an access token (typ={typ!r})")
        authorized_party = claims.get("azp")
        if authorized_party and str(authorized_party) != self._client_id:
            raise InvalidCodeError(
                f"token was issued to client {authorized_party!r}, not {self._client_id!r}")
        subject = str(claims.get("preferred_username") or claims.get("sub") or "")
        if subject.startswith(_SERVICE_ACCOUNT_PREFIX):
            raise InvalidCodeError(
                "a service-account token cannot open a dashboard session; sign in as a user")


__all__ = ["AccessTokenSessionGrant"]