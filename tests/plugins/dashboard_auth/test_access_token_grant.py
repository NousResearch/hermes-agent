"""The Keycloak mixin that makes the **access token** the dashboard session credential.

Why it exists: the stock ``SelfHostedOIDCProvider`` keeps the *ID token* as the session
credential. That is right for a dashboard that only needs identity, and wrong here — the same
session token is forwarded to a resource server (an MCP server) whose audience is not this
dashboard client, so the ID token is rejected there (or, worse, accepted by a server that trusts
the wrong audience). The OAuth access token *is* minted for that resource server.

Three behaviours, each tested for what it stops as much as what it does:

* ``_grant`` reads ``access_token`` from the token response (and still verifies the ID token
  once, for flow integrity);
* ``_claims_for`` verifies the access token (the parent aliases it to ``_verify_id_token``, so
  overriding ``_verify_id_token`` alone would never run);
* ``_reject_non_session_token`` refuses a *machine* token that the realm validly signed.
"""

from __future__ import annotations

import pytest

import plugins.dashboard_auth.keycloak.access_token_grant as grant_module
from hermes_cli.dashboard_auth.base import InvalidCodeError
from plugins.dashboard_auth.keycloak.access_token_grant import (
    AccessTokenSessionGrant,
    _ACCESS_TOKEN_TYPES,
    _SERVICE_ACCOUNT_PREFIX,
)

CLIENT_ID = "ebmedical_work"
ISSUER = "https://sso.example/realms/ebmedical"


class _FakeProvider:
    """The parent surface the mixin relies on, with the network and JWT layers stubbed.

    Subclassing the mixin over this (rather than over the real provider) keeps the mixin's own
    method resolution — the thing under test — while replacing everything it calls out to.
    """

    _client_id = CLIENT_ID
    _id_token_leeway = 0

    def __init__(self, *, claims: dict, id_token_claims: dict | None = None):
        self.claims = claims
        self.id_token_claims = id_token_claims or {"sub": "from-id-token"}
        self.verified_id_tokens: list[str] = []
        self.verified_access_tokens: list[str] = []
        self.session_args: tuple | None = None

    # ---- collaborators ---------------------------------------------------
    def _get_discovery(self):
        return {"token_endpoint": "https://idp/token", "issuer": ISSUER}

    def _token_endpoint_auth(self, disco):
        return {}, {}

    def _get_jwks_client(self):
        return object()

    def _verify_id_token(self, id_token: str):
        self.verified_id_tokens.append(id_token)
        return self.id_token_claims

    def _session(self, token: str, refresh_token: str, claims: dict):
        self.session_args = (token, refresh_token, claims)
        return {"credential": token, "refresh_token": refresh_token, "claims": claims}

    # ---- the JWT layer ---------------------------------------------------
    def _verify_access_token(self, access_token: str):
        self.verified_access_tokens.append(access_token)
        return dict(self.claims)


class _Subject(AccessTokenSessionGrant, _FakeProvider):
    """Mixin first, exactly as the plugin layers it over the real provider."""

    def _claims_for(self, access_token: str) -> dict:
        """The real screening (:meth:`_reject_non_session_token`), minus the JWKS round trip."""
        claims = self._verify_access_token(access_token)
        self._reject_non_session_token(claims)
        return claims


@pytest.fixture
def exchange(monkeypatch):
    """Replace the token-endpoint POST; the response payload is set per test."""
    state = {"payload": {}, "token_key": None}

    def _exchange(url, data, *, headers, bad_request_exc, idp, endpoint, token_key, missing_msg):
        state["token_key"] = token_key
        token = str(state["payload"].get(token_key) or "")
        if not token:
            raise InvalidCodeError("token response missing the requested token")
        return token, state["payload"]

    monkeypatch.setattr(grant_module, "exchange_token", _exchange)
    return state


def _grant(state, *, claims=None, id_token_claims=None, previous_refresh_token="", payload=None):
    state["payload"] = payload if payload is not None else {"access_token": "AT"}
    obj = _Subject(claims=claims if claims is not None else {"sub": "user-1"},
                   id_token_claims=id_token_claims)
    result = obj._grant({}, bad_request_exc=InvalidCodeError,
                        previous_refresh_token=previous_refresh_token)
    return obj, result


# --------------------------------------------------------------------------- _grant

class TestGrantKeepsTheAccessToken:
    def test_the_access_token_becomes_the_session_credential(self, exchange):
        obj, _ = _grant(exchange, payload={"access_token": "AT-real", "id_token": "ID-real",
                                           "refresh_token": "RT-real"})
        assert obj.session_args[0] == "AT-real"      # not the ID token
        assert obj.session_args[1] == "RT-real"

    def test_it_asks_the_endpoint_for_the_access_token(self, exchange):
        """Reading ``id_token`` here is exactly what made the forwarded credential the wrong
        one: an ID token's audience is this dashboard client, not the resource server."""
        _grant(exchange)
        assert exchange["token_key"] == "access_token"

    def test_the_id_token_is_still_verified_for_flow_integrity(self, exchange):
        """A tampered or truncated code exchange cannot yield a valid ID token."""
        obj, _ = _grant(exchange, payload={"access_token": "AT", "id_token": "ID-real"})
        assert obj.verified_id_tokens == ["ID-real"]

    def test_an_absent_id_token_is_not_fatal(self, exchange):
        """Some deployments omit it; the access token is the credential either way."""
        obj, _ = _grant(exchange, payload={"access_token": "AT"})
        assert obj.verified_id_tokens == []

    def test_the_id_tokens_claims_are_not_used(self, exchange):
        """The access token carries the same realm user claims; mixing them in would create a
        second, unverified source of truth."""
        obj, _ = _grant(exchange, claims={"sub": "from-access-token"},
                        id_token_claims={"sub": "from-id-token"},
                        payload={"access_token": "AT", "id_token": "ID"})
        assert obj.session_args[2] == {"sub": "from-access-token"}

    def test_a_rotated_refresh_token_is_preferred(self, exchange):
        obj, _ = _grant(exchange, payload={"access_token": "AT", "refresh_token": "RT-new"},
                        previous_refresh_token="RT-old")
        assert obj.session_args[1] == "RT-new"

    def test_an_idp_that_does_not_rotate_keeps_the_previous_refresh_token(self, exchange):
        obj, _ = _grant(exchange, payload={"access_token": "AT"},
                        previous_refresh_token="RT-old")
        assert obj.session_args[1] == "RT-old"

    def test_a_missing_access_token_is_an_error(self, exchange):
        """An ID-token-only response would leave nothing a resource server accepts."""
        with pytest.raises(InvalidCodeError, match="missing the requested token"):
            _grant(exchange, payload={"id_token": "ID-only"})


# --------------------------------------------------------------------------- token screening

class TestWhichTokensMayOpenASession:
    """Signature verification is not authorisation: the realm signs *every* token the same way,
    including client credentials. These checks are what separate a person from a machine."""

    @pytest.mark.parametrize("typ", [None, "Bearer", "bearer", "at+jwt", "AT+JWT"])
    def test_access_token_types_are_accepted(self, exchange, typ):
        claims = {"sub": "user-1"}
        if typ is not None:
            claims["typ"] = typ
        assert _grant(exchange, claims=claims)[1] is not None

    @pytest.mark.parametrize("typ", ["ID", "id", "JWT", "access"])
    def test_non_access_token_types_are_refused(self, exchange, typ):
        with pytest.raises(InvalidCodeError, match="not an access token"):
            _grant(exchange, claims={"sub": "user-1", "typ": typ})

    def test_a_token_with_no_typ_is_accepted(self, exchange):
        """A realm may be configured without the header; absence is not evidence of misuse."""
        assert _grant(exchange, claims={"sub": "user-1"})[1] is not None

    def test_another_clients_token_is_refused(self, exchange):
        """``azp`` is the client the token was issued to, so another client's token cannot be
        replayed here even when the audience entry is shared."""
        with pytest.raises(InvalidCodeError, match="issued to client"):
            _grant(exchange, claims={"sub": "user-1", "azp": "some-other-client"})

    def test_our_own_client_is_accepted(self, exchange):
        assert _grant(exchange, claims={"sub": "user-1", "azp": CLIENT_ID})[1] is not None

    @pytest.mark.parametrize("claim", ["preferred_username", "sub"])
    def test_a_service_account_token_is_refused(self, exchange, claim):
        """A ``client_credentials`` token is shared by every holder, so opening a person's
        session with one would make the integration account a login."""
        with pytest.raises(InvalidCodeError, match="service-account"):
            _grant(exchange, claims={claim: f"{_SERVICE_ACCOUNT_PREFIX}{CLIENT_ID}"})

    def test_preferred_username_takes_precedence_over_sub(self, exchange):
        """Keycloak puts the human name in ``preferred_username``, so a machine token whose
        ``sub`` is a bare UUID must still be caught by that field."""
        with pytest.raises(InvalidCodeError, match="service-account"):
            _grant(exchange, claims={
                "sub": "uuid-1234",
                "preferred_username": f"{_SERVICE_ACCOUNT_PREFIX}{CLIENT_ID}"})

    def test_a_user_whose_name_merely_starts_with_the_word_is_not_locked_out(self, exchange):
        assert _grant(exchange, claims={"sub": "service-accounting-team"})[1] is not None

    def test_the_constant_lists_the_access_token_types(self):
        assert {"bearer", "at+jwt"} <= _ACCESS_TOKEN_TYPES


# --------------------------------------------------------------------------- the wiring

class TestTheMixinIsLayeredFirst:
    """The plugin puts ``AccessTokenSessionGrant`` before ``SelfHostedOIDCProvider``; if that
    order flips, the parent's ID-token behaviour silently wins again."""

    def test_the_keycloak_provider_resolves_the_mixin_methods(self):
        from plugins.dashboard_auth.keycloak import KeycloakDashboardAuthProvider

        assert KeycloakDashboardAuthProvider._grant is AccessTokenSessionGrant._grant
        assert KeycloakDashboardAuthProvider._claims_for is AccessTokenSessionGrant._claims_for
        assert (KeycloakDashboardAuthProvider._reject_non_session_token
                is AccessTokenSessionGrant._reject_non_session_token)

    def test_the_parent_alias_is_what_makes_verify_session_use_our_claims(self):
        """``SelfHostedOIDCProvider`` aliases ``_claims_for = _verify_id_token`` at class
        creation, so overriding ``_verify_id_token`` alone would never run."""
        from plugins.dashboard_auth.self_hosted import SelfHostedOIDCProvider

        assert SelfHostedOIDCProvider._claims_for is SelfHostedOIDCProvider._verify_id_token
        # The mixin defines its own ``_claims_for`` and no ``_verify_id_token`` at all, so the
        # alias above can never route past it.
        assert "_claims_for" in AccessTokenSessionGrant.__dict__
        assert "_verify_id_token" not in AccessTokenSessionGrant.__dict__