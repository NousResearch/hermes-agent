"""On the gateway's Webapp behind the sign-in gate, a Portal token-exchange token works on its own.

The hosted dashboard signs the user in on its own host, trades that sign-in at Nous Portal (RFC 8693
token exchange) for a token whose audience is this agent, and relays the browser's requests here with
it as a bearer. Portal marks such a token with an ``act`` claim. The Webapp's listener lets such a
token past the rule that keeps a browser login's tokens to its cookies only when the Nous provider
verifies it as its own exchanged token: a dashboard token for this agent without ``act`` (what a
browser login holds) is still refused on its own; a token with ``act`` that is signed by another key,
minted for another agent, or issued by another provider gets nothing; a Portal outage refuses it.
"""
import base64
import json
import time

import jwt
from jwt.algorithms import RSAAlgorithm
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.hermes_cli import test_webapp_gated_sign_out as webapp_gate
from tests.hermes_cli.test_webapp_gated_sign_out import _bearer, _gateway_socket_with, _refused

gated = webapp_gate.gated  # the Webapp behind the sign-in gate, password and OAuth providers registered

PORTAL = "https://portal.example.test"
AGENT = "agent:inst-alpha"
ACTOR = {"sub": "hermes-webapp"}


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def portal_key():
    return _key()


@pytest.fixture
def nous(gated, portal_key):
    """The bundled Nous provider, registered beside the gate's own, trusting *portal_key* as Portal's
    signing key. Only its key lookup is replaced: the JWKS fetch is the one network call."""
    from hermes_cli.dashboard_auth import register_provider
    from plugins.dashboard_auth.nous import NousDashboardAuthProvider

    provider = NousDashboardAuthProvider(client_id=AGENT, portal_url=PORTAL)
    public = jwt.PyJWK.from_dict(json.loads(RSAAlgorithm.to_jwk(portal_key.public_key())))

    class PortalKeys:
        def get_signing_key_from_jwt(self, _token):
            return public

    provider._jwks_client = PortalKeys()
    register_provider(provider)
    return provider


def _token(key, *, act=ACTOR, aud=AGENT, algorithm="RS256", **extra):
    """A Portal dashboard token for *aud* (the claims the agent's own grant mints), with *act* when given."""
    now = int(time.time())
    claims = {"iss": PORTAL, "aud": aud, "sub": "user_1", "iat": now, "exp": now + 900,
              "scope": "agent_dashboard:access", "agent_instance_id": aud.split(":", 1)[1],
              "oauth_contract_version": 1, "org_id": "org_1", **extra}
    if act is not None:
        claims["act"] = act
    signing = key if algorithm.startswith("RS") else "k" * 32
    return jwt.encode(claims, signing, algorithm=algorithm)


def test_a_dashboard_token_without_act_is_still_refused_on_its_own(gated, nous, portal_key):
    """What a browser login on this listener holds: Portal's token for this agent, no ``act``."""
    token = _token(portal_key, act=None)
    assert nous.verify_session(access_token=token) is not None  # a valid token, refused only for being alone
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))
    assert _gateway_socket_with(gated, token) == 4401


def test_an_exchanged_token_works_on_its_own_and_the_provider_verifies_it(gated, nous, portal_key):
    token = _token(portal_key)
    me = gated.browser().get("/api/auth/me", headers=_bearer(token))
    assert me.status_code == 200, me.text
    assert (me.json()["provider"], me.json()["user_id"]) == ("nous", "user_1")
    assert _gateway_socket_with(gated, token) is None


@pytest.mark.parametrize("forged", ["other_key", "other_agent", "expired"])
def test_an_exchanged_token_the_provider_does_not_verify_gets_nothing(gated, nous, portal_key, forged):
    token = {
        "other_key": lambda: _token(_key()),
        "other_agent": lambda: _token(portal_key, aud="agent:inst-beta"),
        "expired": lambda: _token(portal_key, exp=int(time.time()) - 3600),
    }[forged]()
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))
    assert _gateway_socket_with(gated, token) == 4401


def test_the_refresh_route_refuses_an_exchanged_token_before_asking_any_provider(gated, nous, portal_key,
                                                                                 monkeypatch):
    """An exchanged token is an access token: the refresh route keeps main's rule and refuses it
    without calling Portal, so it cannot turn into a refresh grant or a 503."""
    calls = []
    monkeypatch.setattr(nous, "refresh_session", lambda **kw: calls.append(kw))
    response = gated.browser().post("/auth/native/refresh", json={"refresh_token": _token(portal_key)})
    assert response.status_code == 401, response.text
    assert calls == []


def _b64(value) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()


def test_another_issuers_token_with_an_actor_stays_bound_to_its_login(gated, nous):
    """What ``act`` means is Portal's to say: a provider that verifies JWTs carrying ``act`` but
    does not vouch for exchanged tokens (a self-hosted OIDC issuer) gets no exemption."""
    from hermes_cli.dashboard_auth import Session, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    other_key = _key()

    class OtherIssuer(StubAuthProvider):
        name = "other-oidc"
        display_name = "Other OIDC"

        def verify_session(self, *, access_token):
            try:
                claims = jwt.decode(access_token, other_key.public_key(), algorithms=["RS256"], audience=AGENT)
            except jwt.PyJWTError:
                return None
            return Session(user_id=claims["sub"], email="", display_name="", org_id="",
                           provider=self.name, expires_at=claims["exp"], access_token=access_token,
                           refresh_token="")

    register_provider(OtherIssuer())
    token = _token(other_key)
    assert OtherIssuer().verify_session(access_token=token) is not None
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))
    assert _gateway_socket_with(gated, token) == 4401


def test_an_unreachable_portal_refuses_an_exchanged_token(gated, nous, portal_key, monkeypatch):
    from hermes_cli.dashboard_auth import ProviderError

    def unreachable(*, token, unverified_claims=None):
        raise ProviderError("JWKS unreachable")

    monkeypatch.setattr(nous, "is_exchanged_token", unreachable)
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(_token(portal_key))))


@pytest.mark.parametrize("addressed", ["another_issuer", "another_agent", "another_instance"])
def test_a_token_addressed_elsewhere_fetches_no_signing_key(gated, nous, portal_key, addressed):
    """An anonymous caller cannot make the listener look up Portal's keys for a token that is not
    addressed to this agent: the provider turns it away on its unverified claims."""
    lookups = []

    class CountingKeys:
        def get_signing_key_from_jwt(self, token):
            lookups.append(token)
            raise AssertionError("no key lookup expected")

    nous._jwks_client = CountingKeys()
    token = {
        "another_issuer": lambda: _token(portal_key, iss="https://elsewhere.example"),
        "another_agent": lambda: _token(portal_key, aud="agent:inst-beta"),
        "another_instance": lambda: _token(portal_key, agent_instance_id="inst-beta"),
    }[addressed]()
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))
    assert lookups == []


@pytest.mark.parametrize("header", [{"alg": []}, {"alg": {"x": 1}}, {"alg": None}])
def test_a_malformed_header_is_refused_not_an_error(gated, nous, header):
    token = f"{_b64(header)}.{_b64({'act': ACTOR})}.c2ln"
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))


def test_deeply_nested_claims_are_refused_not_an_error(gated, nous):
    deep = base64.urlsafe_b64encode(b"[" * 5000 + b"]" * 5000).rstrip(b"=").decode()
    token = f"{_b64({'alg': 'RS256'})}.{deep}.c2ln"
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(token)))


@pytest.mark.parametrize(("label", "token"), [
    ("no act", f"{_b64({'alg': 'RS256'})}.{_b64({'sub': 'u'})}.c2ln"),
    ("act not an object", f"{_b64({'alg': 'RS256'})}.{_b64({'act': 'hermes-webapp'})}.c2ln"),
    ("act without a subject", f"{_b64({'alg': 'RS256'})}.{_b64({'act': {}})}.c2ln"),
    ("symmetric algorithm", f"{_b64({'alg': 'HS256'})}.{_b64({'act': ACTOR})}.c2ln"),
    ("unsigned", f"{_b64({'alg': 'none'})}.{_b64({'act': ACTOR})}."),
    ("padded segments", f"{_b64({'alg': 'RS256'})}.{_b64({'act': ACTOR})}=.c2ln"),
    ("not a JWT", "opaque-session-token-" + "x" * 32),
    ("claims not JSON", f"{_b64({'alg': 'RS256'})}.bm90LWpzb24.c2ln"),
])
def test_only_a_signed_jwt_with_an_actor_reads_as_exchanged(label, token):
    from hermes_cli.webapp_sign_in import reads_as_exchanged

    assert not reads_as_exchanged(token), label
    assert reads_as_exchanged(f"{_b64({'alg': 'RS256'})}.{_b64({'act': ACTOR})}.c2ln")


def test_a_password_login_token_is_never_read_as_exchanged(gated):
    """The password provider's tokens are base64 that its decoder reads leniently; no spelling of one
    reads as an exchanged token, so the binding still keeps it to its cookies."""
    from hermes_cli.webapp_sign_in import reads_as_exchanged
    from tests.hermes_cli.test_webapp_gated_sign_out import _sign_in_with_password, _tokens

    browser = gated.browser()
    _sign_in_with_password(browser)
    access, refresh = _tokens(browser)
    assert not reads_as_exchanged(access) and not reads_as_exchanged(refresh)
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(access)))
