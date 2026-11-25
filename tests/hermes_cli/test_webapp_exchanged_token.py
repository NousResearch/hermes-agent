"""On the gateway's Webapp behind the sign-in gate, a Portal token-exchange token works on its own.

The hosted dashboard signs the user in on its own host, trades that sign-in at Nous Portal (RFC 8693
token exchange) for a token whose audience is this agent, and relays the browser's requests here with
it as a bearer. Portal marks such a token with an ``act`` claim. The Webapp's listener lets a token
carrying ``act`` past the rule that keeps a browser login's tokens to its cookies, and the Nous
provider then verifies it like any bearer: a dashboard token for this agent without ``act`` (what a
browser login holds) is still refused on its own, and a token with ``act`` that is signed by another
key, or minted for another agent, gets nothing.
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
    from hermes_cli.webapp_sign_in import is_exchanged_token

    assert not is_exchanged_token(token), label
    assert is_exchanged_token(f"{_b64({'alg': 'RS256'})}.{_b64({'act': ACTOR})}.c2ln")


def test_a_password_login_token_is_never_read_as_exchanged(gated):
    """The password provider's tokens are base64 that its decoder reads leniently; no spelling of one
    reads as an exchanged token, so the binding still keeps it to its cookies."""
    from hermes_cli.webapp_sign_in import is_exchanged_token
    from tests.hermes_cli.test_webapp_gated_sign_out import _sign_in_with_password, _tokens

    browser = gated.browser()
    _sign_in_with_password(browser)
    access, refresh = _tokens(browser)
    assert not is_exchanged_token(access) and not is_exchanged_token(refresh)
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(access)))
