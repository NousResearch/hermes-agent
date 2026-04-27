"""Bot Framework auth tests for the Microsoft Teams adapter."""

from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from gateway.platforms.msteams import auth


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHTTPClient:
    def __init__(self, get_payloads=None, post_payloads=None):
        self.get_payloads = list(get_payloads or [])
        self.post_payloads = list(post_payloads or [])
        self.gets = []
        self.posts = []

    async def get(self, url):
        self.gets.append(url)
        payload = self.get_payloads.pop(0)
        return _FakeResponse(payload)

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        payload = self.post_payloads.pop(0)
        return _FakeResponse(payload)


def _rsa_key_and_jwk(kid="kid-1"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return private_key, jwk


def _signed_token(private_key, *, app_id="app-id", kid="kid-1", service_url=None):
    payload = {
        "iss": "https://api.botframework.com",
        "aud": app_id,
        "exp": int(time.time()) + 300,
    }
    if service_url:
        payload["serviceurl"] = service_url
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


@pytest.mark.asyncio
async def test_jwt_validator_accepts_bot_framework_token():
    private_key, jwk = _rsa_key_and_jwk()
    metadata = {"jwks_uri": "https://login.botframework.com/keys"}
    jwks = {"keys": [jwk]}
    client = _FakeHTTPClient(get_payloads=[metadata, jwks])
    validator = auth.BotFrameworkJWTValidator("app-id", client)
    token = _signed_token(
        private_key,
        app_id="app-id",
        service_url="https://smba.trafficmanager.net/amer/",
    )

    assert await validator.validate_authorization_header(
        f"Bearer {token}",
        service_url="https://smba.trafficmanager.net/amer/",
    ) is True
    assert client.gets == [
        auth.BOT_FRAMEWORK_OPENID_METADATA_URL,
        "https://login.botframework.com/keys",
    ]


@pytest.mark.asyncio
async def test_jwt_validator_rejects_missing_bearer_header():
    private_key, jwk = _rsa_key_and_jwk()
    client = _FakeHTTPClient(
        get_payloads=[
            {"jwks_uri": "https://login.botframework.com/keys"},
            {"keys": [jwk]},
        ]
    )
    validator = auth.BotFrameworkJWTValidator("app-id", client)
    token = _signed_token(private_key)

    assert await validator.validate_authorization_header(token) is False
    assert client.gets == []


@pytest.mark.asyncio
async def test_jwt_validator_rejects_wrong_audience():
    private_key, jwk = _rsa_key_and_jwk()
    client = _FakeHTTPClient(
        get_payloads=[
            {"jwks_uri": "https://login.botframework.com/keys"},
            {"keys": [jwk]},
        ]
    )
    validator = auth.BotFrameworkJWTValidator("expected-app", client)
    token = _signed_token(private_key, app_id="other-app")

    assert await validator.validate_authorization_header(f"Bearer {token}") is False


@pytest.mark.asyncio
async def test_token_provider_posts_client_credentials_and_caches():
    client = _FakeHTTPClient(
        post_payloads=[{"access_token": "outbound-token", "expires_in": 3600}]
    )
    provider = auth.BotFrameworkTokenProvider(
        app_id="app-id",
        app_password="secret",
        tenant_id="tenant-id",
        http_client=client,
    )

    token1 = await provider.get_token(auth.BOT_FRAMEWORK_SCOPE)
    token2 = await provider.get_token(auth.BOT_FRAMEWORK_SCOPE)

    assert token1 == token2 == "outbound-token"
    assert len(client.posts) == 1
    url, kwargs = client.posts[0]
    assert url == "https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token"
    assert kwargs["data"]["grant_type"] == "client_credentials"
    assert kwargs["data"]["client_id"] == "app-id"
    assert kwargs["data"]["client_secret"] == "secret"
    assert kwargs["data"]["scope"] == auth.BOT_FRAMEWORK_SCOPE


def test_token_provider_requires_app_password():
    with pytest.raises(auth.AuthError, match="MSTEAMS_APP_PASSWORD"):
        auth.BotFrameworkTokenProvider(
            app_id="app-id",
            app_password="",
            http_client=_FakeHTTPClient(),
        )
