"""Regression test: OAuth/Bearer clients must not pick up ANTHROPIC_API_KEY.

When build_anthropic_client() authenticates via ``auth_token`` and omits
``api_key``, the Anthropic SDK constructor falls back to reading
ANTHROPIC_API_KEY from the environment. The client then sends BOTH
X-Api-Key and Authorization headers, and Anthropic bills the API key
instead of the OAuth subscription. The adapter must null out
``client.api_key`` on that path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent.anthropic_adapter import build_anthropic_client

OAUTH_TOKEN = "sk-ant-oat01-test-token"
API_KEY = "sk-ant-api03-env-key"


@pytest.mark.parametrize(
    "token, base_url",
    [
        pytest.param(OAUTH_TOKEN, None, id="native-oauth"),
        pytest.param(
            "minimax-secret-123", "https://api.minimax.io/anthropic", id="minimax"
        ),
        pytest.param(
            "azure-foundry-secret-123",
            "https://my-resource.openai.azure.com/anthropic",
            id="azure",
        ),
    ],
)
def test_bearer_client_does_not_inherit_env_api_key(monkeypatch, token, base_url):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    with build_anthropic_client(token, base_url=base_url) as client:
        assert client.auth_token == token
        assert client.api_key is None
        headers = {key.lower(): value for key, value in client.default_headers.items()}
        assert headers["authorization"] == f"Bearer {token}"
        assert "x-api-key" not in headers


def test_regular_api_key_client_unaffected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-other")
    client = build_anthropic_client(API_KEY)
    assert client.api_key == API_KEY
