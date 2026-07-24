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


def test_oauth_client_does_not_inherit_env_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    client = build_anthropic_client(OAUTH_TOKEN)
    assert client.auth_token == OAUTH_TOKEN
    assert client.api_key is None


def test_regular_api_key_client_unaffected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-other")
    client = build_anthropic_client(API_KEY)
    assert client.api_key == API_KEY
