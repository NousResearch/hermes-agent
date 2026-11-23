"""A native-anthropic client at an Azure Foundry host with no key of its own must never be filled
with the Claude Code OAuth token (#154242). Covers the three builders that fill an empty key:
agent init, fallback activation and a mid-session model switch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
OAUTH_TOKEN = "claude-code-oauth-token"


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda **_k: OAUTH_TOKEN)
    calls = []

    def _fake_build(token, base_url, **_kwargs):
        calls.append((token, base_url))
        return MagicMock()

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", _fake_build)
    return calls


def _agent(**extra):
    return SimpleNamespace(provider="anthropic", model="claude-x", quiet_mode=True, api_key="", **extra)


def test_agent_init_never_fills_a_foundry_client_with_the_oauth_token(built):
    from agent.agent_init import _init_anthropic_client
    _init_anthropic_client(_agent(), "", FOUNDRY_URL, None)
    assert built and all(token != OAUTH_TOKEN for token, _url in built)


def test_fallback_activation_never_fills_a_foundry_client_with_the_oauth_token(built):
    from agent.client_lifecycle import _swap_fallback_clients
    fb_client = SimpleNamespace(api_key="")
    _swap_fallback_clients(_agent(), fb_client, "anthropic", "claude-x", FOUNDRY_URL, "anthropic_messages")
    assert built and all(token != OAUTH_TOKEN for token, _url in built)


def test_model_switch_never_fills_a_foundry_client_with_the_oauth_token(built):
    from agent.agent_runtime_helpers import _build_switched_client
    _build_switched_client(_agent(), "anthropic", "", FOUNDRY_URL, "anthropic_messages", "anthropic")
    assert built and all(token != OAUTH_TOKEN for token, _url in built)


def test_native_anthropic_endpoint_still_falls_back_to_the_oauth_token(built):
    from agent.agent_init import _init_anthropic_client
    _init_anthropic_client(_agent(), "", "https://api.anthropic.com", None)
    assert built == [(OAUTH_TOKEN, "https://api.anthropic.com")]
