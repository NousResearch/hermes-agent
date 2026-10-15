"""Key-less non-native anthropic_messages providers resolve their own key (#139155).

Kanban workers construct ``AIAgent(api_key=None)``; the anthropic_messages branch then froze
``api_key=""`` for any provider that is not native Anthropic (the ANTHROPIC_TOKEN fallback is
native-only, #1739), so the SDK sent ``Authorization: Bearer `` and every worker request died
as a misleading transient ``APIConnectionError``. The OpenAI-wire branch already resolves such
keys through the provider router; these tests pin the same fallback for the Anthropic wire —
and that it is never consulted for native Anthropic or an explicit key.
"""

import secrets
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.agent_init import _init_anthropic_client

MODEL = "MiniMax-M2.7"
BASE_URL = "https://api.minimax.io/anthropic"
# Dummy markers (not credentials), generated per run so no literal key-looking value lives here.
ROUTER_KEY = "router-" + secrets.token_hex(12)
EXPLICIT_KEY = "explicit-" + secrets.token_hex(12)


def _agent(provider, api_key=None):
    return SimpleNamespace(
        provider=provider,
        model=MODEL,
        api_key=api_key,
        quiet_mode=True,
        client="untouched",
        _client_kwargs={},
        _anthropic_base_url=None,
        _anthropic_api_key=None,
        _anthropic_client=None,
        _is_anthropic_oauth=False,
    )


@pytest.fixture
def wire(monkeypatch):
    """Stub the wire construction; record the (key, base_url) ``build_anthropic_client`` got."""
    built = {}
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda key, base_url, timeout=None: (
            built.update(key=key, base_url=base_url) or "built"
        ),
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda **_: None
    )
    return built


def _router(monkeypatch, client):
    mock = MagicMock(return_value=(client, MODEL))
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", mock)
    return mock


def test_keyless_custom_provider_resolves_its_router_key(monkeypatch, wire):
    _router(monkeypatch, SimpleNamespace(api_key=ROUTER_KEY))
    agent = _agent("custom:minimax")
    _init_anthropic_client(agent, None, BASE_URL, None)
    assert agent.api_key == ROUTER_KEY
    assert agent._anthropic_api_key == ROUTER_KEY
    assert wire["key"] == ROUTER_KEY


def test_router_without_credentials_keeps_the_empty_key(monkeypatch, wire):
    _router(monkeypatch, None)
    agent = _agent("custom:minimax")
    _init_anthropic_client(agent, None, BASE_URL, None)
    assert agent.api_key == ""
    assert wire["key"] == ""


def test_router_failure_keeps_the_empty_key(monkeypatch, wire):
    mock = MagicMock(side_effect=RuntimeError("router unavailable"))
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", mock)
    agent = _agent("custom:minimax")
    _init_anthropic_client(agent, None, BASE_URL, None)
    assert agent.api_key == ""


def test_native_anthropic_never_consults_the_router(monkeypatch, wire):
    router = _router(monkeypatch, SimpleNamespace(api_key=ROUTER_KEY))
    agent = _agent("anthropic")
    _init_anthropic_client(agent, None, None, None)
    router.assert_not_called()
    assert agent.api_key == ""


def test_explicit_key_skips_the_router(monkeypatch, wire):
    router = _router(monkeypatch, SimpleNamespace(api_key=ROUTER_KEY))
    agent = _agent("custom:minimax", api_key=EXPLICIT_KEY)
    _init_anthropic_client(agent, EXPLICIT_KEY, BASE_URL, None)
    router.assert_not_called()
    assert agent.api_key == EXPLICIT_KEY
