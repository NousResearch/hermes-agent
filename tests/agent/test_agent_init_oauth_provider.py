"""agent_init must hand the Anthropic client a token PROVIDER, not a frozen string.

This is where the 401 "OAuth access token has been revoked" came from: the token was
resolved once at session start and baked into the SDK client, so a mid-session refresh by
`claude` never reached the wire.
"""
import types

import pytest

from agent import agent_init


class _Agent:
    """The attributes _init_anthropic_client touches."""

    def __init__(self, provider="anthropic", model="claude-opus-5"):
        self.provider = provider
        self.model = model
        self.quiet_mode = True


@pytest.fixture
def captured(monkeypatch):
    """Capture the credential handed to build_anthropic_client."""
    seen = {}

    def fake_build(credential, base_url=None, timeout=None, **kw):
        seen["credential"] = credential
        return types.SimpleNamespace(base_url=base_url)

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", fake_build)

    return seen


def test_a_subscription_token_is_passed_as_a_provider(captured, monkeypatch):
    """The regression: an OAuth token must arrive as a callable, re-read per request."""
    tokens = iter(["sk-ant-oat01-first", "sk-ant-oat01-second"])
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda *a, **k: next(tokens)
    )

    agent = _Agent()
    agent_init._init_anthropic_client(agent, None, "https://api.anthropic.com", None)
    credential = captured["credential"]

    assert callable(credential), "a frozen string cannot follow a refresh"
    # The first token was consumed resolving the session's identity; the provider must
    # return what is on disk NOW, not that first value.
    assert credential() == "sk-ant-oat01-second"


def test_an_explicit_api_key_stays_a_string(captured, monkeypatch):
    """An ANTHROPIC_API_KEY does not expire, so it must not be wrapped."""
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda *a, **k: "unused"
    )

    agent = _Agent()
    agent_init._init_anthropic_client(agent, "sk-ant-api03-static", "https://api.anthropic.com", None)

    assert captured["credential"] == "sk-ant-api03-static"


def test_a_third_party_endpoint_stays_a_string(captured, monkeypatch):
    """Third-party anthropic_messages providers must never trip the OAuth path (#1739)."""
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda *a, **k: "sk-ant-oat01-x"
    )

    agent = _Agent(provider="minimax")
    agent_init._init_anthropic_client(agent, "minimax-key", "https://api.minimax.io/anthropic", None)

    assert captured["credential"] == "minimax-key"


def test_a_read_failure_falls_back_to_the_known_token(captured, monkeypatch):
    """A credential file that goes unreadable mid-session must not crash the turn."""
    state = {"first": True}

    def flaky(*a, **k):
        if state["first"]:
            state["first"] = False
            return "sk-ant-oat01-known"
        raise OSError("credentials file vanished")

    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", flaky)

    agent = _Agent()
    agent_init._init_anthropic_client(agent, None, "https://api.anthropic.com", None)

    assert captured["credential"]() == "sk-ant-oat01-known"
