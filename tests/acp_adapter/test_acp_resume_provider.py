"""Resuming an ACP session must not lose a named custom provider's credentials.

``_persist`` records the agent's provider KIND ("custom"), not the name of the
entry under ``providers:``. Feeding that kind back into ``resolve_runtime_provider``
on restore resolves a different endpoint and no api_key — and only base_url and
api_mode have a stored value to fall back on, so the resumed session sends its
model somewhere that has never heard of it. Every turn then answers with that
endpoint's rejection, as ordinary assistant text, for the life of the session.
"""

from __future__ import annotations

import pytest

from acp_adapter.session import SessionManager


GATEWAY_URL = "https://gateway.example.com/v1"


class RecordingAgent:
    """Stand-in for AIAgent that keeps the kwargs it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.session_id = kwargs.get("session_id")


@pytest.fixture
def resolver_calls(monkeypatch):
    """Patch the pieces ``_make_agent`` reaches for, and record the resolution."""
    calls: list[dict] = []

    def fake_resolve(*, requested=None, explicit_base_url=None, **_kw):
        calls.append({"requested": requested, "explicit_base_url": explicit_base_url})
        # Only the provider's own name resolves to its credentials. The bare
        # kind resolves to whatever else is configured, without a key — this is
        # what the real resolver does with a named custom provider.
        if requested == "my-gateway":
            return {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": GATEWAY_URL,
                "api_key": "gateway-key",
            }
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://elsewhere.example.com/v1",
            "api_key": "",
        }

    def fake_load_config():
        return {"model": {"default": "some-model", "provider": "my-gateway"}}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)
    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr("run_agent.AIAgent", RecordingAgent)
    return calls


def test_restore_does_not_feed_the_provider_kind_back_in(resolver_calls, tmp_path):
    """A resumed session keeps the credentials it was created with."""
    manager = SessionManager(db=None)

    agent = manager._make_agent(
        session_id="s-1",
        cwd=str(tmp_path),
        model="some-model",
        # What _restore reads back out of the session row.
        requested_provider="custom",
        base_url=GATEWAY_URL,
        api_mode="chat_completions",
    )

    assert resolver_calls, "the provider was never resolved"
    assert resolver_calls[0]["requested"] == "my-gateway", (
        '"custom" is a provider kind, not a configured provider name; resolving '
        "it picks a different endpoint and no credentials"
    )
    assert resolver_calls[0]["explicit_base_url"] == GATEWAY_URL, (
        "the session's own endpoint must be passed so a session pinned to one "
        "other than the configured default keeps it"
    )
    assert agent.kwargs["api_key"] == "gateway-key"
    assert agent.kwargs["base_url"] == GATEWAY_URL


def test_a_fresh_session_still_uses_the_configured_provider(resolver_calls, tmp_path):
    """The path that always worked keeps working: nothing stored, nothing forced."""
    manager = SessionManager(db=None)

    agent = manager._make_agent(session_id="s-2", cwd=str(tmp_path))

    assert resolver_calls[0]["requested"] == "my-gateway"
    assert agent.kwargs["api_key"] == "gateway-key"


def test_an_explicitly_named_provider_is_still_honoured(resolver_calls, tmp_path):
    """Only the ambiguous kind is overridden — a real provider name is a request."""
    manager = SessionManager(db=None)

    manager._make_agent(
        session_id="s-3",
        cwd=str(tmp_path),
        requested_provider="my-gateway",
        base_url=GATEWAY_URL,
    )

    assert resolver_calls[0]["requested"] == "my-gateway"
