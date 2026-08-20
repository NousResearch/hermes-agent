"""MoA client lifecycle must never fall through to the OpenAI SDK factory."""

from __future__ import annotations

import pytest

from run_agent import AIAgent


def _bare_moa_agent(client):
    agent = object.__new__(AIAgent)
    setattr(agent, "provider", "moa")
    setattr(agent, "client", client)
    setattr(agent, "_client_kwargs", {})
    return agent


def test_replace_primary_client_is_a_noop_for_moa() -> None:
    facade = object()
    agent = _bare_moa_agent(facade)

    assert agent._replace_primary_openai_client(reason="stale_stream") is True
    assert agent.client is facade


def test_ensure_primary_client_returns_existing_moa_facade() -> None:
    facade = object()
    agent = _bare_moa_agent(facade)

    assert agent._ensure_primary_openai_client(reason="request") is facade


def test_ensure_primary_client_fails_with_moa_specific_error_when_missing() -> None:
    agent = _bare_moa_agent(None)

    with pytest.raises(RuntimeError, match="MoA primary client is None"):
        agent._ensure_primary_openai_client(reason="request")
