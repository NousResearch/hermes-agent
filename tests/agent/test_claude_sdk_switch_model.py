"""Regression coverage for in-place switches into the Claude Agent SDK lane.

The SDK is a subprocess-owned subscription transport and intentionally has no
HTTP base URL.  Generic provider-switch validation must clear the previous
provider's endpoint rather than reject a valid SDK route or build an OpenAI
client against the stale endpoint.
"""

from __future__ import annotations

import types


def _agent():
    return types.SimpleNamespace(
        model="gpt-5.6-terra",
        provider="openai-codex",
        requested_provider="openai-codex",
        api_mode="codex_responses",
        api_key="old-codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=object(),
        _client_kwargs={"base_url": "https://chatgpt.com/backend-api/codex"},
        _config_context_length=128000,
        _transport_cache={},
        _credential_pool=None,
        _credential_pool_entry_id=None,
        # Upstream's transactional switch now refreshes this flag before the
        # SDK client branch. Supply the new production contract so this test's
        # intentionally broad post-swap AttributeError catch cannot roll the
        # core fields back before reaching the behavior it is meant to pin.
        _read_reasoning_echo_from_config=lambda: False,
        quiet_mode=True,
    )


def test_switch_to_claude_agent_sdk_allows_empty_endpoint(monkeypatch):
    """SDK switch clears a prior HTTP endpoint and installs no HTTP client."""
    from agent import agent_runtime_helpers as arh

    monkeypatch.setattr(arh, "load_pool", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(
        arh, "sync_credential_pool_entry_id", lambda *_a, **_kw: None, raising=False
    )
    agent = _agent()

    try:
        arh.switch_model(
            agent,
            new_model="claude-sonnet-5",
            new_provider="claude-agent-sdk",
            api_key="",
            base_url="",
            api_mode="claude_agent_sdk",
        )
    except AttributeError:
        # The minimal fixture lacks post-swap compressor/context helpers.  The
        # core swap completed before those optional refreshes.
        pass

    assert agent.model == "claude-sonnet-5"
    assert agent.provider == "claude-agent-sdk"
    assert agent.api_mode == "claude_agent_sdk"
    assert agent.base_url == ""
    assert agent.api_key == "claude-subscription-oauth"
    assert agent.client is None
    assert agent._client_kwargs == {}
