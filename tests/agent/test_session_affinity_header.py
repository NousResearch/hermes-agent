"""Tests for per-provider session_affinity_header in agent and auxiliary requests.

Closes #104449: custom gateways (such as LiteLLM x-litellm-session-id) receive
the turn's affinity scope to pin requests to warm prompt caches.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from hermes_cli.config_providers import (
    _normalize_custom_provider_entry,
    get_custom_provider_session_affinity_header,
)
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hello"}]


def _agent(provider, model, base_url, api_mode=None, session_id="sess-affinity-test-1"):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
    )
    if api_mode:
        agent.api_mode = api_mode
        agent._transport = None
        agent._anthropic_base_url = base_url
    return agent


def test_normalization_and_camel_case_alias():
    entry = _normalize_custom_provider_entry(
        {
            "name": "litellm-lan",
            "base_url": "http://localhost:4000/v1",
            "sessionAffinityHeader": "x-litellm-session-id",
        }
    )
    assert entry is not None
    assert entry.get("session_affinity_header") == "x-litellm-session-id"


def test_lookup_by_provider_or_base_url():
    custom_providers = [
        {
            "name": "litellm-lan",
            "provider_key": "litellm-lan",
            "base_url": "http://localhost:4000/v1",
            "session_affinity_header": "x-litellm-session-id",
        }
    ]
    # Matched by provider name
    assert (
        get_custom_provider_session_affinity_header(
            provider="litellm-lan", custom_providers=custom_providers
        )
        == "x-litellm-session-id"
    )
    # Matched by base_url
    assert (
        get_custom_provider_session_affinity_header(
            base_url="http://localhost:4000/v1", custom_providers=custom_providers
        )
        == "x-litellm-session-id"
    )
    # Unmatched
    assert (
        get_custom_provider_session_affinity_header(
            provider="unconfigured",
            base_url="http://other:8000/v1",
            custom_providers=custom_providers,
        )
        is None
    )


def test_main_turn_sends_session_affinity_header():
    mock_providers = [
        {
            "name": "litellm-lan",
            "provider_key": "litellm-lan",
            "base_url": "http://localhost:4000/v1",
            "session_affinity_header": "x-litellm-session-id",
        }
    ]
    agent = _agent("litellm-lan", "claude-3-5-sonnet", "http://localhost:4000/v1")
    with patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=mock_providers,
    ):
        kwargs = build_api_kwargs(agent, _MSGS)
        extra = kwargs.get("extra_headers") or {}
        assert extra.get("x-litellm-session-id") == "sess-affinity-test-1"


def test_auxiliary_call_inherits_session_affinity_header():
    mock_providers = [
        {
            "name": "litellm-lan",
            "provider_key": "litellm-lan",
            "base_url": "http://localhost:4000/v1",
            "session_affinity_header": "x-litellm-session-id",
        }
    ]
    token = aux.set_runtime_main(
        "litellm-lan",
        "claude-3-5-sonnet",
        base_url="http://localhost:4000/v1",
        session_id="sess-affinity-test-1",
    )
    try:
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=mock_providers,
        ):
            kwargs = aux._build_call_kwargs(
                "litellm-lan",
                "claude-3-5-sonnet",
                _MSGS,
                base_url="http://localhost:4000/v1",
            )
            extra = kwargs.get("extra_headers") or {}
            assert extra.get("x-litellm-session-id") == "sess-affinity-test-1"
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_unconfigured_custom_provider_does_not_inject_header():
    mock_providers = [
        {
            "name": "plain-proxy",
            "base_url": "http://localhost:5000/v1",
        }
    ]
    agent = _agent("plain-proxy", "model-a", "http://localhost:5000/v1")
    with patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=mock_providers,
    ):
        kwargs = build_api_kwargs(agent, _MSGS)
        extra = kwargs.get("extra_headers") or {}
        assert "x-litellm-session-id" not in extra
        assert "x-opencode-session" not in extra


def test_caller_pinned_header_wins():
    mock_providers = [
        {
            "name": "litellm-lan",
            "base_url": "http://localhost:4000/v1",
            "session_affinity_header": "x-litellm-session-id",
        }
    ]
    agent = _agent("litellm-lan", "claude-3-5-sonnet", "http://localhost:4000/v1")
    agent.request_overrides = {"extra_headers": {"x-litellm-session-id": "pinned-by-caller"}}
    with patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=mock_providers,
    ):
        kwargs = build_api_kwargs(agent, _MSGS)
        extra = kwargs.get("extra_headers") or {}
        assert extra.get("x-litellm-session-id") == "pinned-by-caller"
