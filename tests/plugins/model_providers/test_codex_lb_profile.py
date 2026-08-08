"""Tests for Codex LB conversation-affinity headers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock

import pytest

from agent.portal_tags import reset_conversation_context, set_conversation_context


CODEX_LB_BASE_URL = "http://127.0.0.1:2455/v1"


def _configure_codex_lb(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {
            "providers": {
                "codex-lb": {
                    "name": "Codex LB",
                    "api": CODEX_LB_BASE_URL,
                    "model": "gpt-5.6",
                }
            }
        },
    )


def _make_live_custom_agent(monkeypatch, base_url):
    import run_agent
    from run_agent import AIAgent

    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="custom",
        model="gpt-5.6",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@pytest.fixture
def codex_lb_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("codex-lb")
    assert profile is not None, "codex-lb provider profile must be registered"
    assert profile.name == "codex-lb"
    return profile


def test_main_and_auxiliary_calls_share_ambient_lineage_root(codex_lb_profile):
    from agent.auxiliary_client import _build_call_kwargs
    from agent.transports.chat_completions import ChatCompletionsTransport

    token = set_conversation_context("lineage-root")
    try:
        main_kwargs = ChatCompletionsTransport().build_kwargs(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "main"}],
            tools=None,
            provider_profile=codex_lb_profile,
            session_id="rotated-session",
        )
        auxiliary_kwargs = _build_call_kwargs(
            provider="custom:codex-lb",
            model="gpt-5.6",
            messages=[{"role": "user", "content": "auxiliary"}],
        )
    finally:
        reset_conversation_context(token)

    expected = {"session_id": "lineage-root"}
    assert main_kwargs["extra_headers"] == expected
    assert auxiliary_kwargs["extra_headers"] == expected


def test_live_bare_custom_runtime_recovers_named_profile(monkeypatch):
    from agent.auxiliary_client import _build_call_kwargs

    _configure_codex_lb(monkeypatch)
    agent = _make_live_custom_agent(monkeypatch, CODEX_LB_BASE_URL)
    assert agent.provider == "custom"

    token = set_conversation_context("webui-conversation")
    try:
        main_kwargs = agent._build_api_kwargs(
            [{"role": "user", "content": "main"}]
        )
        auxiliary_kwargs = _build_call_kwargs(
            provider="custom",
            base_url=CODEX_LB_BASE_URL,
            model="gpt-5.6",
            messages=[{"role": "user", "content": "auxiliary"}],
        )
    finally:
        reset_conversation_context(token)

    expected = {"session_id": "webui-conversation"}
    assert main_kwargs["extra_headers"] == expected
    assert auxiliary_kwargs["extra_headers"] == expected


def test_unmatched_bare_custom_runtime_keeps_generic_profile(monkeypatch):
    from agent.auxiliary_client import _build_call_kwargs

    _configure_codex_lb(monkeypatch)
    unmatched_url = "http://127.0.0.1:9999/v1"
    agent = _make_live_custom_agent(monkeypatch, unmatched_url)

    token = set_conversation_context("webui-conversation")
    try:
        main_kwargs = agent._build_api_kwargs(
            [{"role": "user", "content": "main"}]
        )
        auxiliary_kwargs = _build_call_kwargs(
            provider="custom",
            base_url=unmatched_url,
            model="gpt-5.6",
            messages=[{"role": "user", "content": "auxiliary"}],
        )
    finally:
        reset_conversation_context(token)

    assert "extra_headers" not in main_kwargs
    assert "extra_headers" not in auxiliary_kwargs


def test_bare_custom_profile_lookup_uses_model_when_url_is_unavailable(
    monkeypatch, codex_lb_profile
):
    from providers import get_provider_profile

    _configure_codex_lb(monkeypatch)

    assert (
        get_provider_profile("custom", model="gpt-5.6") is codex_lb_profile
    )


def test_explicit_session_id_is_used_without_ambient_context(codex_lb_profile):
    extra_body, top_level = codex_lb_profile.build_api_kwargs_extras(
        session_id="explicit-session"
    )

    assert extra_body == {}
    assert top_level == {"extra_headers": {"session_id": "explicit-session"}}


def test_affinity_header_is_omitted_without_conversation_or_session(codex_lb_profile):
    extra_body, top_level = codex_lb_profile.build_api_kwargs_extras()

    assert extra_body == {}
    assert top_level == {}


def test_named_custom_runtime_identity_resolves_same_profile(codex_lb_profile):
    from providers import get_provider_profile

    assert get_provider_profile("custom:codex-lb") is codex_lb_profile


def test_concurrent_conversations_keep_distinct_affinity_headers(codex_lb_profile):
    ready = Barrier(2)

    def affinity_header(conversation_id: str) -> str:
        token = set_conversation_context(conversation_id)
        try:
            ready.wait()
            _, top_level = codex_lb_profile.build_api_kwargs_extras()
            return top_level["extra_headers"]["session_id"]
        finally:
            reset_conversation_context(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        headers = set(pool.map(affinity_header, ("conversation-a", "conversation-b")))

    assert headers == {"conversation-a", "conversation-b"}
