"""x-opencode-session rides on every OpenCode request, on every transport."""

from __future__ import annotations

import pytest

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


def _agent(provider, model, base_url, api_mode=None):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="sess-affinity-1",
    )
    if api_mode:
        agent.api_mode = api_mode
        agent._transport = None
        agent._anthropic_base_url = base_url
    return agent


@pytest.mark.parametrize(
    "provider, model, base_url, api_mode",
    [
        ("opencode-go", "glm-5", "https://opencode.ai/zen/go/v1", None),  # chat_completions
        ("opencode-go", "gpt-5.6-luna", "https://opencode.ai/zen/go/v1", None),  # codex_responses
        ("opencode-go", "minimax-m2.7", "https://opencode.ai/zen/go/v1", "anthropic_messages"),
        ("opencode-free", "laguna-s-2.1-free", "https://opencode.ai/zen/v1", None),
        ("custom", "glm-5", "https://opencode.ai/zen/go/v1", None),  # URL-only detection
    ],
)
def test_main_turn_sends_stable_session_header_on_every_transport(provider, model, base_url, api_mode):
    agent = _agent(provider, model, base_url, api_mode)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"]["x-opencode-session"]
    assert first == second == "sess-affinity-1"

    other = _agent("openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1")
    assert "x-opencode-session" not in (build_api_kwargs(other, _MSGS).get("extra_headers") or {})


def test_auxiliary_calls_share_the_main_turn_session_key():
    token = aux.set_runtime_main(
        "opencode-go", "glm-5", base_url="https://opencode.ai/zen/go/v1", session_id="sess-affinity-1"
    )
    try:
        kwargs = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1")
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-affinity-1"
        other = aux._build_call_kwargs("openrouter", "x", _MSGS, base_url="https://openrouter.ai/api/v1")
        assert "x-opencode-session" not in (other.get("extra_headers") or {})
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_oneshot_and_unparented_opencode_calls_send_ephemeral_session_header():
    from agent.opencode_affinity import opencode_session_headers

    # Unparented opencode call (no active session or contextvar) generates an ephemeral session key
    headers = opencode_session_headers("opencode-go", None, session_id=None)
    assert "x-opencode-session" in headers
    assert headers["x-opencode-session"].startswith("oneshot-")

    # Direct auxiliary call without runtime-main session generates an ephemeral session header
    kwargs = aux._build_call_kwargs("opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1")
    assert "extra_headers" in kwargs
    assert kwargs["extra_headers"]["x-opencode-session"].startswith("oneshot-")

    # Non-OpenCode provider without session still returns empty headers without generating UUIDs
    other = opencode_session_headers("openrouter", "https://openrouter.ai/api/v1", session_id=None)
    assert other == {}


def test_oneshot_commit_message_with_opencode_runtime(monkeypatch):
    from unittest.mock import MagicMock
    from agent.oneshot import run_oneshot

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "feat(git): add commit helper"
    mock_resp.choices[0].message.reasoning = None
    mock_resp.choices[0].message.reasoning_content = None
    mock_resp.choices[0].message.reasoning_details = None

    captured_kwargs = {}

    def fake_call_llm(**kwargs):
        captured_kwargs.update(kwargs)
        # Verify that when auxiliary_client._build_call_kwargs builds kwargs for this provider,
        # it carries the x-opencode-session header
        call_kwargs = aux._build_call_kwargs(
            kwargs["main_runtime"]["provider"],
            kwargs["main_runtime"]["model"],
            kwargs["messages"],
            base_url=kwargs["main_runtime"].get("base_url"),
        )
        assert "x-opencode-session" in call_kwargs["extra_headers"]
        assert call_kwargs["extra_headers"]["x-opencode-session"].startswith("oneshot-")
        return mock_resp

    monkeypatch.setattr("agent.oneshot.call_llm", fake_call_llm)

    msg = run_oneshot(
        template="commit_message",
        variables={"diff": "diff --git a/a b/a\n+hello"},
        main_runtime={
            "provider": "opencode-go",
            "model": "glm-5",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
    )
    assert msg == "feat(git): add commit helper"
    assert captured_kwargs["main_runtime"]["provider"] == "opencode-go"

