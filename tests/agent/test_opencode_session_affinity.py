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


def _clear_aux_thread_state():
    """Simulate an executor-thread caller: no turn-scoped runtime, scope, or conversation."""
    aux.clear_runtime_main()
    from agent.portal_tags import set_affinity_scope, set_conversation_context

    set_affinity_scope(None)
    set_conversation_context(None)


def test_auxiliary_explicit_session_id_survives_cleared_context():
    """Post-turn callers (goal judge) run without turn contextvars: an explicit session_id
    must still produce the header (issue #105802)."""
    _clear_aux_thread_state()
    kwargs = aux._build_call_kwargs(
        "opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1",
        session_id="sess-judge-1",
    )
    assert kwargs["extra_headers"]["x-opencode-session"] == "sess-judge-1"


def test_auxiliary_bare_context_sends_no_header():
    """Without any session id anywhere there is nothing to pin: no header (the pre-fix
    judge behavior that OpenCode's relay rejects with MissingSessionID)."""
    _clear_aux_thread_state()
    kwargs = aux._build_call_kwargs(
        "opencode-go", "glm-5", _MSGS, base_url="https://opencode.ai/zen/go/v1",
    )
    assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})


def test_goal_judge_threads_session_id_to_call_llm(monkeypatch):
    """judge_goal(session_id=...) reaches call_llm so the header merge sees it."""
    from types import SimpleNamespace

    from hermes_cli import goals as goals_mod

    seen = {}

    def _fake_call_llm(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"verdict": "done", "reason": "all criteria met"}'))]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
    verdict, reason, parse_failed, _wait, transport_failed = goals_mod.judge_goal(
        "ship it", "the feature works", session_id="sess-judge-1",
    )
    assert seen.get("session_id") == "sess-judge-1"
    assert verdict == "done"
    assert transport_failed is False
