"""Provider-neutral policy seam for cache-preserving reasoning-effort changes."""

from types import SimpleNamespace

import pytest

from agent.effort_updates import effort_update, record_effort_switch


def test_policy_selection_becomes_first_baseline_then_durable_marker(monkeypatch):
    from agent import turn_context
    from hermes_cli import plugins as plugins_module

    choices = iter(("low", "high"))
    delivered = []

    monkeypatch.setattr(plugins_module, "has_middleware", lambda kind: True)

    def invoke(kind, **context):
        delivered.append((kind, context))
        return [{"effort": next(choices)}]

    monkeypatch.setattr(plugins_module, "invoke_middleware", invoke)

    patches = []

    class DB:
        def patch_session_model_config(self, session_id, patch):
            patches.append((session_id, patch))

    agent = SimpleNamespace(
        reasoning_config={"enabled": True, "effort": "max"},
        _session_init_model_config={
            "reasoning_config": {"enabled": True, "effort": "max"},
        },
        _session_db=DB(),
        session_id="session-1",
        platform="cli",
        model="gpt-5.6-luna-1",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        capabilities={},
    )

    history = []
    turn_context._apply_reasoning_effort_policy(
        agent,
        history,
        {"role": "user", "content": "hello"},
        task_id="task",
        turn_id="one",
    )
    assert delivered[-1][0] == "reasoning_effort"
    assert delivered[-1][1]["user_message"] == "hello"
    assert delivered[-1][1]["has_conversation_history"] is False
    assert delivered[-1][1]["previous_effort"] is None
    assert "conversation_history" not in delivered[-1][1]
    assert delivered[-1][1]["reasoning_effort_updates_supported"] is True
    assert "low" in delivered[-1][1]["supported_reasoning_efforts"]
    assert patches[-1][1]["reasoning_config"]["effort"] == "low"
    assert patches[-1][1]["_reasoning_effort_baseline"]["effort"] == "low"
    assert record_effort_switch(agent, history) is False

    history.extend([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    turn_context._apply_reasoning_effort_policy(
        agent,
        history,
        {"role": "user", "content": "debug this"},
        task_id="task",
        turn_id="two",
    )
    assert delivered[-1][1]["has_conversation_history"] is True
    assert delivered[-1][1]["previous_effort"] == "low"
    assert record_effort_switch(agent, history) is True
    update = effort_update(history[-1])
    assert update is not None
    assert update["previous"] == "low"
    assert update["effort"] == "high"


@pytest.mark.parametrize(
    ("api_mode", "model", "provider", "base_url", "capabilities", "expected"),
    [
        (
            "codex_responses",
            "gpt-5.6-luna-1",
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            {},
            True,
        ),
        (
            "codex_responses",
            "future-reasoner",
            "custom",
            "https://proxy.invalid/v1",
            {},
            False,
        ),
        (
            "anthropic_messages",
            "claude-opus-5",
            "anthropic",
            "https://api.anthropic.com",
            {},
            True,
        ),
        (
            "anthropic_messages",
            "claude-sonnet-4-6",
            "anthropic",
            "https://api.anthropic.com",
            {},
            False,
        ),
        (
            "chat_completions",
            "gpt-5.6-luna-1",
            "custom",
            "https://proxy.invalid/v1",
            {"reasoning_effort_updates": True},
            False,
        ),
    ],
)
def test_transport_contract_exposes_only_cache_safe_effort_updates(
    api_mode, model, provider, base_url, capabilities, expected,
):
    from agent.transports import get_transport

    transport = get_transport(api_mode)
    assert transport is not None
    context = {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "capabilities": capabilities,
    }
    supported = transport.supports_reasoning_effort_updates(**context)
    levels = transport.reasoning_effort_update_levels(**context)

    assert supported is expected
    assert bool(levels) is expected
    if expected:
        assert "low" in levels and "max" in levels
