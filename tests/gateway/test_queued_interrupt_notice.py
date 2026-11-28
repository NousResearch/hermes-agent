"""Regression for #84207: a queued user steer must not acquire a stop bubble.

The carrier's general interrupted-turn visibility floor is for system stops, not
the intentional handoff to a newer user message. Exercise the gateway's actual
queued-followup delivery boundary, preserving system-stop visibility.
"""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_reason", "system_stop"),
    [
        ("interrupted_by_user", False),
        ("interrupted_during_api_call", False),
        ("interrupted_by_system(session_turn_lease_lost)", True),
        ("interrupted_during_api_call(session_turn_lease_lost)", True),
    ],
)
async def test_queued_followup_emits_only_system_stop_notice(
    monkeypatch, exit_reason, system_stop,
):
    runner = object.__new__(GatewayRunner)
    notices = []
    followups = []

    async def deliver(_ctx, _adapter, text, _result, _stream_task):
        notices.append(text)

    async def refresh(_key, _sid):
        return None

    async def run_followup(**kwargs):
        followups.append(kwargs["message"])
        return {"final_response": "answer to new message", "messages": []}

    monkeypatch.setattr(runner, "_run_agent_deliver_first_response", deliver)
    monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", refresh)
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda _source: None)
    monkeypatch.setattr(runner, "_run_agent", run_followup)

    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="chat"),
        session_id="s", session_key="key", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None,
        channel_prompt=None, context_prompt="", result_holder=[None],
    )
    notice = "Operation interrupted: " + (
        "system stop (session turn lease lost)."
        if system_stop else "stopped for a new user message."
    )
    result = {
        "interrupted": True, "turn_exit_reason": exit_reason,
        "final_response": notice,
    }

    out = await runner._run_agent_queued_followup(
        ctx, None, "new message", None, notice, result, None,
    )

    assert notices == ([notice] if system_stop else [])
    assert followups == ["new message"]
    assert out["final_response"] == "answer to new message"
