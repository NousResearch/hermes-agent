"""An authority-owned recovered human turn must not park its /goal continuation in an undrained adapter slot.

A pending messaging input can resume without an adapter waiter after gateway recovery.
It is still a human turn, but only the session authority can drain its follow-up.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session_authority import initialize_session_authority
from gateway.session_ingress_context import native_callback, register_transport_home
from hermes_cli import goals
from persistence_assembly import open_persistence
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_recovered_human_goal_turn_admits_continuation_on_authority(tmp_path, monkeypatch):
    import gateway.run as gr

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="recovered-goal")
    db = open_persistence(runner.session_store.persistence.scope)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    sent = []
    calls = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(sent)))

    async def respond(**kw):
        message = kw["message"]
        calls.append(message)
        if "[Continuing toward your standing goal]" in message:
            reply = "final goal work complete"
        elif "recovered request" in message:
            reply = "recovered request completed"
        else:
            reply = "route ready"
        return {"final_response": reply, "messages": [], "tools": [],
                "history_offset": 0, "last_prompt_tokens": 0}

    verdicts = iter([
        ("continue", "one last check", False, None, False),
        ("done", "the check is complete", False, None, False),
    ])
    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", lambda *args, **kwargs: next(verdicts))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)

    try:
        await adapter.handle_message(MessageEvent(text="prime route", source=source, message_id="seed"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks,
                               what="first human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        manager = goals.GoalManager(sid)
        manager.set("finish the check")

        # Model a persisted human message recovered without the original adapter delivery task.
        # Calling the authority directly admits its real native event but registers NO native waiter.
        recovered = MessageEvent(text="recovered request", source=source, message_id="recovered-2")
        # The original authenticated callback stamps connector provenance before persistence.
        with native_callback(runner, recovered, tmp_path):
            receipt = await authority.admit_native(recovered)
        assert receipt.admission_id not in authority.native_waiters
        await authority.sessions[sid].task

        rows = authority.persistence.admissions.list(session_id=authority.logical_owner(sid), pending_only=False)
        assert len(rows) == 3, [(row.request_id, row.outcome) for row in rows]
        assert [row.outcome for row in rows] == ["completed"] * 3
        assert "goal-continuation" in rows[2].request_id
        assert len(calls) == 3
        assert "[Continuing toward your standing goal]" in calls[2]
        assert route not in adapter._pending_messages
        assert sent.count(("42", "recovered request completed")) == 1
        assert sent[-2:] == [
            ("42", "final goal work complete"),
            ("42", "✓ Goal achieved: the check is complete"),
        ]
        manager.refresh()
        assert manager.state.status == "done"
    finally:
        db.close()
