"""A /goal continuation queued on the adapter FIFO belongs to the session that judged it.

``_post_turn_goal_continuation`` parks the continuation in the chat's pending slot; if the route
is reset (/new, idle/daily policy) before that slot drains, the stale continuation must not run in
the fresh session, while an unreset route still continues as before. Under a session authority the
drained continuation is admitted as native input (``_hm_admit_to_fifo``); without one it resolves
its session in ``_hmwa_resolve_session``. Both honour the strict pin.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent

CONTINUATION_MARKER = "[Continuing toward your standing goal]"


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_before_drain", [True, False])
@pytest.mark.parametrize("with_authority", [True, False])
async def test_fifo_goal_continuation_never_runs_in_a_reset_session(
        tmp_path, monkeypatch, reset_before_drain, with_authority):
    import gateway.run as gr
    from gateway.config import GatewayConfig
    from gateway.platforms.base import SendResult
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from hermes_cli import goals
    from hermes_state import SessionDB
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    if with_authority:  # native admission (production); else the classic in-process resolver
        await initialize_session_authority(runner, profile_id="default", instance_id="goal-pin", db=db)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    monkeypatch.setattr(adapter, "send", send)
    calls: list[tuple[str, str]] = []  # (session_id, message)

    async def respond(message, context_prompt, history, source, session_id, **kw):
        calls.append((session_id, message))
        return {"final_response": f"reply {len(calls)}", "messages": [], "tools": [],
                "history_offset": 0, "last_prompt_tokens": 0}

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "more work remains", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal",
                        lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))

    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    session_a = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(session_a).set("ship the release")

    enqueued = []
    real_enqueue = runner._enqueue_fifo

    def enqueue_then_maybe_reset(session_key, queued_event, fifo_adapter):
        real_enqueue(session_key, queued_event, fifo_adapter)
        enqueued.append(queued_event.text)
        if reset_before_drain:
            # The route moves on while the continuation still sits in the pending slot.
            runner.session_store.reset_session(session_key)

    monkeypatch.setattr(runner, "_enqueue_fifo", enqueue_then_maybe_reset)

    try:
        await adapter.handle_message(MessageEvent(text="start on the release", source=source, message_id="1"))
        await async_wait_until(
            lambda: enqueued and route not in adapter._pending_messages and not adapter._background_tasks,
            what="continuation enqueued and its slot drained",
        )
        assert enqueued and CONTINUATION_MARKER in enqueued[0]
        session_b = runner.session_store.peek_session_id(route)
        assert (session_b != session_a) is reset_before_drain

        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: any(m == "fresh question" for _, m in calls)
                               and not adapter._background_tasks, what="second user turn ran")

        continuation_runs = [sid for sid, m in calls if CONTINUATION_MARKER in m]
        assert ("fresh question" in [m for sid, m in calls if sid == session_b])
        if reset_before_drain:
            assert continuation_runs == [], f"stale continuation ran after reset: {calls}"
        else:
            assert continuation_runs == [session_a], calls
    finally:
        db.close()
