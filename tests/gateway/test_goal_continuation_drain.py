"""Regression: /goal continuations enqueued via the adapter FIFO are drained
automatically — no extra user message required (#47699).

Issue #47699 reported that a Slack ``/goal`` continuation could be enqueued by
``_post_turn_goal_continuation`` → ``_enqueue_fifo`` (adapter pending slot) but
never consumed until the next real inbound message woke the session.

On the current tree the continuation is enqueued while the adapter's
``_process_message_background`` frame is still live (the runner hook runs
inside ``self._message_handler(event)``), so the adapter's in-band pending
drain — and, for late arrivals, the finally-block late-arrival drain — spawns
the follow-up turn without user input.  These tests pin that contract from the
adapter dispatch layer down, so a future refactor that moves the goal hook
after the drain (or changes FIFO key derivation) fails loudly instead of
silently stalling goal loops on messaging gateways.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _DrainProbeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter that records deliveries."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.SLACK)
        self.sent: list[str] = []

    async def start(self):  # pragma: no cover - unused
        pass

    async def stop(self):  # pragma: no cover - unused
        pass

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)

        class _R:
            success = True
            message_id = "m1"

        return _R()

    async def send_typing(self, chat_id, metadata=None):
        pass


def _slack_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="U1",
        chat_id="C1",
        user_name="tester",
        chat_type="channel",
        thread_id="1718600000.000100",
    )


CONTINUATION_TEXT = "[Continuing toward your standing goal]\nGoal: ship it"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from this sync (non-loop) context so the
    # async tests' GoalManager.set() never races the bounded loop-thread
    # bootstrap window on loaded CI runners (goal silently not persisted →
    # continuation never enqueued; flaked on main run 33455779041).
    goals._get_session_db()
    yield home
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_fifo_enqueued_continuation_is_drained_without_new_user_message():
    """A continuation placed in the adapter pending slot during the handler
    frame (exactly what _post_turn_goal_continuation does via _enqueue_fifo)
    must start a second turn automatically."""
    adapter = _DrainProbeAdapter()
    src = _slack_thread_source()
    key = build_session_key(src)
    handled: list[str] = []

    async def handler(event):
        handled.append(event.text)
        if len(handled) == 1:
            # Mirror the runner's goal hook: enqueue the synthetic
            # continuation into the adapter FIFO while this frame is live.
            cont = MessageEvent(
                text=CONTINUATION_TEXT,
                message_type=MessageType.TEXT,
                source=src,
            )
            adapter._pending_messages[key] = cont
        return f"reply-{len(handled)}"

    adapter.set_message_handler(handler)
    event = MessageEvent(text="do the thing", message_type=MessageType.TEXT, source=src)

    await adapter._process_message_background(event, key)
    # The in-band drain hands off to a fresh task (#17758); let it run.
    for _ in range(40):
        # Wait for the SENDS, not just the handler calls: the delivery
        # ledger hops to worker threads around each send, so the handler
        # can return while reply-2's send is still in flight.
        if len(adapter.sent) >= 2:
            break
        await asyncio.sleep(0.05)

    assert handled == ["do the thing", CONTINUATION_TEXT], (
        "goal continuation was enqueued but never drained — "
        "a user nudge would be required (#47699)"
    )
    assert adapter.sent == ["reply-1", "reply-2"]
    # The drain task must have released the session guard when the chain ended.
    for _ in range(40):
        if key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.05)
    assert key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_runner_goal_hook_enqueues_into_the_key_the_adapter_drains(hermes_home):
    """_post_turn_goal_continuation resolves the FIFO key via
    _session_key_for_source; the adapter drain uses build_session_key on the
    event source. These must agree or the continuation is orphaned under a
    key nobody drains (the silent-stall shape from #47699)."""
    from unittest.mock import MagicMock, patch
    from datetime import datetime
    import uuid

    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry
    from hermes_cli.goals import GoalManager

    src = _slack_thread_source()
    adapter_key = build_session_key(src)

    runner = object.__new__(GatewayRunner)
    from gateway.config import GatewayConfig

    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")},
    )
    session_entry = SessionEntry(
        session_key=adapter_key,
        session_id=f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = adapter_key

    adapter = _DrainProbeAdapter()
    runner.adapters = {Platform.SLACK: adapter}

    GoalManager(session_entry.session_id).set("ship it")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "still needs work", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="partial progress",
        )
        await asyncio.sleep(0.05)

    assert adapter_key in adapter._pending_messages, (
        "continuation enqueued under a different key than the adapter "
        f"drains: pending keys={list(adapter._pending_messages)} "
        f"expected={adapter_key}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route_available", [True, False])
async def test_telegram_background_completion_continues_goal_without_user_nudge(tmp_path, monkeypatch, route_available):
    """A producer-owned Telegram turn has no adapter FIFO reader."""
    import gateway.run as gr
    from gateway.config import GatewayConfig
    from gateway.platforms.base import SendResult
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from hermes_cli import goals
    from hermes_state import SessionDB
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="goal-return", db=db)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(sent)))

    monkeypatch.setattr(adapter, "send", send)
    if not route_available:
        from gateway import session_automation

        async def refuse_wake(*args, **kwargs):
            return False

        monkeypatch.setattr(session_automation, "admit_route_wake", refuse_wake)
    calls = []

    async def respond(**kw):
        message = kw["message"]
        calls.append(message)
        if "[Continuing toward your standing goal]" in message:
            reply = "final read completed"
        elif "background result arrived" in message:
            reply = "worker result read"
        else:
            reply = "route ready"
        return {"final_response": reply, "messages": [], "tools": [], "history_offset": 0, "last_prompt_tokens": 0}

    verdicts = iter([
        ("continue", "one final read remains", False, None, False),
        ("done", "the read is complete", False, None, False),
    ])
    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", lambda *args, **kwargs: next(verdicts))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)

    try:
        await adapter.handle_message(MessageEvent(text="seed the read-only route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        manager = goals.GoalManager(sid)
        manager.set("complete the read-only check")
        live = authority.sessions[sid]
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_fixture", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task

        rows = list_session_admissions(db, session_id=authority.logical_owner(sid), pending_only=False)
        expected_turns = 3 if route_available else 2
        assert len(rows) == expected_turns, [(row["request_id"], row["outcome"]) for row in rows]
        assert "deleg_fixture" in rows[1]["request_id"]
        assert [row["outcome"] for row in rows] == ["completed"] * expected_turns
        assert len(calls) == expected_turns
        assert sent.count(("42", "worker result read")) == 1
        assert route not in adapter._pending_messages
        manager.refresh()
        if route_available:
            assert "goal-continuation" in rows[2]["request_id"]
            assert "[Continuing toward your standing goal]" in calls[-1]
            assert sent.count(("42", "final read completed")) == 1
            assert sent[-2:] == [("42", "final read completed"), ("42", "✓ Goal achieved: the read is complete")]
            assert manager.state.status == "done"
        else:
            assert sent[-2:] == [
                ("42", "worker result read"),
                ("42", "Goal is still active, but Hermes couldn’t queue the next turn."),
            ]
            assert not any("Continuing toward goal" in reply for _, reply in sent)
            assert manager.state.status == "active"
    finally:
        db.close()
