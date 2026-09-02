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
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
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
    runner._queued_events = {}
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
        # The continuation hand-off can cross an adapter task boundary on a
        # loaded worker; wait for the declared FIFO effect rather than assuming
        # a single scheduler turn is enough.
        for _ in range(40):
            if adapter_key in adapter._pending_messages:
                break
            await asyncio.sleep(0.05)

    assert adapter_key in adapter._pending_messages, (
        "continuation enqueued under a different key than the adapter "
        f"drains: pending keys={list(adapter._pending_messages)} "
        f"expected={adapter_key}"
    )
    assert adapter._pending_messages[adapter_key].text.startswith(
        "[Continuing toward your standing goal]"
    )
    continuation = adapter._pending_messages[adapter_key]
    assert continuation.internal is True
    assert continuation.allow_gateway_control is False
    assert continuation.metadata["gateway_session_key"] == adapter_key
    assert continuation.metadata["gateway_session_id"] == session_entry.session_id
    assert continuation.metadata["gateway_session_strict"] is True
    assert continuation.metadata["goal_continuation"]["goal_id"]
    assert continuation.metadata["goal_continuation"]["checkpoint_revision"] > 0
    assert continuation.metadata["goal_continuation"]["continuation_token"]


@pytest.mark.asyncio
async def test_generation_scoped_continuation_consumes_only_its_checkpoint(hermes_home):
    """The agent-start seam rejects a FIFO item superseded by user work."""
    from gateway.run import GatewayRunner
    from hermes_cli import goals

    session_id = "goal-generation-scope"
    mgr = goals.GoalManager(session_id)
    mgr.set("finish bounded work")
    mgr.evaluate_after_turn(
        "", turn_outcome=goals.EXECUTION_FAILED,
        turn_metadata={"reason": "first failure"},
    )
    stale = {
        "goal_id": mgr.state.goal_id,
        "checkpoint_revision": mgr.state.checkpoint_revision,
        "continuation_token": mgr.state.continuation_token,
    }
    assert mgr.start_continuation() is True
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            goals,
            "judge_goal",
            lambda *args, **kwargs: ("continue", "new work remains", False, None, False),
        )
        mgr.evaluate_after_turn("real user progress")

    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text=CONTINUATION_TEXT,
        message_type=MessageType.TEXT,
        source=_slack_thread_source(),
        internal=True,
        metadata={"goal_continuation": stale},
    )
    entry = SimpleNamespace(session_id=session_id)
    assert await runner._consume_goal_continuation_at_turn_start(event, entry) is False
    assert goals.GoalManager(session_id).state.continuation_pending is True

    current = goals.GoalManager(session_id).state
    event.metadata["goal_continuation"] = {
        "goal_id": current.goal_id,
        "checkpoint_revision": current.checkpoint_revision,
        "continuation_token": current.continuation_token,
    }
    assert await runner._consume_goal_continuation_at_turn_start(event, entry) is True
    assert goals.GoalManager(session_id).state.continuation_pending is False
