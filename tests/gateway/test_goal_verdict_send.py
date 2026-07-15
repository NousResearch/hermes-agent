"""Tests for gateway /goal outcome-message delivery.

The primary model's outcome message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import uuid
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


def _make_runner_with_adapter(session_id: str = None):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    src = _make_source()
    # Default to a unique session_id so xdist parallel runs on the same worker
    # don't see each other's GoalManager state (DEFAULT_DB_PATH gets frozen at
    # module-import time, defeating per-test HERMES_HOME monkeypatches).
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


def _begin_goal_turn(mgr, outcome: str | None = None, reason: str = ""):
    """Create the exact turn/generation authority used by gateway delivery."""
    turn_id = f"turn-{uuid.uuid4().hex}"
    generation_id = mgr.begin_model_turn(turn_id)
    assert generation_id
    if outcome is not None:
        assert mgr.record_model_outcome(
            outcome,
            reason,
            originating_turn_id=turn_id,
            goal_generation_id=generation_id,
        )
    return turn_id, generation_id


@pytest.mark.asyncio
async def test_goal_verdict_done_sent_via_adapter_send(hermes_home):
    """When the primary model records completion, the achieved message reaches
    the user through the adapter's ``send()`` method."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("ship the feature")
    turn_id, generation_id = _begin_goal_turn(
        mgr, "complete", "the feature shipped"
    )

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="I shipped the feature.",
        goal_session_id=session_entry.session_id,
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    # fire-and-forget create_task — give the loop a tick
    await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    msg = adapter.sends[0]
    assert msg["chat_id"] == "c1"
    assert "Goal achieved" in msg["content"]
    assert "the feature shipped" in msg["content"]


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the primary model records continue, both the status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")
    turn_id, generation_id = _begin_goal_turn(
        mgr, "continue", "still needs work"
    )

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="here's a partial edit",
        goal_session_id=session_entry.session_id,
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    await asyncio.sleep(0.05)

    # Status line sent back
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    # Continuation prompt enqueued for next turn
    assert adapter._pending_messages, "continuation prompt must be enqueued in pending_messages"


@pytest.mark.asyncio
async def test_goal_verdict_budget_exhausted_sends_pause(hermes_home):
    """When the budget is exhausted, a '⏸ Goal paused' message must be sent
    and no further continuation enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager, save_goal

    mgr = GoalManager(session_entry.session_id, default_max_turns=2)
    state = mgr.set("tiny goal", max_turns=2)
    state.turns_used = 1
    save_goal(session_entry.session_id, state)
    turn_id, generation_id = _begin_goal_turn(mgr, "continue", "keep going")

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="still partial",
        goal_session_id=session_entry.session_id,
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages


@pytest.mark.asyncio
async def test_goal_verdict_skipped_when_no_active_goal(hermes_home):
    """No goal set → the hook is a no-op. Nothing is sent, nothing enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="anything",
        goal_session_id=session_entry.session_id,
        originating_turn_id="no-active-turn",
        goal_generation_id="no-active-generation",
    )
    await asyncio.sleep(0.05)

    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_goal_verdict_survives_adapter_without_send(hermes_home):
    """Bad adapter (no ``send`` attribute) must not crash the goal hook."""
    runner, _adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("survive missing send")
    turn_id, generation_id = _begin_goal_turn(mgr, "complete", "ok")

    class _NoSendAdapter:
        def __init__(self):
            self._pending_messages: dict = {}

    runner.adapters[Platform.TELEGRAM] = _NoSendAdapter()

    # must not raise
    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="whatever",
        goal_session_id=session_entry.session_id,
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_authority", ["turn", "generation"])
async def test_goal_verdict_stale_authority_cannot_mutate_goal(
    hermes_home, stale_authority
):
    """A late worker cannot consume or advance another turn's goal state."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("preserve exact turn authority")
    turn_id, generation_id = _begin_goal_turn(
        mgr, "continue", "the exact turn still has work"
    )

    supplied_turn = "stale-turn" if stale_authority == "turn" else turn_id
    supplied_generation = (
        "stale-generation" if stale_authority == "generation" else generation_id
    )
    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="late response",
        goal_session_id=session_entry.session_id,
        originating_turn_id=supplied_turn,
        goal_generation_id=supplied_generation,
    )

    durable = GoalManager(session_entry.session_id).state
    assert durable is not None
    assert durable.status == "active"
    assert durable.turns_used == 0
    assert durable.active_model_turn_id == turn_id
    assert durable.pending_model_outcome == "continue"
    assert durable.pending_model_turn_id == turn_id
    assert durable.pending_model_generation_id == generation_id
    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_goal_verdict_uses_effective_session_after_compression(hermes_home):
    """The post-compression session id, not the pre-turn entry, owns outcome."""
    runner, adapter, old_entry, src = _make_runner_with_adapter()
    effective_session_id = f"compressed-goal-{uuid.uuid4().hex}"

    from hermes_cli.goals import GoalManager

    old_mgr = GoalManager(old_entry.session_id)
    old_mgr.set("old session goal must remain untouched")

    effective_mgr = GoalManager(effective_session_id)
    effective_mgr.set("finish work after compression")
    turn_id, generation_id = _begin_goal_turn(
        effective_mgr, "complete", "completed in the effective session"
    )

    await runner._finalize_gateway_goal_turn(
        source=src,
        response={
            "final_response": "done after compression",
            "session_id": effective_session_id,
            "turn_id": turn_id,
            "goal_generation_id": generation_id,
        },
        agent=SimpleNamespace(
            session_id=old_entry.session_id,
            _current_turn_id="stale-pre-compression-turn",
            _current_goal_generation_id="stale-pre-compression-generation",
        ),
        fallback_session_id=old_entry.session_id,
    )
    await asyncio.sleep(0.05)

    effective_state = GoalManager(effective_session_id).state
    old_state = GoalManager(old_entry.session_id).state
    assert effective_state is not None and effective_state.status == "done"
    assert effective_state.turns_used == 1
    assert old_state is not None and old_state.status == "active"
    assert old_state.turns_used == 0
    assert len(adapter.sends) == 1
    assert "completed in the effective session" in adapter.sends[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unhealthy_fields",
    [
        {"final_response": ""},
        {"final_response": "provider error", "error": "provider unavailable"},
        {"final_response": "interrupted response", "interrupted": True},
        {"final_response": "partial response", "partial": True},
        {"final_response": "failed response", "failed": True},
    ],
    ids=["empty", "error", "interrupted", "partial", "failed"],
)
async def test_unhealthy_gateway_turn_abandons_exact_authority(
    hermes_home, unhealthy_fields
):
    """Unhealthy turns revoke their authority without consuming goal budget."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("survive an unhealthy transport turn")
    turn_id, generation_id = _begin_goal_turn(mgr)
    response = {
        "session_id": session_entry.session_id,
        "turn_id": turn_id,
        "goal_generation_id": generation_id,
        **unhealthy_fields,
    }

    await runner._finalize_gateway_goal_turn(
        source=src,
        response=response,
        fallback_session_id="wrong-fallback-session",
    )

    durable = GoalManager(session_entry.session_id).state
    assert durable is not None
    assert durable.status == "active"
    assert durable.turns_used == 0
    assert durable.active_model_turn_id is None
    assert durable.pending_model_outcome is None
    assert durable.pending_model_turn_id is None
    assert durable.pending_model_generation_id is None
    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_location", ["pending-slot", "overflow"])
async def test_goal_continuation_never_displaces_queued_user_input(
    hermes_home, queue_location
):
    """Real queued input suppresses, rather than merely precedes, auto-continue.

    Leaving a synthetic continuation anywhere behind the real event is unsafe:
    FIFO promotion can expose it while the real turn is running and the adapter's
    interrupt monitor would then interrupt that user-authored turn.
    """
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("continue without overtaking the user")
    turn_id, generation_id = _begin_goal_turn(
        mgr, "continue", "one more model turn is required"
    )

    session_key = build_session_key(src)
    user_event = MessageEvent(
        text="new instruction from the user",
        message_type=MessageType.TEXT,
        source=src,
        message_id="real-user-message",
    )
    if queue_location == "pending-slot":
        adapter._pending_messages[session_key] = user_event
    else:
        runner._queued_events[session_key] = [user_event]

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="continuing",
        goal_session_id=session_entry.session_id,
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )

    if queue_location == "pending-slot":
        assert adapter._pending_messages[session_key] is user_event
        assert session_key not in runner._queued_events
    else:
        assert session_key not in adapter._pending_messages
        assert runner._queued_events[session_key] == [user_event]
