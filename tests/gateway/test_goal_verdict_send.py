"""Tests for gateway /goal verdict-message delivery.

The judge verdict message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from this SYNC context. The tests call
    # GoalManager.set() on the event-loop thread, where _get_session_db()
    # refuses to construct SessionDB inline (loop-liveness guard) and only
    # waits _DB_BOOTSTRAP_LOOP_WAIT_S for a background bootstrap. On a loaded
    # CI runner the init overruns that window, the goal write is silently
    # dropped by design, and the continuation path no-ops — the recurring
    # sends == [] flake. Warming here uses the direct construction path, so
    # the loop-thread set() always finds a cached DB.
    goals._get_session_db()
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
    import uuid

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


async def _drain_until(condition, timeout=5.0):
    """Yield to the event loop until ``condition()`` is truthy (bounded).

    The goal-continuation path finishes its sends/enqueues on spawned tasks;
    a fixed 0.05s sleep raced them on loaded CI runners (#88975). Returns as
    soon as the condition holds — the asserts after the call stay exact.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while not condition() and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the judge says continue, both the 'continuing' status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await _drain_until(lambda: adapter.sends and adapter._pending_messages)

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
    state.turns_used = 2
    save_goal(session_entry.session_id, state)

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="still partial",
        )
        await _drain_until(lambda: adapter.sends)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages




def _blocked_goal(session_id: str):
    """Set a goal and drive it to the judge's BLOCKED auto-pause (needs user input)."""
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id)
    mgr.set("run both agent sessions end to end")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("blocked", "needs the user to approve the login card", False, None, False),
    ):
        assert mgr.evaluate_after_turn("May I use your login?")["status"] == "paused"
    return mgr


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    "success", "cancelled", "blocked-again", "stop", "exception", "failed-result", "interrupted-result",
])
async def test_user_turn_revives_blocked_goal_before_gateway_run(hermes_home, monkeypatch, tmp_path, outcome):
    """Admission, not successful delivery, owns recovery; a new in-run pause wins."""
    from hermes_cli.goals import GoalManager
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event

    runner = _bootstrap(monkeypatch, tmp_path)
    event = _event()
    entry = runner.session_store.get_or_create_session.return_value
    adapter = _RecordingAdapter()
    runner.adapters[event.source.platform] = adapter
    _blocked_goal(entry.session_id)
    before = GoalManager(entry.session_id).state
    observed = []
    loop_cleanup = AsyncMock()
    runner._post_turn_loop_completion = loop_cleanup

    async def run_agent(**kwargs):
        observed.append(GoalManager(kwargs["session_id"]).state)
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "exception":
            raise RuntimeError("fixture provider unavailable")
        if outcome == "failed-result":
            return {"final_response": "Provider failed before completion", "messages": [],
                    "history_offset": 0, "failed": True, "error": "fixture provider unavailable"}
        if outcome == "interrupted-result":
            return {"final_response": "Interrupted before completion", "messages": [],
                    "history_offset": 0, "interrupted": True, "completed": False}
        if outcome in {"blocked-again", "stop"}:
            GoalManager(entry.session_id).pause(
                reason=before.paused_reason if outcome == "blocked-again" else "user-paused")
        return {"final_response": "repair underway", "messages": [], "history_offset": 0}

    runner._run_agent = run_agent
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "more work", False, None, False)) as judge:
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await runner._handle_message(event)
        else:
            await runner._handle_message(event)
    assert len(observed) == 1
    assert observed[0].status == "active"
    assert observed[0].paused_reason is None
    assert observed[0].turns_used == before.turns_used
    assert observed[0].max_turns == before.max_turns
    state = GoalManager(entry.session_id).state
    assert any(s["content"].startswith("▶ Goal resumed") for s in adapter.sends)
    if outcome != "cancelled":
        loop_cleanup.assert_awaited_once()
    if outcome in {"blocked-again", "stop"}:
        assert state.status == "paused"
        judge.assert_not_called()
        assert not adapter._pending_messages
    elif outcome in {"cancelled", "exception", "failed-result", "interrupted-result"}:
        assert state.status == "active"
        assert state.turns_used == before.turns_used
        judge.assert_not_called()
        assert not adapter._pending_messages
    else:
        assert state.status == "active"
        assert adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "internal", "continuation", "heartbeat", "proactive", "user-paused", "user-interrupted (Ctrl+C)",
    "budget", "unauthorized", "rejected-preparation", "lease-timeout",
])
async def test_gateway_non_user_or_rejected_input_cannot_revive(hermes_home, monkeypatch, tmp_path, case):
    from unittest.mock import AsyncMock
    from hermes_cli.goals import GoalManager
    from gateway.turn_lease import TurnLeaseTimeoutError
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event

    runner = _bootstrap(monkeypatch, tmp_path)
    event = _event()
    entry = runner.session_store.get_or_create_session.return_value
    _blocked_goal(entry.session_id)
    if case in {"user-paused", "user-interrupted (Ctrl+C)"}:
        GoalManager(entry.session_id).pause(reason=case)
    elif case == "budget":
        GoalManager(entry.session_id).pause(reason="turn budget exhausted (20/20)")
    elif case == "internal":
        event.internal = True
    elif case == "continuation":
        event.text = "[Continuing toward your standing goal]\nGoal: repair"
    elif case == "heartbeat":
        event = runner._synthetic_prompt_event(event.source, "[Heartbeat — periodic check]")
        event._heartbeat_session_id = entry.session_id
        event._heartbeat_execution_started = False
        runner.session_store.lookup_by_session_key.return_value = entry
    elif case == "proactive":
        event.allow_gateway_control = False
    elif case == "unauthorized":
        runner._is_user_authorized = lambda source: False
    elif case == "rejected-preparation":
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None)
    elif case == "lease-timeout":
        runner._hmwa_acquire_turn_lease = AsyncMock(side_effect=TurnLeaseTimeoutError(
            entry.session_id, owner_key=entry.session_key, generation=1, wait_seconds=0))
    before = GoalManager(entry.session_id).state
    observed = []

    async def run_agent(**kwargs):
        observed.append(GoalManager(kwargs["session_id"]).state)
        return {"final_response": "noted", "messages": [], "history_offset": 0}

    runner._run_agent = run_agent
    with patch("hermes_cli.goals.judge_goal") as judge:
        await runner._handle_message(event)
    if case in {"unauthorized", "rejected-preparation", "lease-timeout"}:
        assert not observed
    else:
        assert len(observed) == 1
        assert observed[0].status == before.status
        assert observed[0].paused_reason == before.paused_reason
    after = GoalManager(entry.session_id).state
    assert after.status == before.status
    assert after.paused_reason == before.paused_reason
    assert after.turns_used == before.turns_used
    judge.assert_not_called()
