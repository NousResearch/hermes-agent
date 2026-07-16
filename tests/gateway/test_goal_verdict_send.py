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


@pytest.mark.asyncio
@pytest.mark.parametrize("enqueue_failure", ["raises", "drops"])
async def test_goal_continuation_enqueue_failure_is_visible_and_preserves_goal(
    hermes_home, caplog, enqueue_failure
):
    """A mechanical queue failure must never look like a running goal loop.

    The primary model has already authored ``continue`` by this boundary, so
    the durable goal remains active.  If the gateway cannot enqueue the next
    turn, it must emit an operator-visible receipt instead of swallowing the
    failure at DEBUG while telling the user that work is continuing.
    """

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id, default_max_turns=0)
    mgr.set("finish the approved plan", max_turns=0)
    turn_id, generation_id = _begin_goal_turn(
        mgr, "continue", "verified work remains"
    )

    def _broken_enqueue(*_args, **_kwargs):
        if enqueue_failure == "raises":
            raise RuntimeError("queue storage unavailable")
        return None

    runner._enqueue_fifo = _broken_enqueue

    with caplog.at_level("WARNING"):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I completed the first verified step.",
            goal_session_id=session_entry.session_id,
            originating_turn_id=turn_id,
            goal_generation_id=generation_id,
        )

    durable = GoalManager(session_entry.session_id).state
    assert durable is not None
    assert durable.status == "active"
    assert durable.turns_used == 1
    assert durable.last_verdict == "continue"
    assert adapter._pending_messages == {}
    assert any(
        "automatic continuation was not queued" in item["content"].lower()
        for item in adapter.sends
    )
    assert "goal continuation: enqueue failed" in caplog.text.lower()


@pytest.mark.asyncio
async def test_production_goal_release_acceptance_continues_recovers_and_completes(
    hermes_home, monkeypatch
):
    """Production standing goals keep one model authority through completion.

    This acceptance path deliberately does not use the capability-canary
    shortcut.  It proves two automatic continuation turns, exact durable
    recovery after a simulated gateway restart, Canonical Task Workspace
    hydration, real-user preemption, and model-authored completion.  Goal
    orchestration may add only ordinary user-role continuation events; the
    cached system prompt, model route, and tool schema remain byte/identity
    stable within a process and equal after restart reconstruction.
    """

    from gateway import canonical_brain_task_workspace as workspace
    from hermes_cli.goals import GoalManager
    from tools.todo_tool import TodoStore

    session_id = f"production-goal-acceptance-{uuid.uuid4().hex}"
    runner, adapter, session_entry, src = _make_runner_with_adapter(session_id)
    session_key = build_session_key(src)

    stable_identity = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "cached_system_prompt": "stable production system prompt",
        "tools": (
            ("todo", "model-authored-plan-and-goal-outcome"),
            ("terminal", "mechanical-execution"),
        ),
    }

    def _agent_for_process():
        return SimpleNamespace(
            session_id=session_id,
            model=stable_identity["model"],
            provider=stable_identity["provider"],
            api_mode=stable_identity["api_mode"],
            base_url=stable_identity["base_url"],
            _cached_system_prompt=stable_identity["cached_system_prompt"],
            tools=stable_identity["tools"],
        )

    def _identity(agent):
        return {
            "model": agent.model,
            "provider": agent.provider,
            "api_mode": agent.api_mode,
            "base_url": agent.base_url,
            "cached_system_prompt": agent._cached_system_prompt,
            "tools": agent.tools,
        }

    async def _finish_turn(
        active_runner,
        agent,
        *,
        outcome: str,
        reason: str,
        final_response: str,
    ):
        durable_manager = GoalManager(session_id, default_max_turns=0)
        turn_id, generation_id = _begin_goal_turn(
            durable_manager, outcome, reason
        )
        before = _identity(agent)
        await active_runner._finalize_gateway_goal_turn(
            source=src,
            response={
                "session_id": session_id,
                "turn_id": turn_id,
                "goal_generation_id": generation_id,
                "final_response": final_response,
            },
            agent=agent,
            fallback_session_id="must-not-be-used",
        )
        assert _identity(agent) == before == stable_identity
        return turn_id, generation_id

    goal = GoalManager(session_id, default_max_turns=0)
    initial = goal.set("finish every approved step", max_turns=0)
    original_generation = initial.generation_id
    first_agent = _agent_for_process()

    # Turn 1 -> exact model-authored continue -> automatic continuation #1.
    await _finish_turn(
        runner,
        first_agent,
        outcome="continue",
        reason="implementation remains",
        final_response="The first verified step is complete.",
    )
    automatic_one = adapter._pending_messages.pop(session_key)
    assert runner._is_goal_continuation_event(automatic_one)
    assert automatic_one.metadata == {"hermes_internal_event": "goal_continuation"}

    # Turn 2 consumes continuation #1 and queues continuation #2.
    await _finish_turn(
        runner,
        first_agent,
        outcome="continue",
        reason="verification remains",
        final_response="Implementation is complete; verification remains.",
    )
    automatic_two = adapter._pending_messages.pop(session_key)
    assert runner._is_goal_continuation_event(automatic_two)
    assert automatic_two.text == automatic_one.text

    before_restart = GoalManager(session_id).state
    assert before_restart is not None
    assert before_restart.status == "active"
    assert before_restart.turns_used == 2
    assert before_restart.generation_id == original_generation

    # Simulate a fresh gateway process.  The in-memory queued event is gone,
    # while both the standing goal and the model-authored Canonical plan remain
    # mechanically reconstructable from durable stores.
    restarted_runner, restarted_adapter, restarted_entry, _ = (
        _make_runner_with_adapter(session_id)
    )
    assert restarted_entry.session_id == session_id
    recovered_goal = GoalManager(session_id, default_max_turns=0)
    assert recovered_goal.state is not None
    assert recovered_goal.state.generation_id == original_generation
    assert recovered_goal.next_continuation_prompt() == automatic_two.text

    canonical_case = {
        "case_id": "case:production-goal-acceptance",
        "next_action": {"kind": "task_resume", "next_step_id": "verify"},
        "workspace": {
            "plan_event_id": "event:production-goal-acceptance",
            "plan": {
                "plan_id": "plan:production-goal-acceptance",
                "revision": 2,
                "objective": "finish every approved step",
                "state": "active",
                "current_step_id": "verify",
                "resume_cursor": {
                    "next_step_id": "verify",
                    "summary": "implementation is complete; verify next",
                },
                "steps": [
                    {
                        "id": "implement",
                        "content": "implement",
                        "status": "completed",
                    },
                    {
                        "id": "verify",
                        "content": "verify",
                        "status": "in_progress",
                    },
                ],
            },
            "remaining_step_ids": ["verify"],
            "verifications": [],
            "approvals": [],
            "capability_checks": [],
        },
    }
    monkeypatch.setattr(
        workspace,
        "_candidate_case_ids",
        lambda _thread_id, *, deadline: (
            ["case:production-goal-acceptance"],
            False,
            None,
        ),
    )
    monkeypatch.setattr(
        workspace,
        "_resume_case",
        lambda _case_id, *, deadline: (canonical_case, None),
    )
    recovered_todos = TodoStore()
    canonical_recovery = workspace.prepare_task_workspace_resume(
        thread_id="thread:production-goal-acceptance",
        session_key=session_key,
        todo_store=recovered_todos,
        boundary=workspace.BOUNDARY_RESTART_RESUME,
    )
    assert canonical_recovery["status"] == "exact"
    assert canonical_recovery["todo_hydrated"] is True
    assert recovered_todos.read()[-1]["status"] == "in_progress"

    restarted_agent = _agent_for_process()
    assert _identity(restarted_agent) == stable_identity

    # The recovered continuation records another exact continue, but a real
    # user message already waiting is the next goal turn and must preempt the
    # synthetic event without being displaced or duplicated.
    user_event = MessageEvent(
        text="Use the production-shaped verification evidence I just supplied.",
        message_type=MessageType.TEXT,
        source=src,
        message_id="real-user-preemption",
    )
    restarted_adapter._pending_messages[session_key] = user_event
    await _finish_turn(
        restarted_runner,
        restarted_agent,
        outcome="continue",
        reason="incorporate the user's new verification evidence",
        final_response="I will incorporate the newly supplied evidence.",
    )
    assert restarted_adapter._pending_messages[session_key] is user_event
    assert session_key not in restarted_runner._queued_events

    # That real user turn completes the goal through the primary model's exact
    # structured outcome; no response-text classifier participates.
    restarted_adapter._pending_messages.pop(session_key)
    await _finish_turn(
        restarted_runner,
        restarted_agent,
        outcome="complete",
        reason="all approved steps are verified",
        final_response="All approved steps are verified and complete.",
    )
    completed = GoalManager(session_id).state
    assert completed is not None
    assert completed.status == "done"
    assert completed.turns_used == 4
    assert completed.last_verdict == "done"
    assert completed.last_reason == "all approved steps are verified"
    assert restarted_adapter._pending_messages == {}
    assert restarted_runner._queued_events == {}
