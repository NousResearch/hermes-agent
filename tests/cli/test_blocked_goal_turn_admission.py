"""A repair message resumes a judge-blocked goal before CLI model work."""

import copy
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cli import HermesCLI
from hermes_cli.goals import GoalManager


@pytest.fixture
def blocked_cli(monkeypatch):
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    mgr = GoalManager("cli-repair", default_max_turns=5)
    mgr.set("finish the repair")
    with patch("hermes_cli.goals.judge_goal", return_value=("blocked", "needs user input", False, None, False)):
        mgr.evaluate_after_turn("Please supply the missing input.")
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = mgr.session_id
    cli._goal_manager = mgr
    cli._pending_input = queue.Queue()
    cli.conversation_history = []
    cli.agent = SimpleNamespace(session_id=mgr.session_id)
    cli._active_agent_route_signature = "test"
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda message: {"signature": "test", "model": "test", "runtime": {}}
    cli._init_agent = lambda **kwargs: True
    cli._chat_route_images = lambda message, images: message
    cli._chat_expand_context_references = lambda message: (message, None)
    cli._reset_stream_state = lambda: None
    cli._chat_setup_turn_audio = lambda *args: None
    cli._flush_credit_notices = lambda: None
    cli._chat_monitor_agent_thread = lambda turn, thread: thread.join(timeout=5)
    cli._chat_settle_turn = lambda turn: None
    cli._chat_render_turn = lambda turn, thread, interrupt: turn.result
    monkeypatch.setattr("cli.ChatConsole", Mock())
    yield cli, mgr
    goals._DB_CACHE.clear()


@pytest.mark.parametrize("outcome", ["success", "raise", "blocked-again", "stop"])
def test_repair_is_active_during_cli_run_even_when_it_fails(blocked_cli, outcome):
    cli, mgr = blocked_cli
    before = copy.deepcopy(mgr.state)
    observed = []

    def run_conversation(**kwargs):
        observed.append(GoalManager(mgr.session_id).state)
        if outcome == "raise":
            raise RuntimeError("provider unavailable")
        if outcome == "blocked-again":
            mgr.pause(reason=before.paused_reason)
        elif outcome == "stop":
            mgr.pause(reason="user-interrupted (Ctrl+C)")
        return {"final_response": "repair underway", "messages": []}

    cli.agent.run_conversation = run_conversation
    result = cli.chat("Here is the missing input")
    assert len(observed) == 1
    assert observed[0].status == "active"
    assert observed[0].paused_reason is None
    assert observed[0].turns_used == before.turns_used
    assert observed[0].max_turns == before.max_turns
    # Completion must not revive a NEW pause made during this run.
    cli.conversation_history.append({"role": "assistant", "content": "" if outcome == "raise" else "repair underway"})
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "more work", False, None, False)) as judge:
        cli._maybe_continue_goal_after_turn()
    if outcome in {"blocked-again", "stop"}:
        assert mgr.state.status == "paused"
        judge.assert_not_called()
        assert cli._pending_input.empty()
    elif outcome == "raise":
        assert result["failed"] is True
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == before.turns_used
        judge.assert_not_called()
    else:
        assert mgr.state.status == "active"
        assert not cli._pending_input.empty()


@pytest.mark.parametrize("case", [
    "user-paused", "user-interrupted (Ctrl+C)", "budget", "rejected", "blank",
    "goal", "heartbeat", "loop", "notification", "batch", "watch", "multimodal",
])
def test_cli_non_user_or_rejected_input_cannot_revive(blocked_cli, case):
    from tools.process_registry_notifications import (
        ProcessNotificationBatch, SubagentNotification, format_process_notification,
    )

    cli, mgr = blocked_cli
    message = "Here is the missing input"
    if case in {"user-paused", "user-interrupted (Ctrl+C)"}:
        mgr.pause(reason=case)
    elif case == "budget":
        mgr.pause(reason="turn budget exhausted (5/5)")
    elif case == "rejected":
        cli._chat_expand_context_references = lambda message: (message, "Context injection refused")
    elif case == "blank":
        message = "  "
    elif case == "goal":
        message = "[Continuing toward your standing goal]\nGoal: finish the repair"
    elif case == "heartbeat":
        message = "[Heartbeat — periodic check]"
    elif case == "loop":
        message = "[/loop wakeup #1]"
    elif case == "notification":
        message = SubagentNotification("unframed worker result", {"goal": "repair", "status": "completed"})
    elif case == "batch":
        message = ProcessNotificationBatch((({}, "first result"), ({}, "second result"))).render(
            SimpleNamespace(is_completion_consumed=lambda sid: False))
    elif case == "watch":
        message = format_process_notification({"type": "watch_disabled", "message": "watch limit reached"})
    elif case == "multimodal":
        message = [{"type": "text", "text": "[System: background notification]"}]
    # Enrichment may obscure the synthetic prefix: classify the ORIGINAL input.
    if case == "goal":
        cli._chat_route_images = lambda message, images: [{"type": "text", "text": message}]
    before = copy.deepcopy(mgr.state)
    observed = []
    cli.agent.run_conversation = lambda **kwargs: observed.append(GoalManager(mgr.session_id).state) or {}
    cli.chat(message)
    if case == "rejected":
        assert not observed
    else:
        assert len(observed) == 1
        assert observed[0].status == before.status
        assert observed[0].paused_reason == before.paused_reason
    assert mgr.state.status == before.status
    assert mgr.state.paused_reason == before.paused_reason
    assert mgr.state.turns_used == before.turns_used
