"""A background notice lands in the running turn at its next tool boundary.

``agent.notify_between_tool_calls`` is the background-completion counterpart of ``/steer``: the
text rides the same role-alternation-safe slot (a standalone ``role:user`` row right after the
newest tool result) but carries its OWN marker — a process notice is machine output and must not
borrow the user-authority steer marker.
"""
import threading

from agent.agent_runtime_helpers import apply_pending_notifications_to_tool_results
from agent.interrupt_control import InterruptControlMixin
from agent.prompt_builder import (
    CONTROL_FRAME_OPENERS,
    NOTIFICATION_MARKER_CLOSE,
    NOTIFICATION_MARKER_OPEN,
    STEER_MARKER_OPEN,
    notification_user_row,
)


class _Agent(InterruptControlMixin):
    """Minimal agent carrying only the mid-turn notification slot."""

    def __init__(self):
        self._pending_notifications = []
        self._pending_notifications_lock = threading.Lock()


NOTICE = "[IMPORTANT: Background process proc_abc completed normally (exit code 0).]"


def _tool_row():
    return {"role": "tool", "tool_call_id": "call_1", "content": "done"}


def test_queued_notice_is_delivered_after_the_tool_batch():
    agent = _Agent()
    assert agent.notify_between_tool_calls(NOTICE) is True
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}, _tool_row()]

    apply_pending_notifications_to_tool_results(agent, messages, 1)

    assert len(messages) == 3, messages
    delivered = messages[-1]
    assert delivered["role"] == "user"
    assert delivered["display_kind"] == "internal_notification"
    assert NOTICE in delivered["content"]
    assert NOTIFICATION_MARKER_OPEN in delivered["content"]
    assert delivered["content"].rstrip().endswith(NOTIFICATION_MARKER_CLOSE)
    assert agent._drain_pending_notifications() == [], "the drain must leave nothing queued"


def test_process_notice_never_carries_user_authority():
    """The steer marker grants full user authority; a process notice must never wear it."""
    row = notification_user_row(NOTICE)
    assert STEER_MARKER_OPEN not in row["content"]
    assert "direct message from the user" not in row["content"]
    assert "not written by a human user" in row["content"]


def test_notice_marker_is_a_registered_control_frame():
    """Hosted rooms relabel Hermes' own control frames; the new marker must be in that list."""
    assert any("INTERNAL NOTIFICATION" in opener for opener in CONTROL_FRAME_OPENERS)


def test_idle_batch_adds_no_row():
    agent = _Agent()
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}, _tool_row()]
    apply_pending_notifications_to_tool_results(agent, messages, 1)
    assert len(messages) == 2


def test_notice_without_a_tool_result_is_requeued_not_lost():
    """A batch that produced no tool row has no legal anchor: the notice waits for the next one."""
    agent = _Agent()
    agent.notify_between_tool_calls(NOTICE)
    messages = [{"role": "assistant", "content": "thinking"}]

    apply_pending_notifications_to_tool_results(agent, messages, 0)

    assert len(messages) == 1
    assert agent._drain_pending_notifications() == [NOTICE], "the notice was dropped"


def test_batched_notices_each_get_their_own_row_in_arrival_order():
    agent = _Agent()
    first = "[IMPORTANT: Background process proc_1 exited (exit code 7).]"
    second = "[IMPORTANT: Background process proc_2 exited (exit code 0).]"
    agent.notify_between_tool_calls(first)
    agent.notify_between_tool_calls(second)
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}, _tool_row()]

    apply_pending_notifications_to_tool_results(agent, messages, 1)

    appended = [m for m in messages[2:]]
    assert len(appended) == 2, appended
    assert first in appended[0]["content"] and second in appended[1]["content"]


def test_empty_text_is_refused():
    agent = _Agent()
    assert agent.notify_between_tool_calls("") is False
    assert agent.notify_between_tool_calls("   ") is False
    assert agent._drain_pending_notifications() == []


class _Runner:
    """The routing predicate in isolation: only the agent lookup is stubbed."""

    _deliver_notice_mid_turn = __import__(
        "gateway.run_notifications", fromlist=["_deliver_notice_mid_turn"]
    ).GatewayNotificationsMixin._deliver_notice_mid_turn

    def __init__(self, agent):
        self.agent = agent

    def _midturn_notice_agent(self, source):
        return self.agent


def _src():
    return type("S", (), {"chat_id": "chat-1"})()


def test_completion_is_steered_into_a_running_turn():
    agent = _Agent()
    runner = _Runner(agent)
    assert runner._deliver_notice_mid_turn({"type": "completion", "session_id": "p1"}, NOTICE, _src()) is True
    assert agent._drain_pending_notifications() == [NOTICE]


def test_watch_match_is_steered_into_a_running_turn():
    agent = _Agent()
    runner = _Runner(agent)
    evt = {"type": "watch_match", "session_id": "p1", "pattern": "READY"}
    assert runner._deliver_notice_mid_turn(evt, "matched READY", _src()) is True
    assert agent._drain_pending_notifications() == ["matched READY"]


def test_idle_session_keeps_the_admit_as_a_turn_path():
    assert _Runner(None)._deliver_notice_mid_turn({"type": "completion"}, NOTICE, _src()) is False


def test_delegation_results_keep_their_durable_admit_path():
    """An async_delegation carries a durable delivery claim and a full task wall."""
    agent = _Agent()
    runner = _Runner(agent)
    evt = {"type": "async_delegation", "delegation_id": "d1"}
    assert runner._deliver_notice_mid_turn(evt, "subagent result", _src()) is False
    assert agent._drain_pending_notifications() == []


def test_an_agent_without_the_seam_falls_back_to_a_turn():
    class _NoSeam:
        pass

    assert _Runner(_NoSeam())._deliver_notice_mid_turn({"type": "completion"}, NOTICE, _src()) is False


def test_a_refused_queue_falls_back_to_a_turn():
    """``notify_between_tool_calls`` returning False means the notice was NOT queued mid-turn: the
    caller must still admit it as its own turn rather than report it delivered."""
    agent = _Agent()
    agent.notify_between_tool_calls = lambda text: False

    assert _Runner(agent)._deliver_notice_mid_turn({"type": "completion"}, NOTICE, _src()) is False


def test_a_raising_queue_never_loses_the_notice():
    agent = _Agent()

    def _boom(text):
        raise RuntimeError("queue exploded")

    agent.notify_between_tool_calls = _boom

    assert _Runner(agent)._deliver_notice_mid_turn({"type": "completion"}, NOTICE, _src()) is False

