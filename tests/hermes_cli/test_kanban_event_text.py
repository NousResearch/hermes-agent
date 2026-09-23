"""Notices must name the recorded event, not a different failure."""

from hermes_cli.kanban_event_text import gave_up_count, gave_up_reason, timeout_promises_retry
from tui_gateway.session_notifications import _kb_gave_up, _kb_timed_out


def test_timeout_does_not_promise_a_retry_unless_the_event_says_so():
    assert timeout_promises_retry({"will_retry": False, "retry_status": "ready"}) is False
    assert timeout_promises_retry({}) is False
    assert timeout_promises_retry({"will_retry": True}) is True
    spent = _kb_timed_out(None, {"limit_seconds": 3600, "will_retry": False}, "t")
    assert "will not retry" in spent
    assert "will retry" not in spent.replace("will not retry", "")
    open_retry = _kb_timed_out(None, {"limit_seconds": 3600, "will_retry": True}, "t")
    assert "will retry" in open_retry


def test_one_timeout_is_not_repeated_spawn_failures():
    payload = {
        "failures": 1,
        "effective_limit": 1,
        "trigger_outcome": "timed_out",
        "error": "elapsed 3600s > limit 3600s",
    }
    assert gave_up_reason(payload) == "the time limit was reached"
    assert "repeated" not in gave_up_count(payload)
    text = _kb_gave_up(None, payload, "t")
    assert "spawn" not in text
    assert "time limit" in text
    assert "1 failure" in text
    assert gave_up_reason({"trigger_outcome": "spawn_failed"}) == "it failed to start"
