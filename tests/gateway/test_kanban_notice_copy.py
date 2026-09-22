"""Kanban terminal-event pings must say what state the task is in and name the next command.

`gave_up` means the dispatcher auto-blocked the task (spawn failure, crash or timeout alike) and a
human must act; `crashed`/`timed_out` retry on their own. `timed_out` covers two terminal causes
under one kind — the dispatcher's wall-clock cap (`limit_seconds`) and the worker's own iteration
budget (`agent/turn_finalizer.py`) — and the ping must name the one that actually stopped the task,
never a limit the task does not have. Contract tests over `_EVENT_FORMATTERS`, never whole-string
snapshots.
"""

from types import SimpleNamespace

from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS, timed_out_cause


def _names(task_id="T-123"):
    return SimpleNamespace(task_id=task_id, head=f"[board] Kanban {task_id}", title="Ship it", board_tag="[board] ")


def _event(**payload):
    return SimpleNamespace(payload=payload)


def test_gave_up_says_blocked_and_names_unblock_log_reassign():
    msg, _wake, _reason = _EVENT_FORMATTERS["gave_up"](
        _event(failures=3, error="spawn: profile 'coder' not found"), _names())
    assert "blocked" in msg.lower()
    assert "3 times" in msg
    assert "spawn: profile 'coder' not found" in msg
    for cmd in ("hermes kanban unblock T-123", "hermes kanban log T-123", "hermes kanban reassign T-123"):
        assert f"`{cmd}`" in msg
    assert "spawn failures" not in msg  # wrong for crash/timeout-triggered trips


def test_crashed_and_timed_out_say_retry_and_hide_internals():
    crashed, *_ = _EVENT_FORMATTERS["crashed"](_event(), _names())
    timed_out, *_ = _EVENT_FORMATTERS["timed_out"](_event(limit_seconds=1800), _names())
    for msg in (crashed, timed_out):
        assert "retried automatically" in msg
        assert "pid" not in msg and "max_runtime" not in msg
    assert "30-minute" in timed_out


# The two terminal causes of ``timed_out``, as the dispatcher actually records them:
# ``enforce_max_runtime`` stamps ``limit_seconds``; the worker's budget exit records
# "Iteration budget exhausted (used/max)" in ``error`` and no cap at all.
_WALL_CLOCK_PAYLOAD = {"pid": 4242, "elapsed_seconds": 1875, "limit_seconds": 1800,
                       "sigkill": False, "retry_status": "ready"}
_ITERATION_BUDGET_PAYLOAD = {
    "error": "Iteration budget exhausted (150/150) — task could not complete within the "
             "allowed iterations",
    "failures": 1, "retry_status": "ready",
}


def _timed_out(**payload):
    msg, *_ = _EVENT_FORMATTERS["timed_out"](_event(**payload), _names())
    return msg


def test_timed_out_by_wall_clock_cap_names_its_limit():
    msg = _timed_out(**_WALL_CLOCK_PAYLOAD)
    assert "30-minute" in msg
    assert "retried automatically" in msg
    assert "max_runtime" not in msg and "pid" not in msg
    assert "iteration" not in msg.lower()


def test_timed_out_by_iteration_budget_says_so_and_names_no_limit():
    """A task whose ``max_runtime_seconds`` is NULL has no cap — the ping must not invent one."""
    msg = _timed_out(**_ITERATION_BUDGET_PAYLOAD)
    assert "iteration budget" in msg.lower()
    assert "150/150" in msg
    assert "retried automatically" in msg
    assert "max_runtime" not in msg
    assert "limit" not in msg.lower()


def test_timed_out_without_a_recorded_cause_invents_no_number():
    """Legacy/odd payloads carry neither signal: retry still stands, no cause is claimed."""
    msg = _timed_out(failures=1, retry_status="ready")
    assert "retried automatically" in msg
    assert "max_runtime" not in msg
    assert "0s" not in msg and "limit" not in msg.lower() and "iteration" not in msg.lower()


def test_timed_out_cause_reads_the_payloads_the_dispatcher_writes():
    assert timed_out_cause(_ITERATION_BUDGET_PAYLOAD) == ("iteration_budget", 150, 150)
    assert timed_out_cause(_WALL_CLOCK_PAYLOAD) == ("limit", 1800, 0)
    assert timed_out_cause({"failures": 1, "retry_status": "ready"}) == ("unknown", 0, 0)
    # Malformed input degrades to "unknown" (no cause claimed) instead of raising inside a notifier.
    assert timed_out_cause({"limit_seconds": "not-a-number"}) == ("unknown", 0, 0)
    assert timed_out_cause(None) == ("unknown", 0, 0)
