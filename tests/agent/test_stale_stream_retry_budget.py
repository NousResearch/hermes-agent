"""Per-call no-first-chunk retry budget.

A single 60–70k request used to be stale-killed, retried by the stream loop, then
rebuilt by the outer turn loop, so one wedged request could burn 8–9 × 240s and
the pacer charge for every attempt. The per-call counter caps that within one
``_StreamingCall`` and leaves the turn to the fallback classifier.
"""

import threading
from types import SimpleNamespace

from agent.chat_completion_helpers import _StreamingCall, _stale_stream_budget_exhausted


def _bare_call(attempt: int = 1) -> _StreamingCall:
    call = object.__new__(_StreamingCall)
    call.agent = SimpleNamespace()
    call.stream_attempt_lock = threading.Lock()
    call.stream_attempt_state = {"current": attempt}
    call._stale_counted_attempts = set()
    call._stale_kills = 0
    return call


def test_budget_is_exhausted_at_the_limit():
    assert not _stale_stream_budget_exhausted(0, 2)
    assert not _stale_stream_budget_exhausted(1, 2)
    assert _stale_stream_budget_exhausted(2, 2)
    assert _stale_stream_budget_exhausted(3, 2)


def test_zero_limit_disables_the_per_call_budget():
    assert not _stale_stream_budget_exhausted(99, 0)


def test_counter_counts_each_attempt_once():
    call = _bare_call()

    call._count_stale_attempt()
    call._count_stale_attempt()  # same attempt re-killed while unwinding: not another try

    assert call._stale_kills == 1


def test_counter_tracks_new_attempts():
    call = _bare_call()

    call._count_stale_attempt()
    call.stream_attempt_state["current"] = 2
    call._count_stale_attempt()

    assert call._stale_kills == 2
    assert _stale_stream_budget_exhausted(call._stale_kills, 2)
