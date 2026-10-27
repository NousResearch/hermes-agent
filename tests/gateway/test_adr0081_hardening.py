"""Regression tests for the three defects found in review of ADR-0081's task gate.

Each test names the failure mode it pins down:

1. the per-session task registry grows without bound
2. a close *inferred* from "turn ended, nothing in flight" swallows a result
   whose job was already running -- the completion is lost
3. a refused result is filed on the session's *current* task instead of the task
   that owns it, so the record points at the wrong work
"""
import pytest

from gateway.task_lifecycle import (
    TASK_CANCELLED,
    TASK_DRAINED,
    TaskLifecycleRegistry,
)

SESSION = "agent:main:telegram:dm:12345"


# -- 1. bounded registry ------------------------------------------------

def test_task_history_stays_bounded_across_many_turns():
    reg = TaskLifecycleRegistry()
    for i in range(500):
        reg.begin(SESSION, task_id=f"t{i}")
        reg.close(SESSION, TASK_DRAINED, reason="turn ended without pending work",
                  inferred=True)
    assert len(reg.tasks(SESSION)) <= TaskLifecycleRegistry.MAX_TASKS_PER_SESSION


def test_trimming_never_drops_the_current_task():
    reg = TaskLifecycleRegistry()
    for i in range(200):
        reg.begin(SESSION, task_id=f"t{i}")
    current = reg.get(SESSION)
    assert current is not None
    assert current.task_id == "t199"
    assert current.is_closed is False


def test_refused_history_per_task_stays_bounded():
    reg = TaskLifecycleRegistry()
    reg.begin(SESSION, task_id="t1")
    reg.close(SESSION, TASK_CANCELLED, reason="user stop")
    for i in range(500):
        reg.record_refusal(SESSION, reason="late", detail={"i": i}, task_id="t1")
    assert len(reg.tasks(SESSION)[0].refused) <= TaskLifecycleRegistry.MAX_REFUSED_PER_TASK


def test_closed_sessions_are_evicted_before_active_ones():
    reg = TaskLifecycleRegistry()
    reg.begin("s-old")
    reg.close("s-old", TASK_DRAINED, reason="turn ended without pending work",
              inferred=True)
    reg.begin("s-live")
    reg._evict_closed_sessions(limit=1)
    assert reg.tasks("s-old") == []
    assert reg.tasks("s-live"), "an active session must never be evicted"


# -- 3. refusal attribution --------------------------------------------

def test_refusal_belongs_to_the_owning_task_not_the_current_one():
    reg = TaskLifecycleRegistry()
    owner = reg.begin(SESSION, task_id="t1")
    reg.close(SESSION, TASK_CANCELLED, reason="user stop")
    fresh = reg.begin(SESSION, task_id="t2")  # a new user turn re-armed the session

    reg.record_refusal(SESSION, reason="late", detail={"process_id": "p1"},
                       task_id="t1")

    assert len(owner.refused) == 1, "refusal was not filed on the owning task"
    assert fresh.refused == [], "refusal was filed on the re-armed task"


# -- 2. inferred close must not swallow a legitimate result -------------

def test_inferred_close_still_delivers_a_result_that_started_under_the_task():
    """Closed because the turn ended with nothing in flight -- but the job had
    already started.  The close was a guess; the result is legitimate work."""
    reg = TaskLifecycleRegistry()
    reg.begin(SESSION, task_id="t1")
    started_at = reg.get(SESSION).started_at - 1.0
    reg.close(SESSION, TASK_DRAINED, reason="turn ended without pending work",
              inferred=True)

    assert reg.may_start_work(SESSION, started_at=started_at, task_id="t1") is True


def test_inferred_close_still_refuses_a_result_started_after_the_task_ended():
    """The leniency covers jobs that ALREADY RAN under the task.  A job that
    only started after the task was closed does not belong to it."""
    reg = TaskLifecycleRegistry()
    reg.begin(SESSION, task_id="t1")
    owner = reg.get(SESSION)
    reg.close(SESSION, TASK_DRAINED, reason="turn ended without pending work",
              inferred=True)

    assert reg.may_start_work(SESSION, started_at=owner.closed_at + 10.0,
                              task_id="t1") is False


def test_declared_close_refuses_even_when_the_result_started_under_the_task():
    """A marker or an explicit /stop is a statement, not a guess -- it refuses."""
    reg = TaskLifecycleRegistry()
    reg.begin(SESSION, task_id="t1")
    started_at = reg.get(SESSION).started_at - 1.0
    reg.close(SESSION, TASK_CANCELLED, reason="user stop")

    assert reg.may_start_work(SESSION, started_at=started_at, task_id="t1") is False
