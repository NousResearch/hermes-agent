"""A preempted review retains its original position relative to newer snapshots."""

import threading
from types import SimpleNamespace

import pytest

from agent import review_idle_queue
from run_agent import AIAgent


@pytest.mark.parametrize("retry_first", [False, True])
def test_preempted_retry_cannot_replace_a_newer_snapshot(monkeypatch, retry_first):
    queue = review_idle_queue.ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    monkeypatch.setattr(review_idle_queue, "review_targets_managed_local", lambda *_: True)
    now = [10.0]
    queue._now = lambda: now[0]
    parent = SimpleNamespace(session_id="session", _REVIEW_REQUEUE_MAX_ATTEMPTS=3)
    old = [{"role": "user", "content": "Original task"}]
    new = old + [{"role": "user", "content": "The corrected requirement"}]
    queue.enqueue(parent, "session", {"messages_snapshot": old, "task_cfg": {}})
    now[0] += 2000
    running = queue._pop_dispatchable()
    assert running is not None
    retry = dict(running.kwargs, _requeue_attempts=1)
    cancelled = threading.Event()
    cancelled.set()

    def requeue():
        AIAgent._maybe_requeue_preempted_review(
            parent, SimpleNamespace(cancel_requested=cancelled), retry)

    if retry_first:
        requeue()
    now[0] += 1
    queue.enqueue(parent, "session", {"messages_snapshot": new, "task_cfg": {}})
    admitted_at = queue._pending["session"].enqueued_at
    if not retry_first:
        requeue()
    pending = queue._pending["session"]
    assert pending.kwargs["messages_snapshot"] == new
    assert pending.kwargs.get("_requeue_attempts", 0) == 0
    assert pending.enqueued_at == admitted_at


def test_spawn_preserves_snapshot_age_through_the_worker(monkeypatch):
    from agent import background_review

    done = threading.Event()
    captured = []
    run = SimpleNamespace()
    monkeypatch.setattr(background_review, "prepare_background_review_run", lambda _: run)
    monkeypatch.setattr(background_review, "spawn_background_review_thread", lambda *a, **k: (lambda: None, ""))

    def capture(finished, kwargs):
        captured.append((finished, kwargs))
        done.set()

    parent = SimpleNamespace(_maybe_requeue_preempted_review=capture)
    AIAgent._spawn_background_review_now(
        parent, [{"role": "user", "content": "Task"}], _review_snapshot_created_at=12.5)
    assert done.wait(10)
    assert captured[0][0] is run
    assert captured[0][1]["_review_snapshot_created_at"] == 12.5
    assert captured[0][1]["_requeue_attempts"] == 1
