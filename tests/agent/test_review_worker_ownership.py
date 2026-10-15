"""A completed request is not a completed review: delivery belongs to the worker."""
from __future__ import annotations

import contextvars
import threading
from types import SimpleNamespace

import pytest

from agent import background_review as bg
from run_agent import AIAgent


def _parent():
    agent = object.__new__(AIAgent)
    agent.session_id = "review-owner"
    agent._background_review_lock = threading.Lock()
    agent._background_review_run = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._maybe_requeue_preempted_review = lambda *args: None
    return agent


@pytest.mark.parametrize("surface", ["query", "oneshot"])
def test_exit_waits_through_receipt_delivery(monkeypatch, tmp_path, surface):
    """Drive the actual spawn + CLI teardown, with a gated provider/receipt tail."""
    import cli
    from hermes_cli import oneshot
    agent = _parent()
    request_done, release, receipt, entered, closed = [threading.Event() for _ in range(5)]
    errors = []

    def build(parent, *args, review_run=None, **kwargs):
        def target():
            try:
                assert review_run.begin_request(SimpleNamespace())
                bg.finish_background_review_run(parent, review_run)
                request_done.set()
                assert release.wait(5)
                # Real file I/O at the tail that used to be killed at process exit.
                (tmp_path / "delivered.txt").write_text("review receipt")
                receipt.set()
            except BaseException as exc:
                errors.append(exc)
        return target, "review"

    monkeypatch.setattr(bg, "spawn_background_review_thread", build)
    agent._spawn_background_review_now([], review_skills=True, task_cfg={})
    assert request_done.wait(5)

    def close():
        closed.set()
    agent.close = close
    agent.shutdown_memory_provider = lambda *args: None
    agent._session_messages = []
    if surface == "query":
        monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", lambda *_: entered.set())
        monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda *_: None)
        monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda *_: None)
        monkeypatch.setattr(cli, "_run_cleanup", lambda **_: close())
        shell = SimpleNamespace(agent=agent, _release_active_session=lambda: None)
        shutdown = lambda: cli._finalize_single_query(shell)
    else:
        monkeypatch.setattr(oneshot, "_linger_for_background_completions", lambda: entered.set())
        shutdown = lambda: oneshot._close_agent(agent, None)
    worker = threading.Thread(target=shutdown)
    worker.start()
    try:
        assert entered.wait(5)
        assert not closed.wait(0.1), "teardown raced the review's receipt tail"
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert closed.is_set() and receipt.is_set()
        assert (tmp_path / "delivered.txt").read_text() == "review receipt"
        assert not errors
    finally:
        release.set()
        worker.join(5)
        assert receipt.wait(5)


def test_timeout_fences_queued_startup_and_preserves_profile(monkeypatch, caplog):
    from agent import review_idle_queue as idle
    from agent.review_lifecycle import drain_background_reviews, has_pending_review
    queue = idle.ReviewIdleQueue()
    monkeypatch.setattr(idle, "QUEUE", queue)
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    profile = contextvars.ContextVar("review-test-profile", default="wrong")
    parent = _parent()
    token = profile.set("right")
    try:
        queue.enqueue(parent, "profile::session", {"task_cfg": {"defer_max_age_s": 1}})
    finally:
        profile.reset(token)
    queue._now = lambda: 10**12
    item = queue._pop_dispatchable()
    assert item is not None
    assert item.context.run(profile.get) == "right"
    assert has_pending_review(parent), "pop-to-spawn gap lost ownership"
    assert drain_background_reviews(parent, timeout=0) is False
    assert "timeout" in caplog.text.lower()
    assert bg.prepare_background_review_run(parent) is None
    assert queue.enqueue(parent, "another", {}) is False
    # The dispatcher owns removing its in-flight item even after cancellation.
    with queue._lock:
        queue._dispatching.clear()
    assert not has_pending_review(parent)
