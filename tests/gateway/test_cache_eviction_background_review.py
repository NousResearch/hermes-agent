"""Soft cache eviction must cancel the discarded parent's background review."""

import threading
from types import SimpleNamespace
from typing import Any

from agent.background_review import (
    finish_background_review_run,
    prepare_background_review_run,
)
from gateway.run_agent_cache import GatewayAgentCacheMixin
from run_agent import AIAgent


def test_soft_eviction_cancels_review_before_detaching_child_clients(monkeypatch):
    parent: Any = object.__new__(AIAgent)
    parent._background_review_lock = threading.Lock()
    parent._background_review_run = None
    parent._background_review_agent = None
    parent._active_children_lock = threading.Lock()
    parent._session_messages = [{"role": "user", "content": "saved transcript"}]
    parent._db_flush_scan_prefix = list(parent._session_messages)
    closed_tools = []
    monkeypatch.setattr(
        AIAgent,
        "_close_task_resources",
        lambda self, session_id: closed_tools.append(session_id),
    )
    run = prepare_background_review_run(parent)
    assert run is not None
    started = threading.Event()
    interrupted = threading.Event()
    released = []

    class Review:
        def hard_interrupt(self, message=None, *, tool_reason=None):
            interrupted.set()

        def release_clients(self):
            released.append(run.request_done.is_set())

    review = Review()
    parent._active_children = {review}
    parent._background_review_agent = review
    assert run.begin_request(review)

    def review_worker():
        try:
            started.set()
            interrupted.wait(timeout=5)
        finally:
            finish_background_review_run(parent, run)

    worker = threading.Thread(target=review_worker, daemon=True)
    worker.start()
    try:
        assert started.wait(timeout=5)
        runner = object.__new__(GatewayAgentCacheMixin)
        runner._release_evicted_agent_soft(parent)
        assert interrupted.is_set(), "The discarded parent's review must be interrupted"
        assert released == [True], (
            "Wait for review acknowledgement before detaching clients"
        )
        assert parent._active_children == set()
        assert parent._session_messages == []
        assert parent._db_flush_scan_prefix is None
        assert closed_tools == [], (
            "Soft eviction must preserve resumable tool resources"
        )
    finally:
        interrupted.set()
        worker.join(timeout=5)
    assert not worker.is_alive()


def test_soft_eviction_still_releases_when_review_retirement_fails(monkeypatch):
    from agent import review_idle_queue

    def broken_purge(agent):
        raise RuntimeError("queue purge failed")

    monkeypatch.setattr(review_idle_queue.QUEUE, "discard_parent", broken_purge)
    released = []
    parent = SimpleNamespace(
        _background_review_lock=threading.Lock(),
        _background_review_run=None,
        _background_review_agent=None,
        _session_messages=[{"role": "user", "content": "large transcript"}],
        _db_flush_scan_prefix=[{"role": "user", "content": "large transcript"}],
        release_clients=lambda: released.append(True),
    )
    runner = object.__new__(GatewayAgentCacheMixin)
    runner._release_evicted_agent_soft(parent)
    assert released == [True], "A retirement failure must not skip client release"
    assert parent._session_messages == []
    assert parent._db_flush_scan_prefix is None


def test_soft_eviction_fences_a_review_not_yet_started(monkeypatch):
    from agent import background_review

    monkeypatch.setattr(
        background_review, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0
    )
    released = []
    parent = SimpleNamespace(
        _background_review_lock=threading.Lock(),
        _background_review_run=None,
        _background_review_agent=None,
        release_clients=lambda: released.append(True),
    )
    run = prepare_background_review_run(parent)
    assert run is not None
    try:
        runner = object.__new__(GatewayAgentCacheMixin)
        runner._release_evicted_agent_soft(parent)
        assert not run.begin_request(object()), (
            "A queued review must not start after eviction"
        )
        assert released == [True]
    finally:
        finish_background_review_run(parent, run)
