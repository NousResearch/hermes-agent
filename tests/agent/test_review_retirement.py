"""Discarded parents must not resume managed-local deferred reviews."""

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from agent import background_review, review_idle_queue
from gateway.run_agent_cache import GatewayAgentCacheMixin
import run_agent
from run_agent import AIAgent


@pytest.fixture
def reviews(monkeypatch):
    """Drive real spawn/worker/requeue orchestration with inert forks and a manual clock."""
    queue = review_idle_queue.ReviewIdleQueue()
    clock = [0.0]
    queue._now = lambda: clock[0]
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    monkeypatch.setattr(
        background_review,
        "load_background_review_settings",
        lambda: (True, {"defer": "auto", "defer_max_age_s": 1}),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._managed_local_netloc", lambda: "127.0.0.1:9999"
    )
    monkeypatch.setattr(
        background_review,
        "_resolve_review_runtime",
        lambda agent, cfg: {"base_url": "http://127.0.0.1:9999/v1"},
    )
    monkeypatch.setattr(
        background_review, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0
    )
    targets = []
    calls = []
    during_request = []

    class ManualThread:
        def __init__(self, *, target, daemon=None, name=None):
            self.target, self.name = target, name

        def start(self):
            targets.append((self.name, self.target))

    class Fork:
        _session_messages = []

        def __init__(self, parent):
            self.parent = parent

        def run_conversation(self, **kwargs):
            calls.append((self.parent, kwargs["conversation_history"]))
            if during_request:
                during_request.pop(0)()

        def hard_interrupt(self, message=None, *, tool_reason=None):
            pass

        def release_clients(self):
            pass

    # Only the review worker spawned by run_agent is scheduled manually; every other
    # thread (cancel interrupts, idle-queue internals) stays a real thread.
    monkeypatch.setattr(
        run_agent,
        "threading",
        SimpleNamespace(**{**vars(threading), "Thread": ManualThread}),
    )
    monkeypatch.setattr(
        background_review,
        "build_cache_parity_fork",
        lambda parent, *args, **kwargs: (Fork(parent), {}, False),
    )
    monkeypatch.setattr(
        background_review, "_review_tool_whitelist", lambda *args: (set(), set())
    )

    def parent(session_id="shared-session"):
        agent: Any = object.__new__(AIAgent)
        agent.session_id = session_id
        agent._background_review_lock = threading.Lock()
        agent._background_review_run = agent._background_review_agent = None
        agent._active_children_lock = threading.Lock()
        agent._active_children = []
        agent._session_messages = []
        agent._safe_print = lambda *args: None
        agent.background_review_callback = None
        agent._emit_auxiliary_failure = lambda *args: pytest.fail(
            f"Unexpected review failure: {args}"
        )
        return agent

    def pop():
        clock[0] += 2  # age-out avoids both real time and the server-idle network probe
        item = queue._pop_dispatchable()
        assert item is not None
        return item

    def dispatch():
        item = pop()
        item.context.run(queue._dispatch, item)
        return item

    def drain():
        while targets:
            _, target = targets.pop(0)
            target()

    return SimpleNamespace(
        queue=queue,
        parent=parent,
        pop=pop,
        dispatch=dispatch,
        drain=drain,
        calls=calls,
        targets=targets,
        during_request=during_request,
        evict=object.__new__(GatewayAgentCacheMixin)._release_evicted_agent_soft,
    )


@pytest.mark.parametrize(
    "evicted", [False, True], ids=["foreground-preemption", "cache-eviction"]
)
def test_preempted_managed_local_review_only_requeues_a_live_parent(reviews, evicted):
    parent = reviews.parent()
    snapshot = [{"role": "user", "content": "review this turn"}]
    parent._spawn_background_review(snapshot, review_skills=True)
    reviews.dispatch()
    reviews.during_request.append(
        lambda: (
            reviews.evict(parent)
            if evicted
            else background_review.cancel_background_review_for_live_turn(parent)
        )
    )
    reviews.drain()  # real worker finishes and AIAgent invokes its real requeue callback
    assert reviews.calls == [(parent, snapshot)]
    assert reviews.queue.pending_count() == (0 if evicted else 1)
    if not evicted:
        reviews.dispatch()
        reviews.drain()
        assert reviews.calls == [(parent, snapshot), (parent, snapshot)]


@pytest.mark.parametrize(
    "popped", [False, True], ids=["queued", "popped-before-dispatch"]
)
def test_eviction_drops_discarded_parent_without_losing_same_key_successor(
    reviews, popped
):
    parent = reviews.parent()
    snapshot = [{"role": "user", "content": "discarded turn"}]
    parent._spawn_background_review(snapshot, review_skills=True)
    item = None
    if popped:
        item = reviews.pop()
    else:
        # Session rotation may leave this same instance under more than one queue key.
        parent.session_id = "rotated-session"
        parent._spawn_background_review(snapshot, review_skills=True)
        parent.session_id = "shared-session"
        assert reviews.queue.pending_count() == 2
    reviews.evict(parent)
    assert reviews.queue.pending_count() == 0

    successor = reviews.parent()  # same queue key, different parent identity
    successor_snapshot = [{"role": "user", "content": "successor turn"}]
    successor._spawn_background_review(successor_snapshot, review_skills=True)
    reviews.evict(
        parent
    )  # late/idempotent cleanup must not remove the successor's item
    parent._spawn_background_review(
        snapshot, review_skills=True
    )  # late enqueue cannot replace it
    if item is not None:
        item.context.run(reviews.queue._dispatch, item)
    assert reviews.targets == [], (
        "A discarded queued item must not spawn a review worker"
    )
    assert reviews.queue.pending_count() == 1
    reviews.dispatch()
    reviews.drain()
    assert reviews.calls == [(successor, successor_snapshot)]


def test_eviction_permanently_fences_direct_and_explicit_review_admission(reviews):
    parent = reviews.parent()
    snapshot = [{"role": "user", "content": "discarded turn"}]
    reviews.evict(parent)
    parent._spawn_background_review_now(snapshot, review_skills=True)
    parent._spawn_background_review(snapshot, review_skills=True, explicit=True)
    assert reviews.targets == []
    reviews.drain()
    assert reviews.calls == []
    assert reviews.targets == []
    assert parent._background_review_run is None


def test_eviction_preserves_a_successor_already_coalesced_before_retirement(reviews):
    parent, successor = reviews.parent(), reviews.parent()
    parent._spawn_background_review(
        [{"role": "user", "content": "old"}], review_skills=True
    )
    snapshot = [{"role": "user", "content": "successor"}]
    successor._spawn_background_review(snapshot, review_skills=True)
    reviews.evict(parent)
    parent._spawn_background_review(
        [{"role": "user", "content": "late old"}], review_skills=True
    )
    assert reviews.queue.pending_count() == 1
    reviews.dispatch()
    reviews.drain()
    assert reviews.calls == [(successor, snapshot)]


def test_retirement_during_popped_dispatch_is_rechecked_before_preparation(
    reviews, monkeypatch
):
    parent = reviews.parent()
    parent._spawn_background_review(
        [{"role": "user", "content": "old"}], review_skills=True
    )
    item = reviews.pop()

    def retire_during_enabled_check(item):
        reviews.evict(item.agent)
        return True

    monkeypatch.setattr(reviews.queue, "_still_enabled", retire_during_enabled_check)
    item.context.run(reviews.queue._dispatch, item)
    assert reviews.targets == []
    reviews.drain()
    assert reviews.calls == []
    assert reviews.queue.pending_count() == 0
