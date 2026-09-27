"""Memory-ownership regressions for deferred background reviews."""

import gc
import weakref

from agent import review_idle_queue as riq
from agent.review_idle_queue import ReviewIdleQueue, _retained_snapshot_bytes


class _Agent:
    def __init__(self) -> None:
        self.spawned = []

    def _spawn_background_review_now(self, **kwargs) -> None:
        self.spawned.append(kwargs)


def _queue():
    queue = ReviewIdleQueue()
    clock = {"t": 0.0}
    queue._now = lambda: clock["t"]
    queue._server_idle = lambda: True
    queue._ensure_thread = lambda: None
    return queue, clock


def test_retained_size_counts_nested_snapshot_without_copying_or_looping():
    shared = "x" * 10_000
    snapshot = [{"role": "tool", "content": shared}, {"role": "assistant", "content": shared}]
    snapshot.append(snapshot)

    measured = _retained_snapshot_bytes(snapshot)

    assert measured >= len(shared)
    assert measured < len(shared) * 2 + 4_096  # shared leaf and cycle counted once


def test_queue_does_not_keep_parent_agent_alive():
    queue, _ = _queue()
    agent = _Agent()
    parent_ref = weakref.ref(agent)
    assert queue.enqueue(agent, "s1", {"messages_snapshot": []})

    del agent
    gc.collect()

    assert parent_ref() is None
    with queue._lock:
        assert queue._pending["s1"].parent() is None


def test_queue_pressure_evicts_oldest_and_bounds_snapshot_bytes(monkeypatch):
    monkeypatch.setattr(riq, "_PENDING_SNAPSHOT_BYTES_MAX", 2_000)
    monkeypatch.setattr(riq, "_PENDING_REVIEWS_MAX", 2)
    queue, clock = _queue()
    agents = [_Agent(), _Agent(), _Agent()]

    for index, agent in enumerate(agents):
        clock["t"] = float(index)
        assert queue.enqueue(
            agent, f"s{index}",
            {"messages_snapshot": [{"role": "user", "content": "x" * 120}]},
        )

    assert queue.pending_count() <= 2
    assert queue.pending_snapshot_bytes() <= 2_000
    with queue._lock:
        assert "s0" not in queue._pending
        assert set(queue._pending) == {"s1", "s2"}


def test_oversized_snapshot_is_rejected_without_replacing_existing(monkeypatch):
    monkeypatch.setattr(riq, "_PENDING_SNAPSHOT_BYTES_MAX", 800)
    queue, _ = _queue()
    agent = _Agent()
    assert queue.enqueue(agent, "s1", {"messages_snapshot": [{"content": "small"}]})

    assert not queue.enqueue(agent, "s1", {"messages_snapshot": [{"content": "x" * 2_000}]})
    assert queue.pending_count() == 1
    with queue._lock:
        assert queue._pending["s1"].kwargs["messages_snapshot"] == [{"content": "small"}]


def test_coalescing_replaces_accounting_instead_of_adding_it():
    queue, _ = _queue()
    agent = _Agent()
    assert queue.enqueue(agent, "s1", {"messages_snapshot": [{"content": "x" * 1_000}]})
    first_bytes = queue.pending_snapshot_bytes()

    assert queue.enqueue(agent, "s1", {"messages_snapshot": [{"content": "tiny"}]})

    assert queue.pending_count() == 1
    assert queue.pending_snapshot_bytes() < first_bytes


def test_pop_releases_accounting_and_preserves_snapshot_for_dispatch():
    queue, clock = _queue()
    agent = _Agent()
    messages = [{"role": "user", "content": "remember this"}]
    assert queue.enqueue(agent, "s1", {"messages_snapshot": messages, "task_cfg": {}})
    assert queue.pending_snapshot_bytes() > 0

    queue.note_turn_started()
    queue.note_turn_finished()
    clock["t"] += riq._IDLE_SETTLE_S + 1
    item = queue._pop_dispatchable()

    assert item is not None
    assert queue.pending_count() == 0
    assert queue.pending_snapshot_bytes() == 0
    parent = item.parent()
    assert parent is agent
    parent._spawn_background_review_now(**item.kwargs)
    assert agent.spawned[0]["messages_snapshot"] == messages
