"""``gateway_ingress_observer`` delivery: the bounded queue, the one delivery thread, per-epoch
seals and fault records. Telegram fire sites are covered in tests/plugins/test_telegram_ingress_observer.py."""

import threading
import time

import pytest

from gateway import ingress_observer
from gateway.ingress_observer import HOOK, IngressEpoch
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

WAIT = 5.0
THREAD = "gateway-ingress-observer"


class Observer:
    """A registered plugin callback modelling a consumer: it records each event durably, and can
    hold the delivery thread or fail before or after that record."""

    def __init__(self):
        self.events = []
        self.changed = threading.Condition()
        self.hold = threading.Event()
        self.hold.set()
        self.holding = threading.Event()
        self.fail_before, self.fail_after, self.slow = set(), set(), set()

    def __call__(self, **event):
        if not self.hold.is_set():
            self.holding.set()
            self.hold.wait()
        key = event["event_no"]
        if key in self.slow:
            time.sleep(0.2)
        if key in self.fail_before:
            raise RuntimeError("consumer failed before its durable receipt")
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()
        if key in self.fail_after:
            raise RuntimeError("consumer failed after its durable receipt")

    def wait(self, count):
        with self.changed:
            assert self.changed.wait_for(lambda: len(self.events) >= count, WAIT), self.events
            return list(self.events)

    def block(self, epoch):
        """Park the delivery thread inside the callback for ``epoch``'s next event."""
        self.hold.clear()
        epoch.emit("observed", 0)
        assert self.holding.wait(WAIT)


def _manager(callback=None):
    manager = PluginManager()
    manager._discovered = True
    if callback is not None:
        PluginContext(PluginManifest(name="ingress-audit", source="user"), manager).register_hook(HOOK, callback)
    return manager


def _delivery_threads():
    return sum(thread.name == THREAD for thread in threading.enumerate())


@pytest.fixture
def dispatcher(monkeypatch):
    dispatcher = ingress_observer._Dispatcher()
    monkeypatch.setattr(ingress_observer, "_dispatcher", dispatcher)
    # Lateness is opt-in per test, so a held callback never doubles as a late one.
    monkeypatch.setattr(ingress_observer, "_LATE_AFTER_SECONDS", 60.0)
    return dispatcher


@pytest.fixture
def observer(dispatcher, monkeypatch):
    observer = Observer()
    manager = _manager(observer)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    yield observer
    observer.hold.set()


def _epoch(bot_id="1"):
    return IngressEpoch("telegram", bot_id)


def _numbers(events):
    return [event["event_no"] for event in events]


def _sized(size):
    return lambda epoch, event_no: ({}, size)


def test_each_delivery_carries_its_predecessors_seal_and_end_is_final(observer, dispatcher):
    epoch = _epoch()
    epoch.emit("start", 3)
    epoch.emit("observed", 3)
    epoch.end(4, steps_clean=True)
    epoch.emit("observed", 4)  # refused: the epoch has ended

    start, observed, end = observer.wait(3)
    assert {"platform": "telegram", "bot_id": "1", "epoch": epoch.epoch_id, "kind": "start", "generation": 3,
            "epoch_state": "live", "prev": None, "fault": None}.items() <= start.items()
    assert observed["prev"] == {"event_no": 1, "outcome": "ok"}
    assert end["kind"] == "end" and end["steps_clean"] is True
    assert end["prev"] == {"event_no": 2, "outcome": "ok"} and end["fault"] is None
    assert _numbers([start, observed, end]) == [1, 2, 3]
    marker = _epoch("2")
    marker.emit("start", 0)
    assert _numbers(observer.wait(4)) == [1, 2, 3, 1]
    assert epoch not in dispatcher._seals  # the end retired the epoch's state


@pytest.mark.parametrize("receipt", ["before", "after"])
def test_a_seal_counts_once_its_carrier_is_recorded(observer, receipt):
    """The carrier's own failure is published by the next delivery either way; only a carrier the
    consumer recorded vouches for its predecessor, and no later event re-sends that seal."""
    (observer.fail_before if receipt == "before" else observer.fail_after).add(2)
    epoch = _epoch()
    for _ in range(3):
        epoch.emit("observed", 0)

    recorded = {event["event_no"]: event for event in observer.wait(2 if receipt == "before" else 3)}
    assert recorded[3]["prev"] == {"event_no": 2, "outcome": "failed"}
    assert recorded[3]["fault"] == {"first_event_no": 2, "kinds": ["failed"], "dropped_count": 0}
    vouched = {event["prev"]["event_no"] for event in recorded.values() if event["prev"]}
    assert (1 in vouched) is (receipt == "after")


def test_an_enqueue_drop_leaves_a_gap_and_the_pending_seal_rides_the_next_delivery(observer, monkeypatch):
    monkeypatch.setattr(ingress_observer, "_MAX_QUEUED_EVENTS", 2)
    epoch = _epoch()
    observer.block(epoch)  # 1
    for _ in range(3):
        epoch.emit("observed", 0)  # 2 and 3 queue, 4 is dropped
    observer.hold.set()
    observer.wait(3)
    epoch.emit("observed", 0)  # 5

    events = observer.wait(4)
    assert _numbers(events) == [1, 2, 3, 5]
    assert events[1]["fault"] is None and events[2]["fault"] is None
    assert events[3]["prev"] == {"event_no": 3, "outcome": "ok"}
    assert events[3]["fault"] == {"first_event_no": 4, "kinds": ["dropped"], "dropped_count": 1}


@pytest.mark.parametrize("bound", ["count", "bytes"])
def test_a_blocked_dispatcher_keeps_every_structure_within_its_bound(observer, dispatcher, monkeypatch, bound):
    size = 0 if bound == "count" else 64 * 1024
    capacity = 1024 if bound == "count" else (4 * 1024 * 1024) // (ingress_observer._EVENT_BYTES + size)
    epoch = _epoch()
    observer.block(epoch)
    threads = _delivery_threads()
    for _ in range(3 * capacity):
        epoch.emit("observed", 0, _sized(size))
    for update_id in range(2 * ingress_observer._MAX_ASSOCIATIONS):
        epoch.associate(update_id, 1, "digest")

    assert len(dispatcher._events) == capacity
    assert dispatcher._bytes <= 4 * 1024 * 1024
    assert len(epoch._associations) == ingress_observer._MAX_ASSOCIATIONS
    assert _delivery_threads() == threads
    observer.hold.set()
    observer.wait(1 + capacity)
    epoch.emit("observed", 0)
    last = observer.wait(2 + capacity)[-1]
    assert last["fault"] == {"first_event_no": 2 + capacity, "kinds": ["dropped"], "dropped_count": 2 * capacity}
    assert len(dispatcher._seals) == 1 and dispatcher._bytes == 0


def test_the_drop_counter_saturates_and_identities_never_wrap(observer, monkeypatch):
    monkeypatch.setattr(ingress_observer, "_COUNTER_CAP", 3)
    monkeypatch.setattr(ingress_observer, "_MAX_QUEUED_EVENTS", 1)
    epoch = _epoch()
    observer.block(epoch)  # 1
    for _ in range(10):
        epoch.emit("observed", 0)  # 2 queues; 3..11 are dropped
    for _ in range(5):
        epoch.associate(7, 1, "digest")
    assert epoch.resolve(7)["refetches"] == 3
    observer.hold.set()
    observer.wait(2)
    epoch.emit("observed", 0)

    last = observer.wait(3)[-1]
    assert last["event_no"] == 12
    assert last["fault"] == {"first_event_no": 3, "kinds": ["dropped", "saturated"], "dropped_count": 3}


def test_a_callback_that_outlives_the_deadline_is_sealed_late(observer, monkeypatch):
    monkeypatch.setattr(ingress_observer, "_LATE_AFTER_SECONDS", 0.05)
    observer.slow.add(1)
    epoch = _epoch()
    epoch.emit("fetched", 0)
    epoch.emit("observed", 0)

    second = observer.wait(2)[1]
    assert second["prev"] == {"event_no": 1, "outcome": "late"}
    assert second["fault"] == {"first_event_no": 1, "kinds": ["late"], "dropped_count": 0}


def test_an_event_delivered_after_its_callback_was_unregistered_is_failed():
    assert ingress_observer._deliver(_manager(), {"kind": "observed"}) == "failed"


def test_a_failing_builder_counts_as_a_drop(observer):
    def broken(epoch, event_no):
        raise ValueError("unexpected update shape")

    epoch = _epoch()
    epoch.emit("observed", 0, broken)
    epoch.emit("observed", 0)

    (event,) = observer.wait(1)
    assert event["event_no"] == 2
    assert event["fault"] == {"first_event_no": 1, "kinds": ["dropped"], "dropped_count": 1}


def test_interleaved_epochs_keep_their_own_order_and_seals(observer):
    first, second = _epoch("1"), _epoch("2")
    for epoch in (first, second, first, second, first):
        epoch.emit("observed", 0)

    events = observer.wait(5)
    for epoch in (first, second):
        mine = [event for event in events if event["epoch"] == epoch.epoch_id]
        assert _numbers(mine) == list(range(1, len(mine) + 1))
        assert [event["prev"] and event["prev"]["event_no"] for event in mine] == [None, *range(1, len(mine))]


def test_an_evicted_epoch_fails_closed_and_never_ends_cleanly(observer, dispatcher, monkeypatch):
    monkeypatch.setattr(ingress_observer, "_MAX_CLOSED_EPOCHS", 2)
    observer.block(_epoch("blocker"))
    evicted, *others = (_epoch(str(n)) for n in range(4))
    evicted.emit("start", 0)
    for other in others:
        other.emit("start", 0)
    evicted.emit("observed", 0)
    evicted.end(0, steps_clean=True)
    for other in others:
        other.end(0, steps_clean=True)
    observer.hold.set()

    events = observer.wait(1 + 3 + 2 * len(others))
    late = [event for event in events if event["epoch"] == evicted.epoch_id][1:]
    assert [(event["kind"], event["epoch_state"], event["prev"]) for event in late] == [
        ("observed", "evicted", None), ("end", "evicted", None)]
    assert len(dispatcher._seals) <= 1 + 2


def test_reconnects_behind_a_blocked_dispatcher_reuse_one_thread_and_bounded_state(observer, dispatcher, monkeypatch):
    monkeypatch.setattr(ingress_observer, "_MAX_QUEUED_EVENTS", 8)
    observer.block(_epoch("live"))
    thread = dispatcher._thread
    threads = _delivery_threads()
    for _ in range(50):
        epoch = _epoch()
        epoch.emit("start", 0)
        epoch.emit("observed", 0)
        epoch.end(0, steps_clean=True)

    assert dispatcher._thread is thread and len(dispatcher._events) == 8
    assert _delivery_threads() == threads
    observer.hold.set()
    observer.wait(9)
    assert len(dispatcher._seals) <= 1 + ingress_observer._MAX_CLOSED_EPOCHS


def test_a_failed_end_delivery_still_retires_the_epoch(observer, dispatcher):
    observer.fail_before.add(2)
    epoch = _epoch()
    epoch.emit("start", 0)
    epoch.end(0, steps_clean=False)
    marker = _epoch("2")
    marker.emit("start", 0)

    assert [event["epoch"] for event in observer.wait(2)] == [epoch.epoch_id, marker.epoch_id]
    assert epoch not in dispatcher._seals


def test_associations_keep_the_first_fetch_flag_conflicts_and_evict_oldest(monkeypatch):
    monkeypatch.setattr(ingress_observer, "_MAX_ASSOCIATIONS", 2)
    monkeypatch.setattr(plugins, "get_plugin_manager", _manager)
    epoch = _epoch()
    epoch.associate(10, 1, "a")
    epoch.associate(10, 2, "a")
    epoch.associate(11, 2, "b")
    epoch.associate(11, 3, "c")
    epoch.associate(12, 3, "d")  # evicts 10, the oldest

    assert epoch.resolve(10)["association"] == "missing"
    assert epoch.resolve(11) == {"fetch_event_no": 2, "raw_sha256": "b", "refetches": 0, "association": "conflict"}
    assert epoch.resolve(12) == {"fetch_event_no": 3, "raw_sha256": "d", "refetches": 0, "association": "ok"}
    assert epoch.resolve(12)["association"] == "missing"  # resolved once
    epoch.associate(13, 4, "e")
    epoch.close()
    assert epoch.resolve(13)["association"] == "missing"


def test_without_a_callback_nothing_is_built_queued_or_started(dispatcher, monkeypatch):
    def must_not_build(epoch, event_no):
        raise AssertionError("built without an observer")

    manager = _manager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    epoch = _epoch()
    for _ in range(3):
        epoch.emit("fetched", 0, must_not_build)
    assert dispatcher._thread is None and not dispatcher._events

    observer = Observer()
    PluginContext(PluginManifest(name="late-audit", source="user"), manager).register_hook(HOOK, observer)
    epoch.emit("observed", 0)
    assert _numbers(observer.wait(1)) == [4]  # the unobserved events stay a visible gap


def test_callbacks_run_in_the_scope_that_opened_the_epoch(dispatcher, tmp_path):
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    homes = []
    changed = threading.Condition()

    def observe(**event):
        with changed:
            homes.append(get_hermes_home())
            changed.notify_all()

    token = set_hermes_home_override(tmp_path / "work")
    try:
        PluginContext(PluginManifest(name="ingress-audit", source="user"), plugins.get_plugin_manager()).register_hook(
            HOOK, observe)
        epoch = _epoch()
    finally:
        reset_hermes_home_override(token)
    epoch.emit("observed", 0)  # fired outside that scope, as an adapter task may be

    with changed:
        assert changed.wait_for(lambda: homes, WAIT)
    assert homes == [tmp_path / "work"]
