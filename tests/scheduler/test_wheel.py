"""Logical-clock timing wheel: stride placement, cascade, FIFO order, cancellation."""

import pytest

from scheduler.tasks import TaskSpec
from scheduler.wheel import TimingWheel


def _spec(name="core.t", every=1, phase="post"):
    return TaskSpec(plugin_id="core", name=name, period="step", every=every, phase=phase, handler=lambda ctx: None)


def test_stride_one_fires_every_tick():
    wheel = TimingWheel(now=0)
    wheel.insert(_spec(every=1), 1)
    fired = []
    wheel.advance_to(5, lambda node: (fired.append(wheel.now), wheel.reschedule(node, wheel.now + 1)))
    assert fired == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("stride", [3, 256, 65_536])
def test_large_strides_fire_exactly_at_due(stride):
    wheel = TimingWheel(now=0)
    fired = []
    wheel.insert(_spec(every=stride), stride)
    wheel.advance_to(stride * 2, lambda node: (fired.append(node.due), wheel.reschedule(node, wheel.now + stride)))
    assert fired == [stride, stride * 2]


def test_one_shot_delay_fires_only_once():
    wheel = TimingWheel(now=0)
    fired = []
    wheel.insert(_spec(every=500), 500)
    wheel.advance_to(1000, lambda node: (fired.append(node.due), wheel.cancel(node)))
    assert fired == [500]


def test_fifo_order_inside_a_slot_and_after_reinsertion():
    wheel = TimingWheel(now=0)
    fired = []
    wheel.insert(_spec(name="core.a", every=5), 5)
    wheel.insert(_spec(name="core.b", every=5), 5)
    wheel.advance_to(10, lambda node: (
        fired.append((wheel.now, node.spec.name)), wheel.reschedule(node, wheel.now + node.spec.every)))
    assert fired == [(5, "core.a"), (5, "core.b"), (10, "core.a"), (10, "core.b")]


def test_cascade_handles_bucket_boundaries():
    wheel = TimingWheel(now=0)
    fired = []
    for due in (255, 256, 257, 65_535, 65_536):
        wheel.insert(_spec(name=f"core.n{due}", every=10**9), due)
    wheel.advance_to(65_536, lambda node: fired.append(node.due))
    assert fired == [255, 256, 257, 65_535, 65_536]


def test_cancelled_nodes_never_fire_and_slots_stay_sparse():
    wheel = TimingWheel(now=0)
    node = wheel.insert(_spec(), 10)
    wheel.cancel(node)
    fired = []
    wheel.advance_to(20, fired.append)
    assert fired == []

    big = TimingWheel(now=0)
    for index in range(1000):
        big.insert(_spec(name=f"core.{index}", every=10**6), 10**6 + index)
    assert big.pending() == 1000
    assert sum(1 for level in big._slots for slot in level.values() if slot) <= 8
