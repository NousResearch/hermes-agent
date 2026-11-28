"""ReplacementCoordinator: disposing a superseded generation must free it without changing restores."""
from __future__ import annotations

import gc
import weakref

from registration_lifecycle import ReplacementCoordinator


class _Value:
    def __init__(self, label):
        self.label = label


class _Slot:
    """A one-entry registry with identity-conditional restore, like ``tools.registry``."""

    def __init__(self, value):
        self.value = value

    def restore(self, current, replacement):
        if self.value is current:
            self.value = replacement
            return True
        return False


def _register(coordinator, slot, value):
    previous, slot.value = slot.value, value
    return coordinator.acquire("k", current=value, previous=previous,
                               restore=lambda replacement, v=value: slot.restore(v, replacement))


def test_disposed_middle_generation_is_released():
    coordinator, original = ReplacementCoordinator(), _Value("orig")
    slot = _Slot(original)
    a = _register(coordinator, slot, _Value("a"))
    b_value = _Value("b")
    b = _register(coordinator, slot, b_value)
    c = _register(coordinator, slot, _Value("c"))
    b_ref = weakref.ref(b_value)
    b.dispose()
    del b, b_value
    gc.collect()
    assert b_ref() is None, "a live successor still pins the disposed generation"
    assert slot.value.label == "c"
    c.dispose()
    assert slot.value.label == "a"  # nearest live predecessor, exactly as before
    a.dispose()
    assert slot.value is original


def test_restore_after_all_predecessors_disposed_returns_original():
    coordinator, original = ReplacementCoordinator(), _Value("orig")
    slot = _Slot(original)
    leases = [_register(coordinator, slot, _Value(str(i))) for i in range(5)]
    for lease in leases[:-1]:
        lease.dispose()
    assert slot.value.label == "4"
    leases[-1].dispose()
    assert slot.value is original
    assert coordinator._active == {}


def test_repeated_supersede_keeps_one_live_lease_and_no_chain():
    coordinator, original = ReplacementCoordinator(), _Value("orig")
    slot = _Slot(original)
    refs, previous = [], None
    for i in range(10):
        value = _Value(str(i))
        refs.append(weakref.ref(value))
        lease = _register(coordinator, slot, value)
        if previous is not None:
            previous.dispose()
        previous = lease
        del value
    gc.collect()
    assert previous is not None
    assert len(coordinator._active["k"]) == 1
    assert previous.predecessor is None and previous.previous is original
    assert [r() is None for r in refs[:-1]] == [True] * 9
