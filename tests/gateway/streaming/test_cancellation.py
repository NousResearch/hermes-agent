"""Direct coverage of CancellationScope semantics (final-review backlog item).

The cooperative-cancellation primitive was previously exercised only indirectly
via test_brain.py and the scenario tests. This file proves its contract directly.
"""
from __future__ import annotations

import pytest

from gateway.calls.native.streaming.cancellation import (
    CallTurnCancelled,
    CancellationScope,
)


def test_starts_uncancelled():
    scope = CancellationScope()
    assert scope.cancelled is False
    assert scope.reason == ""


def test_cancel_sets_state():
    scope = CancellationScope()
    scope.cancel("barge_in")
    assert scope.cancelled is True
    assert scope.reason == "barge_in"


def test_cancel_is_idempotent():
    scope = CancellationScope()
    fired: list[str] = []
    scope.add_listener(fired.append)
    scope.cancel("first")
    scope.cancel("second")  # ignored
    assert scope.reason == "first"
    assert fired == ["first"]  # listener fired exactly once


def test_listener_fires_on_cancel():
    scope = CancellationScope()
    fired: list[str] = []
    scope.add_listener(fired.append)
    assert fired == []
    scope.cancel("stop")
    assert fired == ["stop"]


def test_listener_added_after_cancel_fires_immediately():
    scope = CancellationScope()
    scope.cancel("already")
    fired: list[str] = []
    scope.add_listener(fired.append)
    assert fired == ["already"]


def test_listener_exception_is_swallowed():
    scope = CancellationScope()

    def boom(_reason: str) -> None:
        raise RuntimeError("listener blew up")

    good: list[str] = []
    scope.add_listener(boom)
    scope.add_listener(good.append)
    scope.cancel("x")  # must not raise despite boom()
    assert good == ["x"]


def test_raise_if_cancelled():
    scope = CancellationScope()
    scope.raise_if_cancelled()  # no-op while uncancelled
    scope.cancel("done")
    with pytest.raises(CallTurnCancelled):
        scope.raise_if_cancelled()
