"""Regression tests for flush_subscribers_safely (issue #81617).

The gateway crashed with an uncaught ``RuntimeError: cannot block a running
asyncio event loop`` because ``relay.subscribers.flush()`` is synchronous and
was invoked inside the running event loop on every agent-mode session close.
These tests pin the shared helper's three behaviours:

- outside a running loop the underlying flush is called directly;
- the asyncio RuntimeError is diverted to a worker thread and does not raise;
- any other error is re-raised for the caller to surface.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent import relay_runtime


class _FakeSubscribers:
    """A subscribers facade whose flush can be scripted for each case."""

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        raise_once: bool = False,
    ) -> None:
        self.error = error
        self.raise_once = raise_once
        self.calls: list[str] = []
        self.threads: list[int] = []

    def flush(self) -> None:
        self.calls.append("flush")
        self.threads.append(threading.get_ident())
        if self.error is not None:
            error = self.error
            if self.raise_once:
                self.error = None
            raise error


def _fake_relay(subscribers: _FakeSubscribers) -> object:
    return SimpleNamespace(subscribers=subscribers)


def test_flush_calls_direct_outside_loop(monkeypatch) -> None:
    subscribers = _FakeSubscribers()
    relay = _fake_relay(subscribers)

    def _explode(*args, **kwargs):  # pragma: no cover - asserted below.
        raise AssertionError("direct flush must not touch the thread pool")

    monkeypatch.setattr(relay_runtime._subscribers_executor, "submit", _explode)
    relay_runtime.flush_subscribers_safely(relay)
    assert subscribers.calls == ["flush"]


def test_flush_asyncio_runtime_error_delegates_to_worker_thread() -> None:
    # The asyncio error is loop-thread-specific: it fires on the direct call
    # on the caller thread, and the worker-thread retry succeeds.
    subscribers = _FakeSubscribers(
        error=RuntimeError("cannot block a running asyncio event loop"),
        raise_once=True,
    )
    relay = _fake_relay(subscribers)
    relay_runtime.flush_subscribers_safely(relay)
    # The direct call raised, but a retry on the worker thread flushed.
    assert subscribers.calls == ["flush", "flush"]
    # The second flush really ran on a different thread than the caller.
    caller_thread = subscribers.threads[0]
    assert any(thread_id != caller_thread for thread_id in subscribers.threads[1:])


def test_flush_asyncio_error_does_not_raise_under_running_loop() -> None:
    import asyncio

    subscribers = _FakeSubscribers(
        error=RuntimeError("cannot block a running asyncio event loop"),
        raise_once=True,
    )
    relay = _fake_relay(subscribers)

    async def exercise() -> None:
        # The helper must survive an invocation made from a live event loop,
        # which is the exact gateway scenario from the issue.
        relay_runtime.flush_subscribers_safely(relay)

    asyncio.run(exercise())
    assert subscribers.calls == ["flush", "flush"]


def test_flush_non_asyncio_runtime_error_is_re_raised() -> None:
    subscribers = _FakeSubscribers(error=RuntimeError("Relay not initialised"))
    relay = _fake_relay(subscribers)
    with pytest.raises(RuntimeError, match="Relay not initialised"):
        relay_runtime.flush_subscribers_safely(relay)


def test_flush_other_exception_types_propagate_untransformed() -> None:
    """Only RuntimeError naming asyncio is special-cased; OSError and
    ValueError bubble up untouched."""
    subscribers = _FakeSubscribers(error=OSError("io"))
    with pytest.raises(OSError, match="io"):
        relay_runtime.flush_subscribers_safely(_fake_relay(subscribers))
    subscribers = _FakeSubscribers(error=ValueError("bad"))
    with pytest.raises(ValueError, match="bad"):
        relay_runtime.flush_subscribers_safely(_fake_relay(subscribers))
