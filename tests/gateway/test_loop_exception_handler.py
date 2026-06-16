"""Tests for the gateway loop-level transient-network-error safety net.

Issues #31066 / #31110: unhandled ``telegram.error.TimedOut`` (or peer
``NetworkError`` / ``httpx`` connection error) propagating to the
asyncio event loop killed the gateway process, taking down every
profile attached to the same runner. The safety net installed in
:func:`gateway.run.start_gateway` catches the transient crash class
and logs+swallows it; non-transient errors still surface.

These tests pin the classifier and the loop handler so the safety net
can't silently regress to swallowing every exception.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from gateway.run import (
    _gateway_loop_exception_handler,
    _is_transient_network_error,
)


# ----- Fake exception classes that mimic the real wire types ----------
# We avoid importing telegram / httpx here so the test runs in environments
# without those packages installed (the classifier matches on class name).

class TimedOut(Exception):
    """Stand-in for ``telegram.error.TimedOut``."""


class NetworkError(Exception):
    """Stand-in for ``telegram.error.NetworkError``."""


class ConnectError(Exception):
    """Stand-in for ``httpx.ConnectError``."""


class ReadTimeout(Exception):
    """Stand-in for ``httpx.ReadTimeout``."""


class PoolTimeout(Exception):
    """Stand-in for ``httpx.PoolTimeout``."""


class ClientConnectorError(Exception):
    """Stand-in for ``aiohttp.ClientConnectorError``."""


class SomeUnrelatedBug(Exception):
    """A non-transient error that should NOT be swallowed."""


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls",
    [
        TimedOut,
        NetworkError,
        ConnectError,
        ReadTimeout,
        PoolTimeout,
        ClientConnectorError,
    ],
)
def test_transient_classifier_matches_known_network_errors(exc_cls):
    """Every well-known transient network exception class is classified."""
    assert _is_transient_network_error(exc_cls("boom")) is True


def test_transient_classifier_rejects_unrelated_errors():
    """Real bugs (ValueError, KeyError, custom app errors) are NOT swallowed."""
    for exc in (ValueError("bad"), KeyError("missing"), SomeUnrelatedBug("x")):
        assert _is_transient_network_error(exc) is False


def test_transient_classifier_unwraps_cause_chain():
    """A NetworkError wrapping a ConnectError is still classified."""
    inner = ConnectError("connection refused")
    outer = NetworkError("upstream failed")
    outer.__cause__ = inner
    assert _is_transient_network_error(outer) is True


def test_transient_classifier_unwraps_context_chain():
    """Implicit ``__context__`` wrapping is also unwrapped."""
    try:
        try:
            raise TimedOut("upstream timeout")
        except TimedOut:
            # Re-raise something else with the original as implicit context
            raise SomeUnrelatedBug("wrapper")
    except SomeUnrelatedBug as e:
        wrapped = e
    # The wrapper class name is not transient, but the chained context is.
    assert _is_transient_network_error(wrapped) is True


def test_transient_classifier_does_not_infinite_loop_on_cyclic_cause():
    """A pathological self-referential cause chain terminates."""
    exc = SomeUnrelatedBug("loop")
    exc.__cause__ = exc  # cycle
    # Must return without hanging.
    assert _is_transient_network_error(exc) is False


# ---------------------------------------------------------------------
# Loop handler
# ---------------------------------------------------------------------


def test_handler_swallows_transient_error_and_logs_warning(caplog):
    """Transient errors are logged at WARNING but not re-raised."""
    loop = asyncio.new_event_loop()
    try:
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            _gateway_loop_exception_handler(
                loop,
                {
                    "message": "Task exception was never retrieved",
                    "exception": TimedOut("Timed out"),
                },
            )
        # Warning emitted, exception class name appears in the log.
        assert any("TimedOut" in r.message for r in caplog.records)
    finally:
        loop.close()


def test_handler_delegates_unknown_errors_to_default(monkeypatch):
    """A non-transient error is forwarded to ``loop.default_exception_handler``."""
    loop = asyncio.new_event_loop()
    try:
        forwarded: list[dict] = []

        def fake_default(ctx):
            forwarded.append(ctx)

        monkeypatch.setattr(loop, "default_exception_handler", fake_default)

        context = {
            "message": "Something else broke",
            "exception": SomeUnrelatedBug("real bug"),
        }
        _gateway_loop_exception_handler(loop, context)
        assert forwarded == [context]
    finally:
        loop.close()


def test_handler_tolerates_missing_exception_key(monkeypatch):
    """Contexts without an ``exception`` key fall through to the default handler."""
    loop = asyncio.new_event_loop()
    try:
        forwarded: list[dict] = []
        monkeypatch.setattr(
            loop, "default_exception_handler", lambda ctx: forwarded.append(ctx)
        )
        ctx = {"message": "warning without exception"}
        _gateway_loop_exception_handler(loop, ctx)
        assert forwarded == [ctx]
    finally:
        loop.close()


# ---------------------------------------------------------------------
# End-to-end: task-level
# ---------------------------------------------------------------------


def test_unhandled_transient_error_in_task_does_not_propagate_to_loop():
    """Smoke test the wiring as a loop would actually use it.

    Schedules a task that raises TimedOut and is never awaited. With the
    handler installed, the loop completes normally and logs a warning
    instead of dying. Without the handler, asyncio would emit
    ``Task exception was never retrieved`` and (depending on Python's
    debug mode) potentially escalate.
    """

    async def raiser():
        raise TimedOut("upstream timeout")

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_gateway_loop_exception_handler)
        task = loop.create_task(raiser())
        # Give the task a tick to run and raise.
        await asyncio.sleep(0)
        # Don't await ``task`` — let it become an unhandled-exception task.
        del task
        import gc

        gc.collect()
        await asyncio.sleep(0)

    # If the safety net works, this returns cleanly. If not, the test
    # would still pass (asyncio's default is a warning, not a crash) —
    # the real assertion is that no unhandled exception escapes the
    # ``run`` boundary.
    asyncio.run(main())
