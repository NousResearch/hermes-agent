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

def test_transient_classifier_tolerates_missing_chain_attributes():
    """Regression for #57298: objects without __cause__/__context__ don't crash.

    A ``traceback.TracebackException`` (or any duck-typed exception-like
    object) may lack ``__cause__`` and ``__context__`` as instance
    attributes.  The classifier must return ``False`` instead of raising
    ``AttributeError``.
    """

    class TracebackLike:
        """Mimics an object that has no exception-chain attributes."""
        pass

    assert _is_transient_network_error(TracebackLike()) is False


# ---------------------------------------------------------------------
# Loop handler
# ---------------------------------------------------------------------

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

@pytest.mark.parametrize("shape", ["missing", "cause_only", "context_only", "cycle"])
def test_loop_handler_preserves_incomplete_exception_context(monkeypatch, shape):
    """A third-party loop context must reach the default handler intact."""
    from types import SimpleNamespace

    exc = SimpleNamespace()
    if shape == "cause_only":
        exc.__cause__ = None
    elif shape == "context_only":
        exc.__context__ = ValueError("not transient")
    elif shape == "cycle":
        exc.__cause__ = exc
    context = {"message": "adapter diagnostic", "exception": exc}
    loop = asyncio.new_event_loop()
    try:
        forwarded = []
        monkeypatch.setattr(loop, "default_exception_handler", forwarded.append)
        loop.set_exception_handler(_gateway_loop_exception_handler)
        loop.call_exception_handler(context)
        assert forwarded == [context]
    finally:
        loop.close()


@pytest.mark.parametrize("edge", ["__cause__", "__context__"])
def test_classifier_follows_available_chain_attribute(edge):
    from types import SimpleNamespace

    assert _is_transient_network_error(SimpleNamespace(**{edge: TimedOut("retry")}))


def test_classifier_preserves_cause_precedence():
    from types import SimpleNamespace

    exc = SimpleNamespace(__cause__=ValueError("bug"), __context__=TimedOut("retry"))
    assert _is_transient_network_error(exc) is False


def test_loop_handler_still_swallows_wrapped_transient(monkeypatch, caplog):
    exc = RuntimeError("wrapper")
    exc.__cause__ = TimedOut("retry")
    loop = asyncio.new_event_loop()
    try:
        forwarded = []
        monkeypatch.setattr(loop, "default_exception_handler", forwarded.append)
        loop.set_exception_handler(_gateway_loop_exception_handler)
        loop.call_exception_handler({"exception": exc})
        assert forwarded == []
        assert "Gateway swallowed transient network error" in caplog.text
    finally:
        loop.close()


# ---------------------------------------------------------------------
# End-to-end: task-level
# ---------------------------------------------------------------------
