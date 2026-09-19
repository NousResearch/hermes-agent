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

from gateway.platforms.helpers import is_transient_network_error
from gateway.run import (
    _gateway_loop_exception_handler,
    _is_transient_network_error,
)

# ----- REAL wire exception types --------------------------------------
# ``httpx`` is a CORE dependency (pyproject ``[project].dependencies``), so
# its exception types are always importable and are used directly.
#
# ``python-telegram-bot`` is a lazy-install extra (``[messaging]``), so it may
# be absent — but ``tests/gateway/conftest.py`` installs a PTB-22.x-faithful
# ``telegram.error`` hierarchy in that case, preserving the real inheritance
# graph (``TimedOut`` ⊂ ``NetworkError`` ⊂ ``TelegramError``, and the
# counter-intuitive ``BadRequest`` ⊂ ``NetworkError``). Either way we import
# through ``telegram.error`` rather than declaring a local
# ``class TimedOut(Exception)`` stand-in, so the test exercises the real
# hierarchy instead of a flat one that can't distinguish name-matching from
# subclass-matching.
import httpx
from telegram.error import BadRequest, NetworkError, TelegramError, TimedOut

try:  # aiohttp arrives via [messaging]/[homeassistant]/[sms]; core does not pin it.
    import aiohttp
except Exception:  # pragma: no cover - exercised only on a minimal install
    aiohttp = None  # type: ignore[assignment]
_AIOHTTP_AVAILABLE = aiohttp is not None


class SomeUnrelatedBug(Exception):
    """A non-transient error that should NOT be swallowed."""


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        # python-telegram-bot
        TimedOut("telegram read timeout"),
        NetworkError("telegram network error"),
        # httpx (real library, always installed)
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.PoolTimeout("pool exhausted"),
        httpx.ReadError("read failed"),
        httpx.WriteError("write failed"),
        httpx.RemoteProtocolError("peer closed connection"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_transient_classifier_matches_real_network_exception_types(exc):
    """Every well-known transient network exception INSTANCE is classified."""
    assert is_transient_network_error(exc) is True


@pytest.mark.skipif(not _AIOHTTP_AVAILABLE, reason="aiohttp is an optional extra")
def test_transient_classifier_matches_real_aiohttp_types():
    """aiohttp's transient transport errors are classified when installed."""
    assert is_transient_network_error(aiohttp.ServerDisconnectedError()) is True
    assert is_transient_network_error(aiohttp.ClientOSError()) is True


def test_bad_request_is_not_transient_despite_inheriting_networkerror():
    """The load-bearing negative case, and the reason we match by NAME.

    In python-telegram-bot 22.x ``BadRequest`` inherits from ``NetworkError``
    (``BadRequest.__mro__`` contains it). A subclass-based classifier would
    therefore call a permanent "file is unavailable" error transient and
    burn three retries on it. This test only has teeth because it raises the
    REAL type — a local ``class BadRequest(Exception)`` stand-in subclasses
    nothing and would pass under either implementation.
    """
    assert issubclass(BadRequest, NetworkError), (
        "PTB's hierarchy changed: BadRequest no longer inherits NetworkError. "
        "Re-check whether name-matching is still the right discriminator."
    )
    assert is_transient_network_error(BadRequest("file is unavailable")) is False
    assert is_transient_network_error(TelegramError("generic api error")) is False
    assert is_transient_network_error(SomeUnrelatedBug("real bug")) is False


def test_transient_classifier_walks_the_real_cause_chain():
    """A wrapped transient error is classified through ``__cause__``.

    This is the shape PTB actually produces: a ``NetworkError`` raised
    ``from`` an underlying ``httpx`` transport failure.
    """
    try:
        try:
            raise httpx.ConnectError("connection refused")
        except httpx.ConnectError as inner:
            raise SomeUnrelatedBug("wrapper") from inner
    except SomeUnrelatedBug as outer:
        assert is_transient_network_error(outer) is True


def test_gateway_run_alias_delegates_to_the_shared_classifier():
    """``gateway.run._is_transient_network_error`` stays a working alias.

    The classifier moved to ``gateway.platforms.helpers`` (#84210) but the
    old private name is re-exported for existing importers, so both must
    return identical verdicts.
    """
    for exc in (
        TimedOut("t"),
        httpx.ConnectError("c"),
        BadRequest("permanent"),
        SomeUnrelatedBug("bug"),
    ):
        assert _is_transient_network_error(exc) is is_transient_network_error(exc)


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
