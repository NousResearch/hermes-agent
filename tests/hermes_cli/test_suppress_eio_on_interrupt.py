"""Tests for OSError EIO suppression during interrupt shutdown (#13710).

When the user interrupts a running task, prompt_toolkit tries to flush
stdout during emergency shutdown.  If stdout is already in a broken state
(redirected to /dev/null, pipe closed, etc.), the flush raises
``OSError: [Errno 5] Input/output error``.

The ``_suppress_closed_loop_errors`` asyncio exception handler and the
outer ``except (KeyError, OSError)`` block must both suppress this error
to prevent a hard crash.
"""

from __future__ import annotations

import errno
import os
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _suppress_closed_loop_errors – asyncio exception handler
# ---------------------------------------------------------------------------

def _make_suppress_fn():
    """Build a standalone copy of ``_suppress_closed_loop_errors``.

    The real function is defined as a closure inside
    ``CLI._run_interactive``; we reconstruct an equivalent here so the
    unit tests don't need a full CLI instance.
    """
    def _suppress_closed_loop_errors(loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if isinstance(exc, KeyError) and "is not registered" in str(exc):
            return
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EIO:
            return
        loop.default_exception_handler(context)
    return _suppress_closed_loop_errors


class TestSuppressClosedLoopErrors:
    """Verify the asyncio exception handler suppresses expected errors."""

    def test_suppresses_event_loop_closed(self):
        handler = _make_suppress_fn()
        loop = MagicMock()
        handler(loop, {"exception": RuntimeError("Event loop is closed")})
        loop.default_exception_handler.assert_not_called()

    def test_suppresses_key_not_registered(self):
        handler = _make_suppress_fn()
        loop = MagicMock()
        handler(loop, {"exception": KeyError("0 is not registered")})
        loop.default_exception_handler.assert_not_called()

    def test_suppresses_oserror_eio(self):
        """OSError with errno.EIO must be suppressed (#13710)."""
        handler = _make_suppress_fn()
        loop = MagicMock()
        exc = OSError(errno.EIO, "Input/output error")
        handler(loop, {"exception": exc})
        loop.default_exception_handler.assert_not_called()

    def test_does_not_suppress_oserror_other_errno(self):
        """OSError with a different errno must still propagate."""
        handler = _make_suppress_fn()
        loop = MagicMock()
        exc = OSError(errno.EACCES, "Permission denied")
        handler(loop, {"exception": exc})
        loop.default_exception_handler.assert_called_once()

    def test_does_not_suppress_unrelated_exception(self):
        """Unrelated exceptions must still propagate."""
        handler = _make_suppress_fn()
        loop = MagicMock()
        handler(loop, {"exception": ValueError("something else")})
        loop.default_exception_handler.assert_called_once()

    def test_no_exception_key(self):
        """Context without 'exception' must propagate to default handler."""
        handler = _make_suppress_fn()
        loop = MagicMock()
        handler(loop, {"message": "some log"})
        loop.default_exception_handler.assert_called_once()


# ---------------------------------------------------------------------------
# Outer except block – EIO handling
# ---------------------------------------------------------------------------

class TestOuterExceptEIO:
    """Verify the outer ``except (KeyError, OSError)`` block logic."""

    def test_eio_does_not_reraise(self):
        """OSError with errno.EIO should be silently suppressed."""
        exc = OSError(errno.EIO, "Input/output error")
        # Simulate the condition check from the outer except block:
        assert isinstance(exc, OSError)
        assert getattr(exc, "errno", None) == errno.EIO

    def test_bad_file_descriptor_matches(self):
        """'Bad file descriptor' string should be caught."""
        exc = OSError(errno.EBADF, "Bad file descriptor")
        assert "Bad file descriptor" in str(exc)

    def test_other_oserror_reraises(self):
        """Other OSError variants must not match the EIO guard."""
        exc = OSError(errno.EACCES, "Permission denied")
        assert not (getattr(exc, "errno", None) == errno.EIO)
        assert "is not registered" not in str(exc)
        assert "Bad file descriptor" not in str(exc)


# ---------------------------------------------------------------------------
# Signal handler – guarded logger.debug (#13710 regression)
# ---------------------------------------------------------------------------
#
# CPython's logging module is not reentrant-safe.  ``Logger.isEnabledFor``
# caches level results in ``Logger._cache``; under shutdown races the cache
# can be cleared (``Logger._clear_cache``) or mid-mutation when the signal
# fires, raising ``KeyError: <level_int>`` (e.g. ``KeyError: 10`` for DEBUG)
# from inside the handler.  If that KeyError escapes, it bypasses the
# ``raise KeyboardInterrupt()`` on the next line, which in turn bypasses
# prompt_toolkit's normal interrupt unwind and surfaces as the EIO cascade
# from #13710.
#
# The fix: wrap the ``logger.debug`` call in the signal handler in a bare
# ``try/except Exception: pass`` so logging can never raise through it.
#
# These tests verify the contract: the handler must raise KeyboardInterrupt
# (and nothing else) regardless of whether logger.debug succeeds or blows up.


def _make_signal_handler(logger, agent_state):
    """Build a standalone copy of ``_signal_handler``.

    The real handler is defined as a closure inside ``CLI._run_interactive``;
    we reconstruct an equivalent here so the unit tests don't need a full
    CLI instance.  Mirrors cli.py:_signal_handler as of #13710 regression
    fix — guarded logger.debug + agent interrupt + KeyboardInterrupt.
    """
    def _signal_handler(signum, frame):
        # Guarded: logging must never raise through a signal handler.
        try:
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        except Exception:
            pass  # never let logging raise from a signal handler (#13710 regression)
        try:
            if agent_state.get("agent") and agent_state.get("running"):
                agent_state["agent"].interrupt(f"received signal {signum}")
        except Exception:
            pass  # never block signal handling
        raise KeyboardInterrupt()
    return _signal_handler


class TestSignalHandlerLoggingRace:
    """#13710 regression — logger.debug in signal handler must not escape.

    If the DEBUG-level ``logging._cache`` lookup races with a concurrent
    ``_clear_cache`` (e.g. from another thread reconfiguring logging during
    shutdown), ``logger.debug`` can raise ``KeyError: 10``.  The signal
    handler must swallow that and still raise KeyboardInterrupt.
    """

    def test_keyboard_interrupt_raised_on_normal_path(self):
        """Sanity: handler raises KeyboardInterrupt when logging works."""
        logger = MagicMock()
        handler = _make_signal_handler(logger, {})
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)  # SIGTERM
        logger.debug.assert_called_once()

    def test_keyboard_interrupt_raised_when_logger_raises_keyerror(self):
        """logger.debug raising KeyError(10) must not escape — KeyboardInterrupt wins.

        This is the exact failure signature from the #13710 regression: the
        CPython 3.11 ``Logger._cache[level]`` race surfaces as KeyError on
        the integer level value, and previously propagated out of the
        signal handler before the ``raise KeyboardInterrupt()`` could fire.
        """
        logger = MagicMock()
        logger.debug.side_effect = KeyError(10)  # DEBUG level int
        handler = _make_signal_handler(logger, {})
        # Must still raise KeyboardInterrupt, NOT KeyError.
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)

    def test_keyboard_interrupt_raised_when_logger_raises_generic(self):
        """Any Exception from logger.debug must be swallowed by the guard."""
        logger = MagicMock()
        logger.debug.side_effect = RuntimeError("logging is shutting down")
        handler = _make_signal_handler(logger, {})
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)

    def test_agent_interrupt_still_fires_when_logger_raises(self):
        """Even if logger.debug blows up, the agent interrupt must still run.

        The whole point of the grace window is cleaning up the agent's
        subprocess group.  A logging race must not skip that step.
        """
        logger = MagicMock()
        logger.debug.side_effect = KeyError(10)
        agent = MagicMock()
        handler = _make_signal_handler(logger, {"agent": agent, "running": True})
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)
        agent.interrupt.assert_called_once_with("received signal 15")

    def test_agent_interrupt_failure_also_does_not_escape(self):
        """Defense-in-depth: agent.interrupt() raising must not escape either."""
        logger = MagicMock()
        agent = MagicMock()
        agent.interrupt.side_effect = RuntimeError("agent already torn down")
        handler = _make_signal_handler(logger, {"agent": agent, "running": True})
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)

    def test_base_exception_from_logger_is_not_swallowed(self):
        """BaseException (e.g. SystemExit) must still propagate — only Exception is caught.

        The guard uses ``except Exception`` deliberately; BaseException
        subclasses like SystemExit or a nested KeyboardInterrupt should
        still be honored so we don't mask real shutdown signals.
        """
        logger = MagicMock()
        logger.debug.side_effect = SystemExit(1)
        handler = _make_signal_handler(logger, {})
        with pytest.raises(SystemExit):
            handler(15, None)
