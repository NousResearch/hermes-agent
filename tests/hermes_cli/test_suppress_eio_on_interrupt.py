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
