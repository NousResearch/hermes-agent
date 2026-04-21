"""Unit tests for _SafeWriter.fileno() fix (PR #13278).

Verifies that fileno() is wrapped in try/except like other methods,
returning a safe default (1) when the inner stream raises OSError or ValueError.
"""

import io
import os

from run_agent import _SafeWriter


class TestSafeWriterFileno:

    def test_fileno_returns_inner_fd_when_valid(self):
        """fileno() returns the correct fd when the inner stream is healthy."""
        real = io.StringIO()
        # StringIO doesn't have fileno(), so use a real file descriptor
        r_fd, w_fd = os.pipe()
        try:
            inner = os.fdopen(r_fd, "rb")
            wrapped = _SafeWriter(inner)
            assert wrapped.fileno() == inner.fileno()
        finally:
            inner.close()
            # r_fd already closed by inner.close()
            os.close(w_fd)

    def test_fileno_returns_1_on_oserror(self):
        """fileno() returns 1 when inner stream raises OSError (broken pipe)."""
        class BrokenStream:
            def fileno(self):
                raise OSError(5, "Input/output error")
            def write(self, data):
                return len(data)
            def flush(self):
                pass
            def isatty(self):
                return False

        wrapped = _SafeWriter(BrokenStream())
        assert wrapped.fileno() == 1

    def test_fileno_returns_1_on_valueerror(self):
        """fileno() returns 1 when inner stream raises ValueError (closed file)."""
        class ClosedStream:
            def fileno(self):
                raise ValueError("I/O operation on closed file")
            def write(self, data):
                return len(data)
            def flush(self):
                pass
            def isatty(self):
                return False

        wrapped = _SafeWriter(ClosedStream())
        assert wrapped.fileno() == 1

    def test_fileno_does_not_propagate_oserror(self):
        """fileno() never raises, even when inner stream is completely broken."""
        class DeadStream:
            def fileno(self):
                raise OSError(9, "Bad file descriptor")

        wrapped = _SafeWriter(DeadStream())
        # Must not raise
        result = wrapped.fileno()
        assert result == 1

    def test_fileno_does_not_propagate_valueerror(self):
        """fileno() never raises ValueError from closed inner stream."""
        class DeadStream:
            def fileno(self):
                raise ValueError("I/O operation on closed file")

        wrapped = _SafeWriter(DeadStream())
        result = wrapped.fileno()
        assert result == 1
