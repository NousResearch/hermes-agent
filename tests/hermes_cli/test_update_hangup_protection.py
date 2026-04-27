"""Tests for SIGHUP protection and stdout mirroring in ``hermes update``.

Covers ``_UpdateOutputStream``, ``_install_hangup_protection``, and
``_finalize_update_output`` in ``hermes_cli/main.py``.  These exist so
that ``hermes update`` survives a terminal disconnect mid-install
(SSH drop, shell close) without leaving the venv half-installed.
"""

from __future__ import annotations

import io
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main import (
    _UpdateOutputStream,
    _finalize_update_output,
    _install_hangup_protection,
)


# -----------------------------------------------------------------------------
# _UpdateOutputStream
# -----------------------------------------------------------------------------


class TestUpdateOutputStream:
    def test_write_mirrors_to_both_original_and_log(self):
        original = io.StringIO()
        log = io.StringIO()
        stream = _UpdateOutputStream(original, log)

        stream.write("hello world\n")

        assert original.getvalue() == "hello world\n"
        assert log.getvalue() == "hello world\n"

    def test_write_continues_after_broken_original(self):
        """When the terminal disconnects, original.write raises BrokenPipeError.

        The wrapper must catch it, flip the broken flag, and keep writing to
        the log from then on.
        """
        log = io.StringIO()

        class _BrokenStream:
            def write(self, data):
                raise BrokenPipeError("terminal gone")

            def flush(self):
                raise BrokenPipeError("terminal gone")

        stream = _UpdateOutputStream(_BrokenStream(), log)

        # First write triggers the broken-pipe path.
        stream.write("first line\n")
        # Subsequent writes take the fast broken path (no exception).
        stream.write("second line\n")

        assert log.getvalue() == "first line\nsecond line\n"
        assert stream._original_broken is True

    def test_write_tolerates_oserror_and_valueerror(self):
        """OSError (EIO) and ValueError (closed file) should also be absorbed."""
        log = io.StringIO()

        class _RaisingStream:
            def __init__(self, exc):
                self._exc = exc

            def write(self, data):
                raise self._exc

            def flush(self):
                raise self._exc

        for exc in (OSError("EIO"), ValueError("closed file")):
            stream = _UpdateOutputStream(_RaisingStream(exc), log)
            stream.write("x\n")
            assert stream._original_broken is True

    def test_log_failure_does_not_abort_write(self):
        """Even if the log file write raises, the original write must still happen."""
        class _BrokenLog:
            def write(self, data):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

        original = io.StringIO()
        stream = _UpdateOutputStream(original, _BrokenLog())

        stream.write("data\n")

        assert original.getvalue() == "data\n"

    def test_flush_tolerates_broken_original(self):
        class _BrokenStream:
            def write(self, data):
                return len(data)

            def flush(self):
                raise BrokenPipeError("gone")

        log = io.StringIO()
        stream = _UpdateOutputStream(_BrokenStream(), log)
        stream.flush()  # must not raise
        assert stream._original_broken is True

    def test_isatty_delegates_to_original(self):
        class _TtyStream:
            def isatty(self):
                return True

            def write(self, data):
                return len(data)

            def flush(self):
                return None

        stream = _UpdateOutputStream(_TtyStream(), io.StringIO())
        assert stream.isatty() is True

    def test_isatty_returns_false_after_broken(self):
        class _BrokenStream:
            def isatty(self):
                return True

            def write(self, data):
                raise BrokenPipeError()

            def flush(self):
                return None

        stream = _UpdateOutputStream(_BrokenStream(), io.StringIO())
        stream.write("x")  # marks broken
        assert stream.isatty() is False

    def test_getattr_delegates_unknown_attrs(self):
        class _StreamWithEncoding:
            encoding = "utf-8"

            def write(self, data):
                return len(data)

            def flush(self):
                return None

        stream = _UpdateOutputStream(_StreamWithEncoding(), io.StringIO())
        assert stream.encoding == "utf-8"


# -----------------------------------------------------------------------------
# _install_hangup_protection
# -----------------------------------------------------------------------------


class TestInstallHangupProtection:
    def test_gateway_mode_is_noop(self):
        """In gateway mode the process is already detached — don't touch stdio or signals."""
        prev_out, prev_err = sys.stdout, sys.stderr
        prev_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

        state = _install_hangup_protection(gateway_mode=True)

        try:
            assert sys.stdout is prev_out
            assert sys.stderr is prev_err
            assert state["log_file"] is None
            assert state["installed"] is False
            if hasattr(signal, "SIGHUP"):
                assert signal.getsignal(signal.SIGHUP) == prev_sighup
        finally:
            _finalize_update_output(state)

    @pytest.mark.skipif(
        not hasattr(signal, "SIGHUP"), reason="SIGHUP not available on this platform"
    )
    def test_installs_sighup_ignore(self, tmp_path, monkeypatch):
        """SIGHUP should be set to SIG_IGN so SSH disconnect doesn't kill the update."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Clear cached get_hermes_home if present
        import hermes_cli.config as _cfg
        if hasattr(_cfg, "_HERMES_HOME_CACHE"):
            _cfg._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]

        original_handler = signal.getsignal(signal.SIGHUP)
        state = _install_hangup_protection(gateway_mode=False)

        try:
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        finally:
            _finalize_update_output(state)
            # Restore whatever was there before so we don't leak to other tests.
            signal.signal(signal.SIGHUP, original_handler)

    def test_wraps_stdout_and_stderr_with_mirror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Nuke any cached home path
        import hermes_cli.config as _cfg
        if hasattr(_cfg, "_HERMES_HOME_CACHE"):
            _cfg._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]

        prev_out, prev_err = sys.stdout, sys.stderr
        state = _install_hangup_protection(gateway_mode=False)

        try:
            # On Windows (no SIGHUP) we still wrap stdio and create the log.
            assert state["installed"] is True
            assert isinstance(sys.stdout, _UpdateOutputStream)
            assert isinstance(sys.stderr, _UpdateOutputStream)
            assert state["log_file"] is not None

            sys.stdout.write("checking mirror\n")
            sys.stdout.flush()

            log_path = tmp_path / "logs" / "update.log"
            assert log_path.exists()
            contents = log_path.read_text(encoding="utf-8")
            assert "checking mirror" in contents
            assert "hermes update started" in contents
        finally:
            _finalize_update_output(state)
            # Sanity-check restoration
            assert sys.stdout is prev_out
            assert sys.stderr is prev_err

    def test_logs_dir_created_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as _cfg
        if hasattr(_cfg, "_HERMES_HOME_CACHE"):
            _cfg._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]

        # No logs/ dir yet.
        assert not (tmp_path / "logs").exists()

        state = _install_hangup_protection(gateway_mode=False)
        try:
            assert (tmp_path / "logs").is_dir()
            assert (tmp_path / "logs" / "update.log").exists()
        finally:
            _finalize_update_output(state)

    def test_non_fatal_if_log_setup_fails(self, monkeypatch):
        """If get_hermes_home() raises, stdio must be left untouched but SIGHUP still handled."""
        prev_out, prev_err = sys.stdout, sys.stderr

        def _boom():
            raise RuntimeError("no home for you")

        # Patch the import inside _install_hangup_protection.
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", _boom, raising=True
        )

        original_handler = (
            signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None
        )

        state = _install_hangup_protection(gateway_mode=False)

        try:
            assert sys.stdout is prev_out
            assert sys.stderr is prev_err
            assert state["installed"] is False
            # SIGHUP must still be installed even when log setup fails.
            if hasattr(signal, "SIGHUP"):
                assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        finally:
            _finalize_update_output(state)
            if hasattr(signal, "SIGHUP") and original_handler is not None:
                signal.signal(signal.SIGHUP, original_handler)


# -----------------------------------------------------------------------------
# _finalize_update_output
# -----------------------------------------------------------------------------


class TestFinalizeUpdateOutput:
    def test_none_state_is_noop(self):
        _finalize_update_output(None)  # must not raise

    def test_restores_streams_and_closes_log(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.config as _cfg
        if hasattr(_cfg, "_HERMES_HOME_CACHE"):
            _cfg._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]

        prev_out = sys.stdout
        state = _install_hangup_protection(gateway_mode=False)
        log_file = state["log_file"]

        assert sys.stdout is not prev_out
        assert log_file is not None

        _finalize_update_output(state)

        assert sys.stdout is prev_out
        # The log file handle should be closed.
        assert log_file.closed is True

    def test_skipped_install_leaves_stdio_alone(self):
        """When install failed (state['installed']=False) finalize should not
        touch sys.stdout / sys.stderr (they were never wrapped)."""
        # Build a synthetic state that mimics a failed install.
        sentinel_out = object()
        state = {
            "prev_stdout": sentinel_out,
            "prev_stderr": sentinel_out,
            "log_file": None,
            "installed": False,
        }
        before_out, before_err = sys.stdout, sys.stderr

        _finalize_update_output(state)

        assert sys.stdout is before_out
        assert sys.stderr is before_err
