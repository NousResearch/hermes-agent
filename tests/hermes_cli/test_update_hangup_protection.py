"""Tests for SIGHUP protection and stdout mirroring in ``hermes update``.

Covers ``_UpdateOutputStream``, ``_install_hangup_protection``, and
``_finalize_update_output`` in ``hermes_cli/main_dashboard.py``.  These exist so
that ``hermes update`` survives a terminal disconnect mid-install
(SSH drop, shell close) without leaving the venv half-installed.
"""

from __future__ import annotations

import io
import signal
import sys

import pytest

from hermes_cli.main_dashboard import (
    _UpdateOutputStream,
    _finalize_update_output,
    _install_hangup_protection,
)
from hermes_cli.update_cmd import (
    _log_only_write,
    _print_update_completion,
    _run_logged_subprocess,
)


def test_update_completion_includes_bounded_action_identity(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_ACTION_ID", "a" * 32)
    # These tests pin the action-identity receipt contract, not the branch
    # display — neutralize the branch+HEAD suffix added for the 2026-08-17
    # parked-branch incident (covered by test_update_parked_branch_guard.py).
    monkeypatch.setattr("hermes_cli.update_cmd._branch_head_suffix", lambda: "")

    _print_update_completion("✓ Update complete!")

    assert capsys.readouterr().out.splitlines() == [
        "✓ Update complete!",
        f"=== hermes-update completed {'a' * 32} ===",
    ]


def test_update_completion_rejects_untrusted_action_identity(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_ACTION_ID", "not-safe\nforged")
    monkeypatch.setattr("hermes_cli.update_cmd._branch_head_suffix", lambda: "")

    _print_update_completion("✓ Update complete!")

    assert capsys.readouterr().out == "✓ Update complete!\n"


# -----------------------------------------------------------------------------
# _UpdateOutputStream
# -----------------------------------------------------------------------------


class TestUpdateOutputStream:

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


# -----------------------------------------------------------------------------
# _install_hangup_protection
# -----------------------------------------------------------------------------


class TestInstallHangupProtection:

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


# -----------------------------------------------------------------------------
# _log_only_write / _run_logged_subprocess (spam suppression)
# -----------------------------------------------------------------------------


class TestLogOnlyWrite:

    def test_unwrapped_stdout_still_appends_update_log(self, monkeypatch):
        """Unwrapped stdout must still grow logs/update.log (Desktop watchdog)."""
        from hermes_constants import get_hermes_home

        plain = io.StringIO()
        monkeypatch.setattr(sys, "stdout", plain)
        _log_only_write("something")
        assert plain.getvalue() == ""
        log_path = get_hermes_home() / "logs" / "update.log"
        assert "something" in log_path.read_text(encoding="utf-8")

    def test_empty_text_is_noop(self, monkeypatch):
        terminal = io.StringIO()
        log = io.StringIO()
        monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))
        _log_only_write("")
        assert log.getvalue() == ""


class TestRunLoggedSubprocess:
    def test_captures_output_to_log_only(self, monkeypatch):
        terminal = io.StringIO()
        log = io.StringIO()
        monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))

        result = _run_logged_subprocess(
            [sys.executable, "-c", "print('LOUD BUILD OUTPUT')"]
        )

        assert result.returncode == 0
        assert "LOUD BUILD OUTPUT" in (result.stdout or "")
        assert terminal.getvalue() == ""  # not echoed to terminal
        assert "LOUD BUILD OUTPUT" in log.getvalue()  # but kept in the log

    def test_streams_to_log_before_child_exits(self, monkeypatch, tmp_path):
        """Electron/vite output must hit update.log while the child is alive.

        Buffering until subprocess.run returns made a 40-minute Desktop
        rebuild look stalled to the Windows idle watchdog (exit 124).
        """
        import threading
        import time

        terminal = io.StringIO()
        log_path = tmp_path / "update.log"
        log = log_path.open("w+", encoding="utf-8")
        monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))

        script = (
            "import sys, time\n"
            "print('FIRST', flush=True)\n"
            "time.sleep(4)\n"
            "print('SECOND', flush=True)\n"
        )
        seen_first = threading.Event()

        def watch() -> None:
            deadline = time.time() + 2.5
            while time.time() < deadline:
                try:
                    if "FIRST" in log_path.read_text(encoding="utf-8"):
                        seen_first.set()
                        return
                except OSError:
                    pass
                time.sleep(0.05)

        holder: dict = {}

        def run() -> None:
            holder["r"] = _run_logged_subprocess([sys.executable, "-c", script])

        watcher = threading.Thread(target=watch, daemon=True)
        runner = threading.Thread(target=run)
        watcher.start()
        runner.start()
        assert seen_first.wait(timeout=2.5), (
            "FIRST must land in update.log before the child exits"
        )
        assert runner.is_alive(), (
            "child should still be in the 4s sleep when FIRST is logged"
        )
        runner.join(timeout=10)
        result = holder["r"]
        assert result.returncode == 0
        assert "SECOND" in (result.stdout or "")
        assert terminal.getvalue() == ""
        log.close()


class TestUpdateProgressHeartbeat:
    def test_emits_elapsed_line_while_work_runs(self, capsys):
        import time

        from hermes_cli.update_cmd import _update_progress_heartbeat

        with _update_progress_heartbeat(
            "still ({elapsed}s)", interval_seconds=0.12
        ):
            time.sleep(0.4)
        out = capsys.readouterr().out
        assert "still (" in out

    def test_unwrapped_stdout_still_grows_update_log(self, capsys):
        """Unwrapped stdout must still grow logs/update.log (Desktop watchdog)."""
        import time

        from hermes_constants import get_hermes_home
        from hermes_cli.update_cmd import _update_progress_heartbeat

        log_path = get_hermes_home() / "logs" / "update.log"
        before = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        with _update_progress_heartbeat(
            "still ({elapsed}s)", interval_seconds=0.12
        ):
            time.sleep(0.4)
        after = log_path.read_text(encoding="utf-8")
        assert "still (" in after[len(before):]
        assert "still (" in capsys.readouterr().out

