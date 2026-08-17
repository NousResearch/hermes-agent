"""Tests for the autostash restore prompt timeout (issue #85753).

On macOS an unattended `hermes update` could hang forever at the
"Restore local changes now? [Y/n]" prompt because ``input()`` blocks
on an open-but-unanswered TTY. The fix bounds the wait with SIGALRM
on POSIX and falls back to blocking on Windows.
"""

import signal
import subprocess
import sys
from unittest.mock import patch

import pytest

from hermes_cli.update_cmd import (
    _PromptTimeout,
    _input_with_timeout,
    _restore_stashed_changes,
)

_POSIX = hasattr(signal, "SIGALRM")


class TestAutostashPromptTimeout:
    """Guard the autostash restore prompt against unattended-TTY hangs."""

    def test_autostash_prompt_timeout_skips_restore_on_posix(self, tmp_path, capsys):
        """SIGALRM firing skips restore and preserves the stash."""
        with patch(
            "hermes_cli.update_cmd._input_with_timeout",
            side_effect=_PromptTimeout,
        ):
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
                timeout=60,
            )

        assert result is False
        out = capsys.readouterr().out
        assert "timed out" in out
        assert "git stash apply stash@{0}" in out

    def test_autostash_prompt_timeout_zero_blocks_forever(self, tmp_path, capsys):
        """``timeout=0`` takes the blocking path (pre-fix behavior)."""
        with patch("builtins.input", return_value="n") as mock_input:
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
                timeout=0,
            )

        assert result is False
        out = capsys.readouterr().out
        assert "Skipped restoring" in out
        assert mock_input.call_count == 1

    def test_autostash_prompt_timeout_eof_still_skips(self, tmp_path, capsys):
        """EOFError still falls through to the skip path (regression guard)."""
        with patch("builtins.input", side_effect=EOFError()):
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
                timeout=60,
            )

        assert result is False
        out = capsys.readouterr().out
        assert "Skipped restoring" in out

    def test_input_with_timeout_uses_sigalrm_on_posix(self):
        """Helper arms SIGALRM, restores the previous handler, clears alarm."""
        with patch("signal.signal") as mock_signal, patch(
            "signal.alarm", create=True
        ) as mock_alarm, patch("builtins.input", return_value="y"), patch.object(
            signal, "SIGALRM", 14, create=True
        ):
            result = _input_with_timeout("prompt ", timeout=30)

        assert result == "y"
        # alarm armed with the timeout and cleared in cleanup.
        mock_alarm.assert_any_call(30)
        mock_alarm.assert_any_call(0)
        # Previous handler restored.
        assert mock_signal.call_count >= 2

    def test_input_with_timeout_blocks_when_no_sigalrm(self):
        """Windows fallback: no SIGALRM -> blocking input, no alarm armed."""
        # Ensure SIGALRM appears absent even if the test host is POSIX.
        _had_sigalrm = hasattr(signal, "SIGALRM")
        _orig_sigalrm = getattr(signal, "SIGALRM", None)
        if _had_sigalrm:
            del signal.SIGALRM
        try:
            with patch("builtins.input", return_value="y") as mock_input, patch(
                "signal.alarm", create=True
            ) as mock_alarm:
                result = _input_with_timeout("prompt ", timeout=30)
        finally:
            if _had_sigalrm:
                signal.SIGALRM = _orig_sigalrm

        assert result == "y"
        mock_input.assert_called_once_with("prompt ")
        mock_alarm.assert_not_called()


# ---------------------------------------------------------------------------
# Real SIGALRM end-to-end (POSIX only). Windows lacks SIGALRM, so the helper
# falls back to blocking input() there; these tests skip on non-POSIX hosts.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _POSIX, reason="SIGALRM is POSIX-only")
class TestInputWithTimeoutRealSigalrm:
    """Exercise the real signal.alarm path, not a mock.

    These prove the mechanism actually interrupts a blocking read and that
    the previous SIGALRM handler is restored (no interference with any
    caller that set its own alarm).
    """

    def test_real_timeout_raises_prompt_timeout(self):
        """A blocking input() that outlives the alarm raises _PromptTimeout."""
        # input() blocked on a pipe with no writer — will never return
        # before the 1s alarm fires.
        read_fd, write_fd = __import__("os").pipe()

        def _blocked_input(_prompt):
            # Read from a pipe that nobody writes to — blocks forever.
            __import__("os").read(read_fd, 1)
            return "never"

        with patch("builtins.input", side_effect=_blocked_input):
            with pytest.raises(_PromptTimeout):
                _input_with_timeout("prompt ", timeout=1)

        __import__("os").close(read_fd)
        __import__("os").close(write_fd)

    def test_previous_sigalrm_handler_restored(self):
        """A caller's pre-existing SIGALRM handler survives the helper."""

        def _caller_handler(signum, frame):
            pass

        previous = signal.signal(signal.SIGALRM, _caller_handler)
        try:
            with patch("builtins.input", return_value="y"):
                _input_with_timeout("prompt ", timeout=2)
            # After the helper returns, SIGALRM must point back at the
            # caller's handler, not the helper's internal one.
            current = signal.getsignal(signal.SIGALRM)
            assert current is _caller_handler
        finally:
            signal.signal(signal.SIGALRM, previous)
            signal.alarm(0)

    def test_no_alarm_left_running_after_success(self):
        """No pending alarm remains after a prompt that returns promptly."""
        with patch("builtins.input", return_value="y"):
            _input_with_timeout("prompt ", timeout=30)
        # alarm(0) returns the number of seconds remaining on any previously
        # armed alarm; with the helper's cleanup it must be 0.
        remaining = signal.alarm(0)
        assert remaining == 0


# ---------------------------------------------------------------------------
# Caller-continuation contract: when _restore_stashed_changes returns False
# (the timeout / decline / EOF paths), the update flow must still proceed to
# the post-update gateway restart. We can't run the full _cmd_update_impl
# here (it needs git/pip/uv/node/launchd), so we assert the contract the
# restart depends on: the return value is False AND the stash is preserved
# (so the restart block, which runs unconditionally after the finally, is
# reachable). This is the seam the real restart fires through.
# ---------------------------------------------------------------------------


class TestRestoreReturnsFalseMeansRestartStillRuns:
    """Document and guard the contract between restore-skip and restart.

    The post-update gateway restart in _cmd_update_impl lives outside the
    try/finally that wraps _restore_stashed_changes, so it runs as long as
    _restore_stashed_changes RETURNS (any value) rather than hanging. The
    timeout path's job is therefore to RETURN False, not to raise. These
    tests pin that contract so a future refactor that turns the skip into a
    raise() would fail here before the restart regressions in the field.
    """

    def test_timeout_path_returns_false_does_not_raise(self, tmp_path):
        with patch(
            "hermes_cli.update_cmd._input_with_timeout",
            side_effect=_PromptTimeout,
        ):
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True,
                input_fn=None, timeout=60,
            )

        assert result is False

    def test_timeout_path_does_not_touch_stash(self, tmp_path):
        """Skip-on-timeout must NOT drop the stash — the user can recover."""
        drop_calls = []

        def _fake_run(cmd, **kw):
            if cmd[:2] == ["git", "stash"] and len(cmd) >= 3 and cmd[2] == "drop":
                drop_calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch(
            "hermes_cli.update_cmd._input_with_timeout",
            side_effect=_PromptTimeout,
        ), patch("subprocess.run", side_effect=_fake_run):
            _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True,
                input_fn=None, timeout=60,
            )

        assert drop_calls == [], "timeout-skip must not drop the stash"


# ---------------------------------------------------------------------------
# Run the test file directly to sanity-check on a POSIX host (CI).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
