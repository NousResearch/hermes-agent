"""Root selection for ``dashboard --stop`` / update cleanup must spare the caller.

The argv substring scan (``_DASHBOARD_PATTERNS``) matches any process whose command
line merely mentions ``hermes dashboard`` / ``hermes serve`` — including the shell the
``--stop`` was typed into (``bash -c 'hermes dashboard --stop'``). Killing it takes down
the invoking terminal; the historical fix for the hosted-TUI case is descendant hygiene
(#113819), which is orthogonal to root selection.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.dashboard_procs import (
    _caller_ancestor_pids,
    _is_caller_wrapper_shell,
)
from hermes_cli.main_dashboard import _find_stale_dashboard_pids


class TestCallerAncestorPids:
    def test_returns_parents_without_self(self):
        # Can't assert exact pids portably; the invariant that matters is that the
        # caller itself is never in the spare set (chain roots 0/1 are harmless —
        # they never match the dashboard argv patterns).
        import os

        ancestors = _caller_ancestor_pids()
        assert isinstance(ancestors, set)
        assert all(isinstance(p, int) for p in ancestors)
        assert os.getpid() not in ancestors

    def test_psutil_failure_falls_back_to_proc(self):
        import sys

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        if sys.platform == "win32":
            # /proc walk unavailable: empty set is the documented best-effort answer.
            with patch("builtins.__import__", side_effect=fake_import):
                assert _caller_ancestor_pids() == set()
        else:
            with patch("builtins.__import__", side_effect=fake_import):
                assert isinstance(_caller_ancestor_pids(), set)


class TestIsCallerWrapperShell:
    def test_wrapper_ancestor_is_spared(self):
        # bash-headed ancestor carrying the dashboard argv in its own cmdline.
        with patch(
            "hermes_cli.dashboard_procs._argv_head_command", return_value="bash"
        ):
            assert _is_caller_wrapper_shell(4242, {4242}) is True

    def test_backend_ancestor_stays_killed(self):
        # A python-headed ancestor is a real backend (hosted TUI case): must stay
        # stoppable, otherwise `--stop` from inside the dashboard's own TUI breaks.
        with patch(
            "hermes_cli.dashboard_procs._argv_head_command", return_value="python3.12"
        ):
            assert _is_caller_wrapper_shell(4242, {4242}) is False

    def test_wrapper_non_ancestor_stays_killed(self):
        # Same argv shape, but NOT the caller's ancestor: a genuinely unrelated
        # wrapper (e.g. another user's `bash -c 'hermes serve'`) is still a target.
        with patch(
            "hermes_cli.dashboard_procs._argv_head_command", return_value="bash"
        ):
            assert _is_caller_wrapper_shell(9999, {4242}) is False

    def test_unreadable_argv_head_is_not_spared(self):
        # None (unreadable /proc) must not compare equal to anything in the set.
        with patch("hermes_cli.dashboard_procs._argv_head_command", return_value=None):
            assert _is_caller_wrapper_shell(4242, {4242}) is False


class TestFindStaleDashboardPidsSparesCaller:
    def test_scan_results_are_filtered(self, capsys):
        processes = [
            (111, "/opt/hermes/bin/python hermes dashboard --port 9119"),
            (222, "bash -c hermes dashboard --stop"),
        ]
        with (
            patch(
                "hermes_cli.dashboard_procs._scan_dashboard_processes",
                return_value=processes,
            ) as scan,
            patch(
                "hermes_cli.dashboard_procs._caller_ancestor_pids", return_value={222}
            ),
            patch("hermes_cli.dashboard_procs._argv_head_command", return_value="bash"),
        ):
            result = _find_stale_dashboard_pids()
        scan.assert_called_once()
        assert result == [111]


class TestWrapperHeadCommands:
    def test_set_covers_common_shells_and_carriers(self):
        from hermes_cli.dashboard_procs import _WRAPPER_HEAD_COMMANDS

        for head in (
            "bash",
            "sh",
            "zsh",
            "fish",
            "nohup",
            "env",
            "timeout",
            "sudo",
            "tmux",
        ):
            assert head in _WRAPPER_HEAD_COMMANDS
        # A backend head must never be in the set.
        for head in ("python", "python3", "python3.12", "hermes", "uv", "uvx"):
            assert head not in _WRAPPER_HEAD_COMMANDS
