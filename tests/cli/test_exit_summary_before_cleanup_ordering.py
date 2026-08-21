"""Regression test: the CLI must print `_print_exit_summary()` (cost
report + `--resume <session_id>` hint) BEFORE calling `_run_cleanup()` on
every interactive-mode exit path.

Why this matters: `_run_cleanup()` can block for tens of seconds — a
memory provider's `on_session_end` hook can be a network-bound call, and
`shutdown_mcp_servers()` can separately block during MCP server teardown.
Both run inside `_run_cleanup()`, which is "protected" by
`_arm_exit_watchdog()` — a daemon thread that force-exits the process with
`os._exit(0)` after `HERMES_EXIT_WATCHDOG_S` seconds (default 60s as of
this fix; was 30s) if cleanup hasn't finished.

If the watchdog fires while still inside `_run_cleanup()`, the process is
killed via `os._exit(0)` before any code AFTER `_run_cleanup()` runs. If
`_print_exit_summary()` were called after `_run_cleanup()`, a slow-but-real
memory-provider shutdown (or slow MCP teardown) could silently swallow the
cost report and resume hint with zero user-visible error.

Fix: extracted a shared `HermesCLI._finish_interactive_exit()` helper that
both interactive-exit call sites (the stdin-unavailable early return and
the main `run()` finally-block exit path) now call, so the
summary-then-cleanup ordering can't drift out of sync between them. This
test exercises the real method directly (not a source-text regex) and
confirms both the call order and the default watchdog timeout bump.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

import cli as cli_mod
from cli import HermesCLI


def _make_cli() -> HermesCLI:
    """Minimal HermesCLI bound to only what _finish_interactive_exit needs."""
    inst = HermesCLI.__new__(HermesCLI)
    return inst


class TestFinishInteractiveExitOrdering:
    def test_prints_summary_before_cleanup(self):
        cli = _make_cli()
        order: list[str] = []

        with (
            patch.object(cli, "_print_exit_summary", side_effect=lambda: order.append("summary")),
            patch.object(cli_mod, "_run_cleanup", side_effect=lambda: order.append("cleanup")),
        ):
            cli._finish_interactive_exit()

        assert order == ["summary", "cleanup"], (
            "_print_exit_summary() must run before _run_cleanup() -- if the "
            "exit watchdog fires mid-cleanup, any code written after "
            "_run_cleanup() never executes, silently swallowing the cost "
            "report and --resume hint."
        )

    def test_release_session_false_by_default(self):
        cli = _make_cli()
        cli._release_active_session = MagicMock()

        with (
            patch.object(cli, "_print_exit_summary"),
            patch.object(cli_mod, "_run_cleanup"),
        ):
            cli._finish_interactive_exit()

        cli._release_active_session.assert_not_called()

    def test_release_session_true_releases_after_cleanup(self):
        cli = _make_cli()
        order: list[str] = []
        cli._release_active_session = MagicMock(side_effect=lambda: order.append("release"))

        with (
            patch.object(cli, "_print_exit_summary", side_effect=lambda: order.append("summary")),
            patch.object(cli_mod, "_run_cleanup", side_effect=lambda: order.append("cleanup")),
        ):
            cli._finish_interactive_exit(release_session=True)

        assert order == ["summary", "cleanup", "release"]
        cli._release_active_session.assert_called_once()

    def test_cleanup_still_runs_when_print_exit_summary_raises(self):
        """A bare ``print()`` inside ``_print_exit_summary()`` can raise on
        a broken stdout pipe (BrokenPipeError piping to e.g. `head`).
        _run_cleanup() -- and the watchdog arm inside it -- must still run
        even then; skipping cleanup because the print failed would trade
        the original swallowed-summary bug for a worse never-cleaned-up
        one.
        """
        cli = _make_cli()
        order: list[str] = []

        with (
            patch.object(
                cli,
                "_print_exit_summary",
                side_effect=BrokenPipeError("broken stdout"),
            ),
            patch.object(cli_mod, "_run_cleanup", side_effect=lambda: order.append("cleanup")),
        ):
            with pytest.raises(BrokenPipeError):
                cli._finish_interactive_exit()

        assert order == ["cleanup"], (
            "_run_cleanup() must run even when _print_exit_summary() raises "
            "-- it's in the finally block precisely so a broken stdout pipe "
            "can't skip cleanup and leave the watchdog unarmed."
        )

    def test_release_session_still_runs_when_print_exit_summary_raises(self):
        cli = _make_cli()
        cli._release_active_session = MagicMock()

        with (
            patch.object(cli, "_print_exit_summary", side_effect=RuntimeError("boom")),
            patch.object(cli_mod, "_run_cleanup"),
        ):
            with pytest.raises(RuntimeError):
                cli._finish_interactive_exit(release_session=True)

        cli._release_active_session.assert_called_once()


class TestExitWatchdogDefaultTimeout:
    """The default budget must cover realistic worst-case cleanup time
    (memory-provider on_session_end + MCP teardown), not just a bare
    process shutdown.
    """

    def test_arm_exit_watchdog_on_shutdown_signal_doubles_the_60s_default(self, monkeypatch):
        """The signal-armed backstop is 2x the normal default, so bumping
        the normal default to 60s must flow through to 120s here too.
        _arm_exit_watchdog() itself no-ops under pytest before ever
        touching timeout_s (PYTEST_CURRENT_TEST guard), so this call site
        -- which calls it via patch.object, same pattern as
        test_exit_watchdog_signal_arm.py in this same directory -- is the
        one place the new default is actually observable in a test.
        """
        monkeypatch.setattr(cli_mod, "_signal_watchdog_armed", False)
        monkeypatch.delenv("HERMES_EXIT_WATCHDOG_S", raising=False)
        with patch.object(cli_mod, "_arm_exit_watchdog") as arm:
            cli_mod._arm_exit_watchdog_on_shutdown_signal()
        arm.assert_called_once_with(timeout_s=120.0, from_signal=True)
