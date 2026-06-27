"""Regression test: a first Ctrl+C during startup plugin discovery exits
cleanly instead of dumping a KeyboardInterrupt traceback.

`main()` runs `_prepare_agent_startup()` (which triggers `discover_plugins()`)
before any command takes over. That call walks every plugin manifest and
imports heavy third-party SDKs (e.g. the Teams adapter), so it owns most of
startup wall-clock and is where a startup Ctrl+C almost always lands. Before
the guard, the interrupt unwound uncaught out of `main()` and the interpreter
printed a full traceback.

The fix wraps the `_prepare_agent_startup(args)` call site in `main()`
(hermes_cli/main.py) in a `try/except KeyboardInterrupt` that prints a
one-line notice and raises `SystemExit(130)` (128 + SIGINT). Runtime
interrupts during a command/TUI keep their own finer-grained handling
downstream — this guard only covers the boot window.

These tests drive the real `main()` with the pre-discovery boot helpers
stubbed to no-ops, then force `_prepare_agent_startup` to raise
KeyboardInterrupt and assert the contract. `["hermes"]` with no subcommand
reaches `_prepare_agent_startup` after only arg parsing and the `--version`
check, so nothing downstream of the guard runs.
"""

import sys
from unittest.mock import patch

import pytest

import hermes_cli.main as main_mod


def _run_main_with_startup_interrupt(argv):
    """Invoke the real ``main()`` with boot helpers neutralised and plugin
    discovery forced to raise KeyboardInterrupt, mirroring a first Ctrl+C
    landing inside ``_prepare_agent_startup``."""
    with patch.object(sys, "argv", argv), \
            patch.object(main_mod, "_set_process_title", lambda *a, **k: None), \
            patch.object(main_mod, "_cleanup_quarantined_exes", lambda *a, **k: None), \
            patch.object(main_mod, "_recover_from_interrupted_install", lambda *a, **k: None), \
            patch.object(main_mod, "_try_termux_fast_tui_launch", lambda *a, **k: False), \
            patch.object(main_mod, "_try_termux_fast_cli_launch", lambda *a, **k: False), \
            patch.object(main_mod, "_prepare_agent_startup", side_effect=KeyboardInterrupt):
        return main_mod.main()


class TestStartupKeyboardInterrupt:
    def test_first_ctrl_c_during_discovery_exits_130(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_main_with_startup_interrupt(["hermes"])
        assert exc_info.value.code == 130

    def test_first_ctrl_c_prints_notice_to_stderr(self, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_startup_interrupt(["hermes"])
        assert "interrupted" in capsys.readouterr().err

    def test_keyboardinterrupt_does_not_propagate(self):
        # The guard must convert the interrupt into SystemExit; a bare
        # KeyboardInterrupt leaking through means the guard is gone.
        try:
            _run_main_with_startup_interrupt(["hermes"])
        except SystemExit:
            pass
        except KeyboardInterrupt:
            pytest.fail("KeyboardInterrupt leaked out of main() — guard missing")
