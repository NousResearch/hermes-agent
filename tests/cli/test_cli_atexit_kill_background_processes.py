"""Regression tests for #40343 — background processes must be killed when
the CLI exits via ``_run_cleanup`` (atexit handler).

Before the fix, ``_run_cleanup`` cleaned up terminals, browsers, MCP servers,
and memory providers — but did NOT call ``process_registry.kill_all()``.
Background processes started via ``terminal(background=True)`` survived CLI
exit and kept holding ports, causing conflicts on restart.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch


@patch("hermes_cli.plugins.invoke_hook")
@patch("cli._cleanup_all_browsers")
@patch("cli._cleanup_all_terminals")
@patch("cli._reset_terminal_input_modes_on_exit")
def test_cleanup_calls_process_registry_kill_all(
    _mock_reset, _mock_term, _mock_browsers, _mock_hook
):
    """_run_cleanup must call process_registry.kill_all() to reap background
    processes started via terminal(background=True)."""
    import cli as cli_mod

    with patch("tools.process_registry.process_registry") as mock_registry:
        cli_mod._cleanup_done = False
        try:
            cli_mod._run_cleanup()
        finally:
            cli_mod._cleanup_done = False

    mock_registry.kill_all.assert_called_once()


@patch("hermes_cli.plugins.invoke_hook")
@patch("cli._cleanup_all_browsers")
@patch("cli._cleanup_all_terminals")
@patch("cli._reset_terminal_input_modes_on_exit")
def test_cleanup_kill_all_exception_is_swallowed(
    _mock_reset, _mock_term, _mock_browsers, _mock_hook
):
    """A raising process_registry.kill_all must not crash CLI exit — the
    atexit handler must continue with browser / MCP / memory teardown."""
    import cli as cli_mod

    with patch("tools.process_registry.process_registry") as mock_registry:
        mock_registry.kill_all.side_effect = RuntimeError("registry corrupt")
        cli_mod._cleanup_done = False
        try:
            cli_mod._run_cleanup()  # must not raise
        finally:
            cli_mod._cleanup_done = False

    mock_registry.kill_all.assert_called_once()


@patch("hermes_cli.plugins.invoke_hook")
@patch("cli._cleanup_all_browsers")
@patch("cli._cleanup_all_terminals")
@patch("cli._reset_terminal_input_modes_on_exit")
def test_cleanup_kill_all_import_failure_is_swallowed(
    _mock_reset, _mock_term, _mock_browsers, _mock_hook
):
    """If the process_registry module can't be imported, cleanup must not
    crash — it should continue with the remaining teardown steps."""
    import cli as cli_mod

    with patch(
        "builtins.__import__",
        side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(
            ImportError("no module")
        )
        if name == "tools.process_registry"
        else __import__(name, *a, **kw),
    ):
        cli_mod._cleanup_done = False
        try:
            cli_mod._run_cleanup()  # must not raise
        finally:
            cli_mod._cleanup_done = False
