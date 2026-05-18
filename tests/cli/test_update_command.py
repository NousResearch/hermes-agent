"""Tests for the /update slash command in the classic CLI.

Verifies that ``HermesCLI._handle_update_command`` correctly:
- Refuses to run under a managed install (Homebrew, Docker, etc.)
- Cancels cleanly on a "n" answer
- Cancels cleanly on EOF / Ctrl-C at the confirmation prompt
- Relaunches as ``hermes update`` on a "y" answer (the default)

The TUI side (``/update`` → exit-code-42 → relaunch in main.py) is covered
by ``ui-tui/src/__tests__/createSlashHandler.test.ts``; this file covers
the classic CLI path that the TS test cannot reach.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


@pytest.fixture
def cli_stub():
    """Build a minimal HermesCLI-like object bound to the real method.

    We don't run ``__init__`` — we only need ``self`` to dispatch the
    method.  All collaborators it touches (relaunch, is_managed, input,
    print) are patched at call sites.
    """
    return MagicMock(spec=HermesCLI)


def _call(cli_stub):
    """Invoke the real ``_handle_update_command`` on the stub."""
    HermesCLI._handle_update_command(cli_stub)


def test_managed_install_refuses_and_does_not_relaunch(cli_stub, capsys):
    """Under a managed install (brew/docker), /update prints a hint and
    returns without launching anything."""
    with (
        patch("hermes_cli.config.is_managed", return_value=True),
        patch(
            "hermes_cli.config.format_managed_message",
            return_value="Use `brew upgrade hermes-agent` to update.",
        ),
        patch("hermes_cli.relaunch.relaunch") as mock_relaunch,
        patch("builtins.input") as mock_input,
    ):
        _call(cli_stub)

    out = capsys.readouterr().out
    assert "brew upgrade hermes-agent" in out
    mock_input.assert_not_called()
    mock_relaunch.assert_not_called()


@pytest.mark.parametrize("answer", ["n", "N", "no", "NO", " no "])
def test_negative_answer_cancels(cli_stub, answer, capsys):
    """Any "no"-shaped answer cancels without relaunching."""
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch("builtins.input", return_value=answer),
        patch("hermes_cli.relaunch.relaunch") as mock_relaunch,
    ):
        _call(cli_stub)

    mock_relaunch.assert_not_called()
    # Should not have printed "Launching update..."
    assert "Launching update" not in capsys.readouterr().out


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_eof_or_ctrl_c_at_prompt_cancels(cli_stub, exc):
    """EOFError (Ctrl-D) and KeyboardInterrupt (Ctrl-C) cancel cleanly."""
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch("builtins.input", side_effect=exc),
        patch("hermes_cli.relaunch.relaunch") as mock_relaunch,
    ):
        _call(cli_stub)

    mock_relaunch.assert_not_called()


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "", " "])
def test_affirmative_answer_relaunches_as_update(cli_stub, answer):
    """Empty string (default Y) and any "yes"-shaped answer relaunch
    via ``hermes_cli.relaunch.relaunch`` with the ``update`` subcommand
    and ``preserve_inherited=False`` so flags like --tui aren't carried."""
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch("builtins.input", return_value=answer),
        patch("hermes_cli.relaunch.relaunch") as mock_relaunch,
    ):
        _call(cli_stub)

    mock_relaunch.assert_called_once_with(["update"], preserve_inherited=False)
