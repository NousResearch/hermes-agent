"""Tests for the /update slash command in the classic CLI.

Verifies that ``HermesCLI._handle_update_command`` correctly:
- Refuses to run under a managed install (Homebrew, Docker, etc.)
- Cancels cleanly on a "no" answer or unrecognized input
- Cancels cleanly on timeout / no modal response
- Schedules a main-thread relaunch on affirmative answers

The TUI slash handler is covered by ``createSlashHandler.test.ts``; the
Python wrapper handoff for exit code 42 is covered in
``tests/hermes_cli/test_tui_resume_flow.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cli import HermesCLI


@pytest.fixture
def cli_stub():
    """Minimal HermesCLI instance without running ``__init__``."""
    stub = object.__new__(HermesCLI)
    stub._pending_relaunch_argv = None
    stub._app = None
    return stub


def _call(cli_stub):
    """Invoke the real ``_handle_update_command`` on the stub."""
    return HermesCLI._handle_update_command(cli_stub)


def test_managed_install_refuses_and_does_not_schedule_relaunch(cli_stub, capsys):
    with (
        patch("hermes_cli.config.is_managed", return_value=True),
        patch(
            "hermes_cli.config.format_managed_message",
            return_value="Use `brew upgrade hermes-agent` to update.",
        ),
        patch.object(HermesCLI, "_prompt_text_input_modal") as mock_modal,
    ):
        assert _call(cli_stub) is True

    out = capsys.readouterr().out
    assert "brew upgrade hermes-agent" in out
    mock_modal.assert_not_called()
    assert cli_stub._pending_relaunch_argv is None


@pytest.mark.parametrize("answer", ["no", "n", "2"])
def test_negative_answer_cancels(cli_stub, answer):
    cli_stub._pending_relaunch_argv = None
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch.object(HermesCLI, "_prompt_text_input_modal", return_value=answer),
    ):
        assert _call(cli_stub) is True

    assert cli_stub._pending_relaunch_argv is None


@pytest.mark.parametrize("answer", ["nope", "cancel", "maybe"])
def test_unrecognized_answer_cancels(cli_stub, answer, capsys):
    cli_stub._pending_relaunch_argv = None
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch.object(HermesCLI, "_prompt_text_input_modal", return_value=answer),
    ):
        assert _call(cli_stub) is True

    assert cli_stub._pending_relaunch_argv is None
    assert "/update cancelled." in capsys.readouterr().out


def test_modal_timeout_cancels(cli_stub):
    cli_stub._pending_relaunch_argv = None
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch.object(HermesCLI, "_prompt_text_input_modal", return_value=None),
    ):
        assert _call(cli_stub) is True

    assert cli_stub._pending_relaunch_argv is None


@pytest.mark.parametrize("answer", ["yes", "y", "Y", "1", ""])
def test_affirmative_answer_schedules_main_thread_relaunch(cli_stub, answer):
    cli_stub._pending_relaunch_argv = None
    with (
        patch("hermes_cli.config.is_managed", return_value=False),
        patch.object(HermesCLI, "_prompt_text_input_modal", return_value=answer),
    ):
        assert _call(cli_stub) is False

    assert cli_stub._pending_relaunch_argv == ["update"]
