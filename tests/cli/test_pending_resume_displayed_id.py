"""Regression coverage for preserving the displayed resume-selection snapshot."""

from unittest.mock import MagicMock

from cli import HermesCLI


def test_pending_resume_selection_uses_displayed_session_id():
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_resume_sessions = [
        {"id": "row-one"},
        {"id": "row-eleven"},
    ]
    cli._handle_resume_command = MagicMock()

    assert cli._consume_pending_resume_selection("2") is True

    cli._handle_resume_command.assert_called_once_with("/resume row-eleven")
