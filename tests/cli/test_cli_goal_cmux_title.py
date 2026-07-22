from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_cli(session_id: str = "sid-cmux-goal-title"):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli._goal_manager = None
    cli._pending_input = queue.Queue()
    cli.config = {
        "cmux": {
            "auto_rename_workspace_on_goal": True,
            "goal_title_prefix": "Goal: ",
            "goal_title_max_chars": 60,
        }
    }
    return cli


def test_goal_command_updates_cmux_workspace_title(hermes_home):
    cli = _make_cli()

    with patch("hermes_cli.cli_commands_mixin.rename_cmux_workspace_for_goal") as rename:
        cli._handle_goal_command("/goal Improve cmux workspace titles")

    rename.assert_called_once_with(
        "Improve cmux workspace titles",
        config=cli.config,
    )
    assert cli._pending_input.get_nowait() == "Improve cmux workspace titles"


def test_goal_draft_command_updates_cmux_workspace_title(hermes_home):
    cli = _make_cli(session_id="sid-cmux-goal-draft")

    with (
        patch(
            "hermes_cli.goals.draft_contract",
            return_value=None,
        ),
        patch("hermes_cli.cli_commands_mixin.rename_cmux_workspace_for_goal") as rename,
    ):
        cli._handle_goal_command("/goal draft Improve cmux draft titles")

    rename.assert_called_once_with(
        "Improve cmux draft titles",
        config=cli.config,
    )
    assert cli._pending_input.get_nowait() == "Improve cmux draft titles"
