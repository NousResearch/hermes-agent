from __future__ import annotations

from unittest.mock import MagicMock

from tests.cli.test_cli_new_session import _make_cli


def test_queue_command_sets_status_area_preview_when_agent_running():
    cli = _make_cli()
    cli.console = MagicMock()
    cli._invalidate = MagicMock()
    cli._agent_running = True

    cli.process_command("/queue investigate parser state")

    assert cli._queued_turn_preview == "investigate parser state"
    cli._invalidate.assert_called()


def test_extra_tui_widgets_show_queued_preview():
    cli = _make_cli()
    cli._queued_turn_preview = "queued prompt"

    widgets = cli._get_extra_tui_widgets()

    assert len(widgets) == 1
