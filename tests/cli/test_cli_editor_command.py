"""Regression tests for the /editor CLI command."""

from pathlib import Path
from unittest.mock import patch

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    queued = []

    class _Queue:
        def put(self, value):
            queued.append(value)

    cli_obj._pending_input = _Queue()
    return cli_obj, queued


def test_process_command_editor_dispatches_to_handler():
    cli_obj, _ = _make_cli()

    with patch.object(cli_obj, "_handle_editor_command", create=True) as mock_handler:
        assert cli_obj.process_command("/editor") is True

    mock_handler.assert_called_once_with("/editor")


def test_process_command_edit_alias_dispatches_to_handler():
    cli_obj, _ = _make_cli()

    with patch.object(cli_obj, "_handle_editor_command", create=True) as mock_handler:
        assert cli_obj.process_command("/edit") is True

    mock_handler.assert_called_once_with("/edit")


def test_handle_editor_command_queues_editor_contents(monkeypatch):
    cli_obj, queued = _make_cli()
    captured = {}

    monkeypatch.setenv("EDITOR", "code --wait")

    def fake_call(argv):
        captured["argv"] = argv
        path = Path(argv[-1])
        path.write_text("Draft from editor\n", encoding="utf-8")
        return 0

    with patch("subprocess.call", side_effect=fake_call), \
         patch("cli._cprint"):
        cli_obj._handle_editor_command("/editor")

    assert queued == ["Draft from editor\n"]
    assert captured["argv"][:2] == ["code", "--wait"]
    assert captured["argv"][-1].endswith(".md")


def test_handle_editor_command_skips_empty_content(monkeypatch):
    cli_obj, queued = _make_cli()

    monkeypatch.setenv("EDITOR", "vim")

    def fake_call(argv):
        Path(argv[-1]).write_text(" \n", encoding="utf-8")
        return 0

    with patch("subprocess.call", side_effect=fake_call), \
         patch("cli._cprint") as mock_print:
        cli_obj._handle_editor_command("/editor")

    assert queued == []
    assert any("without content" in str(call.args[0]) for call in mock_print.call_args_list)


def test_handle_editor_command_reports_missing_editor(monkeypatch):
    cli_obj, queued = _make_cli()

    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)

    with patch("shutil.which", return_value=None), \
         patch("cli._cprint") as mock_print:
        cli_obj._handle_editor_command("/editor")

    assert queued == []
    assert any("No editor found" in str(call.args[0]) for call in mock_print.call_args_list)
