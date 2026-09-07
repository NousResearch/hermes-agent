"""Tests for CLI /copy command."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-copy-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    return cli_obj


def test_copy_copies_latest_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "latest"},
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        result = cli_obj.process_command("/copy")

    assert result is True
    mock_copy.assert_called_once_with("latest")


def test_copy_with_index_uses_requested_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        cli_obj.process_command("/copy 1")

    mock_copy.assert_called_once_with("one")


def test_copy_strips_reasoning_blocks_before_copy():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {
            "role": "assistant",
            "content": "<REASONING_SCRATCHPAD>internal</REASONING_SCRATCHPAD>\nVisible answer",
        }
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        cli_obj.process_command("/copy")

    mock_copy.assert_called_once_with("Visible answer")




def test_copy_prefers_osc52_in_ssh_sessions():
    """Over SSH, native tools write the REMOTE clipboard — OSC 52 reaches
    the local terminal instead (#31528)."""
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "remote answer"}]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_native, \
         patch("hermes_cli.clipboard.is_remote_shell_session", return_value=True), \
         patch.object(cli_obj, "_write_osc52_clipboard") as mock_osc52:
        cli_obj.process_command("/copy")

    mock_osc52.assert_called_once_with("remote answer")
    mock_native.assert_not_called()


def test_copy_native_first_when_local():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "local answer"}]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_native, \
         patch("hermes_cli.clipboard.is_remote_shell_session", return_value=False), \
         patch.object(cli_obj, "_write_osc52_clipboard") as mock_osc52:
        cli_obj.process_command("/copy")

    mock_native.assert_called_once_with("local answer")
    mock_osc52.assert_not_called()


