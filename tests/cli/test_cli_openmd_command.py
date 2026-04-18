"""Tests for CLI /openmd command."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-openmd-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    return cli_obj


def test_openmd_pipes_latest_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "latest"},
    ]
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()

    with patch("cli.shutil.which", return_value="/usr/bin/openmd"), patch(
        "subprocess.Popen", return_value=mock_proc
    ) as mock_popen, patch("cli._cprint"):
        result = cli_obj.process_command("/openmd")

    assert result is True
    mock_popen.assert_called_once_with(["openmd"], stdin=-1, text=True)
    mock_proc.stdin.write.assert_called_once_with("latest")
    mock_proc.stdin.close.assert_called_once()


def test_openmd_with_index_uses_requested_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()

    with patch("cli.shutil.which", return_value="/usr/bin/openmd"), patch(
        "subprocess.Popen", return_value=mock_proc
    ), patch("cli._cprint"):
        cli_obj.process_command("/openmd 1")

    mock_proc.stdin.write.assert_called_once_with("one")


def test_openmd_strips_reasoning_blocks_before_piping():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {
            "role": "assistant",
            "content": "<REASONING_SCRATCHPAD>internal</REASONING_SCRATCHPAD>\nVisible answer",
        }
    ]
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()

    with patch("cli.shutil.which", return_value="/usr/bin/openmd"), patch(
        "subprocess.Popen", return_value=mock_proc
    ), patch("cli._cprint"):
        cli_obj.process_command("/openmd")

    mock_proc.stdin.write.assert_called_once_with("Visible answer")


def test_openmd_extracts_text_from_multipart_content():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()

    with patch("cli.shutil.which", return_value="/usr/bin/openmd"), patch(
        "subprocess.Popen", return_value=mock_proc
    ), patch("cli._cprint"):
        cli_obj.process_command("/openmd")

    mock_proc.stdin.write.assert_called_once_with("Hello\nworld")


def test_openmd_not_installed_does_not_launch():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "hello"}]

    with patch("cli.shutil.which", return_value=None), patch(
        "subprocess.Popen"
    ) as mock_popen, patch("cli._cprint") as mock_print:
        cli_obj.process_command("/openmd")

    mock_popen.assert_not_called()
    rendered = " ".join(str(arg) for call in mock_print.call_args_list for arg in call.args)
    assert "openmd is not installed" in rendered


def test_openmd_invalid_index_does_not_launch():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "only"}]

    with patch("cli.shutil.which", return_value="/usr/bin/openmd"), patch(
        "subprocess.Popen"
    ) as mock_popen, patch("cli._cprint") as mock_print:
        cli_obj.process_command("/openmd 99")

    mock_popen.assert_not_called()
    assert any("Invalid response number" in str(call) for call in mock_print.call_args_list)
