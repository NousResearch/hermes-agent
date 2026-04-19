"""Tests for CLI /copy command."""

import os
from unittest.mock import MagicMock, call, patch

from cli import HermesCLI, _extract_code_blocks


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


# ---------------------------------------------------------------------------
# Existing tests (unchanged behavior)
# ---------------------------------------------------------------------------

def test_copy_copies_latest_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "latest"},
    ]

    with patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy:
        result = cli_obj.process_command("/copy")

    assert result is True
    mock_copy.assert_called_once_with("latest")


def test_copy_with_index_uses_requested_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]

    with patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy:
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

    with patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy:
        cli_obj.process_command("/copy")

    mock_copy.assert_called_once_with("Visible answer")


def test_copy_falls_back_to_osc52_when_native_unavailable():
    """When _copy_to_system_clipboard returns False, OSC 52 is used."""
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "fallback"}]

    with (
        patch.object(cli_obj, "_copy_to_system_clipboard", return_value=False),
        patch.object(cli_obj, "_write_osc52_clipboard") as mock_osc,
    ):
        cli_obj.process_command("/copy")

    mock_osc.assert_called_once_with("fallback")


def test_copy_invalid_index_does_not_copy():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "only"}]

    with (
        patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy,
        patch("cli._cprint") as mock_print,
    ):
        cli_obj.process_command("/copy 99")

    mock_copy.assert_not_called()
    assert any("Invalid response number" in str(call) for call in mock_print.call_args_list)


# ---------------------------------------------------------------------------
# _extract_code_blocks
# ---------------------------------------------------------------------------

def test_extract_code_blocks_finds_fenced_blocks():
    text = "Here:\n```python\nprint('hi')\n```\nAnd:\n```js\nconsole.log(1)\n```"
    blocks = _extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == ("python", "print('hi')")
    assert blocks[1] == ("js", "console.log(1)")


def test_extract_code_blocks_no_lang_defaults_to_text():
    blocks = _extract_code_blocks("```\nfoo\n```")
    assert blocks == [("text", "foo")]


def test_extract_code_blocks_returns_empty_for_no_blocks():
    assert _extract_code_blocks("just plain text") == []


# ---------------------------------------------------------------------------
# Interactive picker — code blocks trigger picker
# ---------------------------------------------------------------------------

_RESPONSE_WITH_BLOCKS = (
    "Here is some code:\n"
    "```python\nprint('hello')\n```\n"
    "And more:\n"
    "```bash\necho hi\n```"
)


def test_copy_opens_picker_when_code_blocks_present():
    """When the response contains code blocks, _copy_with_picker is called."""
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": _RESPONSE_WITH_BLOCKS},
    ]

    with patch.object(cli_obj, "_copy_with_picker") as mock_picker:
        cli_obj.process_command("/copy")

    mock_picker.assert_called_once()
    args = mock_picker.call_args
    assert args[0][0] == _RESPONSE_WITH_BLOCKS  # full_text
    assert len(args[0][1]) == 2  # two code blocks


def test_copy_no_picker_when_no_code_blocks():
    """Plain text responses skip the picker and copy directly."""
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": "just text, no blocks"},
    ]

    with patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy:
        cli_obj.process_command("/copy")

    mock_copy.assert_called_once_with("just text, no blocks")


def test_copy_with_picker_copies_full_response():
    """Picker selecting index 0 copies full response."""
    cli_obj = _make_cli()
    blocks = [("python", "print('hello')"), ("bash", "echo hi")]

    with (
        patch("hermes_cli.curses_ui.curses_copy_picker", return_value=(0, False)),
        patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy,
    ):
        cli_obj._copy_with_picker(_RESPONSE_WITH_BLOCKS, blocks, 0)

    mock_copy.assert_called_once_with(_RESPONSE_WITH_BLOCKS)


def test_copy_with_picker_copies_specific_block():
    """Picker selecting index 1 copies the first code block."""
    cli_obj = _make_cli()
    blocks = [("python", "print('hello')"), ("bash", "echo hi")]

    with (
        patch("hermes_cli.curses_ui.curses_copy_picker", return_value=(1, False)),
        patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy,
    ):
        cli_obj._copy_with_picker(_RESPONSE_WITH_BLOCKS, blocks, 0)

    mock_copy.assert_called_once_with("print('hello')")


def test_copy_with_picker_cancel():
    """Picker returning None does not copy anything."""
    cli_obj = _make_cli()
    blocks = [("python", "print('hello')")]

    with (
        patch("hermes_cli.curses_ui.curses_copy_picker", return_value=(None, False)),
        patch.object(cli_obj, "_copy_to_system_clipboard", return_value=True) as mock_copy,
        patch("cli._cprint"),
    ):
        cli_obj._copy_with_picker("text", blocks, 0)

    mock_copy.assert_not_called()


def test_copy_with_picker_write_to_file(tmp_path):
    """Picker with write=True writes to disk instead of clipboard."""
    cli_obj = _make_cli()
    blocks = [("python", "print('hello')")]
    dest = tmp_path / "out.py"

    with (
        patch("hermes_cli.curses_ui.curses_copy_picker", return_value=(1, True)),
        patch("cli._cprint"),
        patch("prompt_toolkit.prompt", return_value=str(dest)),
    ):
        cli_obj._copy_with_picker("full text", blocks, 0)

    assert dest.read_text().strip() == "print('hello')"


# ---------------------------------------------------------------------------
# curses_copy_picker — fallback path
# ---------------------------------------------------------------------------

def test_curses_copy_picker_fallback_non_tty():
    """Non-TTY stdin returns (None, False) immediately."""
    from hermes_cli.curses_ui import curses_copy_picker

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False
        result = curses_copy_picker(["Full response", "Block 1"])

    assert result == (None, False)


# ---------------------------------------------------------------------------
# _copy_to_system_clipboard
# ---------------------------------------------------------------------------

def test_system_clipboard_uses_pbcopy_on_macos():
    """On macOS without SSH, pbcopy is used."""
    cli_obj = _make_cli()

    with (
        patch("sys.platform", "darwin"),
        patch.dict(os.environ, {}, clear=False),
        patch("shutil.which", return_value="/usr/bin/pbcopy"),
        patch("subprocess.run") as mock_run,
    ):
        os.environ.pop("SSH_CONNECTION", None)
        result = cli_obj._copy_to_system_clipboard("hello")

    assert result is True
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["pbcopy"]


def test_system_clipboard_skipped_over_ssh():
    """Over SSH, native clipboard is skipped (returns False)."""
    cli_obj = _make_cli()

    with patch.dict(os.environ, {"SSH_CONNECTION": "1.2.3.4 5678 5.6.7.8 22"}):
        result = cli_obj._copy_to_system_clipboard("hello")

    assert result is False
