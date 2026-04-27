"""Tests for prompt-toolkit-safe Rich console rendering."""

from unittest.mock import patch


def test_chat_console_batches_multiline_output_to_one_render_call():
    """Large panels should not be emitted one terminal line at a time."""
    from rich.panel import Panel

    from cli import ChatConsole

    console = ChatConsole()
    with patch("cli._cprint") as mock_cprint:
        console.print(Panel("first line\nsecond line\nthird line", title="History"))

    assert mock_cprint.call_count == 1
    rendered = mock_cprint.call_args.args[0]
    assert "first line" in rendered
    assert "second line" in rendered
    assert "third line" in rendered
    assert "\n" in rendered
