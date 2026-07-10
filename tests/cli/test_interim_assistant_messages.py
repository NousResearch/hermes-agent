"""Tests for CLI interim assistant message rendering.

The agent core emits real assistant commentary between tool calls via
``interim_assistant_callback``.  The CLI's ``_on_interim_assistant``
handler renders these when ``display.interim_assistant_messages`` is true.
"""

from unittest.mock import patch

from cli import HermesCLI


def _make_cli():
    """Bare-metal HermesCLI without side effects."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.interim_assistant_messages = True
    cli.model = "test-model"
    cli.agent = None
    return cli


def test_renders_when_enabled():
    """Visible text is printed when the config flag is on."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("Found the bug at line 42.")
    mock_print.assert_called_once_with("  Found the bug at line 42.")


def test_silent_when_disabled():
    """No output when display.interim_assistant_messages is false."""
    cli = _make_cli()
    cli.interim_assistant_messages = False
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("You should not see this.")
    mock_print.assert_not_called()


def test_skips_already_streamed():
    """When streaming already displayed the text, do not repeat it."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("already shown via stream", already_streamed=True)
    mock_print.assert_not_called()


def test_skips_empty_text():
    """Blank or whitespace-only text is silently dropped."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("   ")
    mock_print.assert_not_called()


def test_strips_whitespace():
    """Leading/trailing whitespace is stripped before printing."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("  ok  ")
    mock_print.assert_called_once_with("  ok")


def test_respects_already_streamed_when_enabled():
    """already_streamed=True takes priority over the config flag."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("streaming handled this", already_streamed=True)
    mock_print.assert_not_called()
