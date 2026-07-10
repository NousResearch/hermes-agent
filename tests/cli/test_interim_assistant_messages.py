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


def test_renders_box_frame():
    """Text is rendered inside a dimmed box frame with left border."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("Found the bug at line 42.")
    calls = [c[0][0] for c in mock_print.call_args_list]
    assert any("Found the bug at line 42." in c for c in calls), f"text not in calls: {calls}"
    # Box frame: top, content, bottom — at least 3 calls
    assert len(calls) >= 3, f"expected >=3 box lines, got {len(calls)}: {calls}"
    # Top line has ╭ and ◆ marker
    assert "╭" in calls[0] and "◆" in calls[0], f"top line missing box: {calls[0]}"
    # Bottom line has ╰
    assert "╰" in calls[-1], f"bottom line missing box: {calls[-1]}"


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


def test_strips_whitespace_in_box():
    """Leading/trailing whitespace is stripped before boxing."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("  ok  ")
    calls = [c[0][0] for c in mock_print.call_args_list]
    # Content line has box padding (│  ...  │) but no raw user whitespace
    content_lines = [c for c in calls if "│" in c]
    assert content_lines, f"no content line found: {calls}"
    # After stripping ANSI and box chars, should be exactly "  ok  "
    # (box padding is "  " on each side; original "  ok  " → stripped "ok")
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', content_lines[0])
    clean = clean.replace('│', '').strip()
    assert clean == "ok", f"expected 'ok', got {clean!r}"


def test_respects_already_streamed_when_enabled():
    """already_streamed=True takes priority over the config flag."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("streaming handled this", already_streamed=True)
    mock_print.assert_not_called()
