from io import StringIO

from rich.console import Console
from rich.markdown import Markdown

from cli import _render_final_assistant_content
from unittest.mock import patch
from hermes_cli.cli_render import ChatConsole


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


def _render_to_ansi_via_chatconsole(renderable) -> str:
    """Render a Markdown object via ChatConsole and capture the ANSI output."""
    captured_lines = []
    def mock_cprint(line):
        captured_lines.append(line)
    # We need to patch cli._cprint to capture the output
    with patch('cli._cprint', side_effect=mock_cprint):
        console = ChatConsole()
        console.print(renderable)
    return ''.join(captured_lines)


def test_final_assistant_content_uses_markdown_renderable():
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert "Title" in output
    assert "one" in output
    assert "two" in output




def test_final_assistant_content_keeps_non_path_markdown_escapes():
    renderable = _render_final_assistant_content(r"1\. Not an ordered list")

    output = _render_to_text(renderable)
    assert "1. Not an ordered list" in output
    assert r"1\." not in output






def test_strip_mode_preserves_lists():
    renderable = _render_final_assistant_content(
        "**Formatting**\n- Ran prettier\n- Files changed\n- Verified clean",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "- Ran prettier" in output
    assert "- Files changed" in output
    assert "- Verified clean" in output
    assert "**" not in output




def test_strip_mode_preserves_blockquotes():
    renderable = _render_final_assistant_content(
        "> This is quoted text\n> Another quoted line",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "> This is quoted" in output
    assert "> Another quoted" in output






def test_strip_mode_preserves_cron_asterisks_in_plain_text():
    renderable = _render_final_assistant_content("* * * * *", mode="strip")

    output = _render_to_text(renderable)
    assert "* * * * *" in output

    # Still treat the canonical 3-asterisk Markdown horizontal rule as decoration.
    renderable = _render_final_assistant_content("* * *", mode="strip")
    output = _render_to_text(renderable)
    assert "* * *" not in output




def test_strip_mode_preserves_intraword_underscores_in_snake_case_identifiers():
    renderable = _render_final_assistant_content(
        "Let me look at test_case_with_underscores and SOME_CONST "
        "then /tmp/snake_case_dir/file_with_name.py",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "test_case_with_underscores" in output
    assert "SOME_CONST" in output
    assert "snake_case_dir" in output
    assert "file_with_name" in output


def test_strip_mode_still_strips_boundary_underscore_emphasis():
    renderable = _render_final_assistant_content(
        "say _hi_ and __bold__ now",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "say hi and bold now" in output




def test_bold_text_color_applied_when_set():
    """When bold_text is set in the skin, strong markdown should be rendered with that color."""
    with patch('hermes_cli.skin_engine.get_active_skin') as mock_get_skin:
        mock_skin = mock_get_skin.return_value
        mock_skin.get_color.side_effect = lambda key, default: {
            'bold_text': '#FF0000',  # red
            'banner_text': '#00FF00',  # green
        }.get(key, default)
        renderable = _render_final_assistant_content("**bold** normal")
        ansi = _render_to_ansi_via_chatconsole(renderable)
        # Check that the red color escape sequence is present (truecolor)
        assert '38;2;255;0;0m' in ansi
        # Check that the bold text is present
        assert 'bold' in ansi
        # Check that the normal text uses the banner_text color (green)
        # We look for the green escape sequence before the word "normal"
        # Note: there might be a space between bold and normal, so we check for the sequence
        # We expect: ... bold [reset] normal
        # But we don't know the exact reset. We'll just check that green appears after the bold part.
        # We split the ANSI by the bold text and look in the remainder.
        parts = ansi.split('bold')
        if len(parts) > 1:
            after_bold = parts[1]
            assert '38;2;0;255;0m' in after_bold
        else:
            # If we didn't split, we just check that green is present somewhere
            assert '38;2;0;255;0m' in ansi


def test_bold_text_fallback_to_banner_text():
    """When bold_text is not set, it should fall back to banner_text."""
    with patch('hermes_cli.skin_engine.get_active_skin') as mock_get_skin:
        mock_skin = mock_get_skin.return_value
        mock_skin.get_color.side_effect = lambda key, default: {
            'bold_text': None,  # not set
            'banner_text': '#00FF00',  # green
        }.get(key, default)
        renderable = _render_final_assistant_content("**bold**")
        ansi = _render_to_ansi_via_chatconsole(renderable)
        # Expect the green color escape sequence (banner_text) to be used for bold
        assert '38;2;0;255;0m' in ansi
        assert 'bold' in ansi
        # Also, we should not see the red color (if we had set it to red in the mock, we didn't)
        # But we set bold_text to None, so it should not be red.
        # We can check that red is not present, but note: the banner_text is green, so red should not be there.
        # We'll just check that the green is used.


def test_bold_text_invalid_value_falls_back():
    """If bold_text is invalid (e.g., empty string), it should fall back to banner_text."""
    with patch('hermes_cli.skin_engine.get_active_skin') as mock_get_skin:
        mock_skin = mock_get_skin.return_value
        mock_skin.get_color.side_effect = lambda key, default: {
            'bold_text': '',  # invalid
            'banner_text': '#00FF00',  # green
        }.get(key, default)
        renderable = _render_final_assistant_content("**bold**")
        ansi = _render_to_ansi_via_chatconsole(renderable)
        # Expect the green color escape sequence (banner_text) to be used for bold
        assert '38;2;0;255;0m' in ansi
        assert 'bold' in ansi


def test_bold_text_null_value_falls_back():
    """If bold_text is null (None), it should fall back to banner_text."""
    with patch('hermes_cli.skin_engine.get_active_skin') as mock_get_skin:
        mock_skin = mock_get_skin.return_value
        mock_skin.get_color.side_effect = lambda key, default: {
            'bold_text': None,  # explicit None
            'banner_text': '#00FF00',  # green
        }.get(key, default)
        renderable = _render_final_assistant_content("**bold**")
        ansi = _render_to_ansi_via_chatconsole(renderable)
        # Expect the green color escape sequence (banner_text) to be used for bold
        assert '38;2;0;255;0m' in ansi
        assert 'bold' in ansi
