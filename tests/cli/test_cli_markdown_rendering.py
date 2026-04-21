from io import StringIO

from rich.console import Console
from rich.markdown import Markdown

from cli import _render_final_assistant_content


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


def test_final_assistant_content_uses_markdown_renderable():
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert "Title" in output
    assert "one" in output
    assert "two" in output


def test_final_assistant_content_strips_ansi_before_markdown_rendering():
    renderable = _render_final_assistant_content("\x1b[31m# Title\x1b[0m")

    output = _render_to_text(renderable)
    assert "Title" in output
    assert "\x1b" not in output


def test_final_assistant_content_can_strip_markdown_syntax():
    renderable = _render_final_assistant_content(
        "***Bold italic***\n~~Strike~~\n- item\n# Title\n`code`",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "Bold italic" in output
    assert "Strike" in output
    assert "item" in output
    assert "Title" in output
    assert "code" in output
    assert "***" not in output
    assert "~~" not in output
    assert "`" not in output


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


def test_strip_mode_preserves_ordered_lists():
    renderable = _render_final_assistant_content(
        "1. First item\n2. Second item\n3. Third item",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "1. First" in output
    assert "2. Second" in output
    assert "3. Third" in output


def test_strip_mode_preserves_blockquotes():
    renderable = _render_final_assistant_content(
        "> This is quoted text\n> Another quoted line",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "> This is quoted" in output
    assert "> Another quoted" in output


def test_strip_mode_preserves_checkboxes():
    renderable = _render_final_assistant_content(
        "- [ ] Todo item\n- [x] Done item",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "- [ ] Todo" in output
    assert "- [x] Done" in output


def test_strip_mode_preserves_table_structure_while_cleaning_cell_markdown():
    renderable = _render_final_assistant_content(
        "| Syntax | Example |\n|---|---|\n| Bold | `**bold**` |\n| Strike | `~~strike~~` |",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "| Syntax | Example |" in output
    assert "|---|---|" in output
    assert "| Bold | bold |" in output
    assert "| Strike | strike |" in output
    assert "**" not in output
    assert "~~" not in output
    assert "`" not in output


def test_final_assistant_content_can_leave_markdown_raw():
    renderable = _render_final_assistant_content("***Bold italic***", mode="raw")

    output = _render_to_text(renderable)
    assert "***Bold italic***" in output


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
