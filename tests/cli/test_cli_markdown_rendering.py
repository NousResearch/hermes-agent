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
