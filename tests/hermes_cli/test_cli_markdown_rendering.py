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


CODE_ANSWER = (
    "Here is the **class**:\n"
    "\n"
    "```python\n"
    "# leading comment\n"
    "class Thing:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        self._private_ = `x`\n"
    "```\n"
    "\n"
    "That is **all**."
)


def test_strip_mode_keeps_fenced_code_byte_exact():
    from cli import _strip_markdown_syntax

    assert _strip_markdown_syntax(CODE_ANSWER) == (
        "Here is the class:\n"
        "\n"
        "# leading comment\n"
        "class Thing:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self._private_ = `x`\n"
        "\n"
        "That is all."
    )


def test_strip_mode_fence_closes_only_on_its_own_marker():
    from cli import _strip_markdown_syntax

    text = "````md\n```\n# not a heading\n```\n````\n# heading"
    assert _strip_markdown_syntax(text) == "```\n# not a heading\n```\nheading"
    assert _strip_markdown_syntax("~~~\n__init__\n~~~") == "__init__"


FENCED_TABLE = "~~~text\n| key | __init__ |\n|---|---|\n| a | b |\n~~~"


def test_strip_mode_keeps_a_fenced_table_byte_exact_and_aligns_a_prose_table():
    text = FENCED_TABLE + "\n\n| **k** | v |\n|---|---|\n| long cell | x |"

    rendered = _render_final_assistant_content(text, "strip").plain

    assert rendered.split("\n")[:3] == ["| key | __init__ |", "|---|---|", "| a | b |"]
    # The prose table is still stripped first and then re-padded.
    assert "| k         | v   |" in rendered
    assert "| long cell | x   |" in rendered


def test_render_mode_keeps_a_fenced_table_byte_exact():
    output = _render_to_text(_render_final_assistant_content(FENCED_TABLE))

    assert "| key | __init__ |" in output
    assert "|---|---|" in output
