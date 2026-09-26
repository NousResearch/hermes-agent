from io import StringIO

from rich.console import Console
from rich.markdown import Markdown

from cli import _render_final_assistant_content


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


OSC_JIRA = "\x1b]8;;https://jira.skala-r.ru/browse/ITT-3320\x1b\\ITT-3320\x1b]8;;\x1b\\"


def test_strip_keeps_hyperlinks_but_drops_markdown_and_colour():
    from cli import _strip_markdown_syntax_keep_links

    source = OSC_JIRA + " — **жирный** и \x1b[31mкрасный\x1b[0m"
    stripped = _strip_markdown_syntax_keep_links(source)

    # Link target survives as an OSC 8 pair around the same visible text.
    assert "\x1b]8;;https://jira.skala-r.ru/browse/ITT-3320\x1b\\" in stripped
    assert stripped.count("\x1b]8;;") == 2
    # Markdown markers and colour SGR are gone, link text is untouched.
    assert "**" not in stripped and "\x1b[31m" not in stripped
    assert "ITT-3320" in stripped and "жирный" in stripped


def test_strip_renderable_carries_link_span_without_escapes():
    renderable = _render_final_assistant_content(OSC_JIRA + " — текст", mode="strip")

    # Visible text stays plain (no escapes), so cell widths are computed correctly.
    assert "\x1b" not in renderable.plain
    assert renderable.plain.startswith("ITT-3320")
    links = [s.style.link for s in renderable.spans if s.style and getattr(s.style, "link", None)]
    assert links == ["https://jira.skala-r.ru/browse/ITT-3320"]


def test_strip_plain_text_is_unchanged_by_link_handling():
    renderable = _render_final_assistant_content("простой текст и **маркеры**", mode="strip")

    assert "\x1b" not in renderable.plain
    assert renderable.plain == "простой текст и маркеры"


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
