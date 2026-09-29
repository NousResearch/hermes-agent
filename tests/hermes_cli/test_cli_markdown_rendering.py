import re
import sys
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown

from cli import _render_final_assistant_content, _strip_markdown_syntax_keep_links


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


def test_strip_turns_markdown_link_into_osc8_pair():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links(
        "Задача [ITT-3320](https://jira.skala-r.ru/browse/ITT-3320) ждёт ревью")

    # The pair is built from markdown: label visible, target in the escape, URL not left as text.
    assert "\x1b]8;;https://jira.skala-r.ru/browse/ITT-3320\x1b\\" in stripped
    assert stripped == ("Задача \x1b]8;;https://jira.skala-r.ru/browse/ITT-3320\x1b\\ITT-3320"
                        "\x1b]8;;\x1b\\ ждёт ревью")


def test_strip_markdown_link_with_title_keeps_only_the_target():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links('[клик](https://example.org/a "подсказка")')

    assert stripped == "\x1b]8;;https://example.org/a\x1b\\клик\x1b]8;;\x1b\\"
    assert "подсказка" not in stripped


def test_strip_markdown_link_label_keeps_its_markup_stripped():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links("ссылка [**ITT-3320**](https://x/y)")

    assert stripped == "ссылка \x1b]8;;https://x/y\x1b\\ITT-3320\x1b]8;;\x1b\\"


def test_strip_no_link_is_invented_inside_a_code_span():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links("пример: `[ITT-3320](https://x/y)`")

    # Literal code keeps the old marker-pass result and never grows a clickable target.
    assert "\x1b]8;;" not in stripped
    assert stripped == "пример: ITT-3320"


def test_strip_image_syntax_does_not_become_a_link():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links("![скрин](https://x/y.png)")

    assert "\x1b]8;;" not in stripped
    assert stripped == "скрин"


def test_strip_keeps_existing_osc8_untouched_next_to_markdown_link():
    from cli import _strip_markdown_syntax_keep_links

    stripped = _strip_markdown_syntax_keep_links(OSC_JIRA + " и [вторая](https://example.org/a_b_c)")

    assert stripped.count("\x1b]8;;") == 4  # two pairs, both balanced
    assert OSC_JIRA in stripped
    assert "\x1b]8;;https://example.org/a_b_c\x1b\\вторая\x1b]8;;\x1b\\" in stripped


def test_strip_renderable_carries_markdown_link_as_span():
    renderable = _render_final_assistant_content(
        "ключ [ITT-3320](https://jira.skala-r.ru/browse/ITT-3320)", mode="strip")

    # Terminals render the span; the panel's visible text stays escape-free for width math.
    assert "\x1b" not in renderable.plain
    assert renderable.plain == "ключ ITT-3320"
    links = [s.style.link for s in renderable.spans if s.style and getattr(s.style, "link", None)]
    assert links == ["https://jira.skala-r.ru/browse/ITT-3320"]


def test_chat_console_keeps_hyperlink_but_scrubs_other_osc(monkeypatch):
    import cli
    from cli import ChatConsole

    captured_links, captured_plain = [], []
    monkeypatch.setattr(cli, "_cprint_links_raw", lambda line: captured_links.append(line))
    monkeypatch.setattr(cli, "_cprint", lambda line: captured_plain.append(line))

    renderable = _render_final_assistant_content(
        "ключ [ITT-3320](https://jira.skala-r.ru/browse/ITT-3320) \x1b]11;?\x1b\\", mode="strip")
    ChatConsole().print(renderable)

    assert len(captured_links) == 1
    # Rich writes the run as "]8;id=<n>;<uri>"; the label sits between the opening and closing runs.
    assert re.search(r"\x1b]8;(?:id=\d+;)?https://jira\.skala-r\.ru/browse/ITT-3320\x1b\\",
                     captured_links[0])
    assert "\x1b]8;;\x1b\\" in captured_links[0]
    assert "ITT-3320" in captured_links[0]
    # A non-hyperlink OSC sequence (terminal query, title) is still scrubbed out.
    assert "]11;?" not in captured_links[0]


def test_cprint_links_raw_writes_the_pair_without_prompt_toolkit(monkeypatch):
    import hermes_cli.cli_render as render
    import cli

    monkeypatch.setattr(cli, "_output_history_recording", lambda: False)
    written = []
    monkeypatch.setattr(sys, "stdout", type("S", (), {"fileno": lambda self: 4242})())
    monkeypatch.setattr(render.os, "write", lambda fd, data: written.append((fd, data)))

    render._cprint_links_raw("\x1b]8;;https://example.org/x\x1b\\клик\x1b]8;;\x1b\\")

    assert written == [(4242, "\x1b]8;;https://example.org/x\x1b\\клик\x1b]8;;\x1b\\\n".encode())]


def test_stream_line_routes_links_around_prompt_toolkit(monkeypatch):
    import cli
    from hermes_cli.cli_stream_mixin import CLIStreamMixin

    routed, plain = [], []
    monkeypatch.setattr(cli, "_cprint_links_raw", lambda line: routed.append(line))
    monkeypatch.setattr(cli, "_cprint", lambda line: plain.append(line))

    class _Stub:
        _stream_text_ansi = ""

    link_line = _strip_markdown_syntax_keep_links("[ITT-3320](https://x/y)")
    CLIStreamMixin._emit_stream_line(_Stub(), link_line)
    CLIStreamMixin._emit_stream_line(_Stub(), "обычная строка")

    assert routed == [link_line]
    assert plain == ["обычная строка"]


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
