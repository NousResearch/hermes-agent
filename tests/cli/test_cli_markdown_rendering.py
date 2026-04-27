from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import cli
from cli import (
    ChatConsole,
    _render_final_assistant_content,
)


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


def test_final_assistant_content_uses_markdown_renderable():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("default")
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert "Title" in output
    assert "one" in output
    assert "two" in output


def test_final_assistant_content_uses_low_background_code_theme():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("default")
    renderable = _render_final_assistant_content("```python\ndef smoke():\n    return True\n```")

    assert isinstance(renderable, Markdown)
    assert renderable.code_theme == "default"
    assert renderable.inline_code_theme == "default"


def test_slate_skin_uses_compact_code_blocks_in_render_mode():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content("```python\ndef smoke():\n    return True\n```")
    )

    assert "python" in output
    assert "def smoke" in output
    assert "1 def smoke" in output
    assert "2     return True" in output
    assert "╭" in output


def test_slate_diff_code_blocks_show_line_numbers():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "```diff\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            "```"
        )
    )

    assert "diff" in output
    assert "1 --- a/README.md" in output
    assert "4 -old line" in output
    assert "5 +new line" in output


def test_default_skin_leaves_markdown_code_blocks_to_rich():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("default")
    output = _render_to_text(
        _render_final_assistant_content("```python\ndef smoke():\n    return True\n```")
    )

    assert "python:" not in output


def test_slate_renders_indented_code_as_bounded_block():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "Config:\n\n    display:\n      final_response_markdown: render\n      streaming: true\n\nDone"
        )
    )

    assert "code" in output
    assert "display:" in output
    assert "final_response_markdown: render" in output
    assert "╭" in output


def test_slate_preserves_nested_lists_while_rendering_indented_code():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "- parent\n"
            "    - nested item\n"
            "    - second nested\n"
            "\n"
            "    const value = true\n"
        )
    )

    assert "nested item" in output
    assert "second nested" in output
    assert "const value = true" in output
    assert "╭" in output


def test_chat_console_print_preserves_multiline_rich_ansi_stream(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print("first\nsecond")

    assert len(calls) == 1
    assert "first" in calls[0]
    assert "\n" in calls[0]
    assert "second" in calls[0]


def test_slate_chat_console_applies_code_background(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(_render_final_assistant_content("```python\nprint('ok')\n```"))

    assert calls
    assert "48;2;30;41;59" in calls[0]


def test_slate_compact_renderer_handles_unclosed_fence(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(
        _render_final_assistant_content(
            "Before\n\n```raw\n**What to check**\n- render mode\n| A | B |\n"
        )
    )

    assert calls
    assert "raw" in calls[0]
    assert "**What to check**" in calls[0]
    assert "48;2;30;41;59" in calls[0]


def test_slate_renders_blockquote_fenced_code_with_code_background(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(
        _render_final_assistant_content(
            "> Before the code.\n>\n> ```python\n> def main():\n>     print('hello')\n> ```\n>\n> After."
        )
    )

    assert calls
    assert "Before the code" in calls[0]
    assert "def" in calls[0]
    assert "print" in calls[0]
    assert "After" in calls[0]
    assert "48;2;30;41;59" in calls[0]


def test_slate_blockquotes_use_bright_markdown_color(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(_render_final_assistant_content("> This is a blockquote.\n> Nested enough."))

    assert calls
    assert "This is a blockquote" in calls[0]
    assert "38;2;167;182;216" in calls[0]


def test_slate_preserves_properly_escaped_markdown_literals(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(
        _render_final_assistant_content(
            "Literal asterisk: \\*not italic\\*\n"
            "Literal bold: \\*\\*not bold\\*\\*\n"
            "Literal backtick: \\`not code\\`\n"
            "Literal hash: \\# not heading\n"
            "Literal bracket: \\[not a link\\](https://example.com)\n"
            "Literal pipe: \\| not table"
        )
    )

    assert calls
    assert "*not italic*" in calls[0]
    assert "**not bold**" in calls[0]
    assert "`not" in calls[0]
    assert "code`" in calls[0]
    assert "# not heading" in calls[0]
    assert "[not a" in calls[0]
    assert "link](https://example.com)" in calls[0]
    assert "| not table" in calls[0]


def test_chat_console_strips_rich_terminal_hyperlink_sequences(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(
        _render_final_assistant_content(
            "- [External link to GitHub](https://github.com)\n- <noizo@example.com>"
        )
    )

    assert calls
    assert "External link to GitHub" in calls[0]
    assert "noizo@example.com" in calls[0]
    assert "\x1b]8;" not in calls[0]
    assert "id=" not in calls[0]


def test_slate_renders_markdown_tables_with_visible_columns():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "| Left | Center | Right |\n"
            "|:---|:---:|---:|\n"
            "| one | two | three |\n"
            "| alpha | beta | gamma |"
        )
    )

    assert "|Left" in output
    assert "| Center |" in output
    assert "|one" in output
    assert "three|" in output


def test_stream_render_mode_buffers_until_flush(monkeypatch):
    from cli import HermesCLI

    rendered: list[Panel] = []
    raw_lines: list[str] = []

    class FakeChatConsole:
        def print(self, renderable, *args, **kwargs):
            rendered.append(renderable)

    hermes_cli = HermesCLI.__new__(HermesCLI)
    hermes_cli.show_reasoning = False
    hermes_cli.final_response_markdown = "render"
    hermes_cli._stream_buf = ""
    hermes_cli._stream_started = False
    hermes_cli._stream_box_opened = False
    hermes_cli._stream_text_ansi = ""
    hermes_cli._stream_prefilt = ""
    hermes_cli._in_reasoning_block = False
    hermes_cli._stream_last_was_newline = True
    hermes_cli._reasoning_box_opened = False
    hermes_cli._reasoning_buf = ""
    hermes_cli._reasoning_preview_buf = ""
    hermes_cli._deferred_content = ""

    monkeypatch.setattr(cli, "_cprint", raw_lines.append)
    monkeypatch.setattr(cli, "ChatConsole", FakeChatConsole)

    hermes_cli._emit_stream_text("# Heading\n\n```python\nprint('ok')\n```")

    assert raw_lines == []
    assert rendered == []
    assert hermes_cli._stream_box_opened is True

    hermes_cli._flush_stream()

    assert raw_lines == []
    assert len(rendered) == 1
    assert hermes_cli._stream_buf == ""


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
