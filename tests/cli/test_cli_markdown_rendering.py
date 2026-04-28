from io import StringIO
from pathlib import Path

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


def _plain_rendered(text: str) -> str:
    rendered = _render_to_text(_render_final_assistant_content(text))
    return "\n".join(line.rstrip() for line in rendered.splitlines())


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
    assert "1 │ def smoke" in output
    assert "2 │     return True" in output
    assert "╭" in output


def test_slate_generic_code_blocks_do_not_show_fake_line_numbers():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "```code\n"
            "=== diff output ===\n"
            "4c4\n"
            "< old line\n"
            "---\n"
            "> new line\n"
            "```"
        )
    )

    assert "=== diff output ===" in output
    assert "4c4" in output
    assert "1 === diff output ===" not in output
    assert "2 4c4" not in output


def test_slate_diff_code_blocks_show_real_diff_line_numbers():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "```diff\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -73,2 +76,2 @@\n"
            "-old line\n"
            "+new line\n"
            "```"
        )
    )

    assert "diff" in output
    assert "--- a/README.md" in output
    assert "+++ b/README.md" in output
    assert "73 │ -old line" in output
    assert "76 │ +new line" in output
    assert "1 --- a/README.md" not in output


def test_slate_diff_code_blocks_keep_code_colors_under_diff_background():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    renderable = _render_final_assistant_content(
        "```diff\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-def old():\n"
        "+def new():\n"
        "```"
    )
    buf = StringIO()
    Console(file=buf, width=100, force_terminal=True, color_system="truecolor", no_color=False).print(renderable)
    output = buf.getvalue()

    assert "38;2;102;217;239;48;2;138;58;69mdef" in output
    assert "38;2;102;217;239;48;2;31;107;58mdef" in output
    assert "38;2;248;113;113;48;2;138;58;69m-" in output
    assert "38;2;74;222;128;48;2;31;107;58m+" in output


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


def test_slate_chat_console_renders_code_without_block_background(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(_render_final_assistant_content("```python\nprint('ok')\n```"))

    assert calls
    assert "print" in calls[0]
    assert "48;2;30;41;59" not in calls[0]


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
    assert "48;2;30;41;59" not in calls[0]


def test_slate_renders_blockquote_fenced_code_without_block_background(monkeypatch):
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
    assert "48;2;30;41;59" not in calls[0]


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
    assert "Center" in output
    assert "|one" in output
    assert "three" in output


def test_slate_renders_inline_markdown_inside_table_cells():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _render_to_text(
        _render_final_assistant_content(
            "| Kind | Example | Link |\n"
            "|:---|:---|---:|\n"
            "| strong | **bold** and *italic* | [Docs](https://example.com) |\n"
            "| code | `ctx_read` keeps `a | b` together | escaped \\| pipe |"
        )
    )

    assert "|Kind" in output
    assert "bold and italic" in output
    assert "**bold**" not in output
    assert "*italic*" not in output
    assert "ctx_read keeps a | b" in output
    assert "together" in output
    assert "Docs" in output
    assert "escaped | pipe" in output


def test_slate_styles_inline_code_inside_table_cells_in_terminal(monkeypatch):
    from hermes_cli.skin_engine import set_active_skin

    calls: list[str] = []
    set_active_skin("slate")
    monkeypatch.setattr(cli, "_cprint", calls.append)

    ChatConsole().print(
        _render_final_assistant_content(
            "| Scenario | Input | Expected | Status |\n"
            "|---|---|---|---|\n"
            "| SQL Injection | `' OR 1=1 --` | Rejected | ✅ |\n"
            "| XSS Attempt | `<script>alert(1)</script>` | Escaped | ✅ |\n"
            "| Long String | `A`*256 chars | Truncated | ⚠️ |\n"
            "| Unicode | 😀 Текст 日本語 | Preserved | ✅ |\n"
            "| Empty Input | `\"\"` | 400 Error | ✅ |\n"
            "| Null Input | `null` | 400 Error | ✅ |\n"
            "| Nested JSON | `{\"a\":{\"b\":{\"c\":1}}}` | Parsed OK | ✅ |"
        )
    )

    assert calls
    rendered = calls[0]
    assert "48;2;30;41;59" in rendered
    assert "' OR 1=1 --" in rendered
    assert "<script>alert(1)</script>" in rendered
    assert '{"a":{"b":{"c":1}}}' in rendered
    assert "`' OR 1=1 --`" not in rendered
    assert "<script>alert(1)…" not in rendered


def test_slate_expands_smoke_emoji_shortcodes_outside_code():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered(
        "Prose :rocket: :white_check_mark:\n\n"
        "`:rocket:` should stay literal\n\n"
        "```code\n:warning:\n```"
    )

    assert "Prose 🚀 ✅" in output
    assert ":rocket: should stay literal" in output
    assert ":warning:" in output


def test_slate_expands_emoji_shortcodes_inside_table_cells():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered(
        "| Scenario | Status |\n"
        "|---|---|\n"
        "| SQL Injection | :white_check_mark: |\n"
        "| Long String | :warning: |"
    )

    assert "✅" in output
    assert "⚠️" in output
    assert ":white_check_mark:" not in output
    assert ":warning:" not in output


def test_slate_task_lists_render_without_extra_bullet_dot():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered("- [x] Completed task\n- [ ] Pending task")

    assert "☑ Completed task" in output
    assert "☐ Pending task" in output
    assert "• ☑" not in output
    assert "[x]" not in output
    assert "[ ]" not in output


def test_slate_nested_ordered_lists_use_distinct_markers():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered(
        "1. First\n"
        "   1. Nested one\n"
        "   2. Nested two\n"
        "      1. Deep one"
    )

    assert "1 First" in output
    assert "a. Nested one" in output
    assert "b. Nested two" in output
    assert "i. Deep one" in output


def test_slate_mixed_unordered_markers_stay_visually_distinct():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered("- Dash item\n* Asterisk item\n+ Plus item")

    assert "Dash item" in output
    assert "◦ Asterisk item" in output
    assert "▪ Plus item" in output


def test_slate_blockquote_table_uses_compact_table_renderer():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered(
        "> | Key | Value |\n"
        "> |---|---|\n"
        "> | one | :warning: |"
    )

    assert "|Key" in output
    assert "⚠️" in output
    assert ":warning:" not in output


def test_slate_blockquote_fenced_code_keeps_quote_gutter():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered("> ```python\n> print('quote')\n> ```")

    assert "│" in output
    assert "python" in output
    assert "print" in output


def test_slate_h1_is_left_aligned_in_compact_rendering():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered("# Heading level 1")

    assert output.splitlines()[0].startswith("Heading level 1")


def test_slate_preserves_multi_paragraph_spacing():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered("This is paragraph one.\n\nThis is paragraph two.")

    assert "This is paragraph one.\n\nThis is paragraph two." in output


def test_slate_degrades_html_and_images_to_terminal_text():
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("slate")
    output = _plain_rendered(
        "<kbd>Ctrl</kbd> + <kbd>C</kbd>\n\n"
        "<details>\n<summary>Open details</summary>\n:rocket: inside\n</details>\n\n"
        "![Alt text](/tmp/example.png)"
    )

    assert "Ctrl + C" in output
    assert "Summary: Open details" in output
    assert "🚀 inside" in output
    assert "🖼 Alt text — /tmp/example.png" in output
    assert "<details>" not in output


def test_markdown_smoke_ascii_diagram_rows_have_equal_width():
    smoke = Path("/Users/noizo/.hermes/skills/markdown-smoke-test/references/smoke-test.md").read_text()
    start = smoke.index("┌")
    end = smoke.index("```", start)
    diagram_lines = smoke[start:end].splitlines()

    assert {len(line) for line in diagram_lines} == {51}


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


def test_stream_render_mode_renders_markdown_after_code_block_on_flush(monkeypatch):
    from cli import HermesCLI

    calls: list[str] = []

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

    monkeypatch.setattr(cli, "_cprint", calls.append)

    hermes_cli._stream_delta(
        "```python\n"
        "trainer = SFTTrainer(\n"
        "    model=model,\n"
        ")\n"
        "trainer.train()\n"
        "```\n\n"
        "- Streaming / render now works without disabling streaming.\n\n"
        "## Investigation workflow\n\n"
        "1. Confirm config: `grep -A3 display ~/.hermes/config.yaml`\n"
        "2. Check fenced code after the heading:\n\n"
        "```python\n"
        "trainer = SFTTrainer(model=model)\n"
        "```\n\n"
        "**Chat template applied automatically.**"
    )
    hermes_cli._flush_stream()

    assert calls
    output = cli._rich_text_from_ansi("\n".join(calls)).plain
    assert "trainer" in output
    assert "SFTTrainer" in output
    assert "Investigation workflow" in output
    assert "Chat template applied automatically" in output
    assert "## Investigation workflow" not in output
    assert "```python" not in output
    assert "- Streaming / render" not in output


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
