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


def test_strip_mode_preserves_dunder_identifiers_in_fenced_code():
    # Regression: #84377 — dunder identifiers and ** operators inside
    # fenced code blocks must render verbatim, not be eaten as emphasis.
    renderable = _render_final_assistant_content(
        "```python\n"
        'if __name__ == "__main__":\n'
        "    total = a**2 + b**2\n"
        "```",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert 'if __name__ == "__main__":' in output
    assert "total = a**2 + b**2" in output


def test_strip_mode_preserves_dunders_in_unterminated_fence():
    # A fence without a closing marker still marks the intent as code.
    renderable = _render_final_assistant_content(
        "```\nvalue = __all__[0]\n",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "value = __all__[0]" in output


def test_strip_mode_preserves_emphasis_in_inline_code():
    # Regression: #84377 — inline code spans keep ** and __ verbatim while
    # prose emphasis around them is still stripped.
    renderable = _render_final_assistant_content(
        "Run `a**2` and guard with `if __name__ == '__main__':` now",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "a**2" in output
    assert "__name__" in output


def test_strip_mode_still_strips_prose_emphasis_outside_code():
    renderable = _render_final_assistant_content(
        "**bold** prose and `**not bold**` code",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "bold prose and" in output
    assert "**not bold**" in output


def test_strip_mode_preserves_code_in_blockquote_fenced_block():
    # Third path beyond #84379/#84502: a fenced block nested inside a
    # blockquote (models quote docs this way) must keep its code verbatim.
    renderable = _render_final_assistant_content(
        "> ```python\n"
        '> if __name__ == "__main__":\n'
        ">     value = a**2 + b**2\n"
        "> ```\n",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert 'if __name__ == "__main__":' in output
    assert "value = a**2 + b**2" in output


def test_strip_mode_streaming_fence_keeps_asterisks_in_code(monkeypatch):
    # Regression: the streaming path stripped each line individually, so
    # `*` / `**` inside fenced code blocks were eaten (e.g. `*.{ts,tsx}`
    # became `.{ts,tsx}`). Fenced lines must be buffered and stripped as
    # one block on close.
    import cli as cli_mod
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.show_timestamps = False
    cli.final_response_markdown = "strip"
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._stream_table_buf = []
    cli._in_stream_table = False
    cli._stream_fence_buf = []
    cli._stream_in_fence = False
    cli._stream_fence_char = ""
    cli._stream_fence_len = 0

    emitted = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(cli, "_scrollback_box_width", lambda: 80)
    monkeypatch.setattr(HermesCLI, "_status_bar_display_width", staticmethod(lambda s: 10))

    fence = "```bash\nrg --pcre2 -g '*.{ts,tsx}' '^import(?:(?!.*@mui\\/material).)*IconButton.*$'\n```"
    # Feed in 3-char chunks to simulate streaming.
    for i in range(0, len(fence), 3):
        cli._emit_stream_text(fence[i : i + 3])
    cli._flush_stream()  # stream end: flush any buffered partial line

    joined = "".join(emitted)
    assert "*.{ts,tsx}" in joined
    assert ".*" in joined
    assert "IconButton" in joined
    assert "```" not in joined


def test_strip_mode_streaming_unterminated_fence_keeps_code(monkeypatch):
    # If the stream ends inside a fenced block, the buffered body is still
    # stripped as one unit instead of line-by-line.
    import cli as cli_mod
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.show_timestamps = False
    cli.final_response_markdown = "strip"
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._stream_table_buf = []
    cli._in_stream_table = False
    cli._stream_fence_buf = []
    cli._stream_in_fence = False
    cli._stream_fence_char = ""
    cli._stream_fence_len = 0

    emitted = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(cli, "_scrollback_box_width", lambda: 80)
    monkeypatch.setattr(HermesCLI, "_status_bar_display_width", staticmethod(lambda s: 10))

    text = "```\nvalue = __all__[0]\nmore = a**2\n"
    for i in range(0, len(text), 3):
        cli._emit_stream_text(text[i : i + 3])
    cli._flush_stream()

    joined = "".join(emitted)
    assert "value = __all__[0]" in joined
    assert "more = a**2" in joined
    assert "```" not in joined
