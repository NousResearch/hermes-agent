import re
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


# Regression for #84377 / #73212: strip mode (the default) must not rewrite code the model wrote.
_CODE_LINES = [
    "# Entry point",
    "class Box:",
    "    def __init__(self, *args, **kwargs):",
    "        self._items_ = list(args)",
    'if __name__ == "__main__":',
    "    print(2**8, 3*4*5)",
]
_REPLY_WITH_CODE = (
    "Here is **the** class:\n```python\n" + "\n".join(_CODE_LINES) + "\n```\n"
    "Call `Box.__init__` directly **only** in tests.\n"
)


def test_strip_mode_keeps_code_verbatim_and_still_strips_prose():
    output = _render_to_text(_render_final_assistant_content(_REPLY_WITH_CODE, mode="strip"))

    lines = output.splitlines()
    for code_line in _CODE_LINES:
        assert code_line in lines
    assert "Here is the class:" in output
    assert "Call Box.__init__ directly only in tests." in output


def test_streamed_strip_output_matches_final_strip_render(monkeypatch):
    import cli as climod
    from cli import HermesCLI, _strip_markdown_syntax

    emitted = []
    monkeypatch.setattr(climod, "_cprint", emitted.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "strip"
    cli.show_timestamps = False
    cli._reset_stream_state()
    cli._spinner_text = ""
    cli._invalidate = lambda *a, **kw: None
    cli._scrollback_box_width = lambda: 80

    for i in range(0, len(_REPLY_WITH_CODE), 5):
        cli._stream_delta(_REPLY_WITH_CODE[i:i + 5])
    cli._flush_stream()

    streamed = [re.sub(r"\x1b\[[0-9;]*m", "", s) for s in emitted][1:-1]  # drop box header/footer
    assert streamed == _strip_markdown_syntax(_REPLY_WITH_CODE).split("\n")
    for code_line in _CODE_LINES:
        assert code_line in streamed
