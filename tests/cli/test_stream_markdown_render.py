"""``display.final_response_markdown: render`` must apply while streaming.

The streaming emitter only ever branched on ``strip``, so ``render`` fell
through to the raw path and printed markdown markers verbatim (upstream
#83233). Rich's block renderer cannot run line-by-line, so streamed lines
get a line-wise ANSI stylizer that mirrors the ``markdown.*`` palette of
``_skin_markdown_theme``.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


@pytest.fixture
def cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "render"
    cli.show_timestamps = False
    cli._reset_stream_state()

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def _run(cli, emitted, text):
    cli._stream_delta(text)
    cli._flush_stream()
    return emitted


def _content(emitted, needle):
    for e in emitted:
        if needle in _strip_ansi(e):
            return e
    raise AssertionError(f"no emitted line containing {needle!r}: {emitted!r}")


def test_heading_markers_are_removed_and_styled(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "## Section title\n")
    line = _content(emitted, "Section title")
    assert "#" not in _strip_ansi(line)
    assert "\x1b[1m" in line, "heading should be bold"


def test_bold_markers_are_removed_and_styled(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "some **strong** words\n")
    line = _content(emitted, "strong")
    assert _strip_ansi(line).strip() == "some strong words"
    assert "\x1b[1m" in line and "\x1b[22m" in line


def test_italic_and_strikethrough_markers_are_removed(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "an *emphasis* and a ~~removal~~\n")
    line = _content(emitted, "emphasis")
    assert _strip_ansi(line).strip() == "an emphasis and a removal"
    assert "\x1b[3m" in line and "\x1b[9m" in line


def test_inline_code_markers_are_removed_and_colored(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "call `do_thing()` now\n")
    line = _content(emitted, "do_thing()")
    assert "`" not in _strip_ansi(line)
    assert "\x1b[38;2;" in line, "inline code should carry a truecolor fg"


def test_list_bullet_is_normalized(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "- first\n  - nested\n")
    assert _strip_ansi(_content(emitted, "first")).rstrip() == "• first"
    assert _strip_ansi(_content(emitted, "nested")).rstrip() == "  • nested"


def test_fenced_code_block_content_is_not_inline_processed(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "```python\nx = a ** b  # `note`\n```\nafter\n")
    line = _content(emitted, "x = a")
    assert _strip_ansi(line).rstrip() == "x = a ** b  # `note`"
    # The fence must close: prose after it is styled again.
    after = _content(emitted, "after")
    assert _strip_ansi(after).rstrip() == "after"


def test_link_text_replaces_the_markdown_target(cli_stub):
    cli, emitted = cli_stub
    _run(cli, emitted, "see [the docs](https://example.com) here\n")
    line = _content(emitted, "the docs")
    assert _strip_ansi(line).strip() == "see the docs here"
    assert "\x1b[4m" in line, "link text should be underlined"


def test_strip_mode_is_unchanged(cli_stub):
    cli, emitted = cli_stub
    cli.final_response_markdown = "strip"
    _run(cli, emitted, "## Title\nsome **strong** words\n")
    assert _strip_ansi(_content(emitted, "Title")).strip() == "Title"
    assert _strip_ansi(_content(emitted, "strong")).strip() == "some strong words"


def test_raw_mode_is_unchanged(cli_stub):
    cli, emitted = cli_stub
    cli.final_response_markdown = "raw"
    _run(cli, emitted, "## Title\nsome **strong** words\n")
    assert _strip_ansi(_content(emitted, "Title")).strip() == "## Title"
    assert _strip_ansi(_content(emitted, "strong")).strip() == "some **strong** words"


def test_trailing_partial_line_is_styled_on_flush(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("tail **bolded**")
    cli._flush_stream()
    line = _content(emitted, "bolded")
    assert _strip_ansi(line).strip() == "tail bolded"
