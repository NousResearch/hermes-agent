"""Streaming with final_response_markdown: strip leaves fenced code blocks as they are."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


@pytest.fixture
def cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "strip"
    cli.show_timestamps = False
    cli._reset_stream_state()
    cli._spinner_text = ""
    cli._invalidate = lambda *a, **kw: None

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def _lines(emitted):
    return [_strip_ansi(e) for e in emitted]


def test_streamed_code_block_is_not_stripped(cli_stub):
    cli, emitted = cli_stub
    for chunk in ["**Code**:\n```py", "thon\n# comment\ndef __init__(self):\n", "    pass\n```\n## after"]:
        cli._emit_stream_text(chunk)
    cli._flush_stream()

    text = "\n".join(_lines(emitted))
    assert "Code:" in text
    assert "# comment" in text
    assert "def __init__(self):" in text
    assert "```" not in text
    assert "## after" not in text and "after" in text


def test_fence_state_resets_between_turns(cli_stub):
    cli, emitted = cli_stub
    cli._emit_stream_text("```\nunclosed __code__")
    cli._flush_stream()
    cli._reset_stream_state()
    emitted.clear()
    cli._emit_stream_text("**bold** text\n")
    cli._flush_stream()

    assert "bold text" in "\n".join(_lines(emitted))


@pytest.mark.parametrize("mode", ["strip", "render"])
def test_streamed_table_shaped_code_is_byte_exact(cli_stub, mode):
    cli, emitted = cli_stub
    cli.final_response_markdown = mode
    for chunk in ["~~~text\n| key | __init__ |\n", "|---|---|\n| a | b |\n~~~\n", "done\n"]:
        cli._emit_stream_text(chunk)
    cli._flush_stream()

    lines = [ln.strip() for ln in _lines(emitted)]
    assert "| key | __init__ |" in lines
    assert "|---|---|" in lines
    assert "| a | b |" in lines
