"""Streamed response lines must be flush-left (no leading indent).

The 4-space ``_STREAM_PAD`` indent made every copied line carry leading
whitespace, so pasting a response out of the terminal produced broken
text. The pad is now empty and the final-response Rich Panel uses zero
horizontal padding for the same reason (July 2026).
"""
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
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli._reset_stream_state()

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def test_stream_pad_is_empty():
    import cli as climod

    assert climod._STREAM_PAD == ""


def test_streamed_content_lines_have_no_leading_whitespace(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("First streamed line of text.\nSecond streamed line.\n")
    cli._flush_stream()
    content = [
        _strip_ansi(e)
        for e in emitted
        if "streamed line" in _strip_ansi(e)
    ]
    assert content, "no content lines captured"
    for line in content:
        assert not line.startswith(" "), f"leading whitespace leaked: {line!r}"


def test_intentional_markdown_indentation_is_preserved(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("- item\n  - nested item\n")
    cli._flush_stream()
    plain = [_strip_ansi(e) for e in emitted]
    assert any(line == "- item" for line in plain)
    assert any(line == "  - nested item" for line in plain)
