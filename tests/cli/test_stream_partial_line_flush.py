"""Streaming display: logical lines are emitted ONLY at real newlines.

The July 2026 TTFT force-flush hard-wrapped long partial lines at
terminal width, baking real '\\n's into every long paragraph — exactly
what polluted highlight-copy/paste. Now paragraphs stay one logical
line (the terminal soft-wraps them and rejoins on copy, matching the
TUI's selection copy), and TTFT perception is served by mirroring the
partial line's tail into the spinner status text instead.
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
    cli._spinner_text = ""
    cli._invalidate = lambda *a, **kw: None

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


class TestLogicalLineStreaming:
    def test_long_paragraph_not_hard_wrapped_before_newline(self, cli_stub):
        cli, emitted = cli_stub
        text = (
            "This is a long opening paragraph that previously got chopped "
            "into terminal-width chunks with real newlines, which is what "
            "made copy/paste come out full of broken lines. "
        ) * 3
        for i in range(0, len(text), 12):
            cli._stream_delta(text[i : i + 12])
        # No newline seen yet → no content lines printed (box header only).
        plain = _strip_ansi("\n".join(emitted))
        assert "opening paragraph" not in plain
        # The paragraph is still buffered as ONE logical line.
        assert cli._stream_buf.startswith("This is a long opening")

    def test_partial_tail_mirrored_into_spinner(self, cli_stub):
        cli, emitted = cli_stub
        text = "A long paragraph streaming in without any newline " * 4
        for i in range(0, len(text), 16):
            cli._stream_delta(text[i : i + 16])
        assert cli._spinner_text.startswith("…")
        assert "newline" in cli._spinner_text


    def test_no_content_lost_across_stream(self, cli_stub):
        cli, emitted = cli_stub
        words = [f"word{i}" for i in range(120)]
        text = " ".join(words)
        for i in range(0, len(text), 7):
            cli._stream_delta(text[i : i + 7])
        cli._flush_stream()
        plain = " ".join(_strip_ansi("\n".join(emitted)).split())
        for w in words:
            assert w in plain, f"lost {w}"


    def test_table_rows_not_previewed_in_spinner(self, cli_stub):
        cli, emitted = cli_stub
        row = "| " + " | ".join(f"cell{i}" for i in range(20)) + " |"
        cli._stream_delta(row)  # no newline
        plain = _strip_ansi("\n".join(emitted))
        assert "cell19" not in plain
        assert cli._spinner_text == ""

    def test_newline_lines_still_emit_normally(self, cli_stub):
        cli, emitted = cli_stub
        cli._stream_delta("line one\nline two\n")
        plain = _strip_ansi("\n".join(emitted))
        assert "line one" in plain
        assert "line two" in plain

    def test_unbreakable_run_stays_single_line(self, cli_stub):
        cli, emitted = cli_stub
        blob = "x" * 300  # no spaces
        cli._stream_delta(blob)
        cli._flush_stream()
        plain = _strip_ansi("\n".join(emitted))
        assert plain.count("x") == 300
        content = [
            _strip_ansi(e) for e in emitted if "x" in _strip_ansi(e)
        ]
        assert len(content) == 1, "unbreakable run was hard-wrapped"
