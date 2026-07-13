"""Streaming display force-flush: long partial lines must paint before the
first newline arrives (TTFT-perception fix, July 2026).

Previously ``_emit_stream_text`` only emitted on ``"\\n"``, so a response
opening with a long paragraph stayed invisible until the model produced a
newline — seconds of blank box on slow models. Now partial lines are
force-flushed at terminal width (mirroring the reasoning box's 80-char rule).
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
    cli.assistant_body_streaming = True
    cli._reset_stream_state()

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    # Deterministic width regardless of the test runner's terminal
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


class TestPartialLineForceFlush:
    def test_long_paragraph_paints_before_first_newline(self, cli_stub):
        cli, emitted = cli_stub
        text = (
            "This is a long opening paragraph that would previously sit "
            "invisible in the buffer until the model finally produced a "
            "newline character, which on a slow model could take seconds. "
        ) * 3
        for i in range(0, len(text), 12):
            cli._stream_delta(text[i : i + 12])
        # Box header + several wrapped lines painted with NO newline seen yet
        assert len(emitted) > 3

    def test_no_content_lost_across_wraps(self, cli_stub):
        cli, emitted = cli_stub
        words = [f"word{i}" for i in range(120)]
        text = " ".join(words)
        for i in range(0, len(text), 7):
            cli._stream_delta(text[i : i + 7])
        cli._flush_stream()
        plain = " ".join(_strip_ansi("\n".join(emitted)).split())
        for w in words:
            assert w in plain, f"lost {w} at a wrap boundary"

    def test_short_partial_stays_buffered(self, cli_stub):
        cli, emitted = cli_stub
        cli._stream_delta("short line, no newline")
        # Under wrap width: the box header may open, but the text itself
        # stays buffered until a newline or the width threshold.
        plain = _strip_ansi("\n".join(emitted))
        assert "short line" not in plain
        assert cli._stream_buf == "short line, no newline"

    def test_table_rows_not_force_flushed(self, cli_stub):
        cli, emitted = cli_stub
        # A long partial table row must stay buffered for block realignment
        row = "| " + " | ".join(f"cell{i}" for i in range(20)) + " |"
        cli._stream_delta(row)  # no newline
        plain = _strip_ansi("\n".join(emitted))
        assert "cell19" not in plain

    def test_newline_lines_still_emit_normally(self, cli_stub):
        cli, emitted = cli_stub
        cli._stream_delta("line one\nline two\n")
        plain = _strip_ansi("\n".join(emitted))
        assert "line one" in plain
        assert "line two" in plain

    def test_unbreakable_run_hard_wraps(self, cli_stub):
        cli, emitted = cli_stub
        blob = "x" * 300  # no spaces
        cli._stream_delta(blob)
        cli._flush_stream()
        plain = _strip_ansi("\n".join(emitted))
        assert plain.count("x") == 300
