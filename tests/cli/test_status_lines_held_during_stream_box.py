"""Agent status lines (subagent ``✓ [set n · i/N]`` completions, background-process notices) arrive on other
threads mid-stream. The CLI's ``agent._print_fn`` must park them while a response box is open and release them
at the footer, never between two paragraphs of the reply."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _plain(s: str) -> str:
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
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 74)
    return cli, emitted


def test_status_line_waits_for_box_footer(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("First paragraph.\n")
    cli._agent_status_print("  ✓ [set 7 · 2/2] worker  (2175.14s)")
    cli._stream_delta("Second paragraph.\n")
    cli._flush_stream()
    lines = [_plain(e) for e in emitted]
    notice = next(i for i, l in enumerate(lines) if "set 7" in l)
    footer = next(i for i, l in enumerate(lines) if l.startswith("╰"))
    second = next(i for i, l in enumerate(lines) if "Second paragraph" in l)
    assert second < footer < notice, lines


def test_status_line_prints_immediately_outside_a_box(cli_stub):
    cli, emitted = cli_stub
    cli._agent_status_print("  ✓ [set 1 · 1/1] worker  (3.0s)")
    assert [_plain(e) for e in emitted] == ["  ✓ [set 1 · 1/1] worker  (3.0s)"]
    assert not getattr(cli, "_held_status_lines", [])
