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


def test_answer_streams_before_turn_end_after_reasoning(cli_stub):
    """#47116: with show_reasoning on, the reasoning box closes on the first content token so the
    answer streams mid-turn instead of being held until end of turn."""
    cli, emitted = cli_stub
    cli.show_reasoning = True
    cli._stream_reasoning_delta("thinking about it\n")
    cli._stream_delta("First answer line.\n")
    # No _flush_stream(): the line must already be on screen mid-turn.
    lines = [_plain(e) for e in emitted]
    answer = [i for i, l in enumerate(lines) if "First answer line." in l]
    assert answer, lines
    reasoning = next(i for i, l in enumerate(lines) if "thinking about it" in l)
    assert reasoning < answer[0]


def test_late_reasoning_does_not_reopen_box_inside_answer(cli_stub):
    cli, emitted = cli_stub
    cli.show_reasoning = True
    cli._stream_reasoning_delta("early thought\n")
    cli._stream_delta("Answer.\n")
    cli._stream_reasoning_delta("late thought\n")
    cli._flush_stream()
    assert not any("late thought" in _plain(e) for e in emitted), emitted


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


def test_adviser_row_renders_a_labelled_block_after_the_box_footer(cli_stub):
    """A plugin-injected adviser row is model-facing guidance, not a human prompt: the CLI
    labels it and parks it behind an open reply box instead of showing a bare user turn."""
    cli, emitted = cli_stub
    row = {
        "content": "[adviser — advisory only]\n- [blocker · reviewer] Stop: the migration drops the index.\n[end adviser]",
        "display_kind": "adviser",
        "plugin_message_id": "abc123",
        "role": "user",
    }

    cli._stream_delta("First paragraph.\n")
    cli._on_adviser_row(row)
    cli._flush_stream()

    lines = [_plain(e) for e in emitted]
    header = next(i for i, l in enumerate(lines) if "adviser · blocker · reviewer" in l)
    body = next(i for i, l in enumerate(lines) if "drops the index" in l)
    footer = next(i for i, l in enumerate(lines) if l.startswith("╰"))
    assert footer < header < body, lines
    assert not any("[adviser — advisory only]" in l for l in lines)
    assert not any(l.startswith("  You:") for l in lines)


def test_adviser_row_without_advice_text_is_ignored(cli_stub):
    cli, emitted = cli_stub
    cli._on_adviser_row({"content": "[adviser — advisory only]\n[end adviser]", "display_kind": "adviser", "role": "user"})
    assert emitted == []


def test_plain_steer_row_is_not_drawn_as_adviser_guidance(cli_stub):
    """The same callback fires for a mid-turn steer; that row is the operator's words."""
    cli, emitted = cli_stub
    cli._stream_delta("Streaming.\n")
    cli._on_adviser_row({"content": "[steer] Ship the smaller patch.", "display_kind": "steer", "role": "user"})
    cli._flush_stream()
    assert not any("adviser" in _plain(e) for e in emitted)


@pytest.mark.parametrize("response, expect_panel", [
    ("Partial answer.", False),  # the streamed text itself: already on screen
    ("Operation interrupted: waiting for model response (3.0s elapsed).", True),  # never streamed
])
def test_interrupted_reply_panel_after_tool_call_boundary(cli_stub, monkeypatch, response, expect_panel):
    """#65666: an interrupted reply streamed before a tool-call boundary reset per-segment stream
    state must not be re-rendered as a Panel, but an unstreamed interrupt status message still is."""
    from types import SimpleNamespace
    import cli as climod

    cli, _ = cli_stub
    printed = []
    monkeypatch.setattr(climod, "ChatConsole", lambda: SimpleNamespace(print=printed.append))
    cli._streamed_text_this_turn = ""
    cli._stream_delta("Partial answer.\n")
    cli._stream_delta(None)  # tool-call boundary: flush + per-segment reset
    cli._last_turn_interrupted = True
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"interrupted": True, "final_response": response})
    cli._chat_print_response_panel(turn, response)
    assert bool(printed) is expect_panel, printed


def test_unstreamed_final_reply_after_streamed_segment_still_prints_panel(cli_stub, monkeypatch):
    """#65666 scope: the turn-level streamed record only suppresses the Panel for interrupted results.
    A turn that streamed text A, crossed a tool boundary, then returned an unstreamed final B must
    still render B."""
    from types import SimpleNamespace
    import cli as climod

    cli, _ = cli_stub
    printed = []
    monkeypatch.setattr(climod, "ChatConsole", lambda: SimpleNamespace(print=printed.append))
    cli._streamed_text_this_turn = ""
    cli._stream_delta("Looking that up.\n")
    cli._stream_delta(None)  # tool-call boundary
    cli._last_turn_interrupted = False
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": "Final B"})
    cli._chat_print_response_panel(turn, "Final B")
    assert len(printed) == 1
