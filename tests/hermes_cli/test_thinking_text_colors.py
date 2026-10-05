"""Showcase colouring for the CLI's thinking surfaces.

The buffered ``[thinking]`` preview, the live ``show_reasoning`` box and that box's
closing tail all render through :mod:`hermes_cli.thinking_colors`, so a token is
painted the same way no matter which of the three the reader is looking at.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.thinking_colors import (  # noqa: E402
    SHOWCASE_GREEN,
    SHOWCASE_ORANGE,
    SHOWCASE_WHITE,
    render_thinking_text,
    split_safe_stream,
    thinking_spans,
)

GREEN = "\x1b[38;2;40;254;20m"
ORANGE = "\x1b[38;2;255;159;10m"
WHITE = "\x1b[38;2;255;255;255m"
RST = "\x1b[0m"


def _colors(text: str) -> list[tuple[str, str]]:
    """(chunk, SGR) pairs from rendered output, droppings removed."""
    return [
        (chunk, sgr)
        for sgr, chunk in re.findall(r"((?:\x1b\[[0-9;]*m)+)([^\x1b]+)", text)
    ]


def test_palette_matches_the_requested_truecolor():
    assert (SHOWCASE_GREEN, SHOWCASE_ORANGE, SHOWCASE_WHITE) == ("#28FE14", "#FF9F0A", "#FFFFFF")


def test_plain_prose_is_green():
    assert _colors(render_thinking_text("just some prose")) == [("just some prose", GREEN)]


def test_order_mark_is_orange_and_the_lookalikes_stay_green():
    for text, expected in (
        ("1. read the file", ("1.", ORANGE)),
        ("2. ship it", ("2.", ORANGE)),
        ("3.14 is pi", ("3.14 is pi", GREEN)),
        ("v1.0.1 ships", ("v1.0.1 ships", GREEN)),
        ("it took 2.5 hours", ("it took 2.5 hours", GREEN)),
        ("step 1. then 2. done", None),
    ):
        spans = _colors(render_thinking_text(text))
        if expected is None:
            assert spans == [("step ", GREEN), ("1.", ORANGE), (" then ", GREEN), ("2.", ORANGE), (" done", GREEN)]
        else:
            assert expected in spans, (text, spans)


def test_pr_number_needs_six_digits():
    assert ("#12345", GREEN) in _colors(render_thinking_text("see #12345"))
    assert ("#123456", WHITE) in _colors(render_thinking_text("see #123456"))


def test_log_mark_is_one_white_span_including_the_pr_inside():
    assert _colors(render_thinking_text("§[2026-10-02] done")) == [
        ("§[2026-10-02]", WHITE), (" done", GREEN)]
    assert _colors(render_thinking_text("§ [#123456] done")) == [
        ("§ [#123456]", WHITE), (" done", GREEN)]


def test_url_is_one_white_span_through_its_fragment():
    assert ("https://x.com/a#frag", WHITE) in _colors(render_thinking_text("at https://x.com/a#frag ok"))


def test_sentence_punctuation_hugging_a_url_stays_green():
    assert _colors(render_thinking_text("(https://x.com/a).")) == [
        ("(", GREEN), ("https://x.com/a", WHITE), (").", GREEN)]


def test_adjacent_same_colour_runs_merge_into_one_span():
    assert _colors(render_thinking_text("§[2026-10-02] §[2026-10-03]")) == [
        ("§[2026-10-02] §[2026-10-03]", WHITE)]


@pytest.mark.parametrize("partial", ["§[2026-10-0", "#12345", "1.", "https://x.co", "htt"])
def test_a_token_that_has_not_finished_arrives_plain(partial):
    assert _colors(render_thinking_text(partial)) == [(partial, GREEN)]


@pytest.mark.parametrize("buffered,ready", [
    ("§[2026-10-0", ""),
    ("#12345", ""),
    ("trailing 1.", "trailing "),
    ("see https://x.co", "see "),
    ("plain tail", "plain tail"),
])
def test_stream_cut_holds_back_the_oldest_unfinished_token(buffered, ready):
    assert split_safe_stream(buffered) == (ready, buffered[len(ready):])


def test_stream_cut_is_idempotent_across_a_token_completing():
    """The held fragment plus what follows it is cut exactly once, so the token is
    painted as one span instead of a prose-coloured head and a coloured tail."""
    first, held = split_safe_stream("open §[2026-10-0")
    assert (first, held) == ("open ", "§[2026-10-0")
    painted = _colors(render_thinking_text(first))
    assert painted == [("open ", GREEN)]
    # …the rest of the token arrives; now it is claimed whole.
    assert _colors(render_thinking_text(held + "2]")) == [("§[2026-10-02]", WHITE)]


def test_spans_preserve_the_original_text_exactly():
    text = "1. open §[2026-10-02], see #123456 at https://x.com/a#f (3.14)"
    assert "".join(chunk for chunk, _ in thinking_spans(text)) == text


@pytest.fixture
def reasoning_cli(monkeypatch):
    """CLI stub with the reasoning-box state the three surfaces read."""
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = True
    cli.verbose = False
    cli.show_timestamps = False
    cli._stream_box_opened = False
    cli._reset_stream_state()
    emitted: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 74)
    return cli, emitted


def _body(emitted: list[str]) -> str:
    return "".join(re.sub(r"\x1b\[[0-9;]*m", "", chunk) for chunk in emitted)


def test_buffered_preview_colours_its_tokens(reasoning_cli):
    cli, emitted = reasoning_cli
    cli._emit_reasoning_preview("1. read §[2026-10-02] then open #123456")
    assert _body(emitted) == "  [thinking] 1. read §[2026-10-02] then open #123456"
    assert (ORANGE, WHITE) not in emitted  # escapes are per-span, not nested
    colors = [sgr for sgr, _ in _colors("".join(emitted))]
    assert colors.count(ORANGE) == 1 and colors.count(WHITE) == 2


def test_preview_label_stays_dim_chrome_and_the_final_answer_is_untouched(reasoning_cli):
    cli, emitted = reasoning_cli
    cli._emit_reasoning_preview("plain thought")
    assert emitted[0].startswith("\x1b[2;3m")  # the "  [thinking] " label
    assert _body(emitted).strip().startswith("[thinking]")


def test_live_box_line_and_closing_tail_are_coloured(reasoning_cli):
    cli, emitted = reasoning_cli
    cli._stream_reasoning_delta("1. first step\n")
    cli._stream_reasoning_delta("open §[2026-10-02]")
    cli._close_reasoning_box()
    body = _body(emitted)
    assert "1. first step" in body and "open §[2026-10-02]" in body
    colors = [sgr for sgr, _ in _colors("".join(emitted))]
    assert ORANGE in colors and WHITE in colors


def test_live_box_never_splits_a_token_across_two_prints(reasoning_cli):
    cli, emitted = reasoning_cli
    # A force-flush happens while the log mark is still arriving.
    cli._stream_reasoning_delta("thinking " + "x" * 80 + " §[2026-10-0")
    cli._stream_reasoning_delta("2] tail\n")
    cli._close_reasoning_box()
    painted = "".join(emitted)
    assert "§[2026-10-02]" in _body(emitted)
    # The stamp appears exactly once, and as a single white span.
    assert _body(emitted).count("§[2026-02]") == 0
    assert _body(emitted).count("§[2026-10-02]") == 1
    assert [chunk for _, chunk in _colors(painted) if "§" in chunk] == ["§[2026-10-02]"]


def test_live_box_flushes_an_unbroken_run_instead_of_going_silent(reasoning_cli):
    cli, emitted = reasoning_cli
    cli._stream_reasoning_delta("y" * 900)
    cli._close_reasoning_box()
    assert "y" in _body(emitted)