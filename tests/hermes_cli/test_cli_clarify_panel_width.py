"""Clarify panels follow the live terminal width (#126970).

The preview wrap used to be hard-coded at 60 columns and ``_panel_box_width``'s
default ``max_width=76`` capped the box at ~67 columns no matter how wide the
terminal was. The clarify panel (one renderer since the questions[] rework)
sizes its preview wrap and box cap from the live width, so a wide terminal
gets a wide panel while a narrow one keeps the old bounded layout.
"""
import os
import shutil

from hermes_cli.cli_tui_mixin import CLITuiMixin

# Space-separated so textwrap can actually re-flow it (break_long_words=False).
LONG_QUESTION = " ".join(["deploy"] * 60)


class _Host(CLITuiMixin):
    """Minimal host: the renderer only touches clarify state + module helpers."""


def _set_terminal(monkeypatch, columns, rows=50):
    size = os.terminal_size((columns, rows))
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): size)


def _visual_lines(fragments):
    text = "".join(t for _style, t in fragments)
    return [line for line in text.split("\n") if line.startswith("│")]


def _widest_line(fragments):
    return max(len(line) for line in _visual_lines(fragments))


def _state(questions):
    first = questions[0]
    return {
        "questions": [
            {
                "qid": f"q{i}",
                "id": None,
                "question": question,
                "choices": first["choices"],
                "choices_offered": first["choices"],
                "multi_select": False,
            }
            for i, question in enumerate(questions)
        ],
        "answers": {},
        "answer_meta": {},
        "active": 0,
        "choices": first["choices"],
        "selected": 0,
        "multi_select": False,
    }


def test_single_question_panel_widens_with_the_terminal(monkeypatch):
    host = _Host()
    host._clarify_freetext = False
    host._clarify_state = _state([{"question": LONG_QUESTION, "choices": ["red", "blue"]}])

    _set_terminal(monkeypatch, 120)
    wide = _widest_line(host._get_clarify_display_fragments())
    _set_terminal(monkeypatch, 200)
    wider = _widest_line(host._get_clarify_display_fragments())

    assert wide > 90, wide          # old 60-column wrap capped the box at ~67
    assert wider > wide, (wide, wider)


def test_multi_question_panel_widens_with_the_terminal(monkeypatch):
    host = _Host()
    host._clarify_freetext = False
    host._clarify_state = _state(
        [
            {"question": LONG_QUESTION, "choices": ["red", "blue"]},
            {"question": "Second question", "choices": ["red", "blue"]},
        ]
    )

    _set_terminal(monkeypatch, 120)
    wide = _widest_line(host._get_clarify_display_fragments())
    _set_terminal(monkeypatch, 200)
    wider = _widest_line(host._get_clarify_display_fragments())

    assert wide > 90, wide
    assert wider > wide, (wide, wider)


def test_panels_stay_inside_a_narrow_terminal(monkeypatch):
    single_host = _Host()
    single_host._clarify_freetext = False
    single_host._clarify_state = _state([{"question": LONG_QUESTION, "choices": ["red", "blue"]}])

    batch_host = _Host()
    batch_host._clarify_freetext = False
    batch_host._clarify_state = _state(
        [
            {"question": LONG_QUESTION, "choices": ["red", "blue"]},
            {"question": "Second question", "choices": ["red", "blue"]},
        ]
    )

    _set_terminal(monkeypatch, 60)
    single = _widest_line(single_host._get_clarify_display_fragments())
    batch = _widest_line(batch_host._get_clarify_display_fragments())

    # The box keeps its margin inside the real terminal instead of hard-wrapping.
    assert single <= 60, single
    assert batch <= 60, batch
