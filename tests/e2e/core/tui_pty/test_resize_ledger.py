"""The Ink TUI keeps every transcript word exactly once, in order, through terminal resizes —
idle and mid-stream, shrinking, growing and dragging — and its content width follows the
terminal (#96372 history cleared on resize, #35804 content width stuck).

A real ``hermes --tui`` (Node frontend + ``tui_gateway`` child + AIAgent + SessionDB) runs in a
private tmux server against the scripted fake provider. tmux is the terminal: it reflows the grid
on every ``resize-window`` and keeps the scrollback we read back with ``capture-pane -S -``.
Both render modes run: the default alternate-screen viewport, and inline mode
(``HERMES_TUI_INLINE=1``, what the dashboard embeds) where finished turns live in tmux's own
scrollback on a short 30-row pane, so a redraw that re-prints or drops history is visible there.

Each resize step is one cell; a word ledger over the captured text classifies every streamed
token as lost / duplicated / misordered.
"""

from __future__ import annotations

import re
import sys
import time

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TITLE, TmuxTui, cell_params, layout_problem, ledger_problems, paragraph, require_tui, run_cells,
    words,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tmux + /proc session scan"),
    pytest.mark.live_system_guard_bypass,
]

TOKEN = r"\br\dw\d{3}\b"
N = {1: 12, 2: 220, 3: 160}
PROMPTS = {t: f"resize turn zq{t}q please" for t in N}
REPLY = {1: " ".join(words("r1", N[1])), 2: paragraph("r2", N[2]), 3: paragraph("r3", N[3])}
DRAG = (110, 95, 80, 70, 90, 85, 100)

CELLS = [
    "midstream_drag_ledger", "midstream_drag_width",
    "idle_shrink_ledger", "idle_shrink_width",
    "idle_grow_ledger", "idle_grow_width",
    "midstream_two_step_shrink_ledger", "midstream_two_step_shrink_width",
    "prompts_once", "persisted_matches_screen", "exits_clean",
]


def _expected(upto: int) -> list[str]:
    return [w for t in range(1, upto + 1) for w in words(f"r{t}", N[t])]


def _frame_problem(tui: TmuxTui, upto: int, para: int) -> str:
    """The settled frame is laid out for the current width: the newest long reply reaches the
    edge, every reply on screen is whole words reading on row to row, and every prompt echo row
    reads exactly its prompt (a frame painted for another width splices rules and other rows'
    cells into them)."""
    rows, cols = tui.rows(), tui.size()[0]
    for t in range(1, upto + 1):
        if problem := layout_problem(rows, cols, f"r{t}", slack=12 if t == para else None):
            return problem
    history = tui.rows(history=True)  # inline mode scrolls earlier prompts off the pane
    for t in range(1, upto + 1):
        echo = [r for r in history if f"zq{t}q" in r]
        if [r.strip() for r in echo] != [f"❯ {PROMPTS[t]}"]:
            return f"prompt {t} echo rows are {echo!r}, want one row reading '❯ {PROMPTS[t]}'"
    return ""


def _persisted_problem(persisted: list[tuple[str, str, str]], screen: str) -> str:
    """state.db holds exactly the conversation, in one session, and the reply words it persisted
    are the reply words on screen, in the same order."""
    users = [c for _s, r, c in persisted if r == "user"]
    replies = [c for _s, r, c in persisted if r == "assistant" and c.strip()]
    if users != list(PROMPTS.values()) or replies != list(REPLY.values()):
        return f"state.db rows differ from the conversation: users={users!r} replies={[r[:30] for r in replies]}"
    if len({s for s, _r, _c in persisted}) != 1:
        return "turns split across sessions"
    db_words = re.findall(TOKEN, " ".join(replies))
    shown = re.findall(TOKEN, screen)
    if shown != db_words:
        first = next((i for i, (a, b) in enumerate(zip(shown, db_words)) if a != b), min(len(shown), len(db_words)))
        return (f"screen shows {len(shown)} reply words, state.db persisted {len(db_words)}; first difference "
                f"at #{first}: screen {shown[first:first + 3]} vs db {db_words[first:first + 3]}")
    return ""


def _scenario(mode: str, root) -> object:
    rows = 30 if mode == "inline" else 120

    def body(cells) -> None:
        script = [Text(REPLY[1]),
                  Text(REPLY[2], chunk_chars=6, delay_per_chunk=0.02),
                  Text(REPLY[3], chunk_chars=6, delay_per_chunk=0.02)]
        with FakeLLMServer(script, aux=lambda _r: Text(TITLE)) as llm:
            tui = TmuxTui(root, llm.base_url, cols=120, rows=rows, inline=mode == "inline")
            try:
                _drive(tui, cells)
                cells.add("exits_clean", tui.exit_problem())
            finally:
                tui.close()

    def step(tui: TmuxTui, cells, name: str, upto: int, para: int) -> None:
        cells.phase = name
        tui.wait_quiet(1.0)
        cells.add(f"{name}_ledger", ledger_problems(tui.text(), _expected(upto), TOKEN), tui.dump())
        cells.add(f"{name}_width", _frame_problem(tui, upto, para), tui.dump())

    def _drive(tui: TmuxTui, cells) -> None:
        tui.wait_ready()
        cells.phase = "turn1"
        tui.submit(PROMPTS[1])
        tui.wait_replies(1)
        cells.phase = "midstream_drag"
        tui.submit(PROMPTS[2])
        tui.wait_for("r2w030")
        for cols in DRAG:  # a window drag: 7 SIGWINCHes in ~0.35 s while the reply streams
            tui.resize(cols)
            time.sleep(0.05)
        tui.wait_replies(2)
        step(tui, cells, "midstream_drag", 2, 2)
        tui.resize(72)
        step(tui, cells, "idle_shrink", 2, 2)
        tui.resize(150)
        step(tui, cells, "idle_grow", 2, 2)
        cells.phase = "midstream_two_step_shrink"
        tui.submit(PROMPTS[3])
        tui.wait_for("r3w020")
        tui.resize(110)
        tui.wait_for("r3w060")
        tui.resize(90)
        tui.wait_replies(3)
        step(tui, cells, "midstream_two_step_shrink", 3, 3)

        cells.phase = "persisted"
        text = tui.text()
        counts = {t: text.count(f"zq{t}q") for t in N}
        cells.add("prompts_once", "" if set(counts.values()) == {1} else f"prompt echo counts {counts}", tui.dump())
        cells.add("persisted_matches_screen", _persisted_problem(tui.messages(), text), tui.dump())
        cells.phase = "exit"

    return run_cells(body)


@pytest.fixture(scope="module")
def runs() -> dict:
    require_tui()
    return {}




@pytest.mark.parametrize("case", cell_params([f"{m}:{c}" for m in ("alt", "inline") for c in CELLS]))
def test_resize_transcript_ledger(runs: dict, case: str, tmp_path_factory) -> None:
    mode, cell = case.split(":")
    if mode not in runs:
        runs[mode] = _scenario(mode, tmp_path_factory.mktemp(f"resize-{mode}"))
    runs[mode].check(cell)
