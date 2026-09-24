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

import sys
import time

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TITLE, TmuxTui, cell_params, ledger_problems, paragraph, require_tui, run_cells, width_problem,
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
        tui.wait_quiet(1.0)
        text, cols = tui.text(), tui.size()[0]
        cells.add(f"{name}_ledger", ledger_problems(text, _expected(upto), TOKEN), tui.dump())
        cells.add(f"{name}_width", width_problem(tui.rows(history=True), rf"\br{para}w\d{{3}}\b", cols),
                  tui.dump())

    def _drive(tui: TmuxTui, cells) -> None:
        tui.wait_ready()
        tui.submit(PROMPTS[1])
        tui.wait_replies(1)
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
        tui.submit(PROMPTS[3])
        tui.wait_for("r3w020")
        tui.resize(110)
        tui.wait_for("r3w060")
        tui.resize(90)
        tui.wait_replies(3)
        step(tui, cells, "midstream_two_step_shrink", 3, 3)

        text = tui.text()
        counts = {t: text.count(f"zq{t}q") for t in N}
        cells.add("prompts_once", "" if set(counts.values()) == {1} else f"prompt echo counts {counts}", tui.dump())
        persisted = tui.messages()
        users = [c for _s, r, c in persisted if r == "user"]
        replies = [c for _s, r, c in persisted if r == "assistant" and c.strip()]
        problem = ""
        if users != list(PROMPTS.values()) or replies != list(REPLY.values()):
            problem = f"state.db rows differ from the conversation: users={users!r} replies={[r[:30] for r in replies]}"
        elif len({s for s, _r, _c in persisted}) != 1:
            problem = "turns split across sessions"
        cells.add("persisted_matches_screen", problem)

    return run_cells(body)


@pytest.fixture(scope="module")
def runs() -> dict:
    require_tui()
    return {}


# "<mode>:<cell>" -> "#<issue> <symptom>"; each entry is a strict xfail that turns red once fixed.
KNOWN: dict[str, str] = {}


@pytest.mark.parametrize("case", cell_params([f"{m}:{c}" for m in ("alt", "inline") for c in CELLS], KNOWN))
def test_resize_transcript_ledger(runs: dict, case: str, tmp_path_factory) -> None:
    mode, cell = case.split(":")
    if mode not in runs:
        runs[mode] = _scenario(mode, tmp_path_factory.mktemp(f"resize-{mode}"))
    runs[mode].check(cell)
