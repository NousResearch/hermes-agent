"""Wide characters and Ctrl+C in the Ink TUI keep the transcript consistent.

CJK / wide characters (#35272): the composer cursor sits exactly after the typed wide text
(tmux's own cursor column, so a width miscount of one column per ideograph shows up), a streamed
CJK reply wraps inside the terminal on every width and each token is on screen exactly once, and
the frame settles after the resizes instead of scrolling forever.

Ctrl+C mid-stream: the turn stops, the partially streamed prefix is on screen once and matches
what state.db persisted, the next turn's wire request carries that same partial reply, and the
next reply renders exactly once after it.
"""

from __future__ import annotations

import re
import sys

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TmuxTui, cell_params, display_width, ledger_problems, paragraph, require_tui, run_cells,
    words,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tmux + /proc session scan"),
    pytest.mark.live_system_guard_bypass,
]

CJK_TYPED = ("今天天气很好", "我们一起去")
CJK_TOKENS = [f"字{i:03d}漢字テスト" for i in range(90)]
CJK_RE = r"字\d{3}漢字テスト"
INTERRUPT_WORDS = 600

CELLS = [
    "cjk_cursor_after_typed_text", "cjk_prompt_submitted_verbatim",
    "cjk_reply_ledger", "cjk_rows_fit_terminal", "cjk_frame_settles",
    "interrupt_turn_stops", "interrupt_prefix_on_screen_once", "interrupt_screen_matches_db",
    "interrupt_next_request_carries_partial", "interrupt_next_reply_once", "exits_clean",
]
# cell -> "#<issue> <symptom>"; each entry is a strict xfail that turns red once fixed.
KNOWN: dict[str, str] = {}


def _cursor_problem(tui: TmuxTui, typed: str) -> str:
    """The cursor column equals the display width of the composer row up to the typed text."""
    x, y = (int(v) for v in tui.fmt("#{cursor_x} #{cursor_y}").split())
    row = tui.rows()[y] if y < len(tui.rows()) else ""
    if typed not in row:
        return f"typed text {typed!r} not on the cursor row {y}: {row!r}"
    want = display_width(row[: row.index(typed) + len(typed)])
    return "" if x == want else f"cursor at col {x}, typed text ends at col {want} (row {row!r})"


def _cjk(tui: TmuxTui, llm: FakeLLMServer, cells) -> None:
    tui.wait_ready()
    tui.type(CJK_TYPED[0])
    tui.wait_for(CJK_TYPED[0], history=False)
    first = _cursor_problem(tui, CJK_TYPED[0])
    tui.type(CJK_TYPED[1])
    both = "".join(CJK_TYPED)
    tui.wait_for(both, history=False)
    cells.add("cjk_cursor_after_typed_text", first or _cursor_problem(tui, both), tui.dump())
    tui.key("Enter")
    tui.wait_for(CJK_TOKENS[20])
    tui.resize(61)
    tui.wait_replies(1)
    sent = [m.get("content") for m in (llm.main_requests()[0].get("messages") or []) if m.get("role") == "user"]
    cells.add("cjk_prompt_submitted_verbatim", "" if sent and both in str(sent[-1]) else f"wire user {sent!r}")
    for cols in (47, 88):
        tui.resize(cols)
        tui.wait_quiet(1.0)
        cols_now = tui.size()[0]
        cells.add("cjk_reply_ledger", ledger_problems(tui.text(), CJK_TOKENS, CJK_RE), tui.dump())
        wide = [r for r in tui.rows() if display_width(r) > cols_now]
        cells.add("cjk_rows_fit_terminal", f"{len(wide)} rows wider than {cols_now}: {wide[:2]}" if wide else "",
                  tui.dump())
        if cells.results["cjk_reply_ledger"][0] is False or wide:
            return
    # An infinite scroll loop keeps repainting: the frame must hold still for several seconds.
    try:
        tui.wait_quiet(3.0, timeout=20)
        cells.add("cjk_frame_settles", "")
    except AssertionError as exc:
        cells.add("cjk_frame_settles", str(exc)[:2000])


def _interrupt(tui: TmuxTui, llm: FakeLLMServer, cells) -> None:
    tui.wait_ready()
    tui.submit("stream then stop ix1")
    tui.wait_for("i1w060")
    tui.key("C-c")
    try:
        tui.wait_for("[interrupted]", timeout=30)
        tui.wait_quiet(1.0)
        cells.add("interrupt_turn_stops", "" if tui.alive() else "Ctrl+C mid-stream exited the TUI")
    except AssertionError as exc:
        cells.add("interrupt_turn_stops", str(exc)[:2000])
        return
    shown = [w for w in re.findall(r"\bi1w\d{3}\b", tui.text())]
    prefix = words("i1", len(set(shown)))
    cells.add("interrupt_prefix_on_screen_once",
              "" if shown == prefix and len(shown) < INTERRUPT_WORDS
              else f"{len(shown)} words shown, not a once-each prefix: {shown[:3]}…{shown[-3:]}", tui.dump())
    rows = tui.messages()
    partial = [c for _s, r, c in rows if r == "assistant"]
    db_words = re.findall(r"\bi1w\d{3}\b", partial[0]) if partial else []
    cells.add("interrupt_screen_matches_db",
              "" if db_words and db_words == shown
              else f"state.db assistant words {len(db_words)} vs screen {len(shown)}: {partial[:1]!r:.200}")

    tui.submit("carry on ix2")
    tui.wait_replies(2)
    tui.wait_quiet(1.0)
    history = llm.main_requests()[-1].get("messages") or []
    carried = [m for m in history if m.get("role") == "assistant" and "i1w000" in str(m.get("content"))]
    carried_words = re.findall(r"\bi1w\d{3}\b", str(carried[0].get("content"))) if carried else []
    cells.add("interrupt_next_request_carries_partial",
              "" if len(carried) == 1 and carried_words == db_words
              else f"{len(carried)} assistant messages with the partial; words {len(carried_words)} vs db {len(db_words)}")
    cells.add("interrupt_next_reply_once",
              ledger_problems(tui.text(), words("n1", 12), r"\bn1w\d{3}\b")
              or ("" if re.findall(r"\bi1w\d{3}\b", tui.text()) == shown else "the partial reply re-rendered"),
              tui.dump())


def _scenario(kind: str, root) -> object:
    scripts = {
        "cjk": [Text(" ".join(CJK_TOKENS), chunk_chars=5, delay_per_chunk=0.02)],
        "interrupt": [Text(paragraph("i1", INTERRUPT_WORDS), chunk_chars=4, delay_per_chunk=0.03),
                      Text(" ".join(words("n1", 12)))],
    }
    drive = {"cjk": _cjk, "interrupt": _interrupt}[kind]

    def body(cells) -> None:
        with FakeLLMServer(scripts[kind]) as llm:
            # Tall enough that the whole conversation stays inside the viewport at 47 cols.
            tui = TmuxTui(root, llm.base_url, cols=100, rows=130)
            try:
                drive(tui, llm, cells)
                if kind == "interrupt":
                    cells.add("exits_clean", tui.exit_problem())
            finally:
                tui.close()

    return run_cells(body)


@pytest.fixture(scope="module")
def runs() -> dict:
    require_tui()
    return {}


@pytest.mark.parametrize("cell", cell_params(CELLS, KNOWN))
def test_wide_chars_and_interrupt(runs: dict, cell: str, tmp_path_factory) -> None:
    kind = "cjk" if cell.startswith("cjk") else "interrupt"
    if kind not in runs:
        runs[kind] = _scenario(kind, tmp_path_factory.mktemp(kind))
    runs[kind].check(cell)
