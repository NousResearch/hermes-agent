"""A long Ink TUI session renders every transcript word at most once after ``/compress`` and
when it is resumed later (``hermes --tui --resume <id>`` and ``/resume <id>`` from a fresh TUI):
a compressed-transcript block is never painted twice (#88906), the protected tail is on screen
exactly once, the compaction summary is persisted exactly once (the TUI folds it away instead of
painting it) and never painted twice, and ``/compress`` reports once. A ``/resume`` typed while a
fresh TUI is still starting keeps the resumed transcript instead of losing it to the startup
session (#121456).

Real ``hermes --tui`` in tmux against the scripted fake provider (the compression summary is an
auxiliary call answered by the fake). The session is compressed in place, then the TUI exits and
fresh TUI processes reopen it from the same state.db.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TmuxTui, cell_params, poll, private_tui_dir, require_tui, run_cells, words,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tmux + /proc session scan"),
    pytest.mark.live_system_guard_bypass,
]

TURNS = 7
WORDS_PER_REPLY = 200  # the replaced span must outweigh the summary template
TOKEN = r"\bs\dw\d{3}\b"
SUMMARY = "SUMMARYMARK compressed account of the earlier turns"
CONFIG = ("compression:\n  enabled: true\n  protect_last_n: 2\n  min_tail_user_messages: 1\n"
          "  threshold_tokens: 1000000\n")
TAIL = [TURNS]  # the protected tail: the last turn survives compression verbatim
# Fresh TUIs that must each keep a /resume typed during startup. The race loses roughly a third
# to two thirds of landed attempts (it is decided on the gateway's RPC pool), so the pinned bug
# almost always shows within these; a lucky run just passes, which the known-failure pin allows.
STARTUP_TRIES = 8
STARTING = "summoning hermes…"  # status bar before the gateway is up

CELLS = [
    "compress_happened", "compress_reported_once", "summary_persisted_once", "first_exits_clean",
    "after_compress_no_word_twice", "after_compress_tail_once",
    "resume_flag_no_word_twice", "resume_flag_tail_once",
    "slash_resume_no_word_twice", "slash_resume_tail_once", "summary_never_twice",
    "slash_resume_at_startup_survives",
]
# cell -> (regex on the bug's own failure message, "#<issue> <symptom>"): XFAILs only while the cell
# fails exactly that way, passes once the fix lands, and any other failure stays red.
KNOWN: dict[str, tuple[str, str]] = {
    "slash_resume_at_startup_survives": (
        r"the resumed transcript is gone after startup: protected tail words not exactly once",
        "#121456 /resume typed while the TUI starts is replaced by the startup session"),
}


def _reply(t: int) -> str:
    return " ".join(words(f"s{t}", WORDS_PER_REPLY))


def _twice(text: str) -> str:
    dup = sorted(w for w, c in Counter(re.findall(TOKEN, text)).items() if c > 1)
    return f"{len(dup)} words rendered more than once: {dup[:6]}" if dup else ""


def _tail_once(text: str) -> str:
    counts = Counter(re.findall(TOKEN, text))
    bad = [w for t in TAIL for w in words(f"s{t}", WORDS_PER_REPLY) if counts[w] != 1]
    return f"protected tail words not exactly once: {bad[:6]} ({len(bad)})" if bad else ""


def _summary_persisted(tui: TmuxTui) -> str:
    n = sum(c.count("SUMMARYMARK") for _s, _r, c in tui.messages())
    return "" if n == 1 else f"compaction summary in the active state.db transcript {n}x (want exactly 1)"


def _session_id(tui: TmuxTui) -> str:
    rows = tui.db_rows("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1")
    return str(rows[0][0]) if rows else ""


def _compacted(tui: TmuxTui) -> int:
    rows = tui.db_rows("SELECT COUNT(*) FROM messages WHERE compacted = 1")
    return int(rows[0][0]) if rows else 0


def _judge(cells, how: str, tui: TmuxTui) -> int:
    """Word cells for one surface; returns how often the summary text is painted there."""
    text = tui.text()
    cells.add(f"{how}_no_word_twice", _twice(text), tui.dump())
    cells.add(f"{how}_tail_once", _tail_once(text), tui.dump())
    return text.count("SUMMARYMARK")


def _first_tui(root, llm, cells, summaries: list[int]) -> str:
    """Run the long session, compress it, exit; return its session id ('' if it never compressed)."""
    first = TmuxTui(root / "a", llm.base_url, cols=140, rows=200, extra_config=CONFIG)
    try:
        first.wait_ready()
        cells.phase = "long_session"
        for t in range(1, TURNS + 1):
            first.submit(f"long session turn zq{t}q")
            first.wait_replies(t)
        first.wait_quiet(1.0)
        cells.phase = "compress"
        first.submit("/compress")
        try:
            poll(lambda: _compacted(first) > 0, timeout=60, what="compacted rows in state.db")
            cells.add("compress_happened", "")
        except AssertionError as exc:
            cells.add("compress_happened", f"{exc}\n{first.dump()}")
            return ""
        first.wait_for("Compressed:", timeout=60)
        first.wait_quiet(1.5)
        n = first.text().count("Compressed:")
        cells.add("compress_reported_once", "" if n == 1 else f"/compress reported {n}x", first.dump())
        cells.add("summary_persisted_once", _summary_persisted(first))
        summaries.append(_judge(cells, "after_compress", first))
        sid = _session_id(first)
        cells.phase = "first_exit"
        cells.add("first_exits_clean", first.exit_problem())
        return sid
    finally:
        first.close()


def _startup_resume(root, llm, home_env: dict[str, str], sid: str, cells) -> None:
    """/resume typed while the status bar still reads 'summoning hermes…' must stick."""
    cells.phase = "slash_resume_at_startup"
    tui_dir = private_tui_dir(root / "startup")
    landed = 0
    last = ""
    for i in range(STARTUP_TRIES * 3):
        tui = TmuxTui(root / "startup" / str(i), llm.base_url, cols=140, rows=200, write_home=False,
                      env_extra=home_env, tui_dir=tui_dir)
        try:
            tui.wait_raw()
            try:  # the status bar paints a moment after raw mode
                status = poll(lambda: tui.status(), timeout=3, interval=0.02, what="status bar")
            except AssertionError:
                status = ""
            if status != STARTING:
                continue  # the gateway came up before we could type: not an attempt
            tui.type(f"/resume {sid}")
            time.sleep(0.15)  # input pacing: text and Enter in one write is a paste
            tui.key("Enter")
            if tui.status() != STARTING:
                continue
            landed += 1
            tui.wait_ready()
            tui.wait_quiet(1.5)
            last = _tail_once(tui.text())
            if last:
                cells.add("slash_resume_at_startup_survives",
                          f"attempt {landed}: the resumed transcript is gone after startup: {last}",
                          tui.dump())
                return
            if landed == STARTUP_TRIES:
                cells.add("slash_resume_at_startup_survives", "")
                return
        finally:
            tui.close()
    if landed:
        cells.add("slash_resume_at_startup_survives", "")
    else:
        cells.harness_error("slash_resume_at_startup_survives",
                            f"no /resume keystroke landed while any of {STARTUP_TRIES * 3} TUIs was starting")


def _scenario(root) -> object:
    script = [Text(_reply(t)) for t in range(1, TURNS + 1)]

    def body(cells) -> None:
        with FakeLLMServer(script, aux=lambda _r: Text(SUMMARY)) as llm:
            summaries: list[int] = []
            sid = _first_tui(root, llm, cells, summaries)
            if not sid:
                return
            home_env = {"HOME": str(root / "a" / "home"), "HERMES_HOME": str(root / "a" / "home" / ".hermes")}
            for how in ("resume_flag", "slash_resume"):
                cells.phase = how
                sub = root / how
                sub.mkdir()
                args = ("--yolo", "--resume", sid) if how == "resume_flag" else ("--yolo",)
                tui = TmuxTui(sub, llm.base_url, cols=140, rows=200, args=args, write_home=False,
                              env_extra=home_env)
                try:
                    tui.wait_ready()
                    if how == "slash_resume":
                        tui.submit(f"/resume {sid}")
                    tui.wait_for(words(f"s{TURNS}", WORDS_PER_REPLY)[-1], timeout=60)
                    tui.wait_quiet(1.5)
                    summaries.append(_judge(cells, how, tui))
                finally:
                    tui.close()
            cells.add("summary_never_twice", "" if max(summaries) <= 1 else f"summary painted {summaries}x")
            _startup_resume(root, llm, home_env, sid, cells)

    return run_cells(body)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    require_tui()
    return _scenario(tmp_path_factory.mktemp("resume"))


@pytest.mark.parametrize("cell", cell_params(CELLS))
def test_long_session_compress_and_resume_render_once(run, cell: str) -> None:
    run.check(cell, KNOWN)
