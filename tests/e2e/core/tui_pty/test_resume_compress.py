"""A long Ink TUI session renders every transcript word at most once after ``/compress`` and
when it is resumed later (``hermes --tui --resume <id>`` and ``/resume <id>`` from a fresh TUI):
a compressed-transcript block is never painted twice (#88906), and the protected tail is on
screen exactly once.

Real ``hermes --tui`` in tmux against the scripted fake provider (the compression summary is an
auxiliary call answered by the fake). The session is compressed in place, then the TUI exits and
two fresh TUI processes reopen it from the same state.db.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TmuxTui, cell_params, poll, require_tui, run_cells, words,
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

CELLS = [
    "compress_happened", "after_compress_no_word_twice", "after_compress_tail_once",
    "resume_flag_no_word_twice", "resume_flag_tail_once",
    "slash_resume_no_word_twice", "slash_resume_tail_once", "summary_never_twice",
]
# cell -> "#<issue> <symptom>"; each entry is a strict xfail that turns red once fixed.
KNOWN: dict[str, str] = {}


def _reply(t: int) -> str:
    return " ".join(words(f"s{t}", WORDS_PER_REPLY))


def _twice(text: str) -> str:
    dup = sorted(w for w, c in Counter(re.findall(TOKEN, text)).items() if c > 1)
    return f"{len(dup)} words rendered more than once: {dup[:6]}" if dup else ""


def _tail_once(text: str) -> str:
    counts = Counter(re.findall(TOKEN, text))
    bad = [w for t in TAIL for w in words(f"s{t}", WORDS_PER_REPLY) if counts[w] != 1]
    return f"protected tail words not exactly once: {bad[:6]} ({len(bad)})" if bad else ""


def _session_id(tui: TmuxTui) -> str:
    rows = tui.db_rows("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1")
    return str(rows[0][0]) if rows else ""


def _compacted(tui: TmuxTui) -> int:
    rows = tui.db_rows("SELECT COUNT(*) FROM messages WHERE compacted = 1")
    return int(rows[0][0]) if rows else 0


def _scenario(root) -> object:
    script = [Text(_reply(t)) for t in range(1, TURNS + 1)]

    def body(cells) -> None:
        with FakeLLMServer(script, aux=lambda _r: Text(SUMMARY)) as llm:
            first = TmuxTui(root / "a", llm.base_url, cols=140, rows=200, extra_config=CONFIG)
            try:
                first.wait_ready()
                for t in range(1, TURNS + 1):
                    first.submit(f"long session turn zq{t}q")
                    first.wait_replies(t)
                first.wait_quiet(1.0)
                first.submit("/compress")
                try:
                    poll(lambda: _compacted(first) > 0, timeout=60, what="compacted rows in state.db")
                    cells.add("compress_happened", "")
                except AssertionError as exc:
                    cells.add("compress_happened", f"{exc}\n{first.dump()}")
                    return
                first.wait_quiet(1.5)
                text = first.text()
                cells.add("after_compress_no_word_twice", _twice(text), first.dump())
                cells.add("after_compress_tail_once", _tail_once(text), first.dump())
                summaries = [text.count("SUMMARYMARK")]
                sid = _session_id(first)
                first.exit_problem()
            finally:
                first.close()

            for how in ("resume_flag", "slash_resume"):
                sub = root / how
                sub.mkdir()
                args = ("--yolo", "--resume", sid) if how == "resume_flag" else ("--yolo",)
                tui = TmuxTui(sub, llm.base_url, cols=140, rows=200, args=args, write_home=False,
                              env_extra={"HOME": str(root / "a" / "home"),
                                         "HERMES_HOME": str(root / "a" / "home" / ".hermes")})
                try:
                    tui.wait_ready()
                    if how == "slash_resume":
                        tui.submit(f"/resume {sid}")
                    tui.wait_for(words(f"s{TURNS}", WORDS_PER_REPLY)[-1], timeout=60)
                    tui.wait_quiet(1.5)
                    text = tui.text()
                    cells.add(f"{how}_no_word_twice", _twice(text), tui.dump())
                    cells.add(f"{how}_tail_once", _tail_once(text), tui.dump())
                    summaries.append(text.count("SUMMARYMARK"))
                finally:
                    tui.close()
            cells.add("summary_never_twice", "" if max(summaries) <= 1 else f"summary counts {summaries}")

    return run_cells(body)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    require_tui()
    return _scenario(tmp_path_factory.mktemp("resume"))


@pytest.mark.parametrize("cell", cell_params(CELLS, KNOWN))
def test_long_session_compress_and_resume_render_once(run, cell: str) -> None:
    run.check(cell)
