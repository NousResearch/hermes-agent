"""A multi-question clarify reaches the Ink TUI as one batch form (TUI-18 in #134889).

The owner gateway sends the whole question set as one ``clarify`` request only to a local viewer
whose turn declared ``clarify_batch``; otherwise it falls back to one card per question, so the
user sees only the current one and Tab/Shift+Tab has nothing to switch to.

Real ``hermes --tui`` in a private tmux server, scripted fake provider issuing one three-question
``clarify``: all three questions are on screen together, Tab/Shift+Tab move between them, and the
three answers reach the tool in one result.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from tests.e2e.core.tui_pty._helpers import TmuxTui, cell_params, require_tui, run_cells, words
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.platforms("linux"),  # tmux + /proc session scan
    pytest.mark.live_system_guard_bypass,
]

CONFIG = "clarify:\n  timeout: 180\n"
QUESTIONS = [
    {"question": "Which colour for alpha?", "choices": ["red", "blue"]},
    {"question": "Which shape for beta?", "choices": ["square", "circle"]},
    {"question": "Which size for gamma?", "choices": ["small", "large"]},
]
REPLY = " ".join(words("cb", 6))

CELLS = ["batch_form_shows_every_question", "tab_switches_question", "answers_reach_tool", "exits_clean"]


def _tool_results(llm: FakeLLMServer) -> list[str]:
    out = []
    for body in llm.main_requests():
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            out.append(str(msgs[-1].get("content") or ""))
    return out


def _active(tui: TmuxTui) -> str:
    """The question text on the ``▸`` row, or '' when no question row is active."""
    for row in tui.rows():
        for q in QUESTIONS:
            if "▸" in row and q["question"] in row:
                return q["question"]
    return ""


def _scenario(root) -> object:
    script = [ToolCall("clarify", {"questions": QUESTIONS}), Text(REPLY)]

    def body(cells) -> None:
        with FakeLLMServer(script) as llm:
            tui = TmuxTui(root, llm.base_url, cols=120, rows=50, extra_config=CONFIG, args=())
            try:
                _drive(tui, llm, cells)
                cells.phase = "exit"
                cells.add("exits_clean", tui.exit_problem())
            finally:
                tui.close()

    def _drive(tui: TmuxTui, llm: FakeLLMServer, cells) -> None:
        tui.wait_ready()
        cells.phase = "clarify"
        tui.submit("ask me three cb1")
        tui.wait_for(QUESTIONS[0]["question"])
        tui.wait_quiet(1.0)
        screen = "\n".join(tui.rows())
        missing = [q["question"] for q in QUESTIONS if q["question"] not in screen]
        cells.add("batch_form_shows_every_question",
                  f"not on screen with the first question: {missing}" if missing
                  else "" if "0/3 answered" in screen else "no '0/3 answered' counter", tui.dump())

        seen = [_active(tui)]
        for key, target in (("Tab", QUESTIONS[1]), ("BTab", QUESTIONS[0])):  # BTab = Shift+Tab
            tui.key(key)
            # A per-question card has nothing to switch to: record where the cursor is, don't abort.
            with contextlib.suppress(AssertionError):
                tui.wait_for(lambda _t, q=target["question"]: _active(tui) == q, history=False, timeout=10)
            seen.append(_active(tui))
        want = [QUESTIONS[0]["question"], QUESTIONS[1]["question"], QUESTIONS[0]["question"]]
        cells.add("tab_switches_question", "" if seen == want else f"active question went {seen}", tui.dump())

        # Enter locks the highlighted (first) choice and moves to the next unanswered question; wait
        # for that move (the lock's round trip) before answering the next one.
        for i, _question in enumerate(QUESTIONS):
            tui.key("Enter")
            if i + 1 < len(QUESTIONS):
                with contextlib.suppress(AssertionError):  # per-question cards: answers_reach_tool reports
                    tui.wait_for(lambda _t, q=QUESTIONS[i + 1]["question"]: _active(tui) == q, history=False)
        tui.wait_replies(1)
        results = _tool_results(llm)
        try:
            got = [r["user_response"] for r in json.loads(results[0])["responses"]]
        except (IndexError, KeyError, TypeError, ValueError):
            got = None
        want_answers = [q["choices"][0] for q in QUESTIONS]
        cells.add("answers_reach_tool", "" if got == want_answers else f"tool results {results[:1]}")

    return run_cells(body)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    require_tui()
    return _scenario(tmp_path_factory.mktemp("clarify_batch"))


@pytest.mark.parametrize("cell", cell_params(CELLS))
def test_clarify_batch_form(run, cell: str) -> None:
    run.check(cell)
