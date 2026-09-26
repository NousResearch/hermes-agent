"""Blocking modals and streaming tool output in the Ink TUI survive terminal resizes: the clarify
and approval cards stay on screen exactly once while the window is resized, the answer given
after the resize reaches the tool (read back from the next wire request the agent sends), and the
transcript around them keeps every word exactly once.

Real ``hermes --tui`` in a private tmux server, ``approvals.mode: manual`` (the shipped default,
never yolo), local terminal backend, scripted fake provider issuing ``clarify`` /
``terminal(rm -rf <victim>)`` / a slow printing ``terminal`` command whose output the TUI shows
once ``/verbose verbose`` is on: every printed line is on screen exactly once after the drags.
"""

from __future__ import annotations

import json
import re
import sys
import time

import pytest

from tests.e2e.core.tui_pty._helpers import (
    TmuxTui, cell_params, ledger_problems, require_tui, run_cells, words,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tmux + /proc session scan"),
    pytest.mark.live_system_guard_bypass,
]

CONFIG = ("approvals:\n  mode: manual\n  timeout: 180\nclarify:\n  timeout: 180\n"
          "terminal:\n  backend: local\n")
QUESTION = "Which colour for zeta?"
# Few and short enough that the verbose Result block shows them all on one unbroken row at 120
# cols (the block is capped at 12 lines / 800 chars; a longer word is hard-wrapped mid-token).
TOOL_LINES = [f"tout{i:02d}" for i in range(1, 11)]
TOOL_RE = r"tout\d{2}(?!\d)"  # the Result block is JSON: lines arrive as "\ntout02"
REPLIES = {k: " ".join(words(k, 8)) for k in ("c1", "a1", "t1")}
TOKEN = r"\b(?:c1|a1|t1)w\d{3}\b"

CELLS = [
    "clarify_card_survives_resize", "clarify_answer_reaches_tool", "clarify_tool_call_rendered_once",
    "clarify_answer_not_a_user_turn",
    "approval_card_survives_resize", "approval_deny_honoured",
    "tool_output_reaches_model_in_order", "tool_output_rendered_once",
    "transcript_ledger", "exits_clean",
]
# cell -> (regex on the bug's own failure message, "#<issue> <symptom>"): XFAILs only while the cell
# fails exactly that way, passes once the fix lands, and any other failure stays red.
KNOWN: dict[str, tuple[str, str]] = {
    "clarify_tool_call_rendered_once": (r"clarify tool call rendered [2-9]x",
                                        "#121267 answering clarify renders its tool call twice"),
}


def _tool_results(llm: FakeLLMServer) -> list[str]:
    out = []
    for body in llm.main_requests():
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            out.append(str(msgs[-1].get("content") or ""))
    return out


def _typed(content) -> object:
    """What the user typed into a wire user message: the install's first message carries the
    API-only first-contact onboarding note (``\\n\\n[System note: ...]``, never persisted), which is
    per-turn context riding the user turn, not a second user turn."""
    return content.split("\n\n[System note: ", 1)[0] if isinstance(content, str) else content


def _visible_once(tui: TmuxTui, needles: list[str]) -> str:
    text = "\n".join(tui.rows())
    bad = {n: text.count(n) for n in needles if text.count(n) != 1}
    return f"modal rows not visible exactly once after resize: {bad}" if bad else ""


def _scenario(root, victim) -> object:
    command = f"for i in $(seq -w 1 {len(TOOL_LINES)}); do echo tout$i; sleep 0.3; done"
    script = [
        ToolCall("clarify", {"question": QUESTION, "choices": ["red", "blue"]}), Text(REPLIES["c1"]),
        ToolCall("terminal", {"command": f"rm -rf {victim}"}), Text(REPLIES["a1"]),
        ToolCall("terminal", {"command": command}), Text(REPLIES["t1"]),
    ]

    def body(cells) -> None:
        with FakeLLMServer(script) as llm:
            tui = TmuxTui(root, llm.base_url, cols=120, rows=70, extra_config=CONFIG, args=())
            try:
                _drive(tui, llm, cells)
                cells.phase = "exit"
                cells.add("exits_clean", tui.exit_problem())
            finally:
                tui.close()

    def _drive(tui: TmuxTui, llm: FakeLLMServer, cells) -> None:
        tui.wait_ready()
        cells.phase = "clarify"
        # 1. clarify card open across a shrink and a grow, answered afterwards.
        tui.submit("pick a colour mq1")
        tui.wait_for("quick pick")
        for cols in (90, 130):
            tui.resize(cols)
            tui.wait_quiet(1.0)
        cells.add("clarify_card_survives_resize",
                  _visible_once(tui, [f"ask {QUESTION}", "1. red", "2. blue"]), tui.dump())
        tui.key("Down")
        tui.wait_for(lambda t: "▸ 2. blue" in t, history=False)
        tui.key("Enter")
        tui.wait_replies(1)
        results = _tool_results(llm)
        answer = json.loads(results[0]).get("user_response") if results else None
        cells.add("clarify_answer_reaches_tool", "" if answer == "blue" else f"tool result {results[:1]}")
        tui.wait_quiet(1.0)
        n = tui.text().count(f'Clarify("{QUESTION}")')
        cells.add("clarify_tool_call_rendered_once", "" if n == 1 else f"clarify tool call rendered {n}x",
                  tui.dump())
        # The answer is the tool's result, never a persisted user message (the TUI echoes it as a
        # bubble by design; state.db and the next wire request must not grow a user turn).
        users = [c for _s, r, c in tui.messages() if r == "user"]
        wire_users = [_typed(m.get("content")) for m in (llm.main_requests()[-1].get("messages") or [])
                      if m.get("role") == "user"]
        cells.add("clarify_answer_not_a_user_turn",
                  "" if users == ["pick a colour mq1"] and wire_users == ["pick a colour mq1"]
                  else f"persisted user rows {users!r}; wire user messages {wire_users!r}")

        # 2. approval card open across resizes, then denied with its quick-pick key.
        cells.phase = "approval"
        tui.submit("remove the dir mq2")
        tui.wait_for("approval required")
        for cols in (140, 85):
            tui.resize(cols)
            tui.wait_quiet(1.0)
        problem = _visible_once(tui, ["approval required", "1. Allow once", "4. Deny"])
        if not victim.exists():
            problem += " / the dangerous command ran before anyone answered"
        cells.add("approval_card_survives_resize", problem, tui.dump())
        tui.key("4")
        tui.wait_replies(2)
        denied = _tool_results(llm)[-1]
        ok = victim.exists() and (victim / "keep.txt").exists() and "denied" in denied.lower()
        cells.add("approval_deny_honoured", "" if ok else f"victim exists={victim.exists()} result={denied[:200]}")

        # 3. a slow printing tool while the window is dragged, its output rendered (verbose tool
        # progress: the default mode shows no tool output at all) and dragged again once shown.
        cells.phase = "tool_output"
        tui.resize(120)
        tui.submit("/verbose verbose")
        tui.wait_for("verbose: verbose")
        tui.submit("print lines mq3")
        llm.wait_for_requests(5, timeout=60)
        for cols in (100, 80, 110, 95, 120):
            tui.resize(cols)
            time.sleep(0.3)
        tui.wait_replies(3)
        tui.wait_for(lambda t: len(re.findall(TOOL_RE, t)) >= len(TOOL_LINES), timeout=30)
        for cols in (90, 70, 130, 120):
            tui.resize(cols)
            tui.wait_quiet(0.5)
        tui.wait_quiet(1.0)
        result = _tool_results(llm)[-1]
        try:
            output = str(json.loads(result).get("output"))
        except ValueError:
            output = result
        problem = ledger_problems(output, TOOL_LINES, TOOL_RE)
        cells.add("tool_output_reaches_model_in_order", problem and f"{problem}\nresult: {result[:400]}")
        cells.add("tool_output_rendered_once", ledger_problems(tui.text(), TOOL_LINES, TOOL_RE), tui.dump())
        cells.add("transcript_ledger",
                  ledger_problems(tui.text(), [w for k in ("c1", "a1", "t1") for w in words(k, 8)], TOKEN),
                  tui.dump())

    return run_cells(body)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    require_tui()
    root = tmp_path_factory.mktemp("modals")
    victim = root / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious", encoding="utf-8")
    return _scenario(root, victim)


@pytest.mark.parametrize("cell", cell_params(CELLS))
def test_modals_and_tool_output_survive_resize(run, cell: str) -> None:
    run.check(cell, KNOWN)
