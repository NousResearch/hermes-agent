"""The classic CLI keeps main's behaviour under the one gateway: the 15-step main-vs-branch comparison.

Ported from the local A/B harness used on #106742 (#134889 T24). That harness ran one scripted
session on a ``main`` tree and on the rewire branch in tmux and diffed the normalized screens;
its MAIN-vs-MAIN self-check was 15/15 identical. With the rewire merged there is no second tree to
diff, so each step here asserts what main's capture showed (the pre-rewire baseline) against the
current checkout, where the gateway daemon runs the agent, tools and session writes.

One real ``hermes`` (``display.interface: cli``) session in a private tmux server, loopback
``FakeLLMServer`` keyed on the ``STEPn`` marker in the latest user message (zero spend):

01 startup · 02 markdown · 03a approval modal · 03b approve once → tool reply · 04 read_file ·
05 /help · 06a /model picker · 06b /title · 07 /usage · 08 Ctrl+C mid-stream · 09 /compress ·
10a /quit exit summary · 10b ``-c`` resume recap · 11a ``chat -q`` on a TTY · 11b ``chat -q`` piped.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.cli_tmux._classic import PROMPT, ClassicHome, Pane, assert_no_traceback, last_user, require_tmux
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.platforms("linux"),  # the harness finds and reaps processes through /proc
]

MARKDOWN_REPLY = ("# Heading Two\n\nSome intro text for the markdown check.\n\n- alpha item\n- beta item\n"
                  "- gamma item\n\n```python\ndef hello():\n    return 'hi'\n```\n\nEnd of STEP2 reply.")
# Flagged "shell command via -c/-lc flag" -> approval modal. The seq makes an EARLY tool result bulky,
# so /compress has a middle worth summarizing.
STEP3_CMD = 'bash -c "echo hi && seq 1 2500"'
STEP3_DONE = "STEP3 done: the command printed hi."
STEP4_DONE = "STEP4 done: notes.txt holds the fixture lines."
LONG_REPLY = "STEP8 story: " + " ".join(f"w{i:03d}" for i in range(600)) + " THE-END"
HELLO_REPLY = "STEP11 hello back from the fake model."
TITLE = "Some Title"

EXTRA_CONFIG = """memory:
  memory_enabled: false
  user_profile_enabled: false
  nudge_interval: 0
compression:
  threshold_tokens: 30000
  protect_last_n: 2
  min_tail_user_messages: 1
"""


def _main(record: dict[str, Any]):
    body = record.get("body") or {}
    msgs = body.get("messages") or []
    after_tool = bool(msgs) and msgs[-1].get("role") == "tool"
    user = last_user(body)
    if "STEP2" in user:
        return Text(MARKDOWN_REPLY)
    if "STEP3" in user:
        return Text(STEP3_DONE) if after_tool else ToolCall("terminal", {"command": STEP3_CMD})
    if "STEP4" in user:
        return Text(STEP4_DONE) if after_tool else ToolCall("read_file", {"path": "notes.txt"})
    if "STEP8" in user:
        return Text(LONG_REPLY, chunk_chars=6, delay_per_chunk=0.04)
    if "hello" in user.lower():
        return Text(HELLO_REPLY)
    return Text("ok")


def _aux(record: dict[str, Any]):
    blob = json.dumps((record.get("body") or {}).get("messages") or [])
    return Text("Scripted session title" if "You name chat sessions" in blob else "Fake summary of the earlier conversation.")


def _session_rows(ch: ClassicHome) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{ch.hermes_home / 'state.db'}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT id, title, message_count FROM sessions ORDER BY started_at")]
    finally:
        conn.close()


class Steps:
    """Runs the steps in order, keeping each step's screen; a failed step names itself."""

    def __init__(self, pane: Pane) -> None:
        self.pane = pane
        self.screens: dict[str, str] = {}

    def capture(self, name: str, *, scrollback: bool = False) -> str:
        text = self.pane.scrollback() if scrollback else self.pane.screen()
        self.screens[name] = text
        assert_no_traceback(text, name)
        return text

    def expect(self, name: str, needles: list[str], text: str) -> None:
        missing = [n for n in needles if n not in text]
        assert not missing, (f"step {name}: main showed {missing!r}; screen now:\n{text[-4000:]}\n"
                             f"{self.pane.ch.diagnostics()}")


def test_classic_cli_matches_main_baseline_across_fifteen_steps(tmp_path: Path) -> None:
    require_tmux()
    llm = FakeLLMServer(_main, aux=_aux)
    ch = ClassicHome(tmp_path / "run")
    pane = Pane(ch)
    s = Steps(pane)
    with llm:
        ch.write(llm.base_url, EXTRA_CONFIG)
        (ch.work / "notes.txt").write_text("fixture line one\nfixture line two\nfixture line three\n", encoding="utf-8")
        try:
            pane.launch()
            assert pane.ready(), f"01_startup: no status bar:\n{pane.screen()}"
            s.expect("01_startup", ["Welcome to Hermes Agent!", "Session:", PROMPT], s.capture("01_startup", scrollback=True))

            pane.submit("STEP2 please show me some markdown")
            assert pane.wait(["End of STEP2 reply."], 60), pane.screen()
            pane.settle()
            md = s.capture("02_markdown")
            # Rendered through Rich inside main's reply box, not raw markdown.
            s.expect("02_markdown", ["\u2624 Hermes", "Heading Two", "- alpha item", "def hello():"], md)
            assert "# Heading Two" not in md and "```" not in md, f"02_markdown: raw markdown:\n{md}"

            pane.submit("STEP3 please run echo hi in the terminal")
            assert pane.wait(["Dangerous Command"], 45), f"03a_approval_modal: no modal:\n{pane.screen()}"
            pane.settle(quiet=1.5, timeout=10)
            s.expect("03a_approval_modal", ["Dangerous Command", STEP3_CMD, "1. Allow once", "4. Deny"],
                     s.capture("03a_approval_modal"))
            pane.keys("Enter")  # main's preselected first choice: Allow once
            assert pane.wait([STEP3_DONE], 60), pane.screen()
            pane.settle()
            s.expect("03b_tool_reply", ["\u2192 allowed once", "\u2624 Hermes", STEP3_DONE], s.capture("03b_tool_reply"))

            pane.submit("STEP4 please read notes.txt")
            assert pane.wait([STEP4_DONE], 60), pane.screen()
            pane.settle()
            s.expect("04_read_file", ["read", "notes.txt", STEP4_DONE], s.capture("04_read_file"))

            pane.submit("/help")
            pane.settle(quiet=2.0, timeout=20)
            s.expect("05_help", ["/quit", "Exit the CLI", "/help skills"], s.capture("05_help"))

            pane.submit("/model")
            assert pane.wait(["Model Picker"], 30), f"06a_model: no picker:\n{pane.screen()}"
            pane.settle(quiet=1.5, timeout=10)
            s.expect("06a_model", ["Model Picker", "Current: fake-model"], s.capture("06a_model"))
            pane.keys("Escape")
            pane.settle(quiet=1.0, timeout=10)

            pane.submit(f"/title {TITLE}")
            assert pane.wait([f"Session title set: {TITLE}"], 30), pane.screen()
            s.capture("06b_title")

            pane.submit("/usage")
            assert pane.wait(["Session Token Usage"], 30), pane.screen()
            pane.settle()
            s.expect("07_usage", ["Session Token Usage", "Model:", "fake-model", "API calls:", "Messages:",
                                  "Compressions:"], s.capture("07_usage"))

            pane.submit("STEP8 tell me a very long story")
            assert pane.wait(["w040"], 60), pane.screen()
            pane.keys("C-c")
            # The interrupt round-trips through the gateway; the notice lands when the turn ends.
            interrupted = pane.wait(["Interrupted"], 45)
            pane.settle(quiet=2.0, timeout=30)
            text = s.capture("08_interrupt", scrollback=True)
            assert not pane.dead(), f"08_interrupt: Ctrl+C killed the CLI (main stays at the prompt):\n{text[-3000:]}"
            assert "THE-END" not in text, "08_interrupt: the turn was not interrupted"
            assert interrupted, f"08_interrupt: main showed an 'Interrupted' notice; screen now:\n{text[-4000:]}"

            pane.submit("/compress")
            assert pane.wait(["Compressed:", "No changes from compression"], 90), pane.screen()
            pane.settle(quiet=3.0, timeout=60)
            s.expect("09_compress", ["Compressing", "Compressed:"], s.capture("09_compress"))

            pane.submit("/quit")
            assert pane.wait_dead(30), f"10a_exit_summary: CLI alive 30 s after /quit:\n{pane.screen()}"
            summary = s.capture("10a_exit_summary", scrollback=True).split("/quit")[-1]
            s.expect("10a_exit_summary", ["Resume this session with:", "hermes --resume ", f'hermes -c "{TITLE}"',
                                          "Session:", f"Title:          {TITLE}", "Duration:", "Messages:"], summary)

            pane.launch("-c")
            assert pane.ready(), f"10b_resume_recap: no prompt:\n{pane.scrollback()[-3000:]}"
            recap = s.capture("10b_resume_recap", scrollback=True)
            s.expect("10b_resume_recap", ["Resumed session", TITLE, "Previous Conversation",
                                          "STEP2 please show me some markdown"], recap)
            assert "END OF CONTEXT SUMMARY" not in recap, "10b: the raw compaction summary was dumped"
            pane.submit("/quit")
            pane.wait_dead(30)

            pane.launch("chat", "-q", "hello")
            assert pane.wait([HELLO_REPLY], 120), f"11a_q_tty:\n{pane.screen()}"
            pane.settle(quiet=3.0, timeout=30)
            s.expect("11a_q_tty", ["\u2624 Hermes", HELLO_REPLY, PROMPT], s.capture("11a_q_tty"))
            pane.kill()

            piped = ch.run("chat", "-q", "hello")
            assert_no_traceback(piped.stdout + piped.stderr, "11b_q_pipe")
            assert piped.returncode == 0, piped.stderr[-3000:]
            s.expect("11b_q_pipe", ["Query: hello", HELLO_REPLY, "Resume this session with:", "Session:",
                                    "Messages:"], piped.stdout)
        finally:
            pane.kill()
            ch.reap()

    # Zero spend and one writer: every model call went to the loopback fake, and the session row
    # the gateway wrote carries the title and the conversation the screens showed.
    assert llm.main_requests(), "no request reached the fake provider"
    rows = _session_rows(ch)
    titled = [r for r in rows if r["title"] == TITLE]
    assert titled and titled[0]["message_count"] >= 6, rows
