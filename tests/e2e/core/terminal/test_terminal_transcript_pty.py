"""C2 on terminal surfaces: what the classic CLI and the Ink TUI finally show is the transcript,
exactly once, unmangled, and matches what was persisted.

Real chain: a real ``hermes chat --cli`` / ``hermes --tui`` process (the TUI spawns its real Node
frontend and real ``tui_gateway`` child) on a real PTY, a real AIAgent + SessionDB on disk, and the
recording fake OpenAI-compatible provider streaming scripted multi-chunk replies, a tool-call turn
and a reasoning turn. The byte stream is rendered through a VT emulator so assertions are made on
the grid a user would see (plus the scrollback they could scroll to), not on repaint history.

Invariants, checked on the settled final frame of every scenario:
  * every scripted assistant reply is on screen exactly once and verbatim (== the scripted text,
    so streamed deltas were neither double-appended, dropped nor reordered);
  * every user prompt is echoed exactly once, and prompts/replies appear in conversation order;
  * a tool call's command is rendered once; reasoning is never rendered twice;
  * rendered == persisted: every user/assistant row in state.db is on screen exactly once, and the
    persisted assistant text equals the scripted stream;
  * ``/exit`` exits 0 within a bounded time and leaves no process behind in the PTY session.

The ``resize`` scenarios change the terminal width while a long reply is streaming (SIGWINCH via
the PTY), the recurring "history re-appended on resize" shape. ``resize_scrollback`` replays it on
a normal 24-row classic-CLI terminal, where earlier turns already sit in scrollback: that case is a
strict xfail for the open live bug #95375 and turns red (XPASS) once it is fixed.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.e2e.core.terminal._pty import REPO_ROOT, PtyHermes, canon
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real PTY + /proc session scan"),
    # Cleanup SIGKILLs only processes whose session id is the PTY child this test spawned; a
    # grandchild reparented to init after its parent died is outside the pid subtree but still ours.
    pytest.mark.live_system_guard_bypass,
]

TITLE = "Scripted session title"


class DuplicateRender(AssertionError):
    """A message rendered more (or fewer) times than once: the C2 failure itself, as opposed to a
    harness timeout, so an xfail for a known duplication bug can never swallow a hang."""


def _long_reply(tag: str, words: int) -> str:
    return f"{tag} begins " + " ".join(f"{tag.lower()}{i:03d}" for i in range(words)) + f" {tag}-END"


@dataclass
class Turn:
    prompt: str
    responses: list  # scripted provider responses consumed by this user turn
    reply: str  # final assistant text the user must see
    resize_to: tuple[int, int] | None = None  # (rows, cols) applied mid-stream
    tool_command: str | None = None
    reasoning: str | None = None


@dataclass
class Scenario:
    name: str
    turns: list[Turn] = field(default_factory=list)
    rows: int = 80  # tall enough that the whole conversation stays on screen / in the TUI viewport


def _scenarios(surface: str) -> dict[str, Scenario]:
    long_a = _long_reply("ALPHA", 140)
    long_r = _long_reply("RESIZE", 120)
    tool_cmd = "echo TOOLMARK-7f3a"
    # Classic CLI resizes are widen-only: a real terminal truncates or reflows narrowed rows
    # (emulator-specific), which is not the invariant under test. Two of them, because the CLI
    # seeds its width baseline on the first SIGWINCH and only repaints on an observed change.
    # The TUI repaints its own fullscreen viewport, so it gets both directions.
    widths = [(116,), (132,)] if surface == "cli" else [(128,), (84,)]

    def resize_turns(tag: str, rows: int) -> list[Turn]:
        return [
            Turn(f"warm up {tag} delta-q1", [Text("DELTA short warmup reply DELTA-END")],
                 "DELTA short warmup reply DELTA-END"),
            *[
                Turn(f"stream while resizing {tag} epsilon-q{i + 2}",
                     [Text(long_r.replace("RESIZE", f"RESIZE{i}"), chunk_chars=5, delay_per_chunk=0.012)],
                     long_r.replace("RESIZE", f"RESIZE{i}"), resize_to=(rows, cols))
                for i, (cols,) in enumerate(widths)
            ],
            Turn(f"after the resize {tag} omega-q9", [Text("OMEGA settled reply OMEGA-END")],
                 "OMEGA settled reply OMEGA-END"),
        ]

    return {
        "turns": Scenario("turns", [
            Turn("first question alpha-q1", [Text(long_a, chunk_chars=7, delay_per_chunk=0.002)], long_a),
            Turn("please run the tool beta-q2",
                 [ToolCall("terminal", {"command": tool_cmd}), Text("BETA after the tool BETA-END")],
                 "BETA after the tool BETA-END", tool_command=tool_cmd),
            Turn("think then answer gamma-q3",
                 [Text("GAMMA answered with reasoning GAMMA-END",
                       reasoning="scripted reasoning about gamma zeta-reason")],
                 "GAMMA answered with reasoning GAMMA-END", reasoning="scripted reasoning about gamma zeta-reason"),
        ]),
        "resize": Scenario("resize", resize_turns("tall", 80)),
        # A normal 24-row terminal: by the time the width changes, earlier turns have already
        # scrolled into the terminal's scrollback, where a redraw must not print them again.
        "resize_scrollback": Scenario("resize_scrollback", resize_turns("short", 24), rows=24),
    }


def _tui_available() -> bool:
    return shutil.which("node") is not None and (REPO_ROOT / "ui-tui" / "dist" / "entry.js").is_file()


SURFACES = {
    "cli": ["chat", "--cli", "--yolo"],
    "tui": ["--tui", "--yolo"],
}


MATRIX = [
    ("cli", "turns"),
    ("cli", "resize"),
    pytest.param("cli", "resize_scrollback", marks=pytest.mark.xfail(
        strict=True, raises=DuplicateRender,
        reason="LIVE BUG #95375 (open): a width-changing resize clears only the visible screen "
               "(CSI 2J) and replays the last 200 output lines, so turns already in scrollback "
               "are printed again. strict: this XPASSes (and fails) once it is fixed.")),
    ("tui", "turns"),
    ("tui", "resize"),
]


@pytest.mark.parametrize(("surface", "scenario"), MATRIX)
def test_terminal_transcript_integrity(surface: str, scenario: str, tmp_path: Path) -> None:
    if surface == "tui" and not _tui_available():
        if os.environ.get("HERMES_E2E_REQUIRE_TUI") == "1":
            pytest.fail("ui-tui/dist/entry.js or node missing but HERMES_E2E_REQUIRE_TUI=1")
        pytest.skip("Ink TUI not built (cd ui-tui && npm run build) or node missing")

    spec = _scenarios(surface)[scenario]
    script = [r for turn in spec.turns for r in turn.responses]
    rows, cols = spec.rows, 100
    with FakeLLMServer(script, aux=lambda _req: Text(TITLE)) as llm:
        term = PtyHermes(tmp_path, SURFACES[surface], llm, rows=rows, cols=cols)
        try:
            term.wait_ready()
            expected_main = 0
            for done, turn in enumerate(spec.turns, start=1):
                term.submit(turn.prompt)
                expected_main += len(turn.responses)
                if turn.resize_to:
                    # Resize once the stream is visibly under way, then keep streaming.
                    first_words = canon(turn.reply)[:24]
                    term.wait_for_text(first_words)
                    term.resize(*turn.resize_to)
                llm.wait_for_requests(expected_main, timeout=60)
                term.wait_turns_persisted(done)
                term.wait_quiet(1.0, timeout=60)

            final = term.lines()
            text = canon("".join(final))
            dump = "\n".join(final[-120:])

            positions = []
            for turn in spec.turns:
                n_reply = text.count(canon(turn.reply))
                if n_reply != 1:
                    raise DuplicateRender(
                        f"[{surface}/{scenario}] assistant reply rendered {n_reply}x (want exactly 1, verbatim): "
                        f"{turn.reply[:60]!r}\n{dump}")
                n_prompt = text.count(canon(turn.prompt))
                if n_prompt != 1:
                    raise DuplicateRender(
                        f"[{surface}/{scenario}] user prompt echoed {n_prompt}x (want 1): {turn.prompt!r}\n{dump}")
                positions += [text.index(canon(turn.prompt)), text.index(canon(turn.reply))]
                if turn.tool_command:
                    n_cmd = text.count(canon(turn.tool_command))
                    assert n_cmd == 1, f"[{surface}/{scenario}] tool call rendered {n_cmd}x\n{dump}"
                if turn.reasoning:
                    n_reason = text.count(canon(turn.reasoning))
                    assert n_reason <= 1, f"[{surface}/{scenario}] reasoning rendered {n_reason}x\n{dump}"
            assert positions == sorted(positions), (
                f"[{surface}/{scenario}] transcript out of conversation order\n{dump}")

            code = term.exit()
            assert code == 0, f"[{surface}/{scenario}] /exit returned {code}\n{term.dump()}"
            leftovers = term.leftover_processes()
            assert not leftovers, f"[{surface}/{scenario}] processes left after /exit: {leftovers}"

            persisted = term.persisted_messages()
            sessions = {s for s, _r, _c in persisted}
            assert len(sessions) == 1, f"turns split across sessions: {sessions}"
            users = [c for _s, r, c in persisted if r == "user"]
            assistants = [c for _s, r, c in persisted if r == "assistant" and c.strip()]
            assert users == [t.prompt for t in spec.turns], f"persisted user rows {users!r}"
            assert assistants == [t.reply for t in spec.turns], "persisted assistant text != scripted stream"
            for content in users + assistants:
                assert text.count(canon(content)) == 1, (
                    f"[{surface}/{scenario}] persisted row not rendered exactly once: {content[:60]!r}")
        finally:
            term.close()
