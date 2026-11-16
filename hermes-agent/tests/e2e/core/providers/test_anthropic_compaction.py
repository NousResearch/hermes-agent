"""Anthropic Messages wire conformance: compaction in a session full of signed thinking.

Real ``hermes chat -q`` / ``--resume`` turns run interleaved [thinking, tool_use] rounds
whose tool results are large, with the compaction trigger pinned low (documented
``compression.threshold_tokens``) so older history is summarised repeatedly. For EVERY request
the fake saw: tool_use/tool_result pairing holds (no orphan tool_result after a
summary cut), the latest assistant message still leads with ITS signed thinking
byte-exact, and the body validates against the SDK schema.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from tests.e2e.core.providers._anthropic_helpers import (
    assistant_messages,
    blocks,
    dump,
    start_rig,
    thinking_of,
    tool_pairing_problems,
)
from tests.fakes.providers.anthropic_messages import Reply, Text, Thinking, ToolUse

pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
              pytest.mark.live_system_guard_bypass]

TURNS = 6
SUMMARY_TOKEN = "E2E-ANTHROPIC-SUMMARY-7f3a"


def _sig(i: int) -> str:
    return f"EqRound{i:02d}+/sigBytes{'x' * i}=="


def _think(i: int) -> str:
    return f"Round {i}: read the next chunk."


def _last_is_tool_result(body: dict) -> bool:
    last = (body.get("messages") or [{}])[-1]
    return last.get("role") == "user" and any(b.get("type") == "tool_result" for b in blocks(last))


@pytest.fixture
def rig(tmp_path: Path):
    lock = threading.Lock()
    state = {"calls": 0, "answers": 0}
    files: list[Path] = []

    def respond(record):
        body = record["body"]
        tokens = len(json.dumps(body)) // 4
        with lock:
            if _last_is_tool_result(body):
                state["answers"] += 1
                n = state["answers"]
                return Reply([Thinking(f"Answer {n}.", _sig(50 + n)), Text(f"TURN-{n}-DONE")], input_tokens=tokens)
            state["calls"] += 1
            i = state["calls"]
        return Reply([Thinking(_think(i), _sig(i)), ToolUse("read_file", {"path": str(files[i - 1])})],
                     input_tokens=tokens)

    def aux(record):
        return Reply([Text(f"{SUMMARY_TOKEN}: the assistant read chunks and is still reading.")],
                     input_tokens=len(json.dumps(record["body"])) // 4)

    r = start_rig(tmp_path / "r", respond, aux=aux, config={
        "model": {"context_length": 64000},
        "compression": {"threshold_tokens": 20000, "protect_last_n": 4},
    })
    for i in range(1, TURNS + 1):
        path = r.project / f"chunk{i}.txt"
        path.write_text("\n".join(f"chunk {i} line {n}: " + "lorem ipsum dolor sit amet " * 6
                                  for n in range(200)), encoding="utf-8")
        files.append(path)
    yield r
    r.stop()


def _drive_turns(rig) -> None:
    first = rig.run("chat", "-q", "Read chunk 1.", "-Q")
    assert first.returncode == 0 and "TURN-1-DONE" in first.stdout, first.stderr[-2000:]
    (session_id,) = rig.session_ids()
    for n in range(2, TURNS + 1):
        proc = rig.run("chat", "--resume", session_id, "-q", f"Read chunk {n}.", "-Q")
        assert proc.returncode == 0 and f"TURN-{n}-DONE" in proc.stdout, (n, proc.stderr[-2000:])


def test_compaction_keeps_pairing_and_signatures(rig) -> None:
    """Six resumed turns of [thinking, tool_use] -> [thinking, text] with ~9K-token tool results
    against a 20K trigger: history is summarised repeatedly. No request may carry an orphan
    tool_result, and every in-flight tool turn still leads with ITS signed thinking byte-exact."""
    _drive_turns(rig)
    mains = [r["body"] for r in rig.srv.main_requests()]
    assert len(mains) == 2 * TURNS, f"expected {2 * TURNS} main requests, saw {len(mains)}"
    summarised = [i for i, b in enumerate(mains) if SUMMARY_TOKEN in json.dumps(b.get("messages"))]
    assert summarised, (
        f"precondition: the trigger never fired, nothing was compacted (last request "
        f"{len(json.dumps(mains[-1])) // 4} est. tokens)\n{rig.log_tail(pattern='compress', chars=6000)}")
    assert not rig.srv.schema_errors(), rig.srv.schema_errors()

    checked = 0
    for n, body in enumerate(mains, start=1):
        assert not tool_pairing_problems(body), (f"request {n}", tool_pairing_problems(body), dump(body))
        if not _last_is_tool_result(body):
            continue
        latest = assistant_messages(body)[-1]
        uses = [b for b in blocks(latest) if b.get("type") == "tool_use"]
        assert uses, f"request {n}: the in-flight tool_use turn was dropped: {dump(body)}"
        i = int(Path(uses[0]["input"]["path"]).stem.removeprefix("chunk"))
        assert blocks(latest)[0].get("type") == "thinking", f"request {n}: signed thinking dropped {dump(body)}"
        assert thinking_of(latest)[0] == (_think(i), _sig(i)), (
            f"request {n}: round {i} thinking/signature altered across compaction: {thinking_of(latest)}")
        checked += n > summarised[0]
    assert checked, "no in-flight tool turn was sent after the first summary; the post-compaction check is vacuous"
