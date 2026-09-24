"""Anthropic Messages wire conformance: thinking signatures + tool pairing.

Real ``hermes`` processes (``-z``, ``chat -q``, ``chat --resume``, the stdio
``tui_gateway``) talk to the SDK-oracle fake on the production native route.
Each test asserts the NEXT wire request Hermes sends: signed thinking blocks
come back byte-exact (text AND signature), parallel ``tool_use`` ids are each
answered in the very next user message, and every request body validates
against the SDK's ``MessageCreateParams``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import queue
from pathlib import Path

import pytest

from tests.e2e.core.providers._anthropic_helpers import (
    TURN_TIMEOUT,
    Rig,
    assistant_messages,
    blocks,
    dump,
    normalised,
    start_rig,
    thinking_of,
    tool_pairing_problems,
)
from tests.fakes.providers.anthropic_messages import Reply, Text, Thinking, ToolUse

# The guard bypass is only for teardown: a daemonised helper the gateway spawns can be reparented
# to init, and kill_tagged() signals nothing but PIDs carrying this test's unique tag.
pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
              pytest.mark.live_system_guard_bypass]

# Signatures are opaque base64 blobs on the real API; include every base64 symbol plus
# padding so any re-encoding, trimming or normalisation shows up as a byte mismatch.
SIG = {n: f"Eq{n}+/AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/{n * 3}==" for n in ("A", "B", "C", "D")}
THINK = {n: f"Reasoning step {n}: weigh the files.\n  Keep  spacing\tand unicode \u00e9\u2014{n}." for n in SIG}


@pytest.fixture
def rig_factory(tmp_path: Path):
    rigs: list[Rig] = []

    def make(script, **kw) -> Rig:
        rig = start_rig(tmp_path / f"r{len(rigs)}", script, **kw)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.stop()


def _assert_conformant(rig: Rig) -> None:
    assert not rig.srv.schema_errors(), f"request bodies violate the SDK schema: {rig.srv.schema_errors()}"


def test_interleaved_thinking_parallel_tools_three_turns(rig_factory) -> None:
    """3 model turns in one user turn: [thinking, 2 parallel tool_use] -> [thinking, tool_use] ->
    [thinking, text]. Every follow-up request replays the latest signed thinking byte-exact
    ahead of its tool_use blocks and answers every tool_use id in the next user message."""
    rig = rig_factory([])
    for name in ("a", "b", "c"):
        (rig.project / f"{name}.txt").write_text(f"content-{name}", encoding="utf-8")
    rig.srv.push(
        Reply([Thinking(THINK["A"], SIG["A"]),
               ToolUse("read_file", {"path": str(rig.project / "a.txt")}),
               ToolUse("read_file", {"path": str(rig.project / "b.txt")})]),
        Reply([Thinking(THINK["B"], SIG["B"]), ToolUse("read_file", {"path": str(rig.project / "c.txt")})]),
        Reply([Thinking(THINK["C"], SIG["C"]), Text("INTERLEAVED-DONE")]),
    )
    proc = rig.run("-z", "Read a.txt and b.txt together, then c.txt.")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "INTERLEAVED-DONE" in proc.stdout
    mains = rig.srv.main_requests()
    assert len(mains) == 3, [r.get("response") for r in rig.srv.requests]
    _assert_conformant(rig)

    for turn, (req, expected) in enumerate(zip(mains[1:], ("A", "B")), start=2):
        body = req["body"]
        assert body.get("thinking", {}).get("type") in ("enabled", "adaptive"), body.get("thinking")
        latest = assistant_messages(body)[-1]
        kinds = [b["type"] for b in blocks(latest)]
        assert kinds[0] == "thinking", f"request {turn}: latest assistant must lead with thinking: {kinds}"
        assert thinking_of(latest)[0] == (THINK[expected], SIG[expected]), (
            f"request {turn}: signed thinking not replayed byte-exact: {thinking_of(latest)}")
        assert not tool_pairing_problems(body), (tool_pairing_problems(body), dump(body))
    parallel_uses = [b for b in blocks(assistant_messages(mains[1]["body"])[-1]) if b["type"] == "tool_use"]
    assert len(parallel_uses) == 2, "both parallel tool_use blocks must stay in ONE assistant message"
    results = [b for b in blocks(mains[1]["body"]["messages"][-1]) if b["type"] == "tool_result"]
    assert {json.dumps(r["content"]).count("content-a") + json.dumps(r["content"]).count("content-b")
            for r in results} == {1}, "each tool_result must carry its own file's content"


def test_signature_replayed_byte_exact_after_resume(rig_factory) -> None:
    """Turn 1 in one process, turn 2 in a fresh ``chat --resume`` process: the resumed request's
    history is the first process's transcript (same blocks, same signature bytes) plus the
    new user message; nothing is re-encoded by the state.db round trip."""
    rig = rig_factory([
        Reply([Thinking(THINK["A"], SIG["A"]), ToolUse("terminal", {"command": "echo resume-probe"})]),
        Reply([Thinking(THINK["B"], SIG["B"]), Text("FIRST-ANSWER")]),
        Reply([Thinking(THINK["C"], SIG["C"]), Text("SECOND-ANSWER")]),
    ])
    first = rig.run("chat", "-q", "Run the probe command.", "-Q")
    assert first.returncode == 0 and "FIRST-ANSWER" in first.stdout, first.stderr[-2000:]
    (session_id,) = rig.session_ids()
    second = rig.run("chat", "--resume", session_id, "-q", "And now?", "-Q")
    assert second.returncode == 0 and "SECOND-ANSWER" in second.stdout, second.stderr[-2000:]
    _assert_conformant(rig)

    mains = [r["body"] for r in rig.srv.main_requests()]
    assert len(mains) == 3
    # Direct-Anthropic policy: only the LATEST assistant message keeps signed thinking (older
    # signatures are dropped by design), so compare everything else block for block.
    in_process = _unsigned(normalised(mains[1]["messages"]))
    resumed = normalised(mains[2]["messages"])
    assert _unsigned(resumed[:len(in_process)]) == in_process, (
        "resumed history diverges from the in-process transcript:\n"
        f"in-process: {json.dumps(in_process)[-1500:]}\nresumed: {json.dumps(resumed)[-1500:]}")
    final_assistant = resumed[len(in_process)]
    assert final_assistant["role"] == "assistant"
    assert thinking_of(final_assistant) == [(THINK["B"], SIG["B"])], (
        f"turn-1 final thinking lost or altered after resume: {final_assistant}")
    assert resumed[-1]["role"] == "user" and "And now?" in json.dumps(resumed[-1])
    assert not tool_pairing_problems(mains[2])


def _unsigned(messages: list[dict]) -> list[dict]:
    return [{**m, "content": [b for b in m["content"] if b.get("type") != "thinking"]} for m in messages]


def _tui_gateway_two_turns(rig: Rig, prompts: list[str]) -> list[str]:
    from tests.e2e.core.parity._drive_rpc import RpcClient, StreamCapture

    proc = subprocess.Popen([sys.executable, "-m", "tui_gateway.entry"], cwd=rig.project, env=rig.env(),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    cap = StreamCapture().start(proc)
    lock = threading.Lock()

    def send(line: str) -> None:
        with lock:
            assert proc.stdin is not None
            proc.stdin.write(line + "\n")
            proc.stdin.flush()

    answers: list[str] = []
    try:
        rpc = RpcClient(send, cap.stdout_lines)
        rpc.wait_event("gateway.ready", timeout=TURN_TIMEOUT)
        sid = rpc.call("session.create", {}, timeout=TURN_TIMEOUT)["session_id"]
        for n, prompt in enumerate(prompts, start=1):
            rpc.call("prompt.submit", {"session_id": sid, "text": prompt}, timeout=TURN_TIMEOUT)
            done = rpc.wait_event("message.complete", lambda e, n=n: e.get("session_id") == sid
                                  and sum(1 for x in rpc.events if x.get("type") == "message.complete") >= n)
            answers.append(str((done.get("payload") or {}).get("text")))
    except (AssertionError, queue.Empty) as exc:
        raise AssertionError(f"{exc}\ntui_gateway stderr: {cap.stderr[-2000:]}") from exc
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return answers


def test_tui_gateway_replays_signed_thinking_across_turns(rig_factory) -> None:
    """The Ink TUI / Desktop backend path: two prompts in one live session over stdio JSON-RPC.
    The second prompt's request replays turn 1's signed thinking byte-exact."""
    rig = rig_factory([
        Reply([Thinking(THINK["A"], SIG["A"]), Text("GW-ONE")]),
        Reply([Thinking(THINK["D"], SIG["D"]), Text("GW-TWO")]),
    ])
    answers = _tui_gateway_two_turns(rig, ["first question", "second question"])
    assert "GW-ONE" in answers[0] and "GW-TWO" in answers[1], answers
    _assert_conformant(rig)
    mains = [r["body"] for r in rig.srv.main_requests()]
    assert len(mains) == 2
    prior = assistant_messages(mains[1])
    assert prior and thinking_of(prior[-1]) == [(THINK["A"], SIG["A"])], dump(mains[1])
    assert "GW-ONE" in json.dumps(prior[-1])
