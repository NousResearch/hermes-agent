"""Copilot ACP wire conformance, part 3: a long deep-reasoning turn must stream progress while in flight.

The fake agent (``tests/fakes/providers/copilot_acp.py``) streams an early message chunk and then
``agent_thought_chunk`` updates for ~4 s before its final chunk and the ``session/prompt`` result, the
shape of a deep-reasoning tier that thinks for minutes. The contract: chunks the agent has already
emitted reach the user's surface BEFORE the turn completes, so a long turn never looks hung.

Two real surfaces, each timestamped against the fake's own clock (same host):

* the Desktop/TUI event stream (``python -m tui_gateway.entry`` over stdio): ``reasoning.delta`` /
  ``message.delta`` events carrying the agent's text (#120550);
* ACP composition — Hermes itself served as an ACP agent (``hermes acp``) on top of the copilot-acp
  provider: the outer client must get ``session/update`` chunks before the inner turn ends (#101507).

Both are red on main (the ACP client buffers the whole response, then replays it as a stream), so
both are strict xfails that raise :class:`KnownSymptom` only for "no progress before the result".
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.providers._native_helpers import TURN_TIMEOUT, KnownSymptom, NativeHome, make_home
from tests.fakes.providers import copilot_acp as acp

pytest.importorskip("acp.schema", reason="the fake validates against the agent-client-protocol package (acp extra)")

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX launcher script + /proc pid checks"),
    pytest.mark.live_system_guard_bypass,
]

HEAD, ANSWER, THOUGHT = "DEEP-HEAD ", "DEEP-ANSWER-DONE", "DEEP-THOUGHT"
THOUGHT_STEPS, STEP_S = 8, 0.5
MIN_SPAN_S = THOUGHT_STEPS * STEP_S * 0.75  # vacuity: the agent really spent seconds before its result
MARKERS = (HEAD.strip(), THOUGHT, ANSWER)


KNOWN: dict[str, str] = {
    "tui_stream": "#120550 copilot-acp buffers the whole turn: no reasoning/message delta reaches the UI in flight",
    "nested_acp": "#101507 hermes acp over copilot-acp forwards inner ACP chunks only after the inner turn ends",
}


def _deep_turn() -> list[dict[str, Any]]:
    thoughts = [acp.thought(f"{THOUGHT}-{i} ", delay=STEP_S) for i in range(THOUGHT_STEPS)]
    return [acp.message(HEAD, delay=0.5), *thoughts, acp.message(ANSWER, delay=STEP_S), acp.result()]


@dataclass
class Observed:
    fake: acp.AcpFake
    received: list[tuple[float, dict[str, Any]]] = field(default_factory=list)  # (wall clock, message)
    final_text: str = ""
    returncode: int | None = None
    stderr: str = ""


def _pump(proc: subprocess.Popen, sink: "queue.Queue[tuple[float, dict[str, Any]] | None]") -> None:
    for line in proc.stdout:  # type: ignore[union-attr]
        try:
            sink.put((time.time(), json.loads(line)))
        except json.JSONDecodeError:
            continue
    sink.put(None)


class _LineRpc:
    """Newline-delimited JSON-RPC 2.0 over a child's stdio; keeps every inbound message with its arrival time."""

    def __init__(self, proc: subprocess.Popen, obs: Observed, on_request: Callable[[dict[str, Any]], Any]):
        self.proc, self.obs, self.on_request = proc, obs, on_request
        self.inbox: queue.Queue[tuple[float, dict[str, Any]] | None] = queue.Queue()
        self.next_id = 0
        threading.Thread(target=_pump, args=(proc, self.inbox), daemon=True).start()

    def send(self, msg: dict[str, Any]) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]

    def until(self, pred: Callable[[dict[str, Any]], bool], timeout: float = TURN_TIMEOUT) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                raise AssertionError(f"child closed stdout (rc={self.proc.poll()})")
            self.obs.received.append(item)
            msg = item[1]
            if "method" in msg and "id" in msg:
                self.send({"jsonrpc": "2.0", "id": msg["id"], **self.on_request(msg)})
            if pred(msg):
                return msg
        raise AssertionError(f"timed out after {timeout}s")

    def call(self, method: str, params: dict[str, Any], timeout: float = TURN_TIMEOUT) -> Any:
        self.next_id += 1
        rid = self.next_id
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        msg = self.until(lambda m: m.get("id") == rid and "method" not in m, timeout)
        assert "error" not in msg, f"{method} failed: {msg['error']}"
        return msg.get("result")


def _refuse(msg: dict[str, Any]) -> dict[str, Any]:
    return {"error": {"code": -32601, "message": f"test client does not implement {msg['method']}"}}


def _run_child(argv: list[str], nh: NativeHome, fake: acp.AcpFake, drive: Callable[[_LineRpc], str]) -> Observed:
    obs = Observed(fake)
    stderr_path = nh.root / "child_stderr.log"
    with open(stderr_path, "w", encoding="utf-8") as err:
        proc = subprocess.Popen(argv, cwd=nh.project, env=nh.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err, text=True, bufsize=1)
        try:
            obs.final_text = drive(_LineRpc(proc, obs, _refuse))
        finally:
            proc.stdin.close()  # type: ignore[union-attr]
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    obs.returncode, obs.stderr = proc.returncode, stderr_path.read_text(encoding="utf-8")[-3000:]
    return obs


def _home(root: Path) -> tuple[NativeHome, acp.AcpFake]:
    fake = acp.AcpFake(root / "acp", [_deep_turn()])
    return make_home(root, {"provider": "copilot-acp", "default": "copilot-acp"}, env_file=fake.env()), fake


def _tui_gateway(root: Path) -> Observed:
    nh, fake = _home(root)

    def drive(rpc: _LineRpc) -> str:
        def event(etype: str) -> Callable[[dict[str, Any]], bool]:
            return lambda m: m.get("method") == "event" and (m.get("params") or {}).get("type") == etype

        rpc.until(event("gateway.ready"), 120)
        sid = rpc.call("session.create", {})["session_id"]
        rpc.call("prompt.submit", {"session_id": sid, "text": "Think deeply, then answer."})
        done = rpc.until(event("message.complete"))
        return str((done["params"].get("payload") or {}).get("text") or "")

    return _run_child([sys.executable, "-m", "tui_gateway.entry"], nh, fake, drive)


def _nested_acp(root: Path) -> Observed:
    nh, fake = _home(root)

    def drive(rpc: _LineRpc) -> str:
        rpc.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "e2e", "version": "1"}}, 120)
        sid = rpc.call("session/new", {"cwd": str(nh.project), "mcpServers": []}, 120)["sessionId"]
        result = rpc.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "Think deeply, then answer."}]})
        assert result.get("stopReason") == "end_turn", result
        return "".join(_chunk_text(m) for _, m in rpc.obs.received)

    return _run_child([sys.executable, "-m", "hermes_cli.main", "acp"], nh, fake, drive)


def _chunk_text(msg: dict[str, Any]) -> str:
    params = msg.get("params") or {}
    if msg.get("method") == "session/update":  # outer ACP surface
        content = (params.get("update") or {}).get("content") or {}
        return str(content.get("text") or "") if isinstance(content, dict) else ""
    if msg.get("method") == "event" and params.get("type") in ("message.delta", "reasoning.delta"):  # TUI surface
        return str((params.get("payload") or {}).get("text") or "")
    return ""


SURFACES: dict[str, Callable[[Path], Observed]] = {"tui_stream": _tui_gateway, "nested_acp": _nested_acp}


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("copilot_acp_stream")
    with ThreadPoolExecutor(max_workers=len(SURFACES)) as pool:
        futures = {name: pool.submit(fn, base / name) for name, fn in SURFACES.items()}
        done = {name: fut.result() for name, fut in futures.items()}
    yield done
    for obs in done.values():
        for pid in obs.fake.pids():
            if os.path.exists(f"/proc/{pid}"):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _agent_timeline(fake: acp.AcpFake) -> tuple[float, float]:
    """(first chunk sent, prompt result sent) on the fake agent's clock, for the single main turn."""
    outs = [r for r in fake.records() if r["dir"] == "out" and r["turn"] == 0]
    first = min(r["t"] for r in outs if (r["msg"] or {}).get("method") == "session/update")
    done = min(r["t"] for r in outs if "stopReason" in ((r["msg"] or {}).get("result") or {}))
    return first, done


@pytest.mark.parametrize("surface", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN[name]))
    for name in SURFACES
])
def test_long_reasoning_turn_streams_progress_before_it_completes(observed, surface):
    obs = observed[surface]
    assert obs.fake.invalid() == [], f"requests rejected by the ACP schema: {obs.fake.invalid()}"
    assert ANSWER in obs.final_text, f"turn did not complete with the agent's answer: {obs.final_text!r}\n{obs.stderr}"
    first, done = _agent_timeline(obs.fake)
    assert done - first >= MIN_SPAN_S, f"vacuity: the agent streamed for only {done - first:.2f}s"
    seen_at = _first_marker_arrival(obs.received)
    if seen_at is None or seen_at >= done - 0.3:
        late = [round(t - done, 2) for t, m in obs.received if _chunk_text(m)]
        raise KnownSymptom(f"{surface}: no agent chunk reached the surface during the {done - first:.1f}s turn; "
                           f"chunk arrivals relative to the result: {late}")
    assert seen_at >= first, "a chunk cannot arrive before the agent sent it"


def _first_marker_arrival(received: list[tuple[float, dict[str, Any]]]) -> float | None:
    """Arrival time of the delta that first completes one of the agent's markers (surfaces may re-split
    chunks, so match on the accumulated text, not per delta)."""
    buffer = ""
    for t, msg in received:
        buffer += _chunk_text(msg)
        if any(k in buffer for k in MARKERS):
            return t
    return None
