"""C8 chaos: agent-turn liveness through the REAL tui_gateway JSON-RPC subprocess.

``python -m tui_gateway.entry`` is the exact process the Ink TUI and Desktop drive over
stdio. Only the LLM provider is faked (``tests/fakes/fake_llm_provider``); the agent,
tools, SessionDB and the RPC dispatcher are the shipped code. Each scenario starts its
own gateway (so process hygiene is judged per fault), opens a session, submits a turn
whose provider or tool misbehaves FOREVER, and then checks the same invariants:

* the turn ends with a terminal ``message.complete`` within TURN_DEADLINE_S (an
  interrupt ends it within INTERRUPT_DEADLINE_S even though every configured timeout is
  LONG_TIMEOUT_S) and the session then settles (``session.info`` running=false);
* provider main calls stay <= MAX_PROVIDER_CALLS_PER_FAILED_TURN (no retry storm);
* the dispatcher keeps answering cheap RPCs DURING the wedged turn (heartbeat);
* the same session accepts a new prompt (``status: streaming``, not queued behind a
  leaked busy flag) and answers it, with the faulted turn still in its history;
* every tool_call has its own result, in state.db and in what is sent back to the model
  (8-way parallel batch: each id carries ITS token, #93251);
* hung tool processes are reaped, EOF on stdin makes the gateway exit by itself, no
  process carrying the scenario tag survives, and state.db passes integrity_check.

Exit scenarios drop the client (stdin EOF) or SIGTERM the gateway WHILE the turn is wedged:
it must exit within EXIT_TIMEOUT_S, take the hung request/tool tree with it, and leave no
unanswered tool_call in state.db.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.chaos._helpers import (
    INTERRUPT_DEADLINE_S,
    LONG_TIMEOUT_S,
    MAX_PROVIDER_CALLS_PER_FAILED_TURN,
    PLAIN_MODEL,
    PROCESS_REAP_S,
    REASONING_MODEL,
    TOOL_TIMEOUT_S,
    TURN_DEADLINE_S,
    describe_pids,
    hermetic_env,
    integrity_ok,
    kill_tagged,
    new_tag,
    persisted_messages,
    tagged_pids,
    tool_results_by_id,
    unanswered_tool_calls,
    wait_no_tagged,
    write_chaos_home,
)
from tests.e2e.core.chaos._tui_rpc import (
    Heartbeat,
    TuiGatewayProcess,
    env_for_gateway,
    gateway_cwd,
)
from tests.fakes.fake_llm_provider import (
    DropMidStream,
    Error,
    FakeLLMServer,
    Hang,
    StallMidStream,
    Text,
    ToolCall,
)

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="orphan scan reads /proc; POSIX stdio gateway"),
]

# The healthy dispatcher answers ping/session.status in ~1 ms; a dispatcher blocked by the
# turn answers only when the fault ends (>= 10 s here, 600 s for interrupt scenarios).
HEARTBEAT_MAX_S = 3.0
HEARTBEAT_INTERVAL_S = 0.15
MIN_HEARTBEATS = 5
READY_TIMEOUT_S = 60.0
SETTLE_TIMEOUT_S = 30.0
EXIT_TIMEOUT_S = 30.0
PARALLEL_CALLS = 8
HUNG_COMMAND = "sleep 3600"
# The first turn pays the cold agent build (imports, tool registry): setup, not the
# invariant under test, so it gets a budget sized for a box that is loaded 3x over.
WARMUP_DEADLINE_S = 300.0

_MARK = re.compile(r"\[\[chaos:(?P<fault>[a-z0-9_]+):(?P<nonce>[0-9a-f]+)\]\]")
_PROBE = re.compile(r"\[\[probe:(?P<nonce>[0-9a-f]+)\]\]")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _deciding_user_text(messages: list[dict[str, Any]]) -> str:
    """The newest user message carrying a scenario/probe marker. Synthetic user turns the
    agent adds itself (stream-continuation nudges after a partial reply) carry neither, so a
    fault keeps applying to the whole turn — "forever" means forever."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _text_of(msg.get("content"))
            if _PROBE.search(text) or _MARK.search(text):
                return text
    return ""


def _token(nonce: str, i: int) -> str:
    return f"tok{nonce}n{i}"


# ── scripted provider: one module-wide server, faults keyed by the prompt marker ────────


def _tools_then_text(first: Callable[[str], ToolCall]) -> Callable[[list, str], Any]:
    def respond(messages: list[dict[str, Any]], nonce: str):
        if messages and messages[-1].get("role") == "tool":
            return Text(f"tools settled {nonce}")
        return first(nonce)
    return respond


def _parallel_batch(nonce: str) -> ToolCall:
    calls = [("terminal", {"command": f"echo {_token(nonce, i)}"}) for i in range(PARALLEL_CALLS)]
    return ToolCall(calls[0][0], calls[0][1], parallel=calls[1:])


FAULTS: dict[str, Callable[[list, str], Any]] = {
    "hang": lambda _m, _n: Hang(),
    "stall": lambda _m, _n: StallMidStream(),
    "err500": lambda _m, _n: Error(500),
    "err429": lambda _m, _n: Error(429),
    "drop": lambda _m, _n: DropMidStream(),
    "parallel": _tools_then_text(_parallel_batch),
    "toolhang": _tools_then_text(
        lambda _n: ToolCall("terminal", {"command": HUNG_COMMAND, "timeout": TOOL_TIMEOUT_S})),
    "toolhang_long": _tools_then_text(
        lambda _n: ToolCall("terminal", {"command": HUNG_COMMAND, "timeout": LONG_TIMEOUT_S})),
}


def responder(record: dict[str, Any]):
    messages = record["body"].get("messages") or []
    deciding = _deciding_user_text(messages)
    if probe := _PROBE.search(deciding):
        return Text(f"alive {probe['nonce']}")
    if mark := _MARK.search(deciding):
        return FAULTS[mark["fault"]](messages, mark["nonce"])
    return Text("unscripted")


def _main_requests_for(srv: FakeLLMServer, needle: str) -> list[dict[str, Any]]:
    return [
        r["body"] for r in list(srv.requests)
        if r["kind"] == "main" and needle in _deciding_user_text(r["body"].get("messages") or [])
    ]


# ── scenario matrix ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario:
    id: str
    fault: str
    model: str = PLAIN_MODEL
    long_timeouts: bool = False
    # Only the stale-stream detector may end the turn (#115024): the transport timeout is long.
    long_request_timeout: bool = False
    wait_for: str | None = None  # act once the fault is live: "provider" | "tool"
    action: str | None = None  # "interrupt" | "eof" | "sigterm"
    reaps_hung_tool: bool = False


SCENARIOS = [
    Scenario("provider_hang", "hang"),
    # Reasoning slug + long transport timeout: only the explicit stale_timeout_seconds can end
    # the stalled stream, and it must win over the reasoning-model floor.
    Scenario("provider_stall_reasoning_model", "stall", model=REASONING_MODEL, long_request_timeout=True),
    Scenario("provider_500_forever", "err500"),
    Scenario("provider_429_forever", "err429"),
    Scenario("provider_drop_mid_stream", "drop"),
    Scenario("parallel_8_terminal_calls", "parallel"),
    Scenario("tool_hang_past_timeout", "toolhang", reaps_hung_tool=True),
    Scenario("interrupt_during_provider_hang", "hang", long_timeouts=True, wait_for="provider", action="interrupt"),
    Scenario("interrupt_during_hung_tool", "toolhang_long", long_timeouts=True,
             wait_for="tool", action="interrupt", reaps_hung_tool=True),
    # The parent (TUI/Desktop) goes away while the turn is wedged: the gateway must still
    # leave on its own and take the hung request / tool process with it.
    Scenario("stdin_eof_during_provider_hang", "hang", long_timeouts=True, wait_for="provider", action="eof"),
    Scenario("stdin_eof_during_hung_tool", "toolhang_long", long_timeouts=True, wait_for="tool", action="eof"),
    Scenario("sigterm_during_hung_tool", "toolhang_long", long_timeouts=True, wait_for="tool", action="sigterm"),
    Scenario("sigterm_during_provider_hang", "hang", long_timeouts=True, wait_for="provider", action="sigterm"),
]
EXIT_ACTIONS = ("eof", "sigterm")


def _is(ev: dict[str, Any], kind: str, sid: str) -> bool:
    return ev.get("type") == kind and ev.get("session_id") == sid


def _hung_tool_pids(tag: str) -> list[int]:
    out = []
    for pid in tagged_pids(tag):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if HUNG_COMMAND in cmd:
            out.append(pid)
    return out


def _wait_until(pred: Callable[[], object], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return bool(pred())


def _assert_parallel_results(messages: list[dict[str, Any]], nonce: str, where: str) -> None:
    results = tool_results_by_id(messages)
    seen = 0
    for msg in messages:
        calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not calls:
            continue
        for call in json.loads(calls) if isinstance(calls, str) else calls:
            args = json.loads(call["function"]["arguments"])
            token = args["command"].split()[-1]
            assert token.startswith(f"tok{nonce}"), (where, args)
            body = results.get(call["id"], "")
            assert token in body and "Result unavailable" not in body, (
                f"{where}: tool_call {call['id']} ({token}) got {body[:200]!r}")
            seen += 1
    assert seen == PARALLEL_CALLS, f"{where}: {seen} tool calls, expected {PARALLEL_CALLS}"


def _wait_settled(gw: TuiGatewayProcess, sid: str, done: dict[str, Any], what: str) -> None:
    """``message.complete`` is emitted BEFORE the worker clears ``running``; the settled
    ``session.info`` (running=false) that follows is the idle signal a client waits for."""
    settled = gw.wait_event(
        lambda e: _is(e, "session.info", sid) and (e.get("payload") or {}).get("running") is False,
        timeout=SETTLE_TIMEOUT_S, start=done["_i"] + 1)
    assert settled is not None, f"{what}: session never settled (running stays true){gw.tail()}"


def _answered_turn(gw: TuiGatewayProcess, sid: str, nonce: str, what: str,
                   deadline: float = TURN_DEADLINE_S) -> float:
    """Submit a probe prompt to an idle session; it must start at once and be answered."""
    start = gw.event_count()
    t0 = time.monotonic()
    reply = gw.call("prompt.submit", {"session_id": sid, "text": f"are you there [[probe:{nonce}]]"})
    assert reply.get("status") == "streaming", f"{what}: session busy instead of starting the turn: {reply}"
    done = gw.wait_event(lambda e: _is(e, "message.complete", sid), timeout=deadline, start=start)
    assert done is not None, f"{what}: prompt never completed within {deadline}s{gw.tail()}"
    payload = done.get("payload") or {}
    assert payload.get("status") == "complete" and f"alive {nonce}" in _text_of(payload.get("text")), (
        f"{what}: {payload}")
    _wait_settled(gw, sid, done, what)
    return time.monotonic() - t0


def _assert_heartbeat(scn: Scenario, hb: Heartbeat, stats: dict[str, Any], turn_s: float) -> None:
    assert not hb.failures, f"{scn.id}: heartbeat RPC failed during the turn: {hb.failures}"
    stats["heartbeat_max_s"] = round(max(hb.latencies, default=0.0), 3)
    stats["heartbeats"] = len(hb.latencies)
    assert stats["heartbeat_max_s"] < HEARTBEAT_MAX_S, (
        f"{scn.id}: dispatcher blocked {stats['heartbeat_max_s']}s during the wedged turn")
    assert len(hb.latencies) >= MIN_HEARTBEATS or turn_s < 2.0, (
        f"{scn.id}: only {len(hb.latencies)} heartbeats answered in a {turn_s:.1f}s turn")


class ToolOutlivedGateway(Exception):
    """The in-flight tool outlived the gateway's exit: its process tree survives and/or its
    tool_call was left with no result in state.db. Deliberately NOT an AssertionError: the
    known-bug xfail matches only this, so an RPC failure, a crash or any other broken invariant
    (heartbeat, exit deadline, integrity) still fails the test."""


def _exit_mid_turn(scn: Scenario, gw: TuiGatewayProcess, hb: Heartbeat, tag: str,
                   state_db: Path, stored: str, stats: dict[str, Any]) -> dict[str, Any]:
    """The client vanishes (stdin EOF) or the supervisor stops us (SIGTERM) mid-wedge.
    Every other invariant is asserted first; the tool leftovers are checked together, last."""
    _assert_heartbeat(scn, hb, stats, turn_s=LONG_TIMEOUT_S)
    t0 = time.monotonic()
    if scn.action == "sigterm":
        gw.proc.send_signal(signal.SIGTERM)
        try:
            rc = gw.proc.wait(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            rc = None
    else:
        rc = gw.close_stdin_and_wait(EXIT_TIMEOUT_S)
    assert rc is not None, f"{scn.id}: gateway still alive {EXIT_TIMEOUT_S}s after {scn.action} mid-turn{gw.tail()}"
    stats["exit_s"] = round(time.monotonic() - t0, 2)
    assert integrity_ok(state_db) == "ok"
    leftovers = []
    if dangling := unanswered_tool_calls(persisted_messages(state_db, stored)):
        leftovers.append(f"state.db keeps tool_call(s) {dangling} with no result (next resume sends them)")
    if survivors := wait_no_tagged(tag):
        leftovers.append(f"orphans: {describe_pids(survivors)}")
    if leftovers:
        raise ToolOutlivedGateway(f"{scn.id} after {scn.action} mid-turn: " + "; ".join(leftovers))
    return stats


def _reap(gw: TuiGatewayProcess, tag: str) -> None:
    """Failure-path cleanup. Kill tagged descendants BEFORE the gateway: once it dies they
    reparent out of the test's process subtree, where the conftest kill guard refuses them."""
    for pid in tagged_pids(tag, exclude=(gw.proc.pid,)):
        with contextlib.suppress(OSError, RuntimeError):
            os.kill(pid, signal.SIGKILL)
    gw.kill()
    with contextlib.suppress(OSError, RuntimeError):
        kill_tagged(tag)


def run_scenario(scn: Scenario, srv: FakeLLMServer, root: Path) -> dict[str, Any]:
    nonce = uuid.uuid4().hex[:10]
    marker = f"[[chaos:{scn.fault}:{nonce}]]"
    probe_marker = f"[[probe:{nonce}]]"
    cfg: dict[str, Any] = {"model": scn.model}
    if scn.long_timeouts:
        cfg.update(request_timeout=LONG_TIMEOUT_S, stale_timeout=LONG_TIMEOUT_S)
    elif scn.long_request_timeout:
        cfg.update(request_timeout=LONG_TIMEOUT_S)
    home, hermes_home = write_chaos_home(root, srv.base_url, **cfg)
    tag = new_tag()
    work = gateway_cwd(root)
    env = env_for_gateway(hermetic_env(home, hermes_home, tag), work)
    gw = TuiGatewayProcess(env, work, root / "gateway-stderr.log")
    stats: dict[str, Any] = {"scenario": scn.id}
    try:
        assert gw.wait_event(lambda e: e.get("type") == "gateway.ready", timeout=READY_TIMEOUT_S), (
            f"gateway never became ready{gw.tail()}")
        created = gw.call("session.create", {"cols": 100})
        sid, stored = created["session_id"], created["stored_session_id"]
        # A healthy first turn builds the agent, so the fault lands mid-session and the
        # heartbeat measures the dispatcher during the fault, not the cold agent build.
        stats["warm_s"] = round(_answered_turn(
            gw, sid, uuid.uuid4().hex[:10], f"{scn.id} warm-up", WARMUP_DEADLINE_S), 2)

        # ── the faulted turn ──
        start = gw.event_count()
        with Heartbeat(gw, [("ping", {}), ("session.status", {"session_id": sid})],
                       interval=HEARTBEAT_INTERVAL_S, per_call_timeout=LONG_TIMEOUT_S) as hb:
            t0 = time.monotonic()
            submitted = gw.call("prompt.submit", {"session_id": sid, "text": f"please work {marker}"})
            assert submitted.get("status") == "streaming", submitted
            if scn.wait_for == "provider":
                assert _wait_until(lambda: _main_requests_for(srv, marker), TURN_DEADLINE_S), (
                    f"provider never saw the turn{gw.tail()}")
            elif scn.wait_for == "tool":
                assert gw.wait_event(lambda e: _is(e, "tool.start", sid), timeout=TURN_DEADLINE_S, start=start), (
                    f"hung tool never started{gw.tail()}")
                assert _wait_until(lambda: _hung_tool_pids(tag), TURN_DEADLINE_S), "hung tool process never spawned"
            deadline = TURN_DEADLINE_S
            if scn.action:
                # Let the fault sit a moment so the heartbeat samples the wedged turn.
                _wait_until(lambda: len(hb.latencies) >= MIN_HEARTBEATS, TURN_DEADLINE_S)
            if scn.action == "interrupt":
                t_int = time.monotonic()
                assert gw.call("session.interrupt", {"session_id": sid}, timeout=INTERRUPT_DEADLINE_S)[
                    "status"] == "interrupted"
                t0, deadline = t_int, INTERRUPT_DEADLINE_S
            done = None if scn.action in EXIT_ACTIONS else gw.wait_event(
                lambda e: _is(e, "message.complete", sid), timeout=deadline, start=start)
            turn_s = time.monotonic() - t0
        if scn.action in EXIT_ACTIONS:
            return _exit_mid_turn(scn, gw, hb, tag, hermes_home / "state.db", stored, stats)
        seen_types = sorted({str(e.get("type")) for e in gw.events_since(start, sid)})
        assert done is not None, (
            f"{scn.id}: no terminal message.complete within {deadline}s "
            f"(events: {seen_types}){gw.tail()}")
        stats["turn_s"] = round(turn_s, 2)
        stats["status"] = (done.get("payload") or {}).get("status")
        if scn.action == "interrupt":
            assert stats["status"] == "interrupted", done
        _wait_settled(gw, sid, done, scn.id)

        calls = len(_main_requests_for(srv, marker))
        stats["provider_calls"] = calls
        assert 1 <= calls <= MAX_PROVIDER_CALLS_PER_FAILED_TURN, f"{scn.id}: {calls} provider calls"

        _assert_heartbeat(scn, hb, stats, turn_s)

        if scn.reaps_hung_tool:
            assert _wait_until(lambda: not _hung_tool_pids(tag), PROCESS_REAP_S), (
                f"{scn.id}: hung tool survived its turn: {describe_pids(_hung_tool_pids(tag))}")

        # ── the same session must take and answer a new prompt ──
        stats["probe_s"] = round(_answered_turn(gw, sid, nonce, f"{scn.id} after the fault"), 2)

        probe_reqs = _main_requests_for(srv, probe_marker)
        assert probe_reqs, f"{scn.id}: follow-up never reached the provider"
        sent = probe_reqs[-1]["messages"]
        assert any(m.get("role") == "user" and marker in _text_of(m.get("content")) for m in sent), (
            f"{scn.id}: follow-up ran without the faulted turn in its history (not the same session)")
        assert unanswered_tool_calls(sent) == [], f"{scn.id}: model was sent unanswered tool_calls"

        state_db = hermes_home / "state.db"
        persisted = persisted_messages(state_db, stored)
        assert unanswered_tool_calls(persisted) == [], f"{scn.id}: state.db has unanswered tool_calls"
        if scn.fault == "parallel":
            _assert_parallel_results(persisted, nonce, "state.db")
            _assert_parallel_results(sent, nonce, "provider request")

        # ── EOF on stdin: the gateway leaves on its own and takes every child with it ──
        rc = gw.close_stdin_and_wait(EXIT_TIMEOUT_S)
        assert rc is not None, f"{scn.id}: gateway ignored stdin EOF for {EXIT_TIMEOUT_S}s{gw.tail()}"
        survivors = wait_no_tagged(tag)
        assert survivors == [], f"{scn.id}: orphans after exit: {describe_pids(survivors)}"
        assert integrity_ok(state_db) == "ok"
        return stats
    finally:
        _reap(gw, tag)


# ── pytest wiring: all scenarios run concurrently (one gateway each), asserted per test ──


@pytest.fixture(scope="module")
def scenario_futures(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    selected = {
        item.callspec.params["scn"].id for item in request.session.items
        if isinstance(getattr(getattr(item, "callspec", None), "params", {}).get("scn"), Scenario)
    }
    with FakeLLMServer(responder) as srv, ThreadPoolExecutor(
            max_workers=max(1, len(selected)), thread_name_prefix="chaos-tui") as pool:
        futures: dict[str, Future] = {
            scn.id: pool.submit(run_scenario, scn, srv, tmp_path_factory.mktemp(scn.id))
            for scn in SCENARIOS if scn.id in selected
        }
        try:
            yield futures
        finally:
            for fut in futures.values():
                fut.cancel()


# Real production bug on base (reported, not fixed here): when the gateway leaves mid-tool —
# client closes stdin or supervisor SIGTERMs — _shutdown_sessions() closes the agents but the
# in-flight foreground terminal command (its own process group) is never killed, so the
# `bash -c ...` + `sleep 3600` tree survives, reparented to init, and its tool_call is left with
# no result in state.db. strict: flips red once fixed. raises= names only the leftovers check's
# exception, so everything before it is asserted normally.
_ORPHANED_FOREGROUND_TOOL = pytest.mark.xfail(
    strict=True, raises=ToolOutlivedGateway,
    reason="tui_gateway exit (EOF/SIGTERM) orphans the running foreground terminal tool's process tree "
           "and leaves its tool_call without a result")
KNOWN_BUGS = {"stdin_eof_during_hung_tool": _ORPHANED_FOREGROUND_TOOL,
              "sigterm_during_hung_tool": _ORPHANED_FOREGROUND_TOOL}


@pytest.mark.parametrize("scn", [
    pytest.param(s, id=s.id, marks=[KNOWN_BUGS[s.id]] if s.id in KNOWN_BUGS else [])
    for s in SCENARIOS])
def test_tui_gateway_turn_stays_live_under_fault(scn: Scenario, scenario_futures) -> None:
    stats = scenario_futures[scn.id].result(timeout=WARMUP_DEADLINE_S + 3 * TURN_DEADLINE_S + 120)
    print(json.dumps(stats))
