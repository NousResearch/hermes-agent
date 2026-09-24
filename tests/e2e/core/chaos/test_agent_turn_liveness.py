"""C8 chaos: agent-turn liveness of a REAL AIAgent (child process) under provider and tool faults.

Each scenario starts ``_agent_driver`` in its own process with a hermetic HOME, a real
SessionDB on disk, config.yaml carrying the real liveness knobs (``agent.api_max_retries``,
``agent.auto_recovery_cycles``, ``agent.max_turns``, ``providers.custom.request_timeout_seconds``
/ ``stale_timeout_seconds``) and the scripted loopback provider as the only fake. The driver
runs the faulted turn, then a PROBE turn on the same agent with the faulted turn as history,
then ``agent.close()``, then exits. Scenarios run concurrently and are asserted per test.

Invariants after EVERY fault (the class, not one bug):

* bounded termination: the faulted turn returns within TURN_DEADLINE_S (the faults last an
  hour) or within INTERRUPT_DEADLINE_S of ``agent.interrupt()`` when every configured timeout
  is LONG_TIMEOUT_S, with a surfaced answer or error;
* bounded provider calls: no retry storm (#92450); a 400 is not retried; a healthy slow
  stream is not retried at all;
* reusability: the same agent answers the PROBE turn (no stuck interrupt flag, wedged client,
  or tripped breaker), and the PROBE request still carries the faulted turn;
* history integrity: every assistant tool_call has its own tool result in state.db and in
  what is sent back to the model; in an 8-way parallel batch each id carries ITS output (#93251);
* process hygiene: the driver exits on its own after close() and no process carrying the
  scenario tag survives (hung tools, background swarms, setsid escapees);
* state.db integrity_check == ok.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.chaos._helpers import (
    API_MAX_RETRIES,
    INTERRUPT_DEADLINE_S,
    LONG_TIMEOUT_S,
    MAX_PROVIDER_CALLS_PER_FAILED_TURN,
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
    python_exe,
    tagged_pids,
    tool_results_by_id,
    unanswered_tool_calls,
    wait_no_tagged,
    write_chaos_home,
)
from tests.fakes.fake_llm_provider import (
    DropMidStream,
    Error,
    FakeLLMServer,
    Hang,
    Raw,
    Response,
    StallMidStream,
    Text,
    ToolCall,
)

pytestmark = [pytest.mark.skipif(sys.platform != "linux", reason="orphan scan reads /proc")]

BOOT_DEADLINE_S = 120.0  # cold import of the agent stack on a loaded runner
PROBE_DEADLINE_S = 60.0
EXIT_DEADLINE_S = 30.0
SETTLE_DEADLINE_S = 60.0
SCENARIO_WORKERS = 6
PARALLEL_CALLS = 8
HUGE_OUTPUT_BYTES = 30_000_000
# What may be persisted / re-sent for a 30 MB tool output (it is truncated or spilled).
MAX_PERSISTED_TOOL_CHARS = 1_000_000
MAX_REQUEST_BYTES = 2_000_000
TRICKLE_CHUNKS = 40
TRICKLE_DELAY_S = 0.25  # 12x inside the 3 s stale timeout; total ~10 s > 3x the timeout
LOOP_BUDGET = 4

_PROBE_RE = re.compile(r"\[\[probe:([0-9a-f]+)\]\]")
_FAULT_RE = re.compile(r"\[\[chaos:([a-z0-9_]+)\]\]")

LONG = {"request_timeout": LONG_TIMEOUT_S, "stale_timeout": LONG_TIMEOUT_S}


# ── scenario table ───────────────────────────────────────────────────────────


@dataclass
class Ctx:
    """What a fault responder may look at: the request, its index in the faulted turn."""

    nonce: str
    n: int
    messages: list[dict[str, Any]]


def _after_tools(first: Callable[[Ctx], Response], then: str = "tools settled") -> Callable[[Ctx], Response]:
    def respond(ctx: Ctx) -> Response:
        if ctx.messages and ctx.messages[-1].get("role") == "tool":
            return Text(f"{then} {ctx.nonce}")
        return first(ctx)
    return respond


def _token(nonce: str, i: int) -> str:
    return f"tok{nonce}n{i}"


def _parallel(ctx: Ctx) -> ToolCall:
    calls = [("terminal", {"command": f"echo {_token(ctx.nonce, i)}"}) for i in range(PARALLEL_CALLS)]
    return ToolCall(calls[0][0], calls[0][1], parallel=calls[1:])


def _trickle_text(nonce: str) -> str:
    body = f"slow-{nonce}-"
    return (body * (TRICKLE_CHUNKS * 2 // len(body) + 1))[: TRICKLE_CHUNKS * 2]


@dataclass
class Scenario:
    id: str
    fault: Callable[[Ctx], Response]
    cfg: dict[str, Any] = field(default_factory=dict)
    # "error" | "answer" (final contains ``answer`` % nonce) | "ended" | "interrupted"
    expect: str = "error"
    answer: str = ""
    call_bound: int = MAX_PROVIDER_CALLS_PER_FAILED_TURN
    exact_calls: int | None = None
    # Poll condition (run, ctx-free) that, once true, sends ``interrupt`` to the driver.
    interrupt_when: Callable[["Run"], bool] | None = None
    tool_tokens: bool = False  # parallel batch: each result must carry its own token
    # "answer": the PROBE gets the scripted reply; "breaker": the stale breaker refuses it
    # immediately with a surfaced error and no provider call.
    probe: str = "answer"
    huge: bool = False
    background: bool = False  # the tool must really have started a tracked background job
    # Background scenarios: the provider holds its post-tool reply until this many tagged
    # processes whose cmdline contains ``settle_on`` exist, so the kill at close() always
    # meets the job in the intended state no matter how slowly its login shell starts.
    settle_on: str = ""
    settle_count: int = 0


def _hung(timeout: float) -> Callable[[Ctx], Response]:
    return _after_tools(lambda c: ToolCall("terminal", {"command": "sleep 3600", "timeout": timeout}))


SCENARIOS: list[Scenario] = [
    # provider faults that last forever: the turn must give up on its own
    Scenario("provider_hang", lambda c: Hang()),
    # request_timeout LONG: only the explicit stale timeout can end it, and it must beat the
    # reasoning-model floor (#115024). The stale streak it leaves behind trips the cross-turn
    # stale breaker (#58962), so the PROBE must be refused at once, surfaced, and unbilled.
    Scenario("provider_stall_reasoning_model", lambda c: StallMidStream(),
             cfg={"model": REASONING_MODEL, "request_timeout": LONG_TIMEOUT_S}, probe="breaker"),
    Scenario("provider_500_forever", lambda c: Error(500)),
    Scenario("provider_429_forever", lambda c: Error(429, retry_after=1)),
    Scenario("provider_400_forever", lambda c: Error(400, "invalid request"), call_bound=2),
    Scenario("provider_garbage_body_forever", lambda c: Raw("{not json at all")),
    Scenario("provider_drop_mid_stream_forever", lambda c: DropMidStream(), expect="ended"),
    # transient faults: a retry must get through
    Scenario("provider_hang_then_recover",
             lambda c: Hang() if c.n == 0 else Text(f"recovered {c.nonce}"),
             expect="answer", answer="recovered {nonce}", call_bound=3),
    Scenario("provider_slow_trickle_is_not_stale",
             lambda c: Text(_trickle_text(c.nonce), chunk_chars=2, delay_per_chunk=TRICKLE_DELAY_S),
             expect="answer", answer="{trickle}", exact_calls=1),
    # malformed model output
    # (undecodable arguments are treated as a cut-off reply: surfaced, never executed)
    Scenario("tool_args_truncated_json",
             _after_tools(lambda c: ToolCall("terminal", '{"command": "echo trunc' + c.nonce)),
             expect="ended", call_bound=6),
    Scenario("tool_args_garbage",
             _after_tools(lambda c: ToolCall("terminal", "{{{ not json")),
             expect="ended", call_bound=6),
    Scenario("tool_name_unknown",
             _after_tools(lambda c: ToolCall("no_such_tool_chaos", {"x": 1})),
             expect="answer", answer="tools settled {nonce}", call_bound=6),
    Scenario("parallel_8_terminal_calls", _after_tools(_parallel),
             expect="answer", answer="tools settled {nonce}", call_bound=4, tool_tokens=True),
    # tool faults
    Scenario("tool_hang_past_timeout", _hung(TOOL_TIMEOUT_S),
             expect="answer", answer="tools settled {nonce}", call_bound=4),
    # stdin must be closed for tools: `cat` returns at EOF; a pipe held open would wedge the
    # turn past its deadline (the tool timeout is LONG).
    Scenario("tool_reads_stdin",
             _after_tools(lambda c: ToolCall("terminal", {
                 "command": f"cat; echo stdin-closed-{c.nonce}", "timeout": LONG_TIMEOUT_S})),
             expect="answer", answer="tools settled {nonce}", call_bound=4),
    Scenario("tool_background_swarm",
             _after_tools(lambda c: ToolCall("terminal", {
                 "command": "for i in 1 2 3 4 5 6 7 8; do sleep 3600 & done; setsid sleep 3600 & wait",
                 "background": True})),
             expect="answer", answer="tools settled {nonce}", call_bound=4, background=True,
             settle_on="sleep 3600", settle_count=9),
    # a job still spawning when it is killed must not leave its late children behind; the
    # TERM trap makes "spawns after the kill snapshot" deterministic (interactive bash keeps
    # running through the grace window just like this); the loop's `sleep 0.2` proves the
    # trap is installed before close() arrives
    Scenario("tool_background_spawns_while_killed",
             _after_tools(lambda c: ToolCall("terminal", {
                 "command": "trap 'for i in 1 2 3 4; do sleep 3600 & done' TERM; while :; do sleep 0.2; done",
                 "background": True})),
             expect="answer", answer="tools settled {nonce}", call_bound=4, background=True,
             settle_on="sleep 0.2", settle_count=1),
    Scenario("tool_huge_output",
             _after_tools(lambda c: ToolCall("terminal", {
                 "command": f"head -c {HUGE_OUTPUT_BYTES} /dev/zero | tr '\\0' A", "timeout": 120})),
             expect="answer", answer="tools settled {nonce}", call_bound=4, huge=True),
    # a model that never stops calling tools is bounded by agent.max_turns
    Scenario("tool_loop_hits_turn_budget",
             lambda c: ToolCall("terminal", {"command": f"echo loop-{c.n}"}),
             cfg={"max_turns": LOOP_BUDGET}, expect="ended", call_bound=LOOP_BUDGET + 2),
    # interrupts: every timeout LONG, so only agent.interrupt() can end the turn in time
    Scenario("interrupt_provider_hang", lambda c: Hang(), cfg=LONG, expect="interrupted",
             interrupt_when=lambda run: run.fault_calls() >= 1),
    Scenario("interrupt_hung_tool", _hung(LONG_TIMEOUT_S), cfg=LONG, expect="interrupted",
             interrupt_when=lambda run: run.hung_tool_running()),
    Scenario("interrupt_endless_tool_loop",
             lambda c: ToolCall("terminal", {"command": f"echo loop-{c.n}"}),
             cfg={**LONG, "max_turns": 1000}, expect="interrupted",
             interrupt_when=lambda run: run.fault_calls() >= 3),
]
BY_ID = {s.id: s for s in SCENARIOS}


# ── one scenario run ─────────────────────────────────────────────────────────


class Run:
    def __init__(self, sc: Scenario, root: Path) -> None:
        self.sc = sc
        self.root = root
        self.tag = new_tag()
        self.nonce = uuid.uuid4().hex[:10]
        self.session_id = f"chaos-{self.nonce}"
        self.events: "queue.Queue[tuple[float, dict[str, Any]]]" = queue.Queue()
        self.seen: list[tuple[float, dict[str, Any]]] = []
        self.srv = FakeLLMServer(self._respond)
        self.fault_msg = f"[[chaos:{sc.id}]] please help ({self.nonce})"
        self.probe_msg = f"[[probe:{self.nonce}]] are you still there?"
        self.report: dict[str, Any] = {"id": sc.id, "nonce": self.nonce}

    # provider side
    def _is_probe(self, messages: list[dict[str, Any]]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            text = msg.get("content") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content"))
            if _PROBE_RE.search(text or ""):
                return True
            if _FAULT_RE.search(text or ""):
                return False
        return False

    def _respond(self, record: dict[str, Any]) -> Response:
        messages = record["body"].get("messages") or []
        record["probe"] = self._is_probe(messages)
        if record["probe"]:
            return Text(f"alive {self.nonce}")
        n = sum(1 for r in self.srv.requests if r["kind"] == "main" and not r.get("probe")) - 1
        if self.sc.settle_on and messages and messages[-1].get("role") == "tool":
            deadline = time.monotonic() + SETTLE_DEADLINE_S
            while time.monotonic() < deadline and sum(
                    self.sc.settle_on in d for d in describe_pids(tagged_pids(self.tag))) < self.sc.settle_count:
                time.sleep(0.05)
        return self.sc.fault(Ctx(self.nonce, n, messages))

    def fault_calls(self) -> int:
        return sum(1 for r in list(self.srv.requests) if r["kind"] == "main" and not r.get("probe"))

    def hung_tool_running(self) -> bool:
        """The faulted turn's ``sleep 3600`` is up (other tagged helpers may start earlier)."""
        return self.fault_calls() >= 1 and any("sleep 3600" in d for d in describe_pids(tagged_pids(self.tag)))

    def probe_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in list(self.srv.requests) if r["kind"] == "main" and r.get("probe")]

    # driver side
    def _reader(self, stream) -> None:
        for raw in stream:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("CHAOS "):
                try:
                    self.events.put((time.monotonic(), json.loads(line[6:])))
                except json.JSONDecodeError:
                    pass

    def _wait(self, pred: Callable[[dict[str, Any]], bool], deadline: float) -> tuple[float, dict[str, Any]] | None:
        for t, ev in self.seen:
            if pred(ev):
                return t, ev
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                t, ev = self.events.get(timeout=min(left, 0.2))
            except queue.Empty:
                if self.proc.poll() is not None and self.events.empty():
                    return None
                continue
            self.seen.append((t, ev))
            if pred(ev):
                return t, ev

    def execute(self) -> dict[str, Any]:
        rep = self.report
        self.srv.start()
        try:
            self._execute(rep)
        finally:
            if getattr(self, "proc", None) is not None and self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=30)
            kill_tagged(self.tag)
            self.srv.stop()
        return rep

    def _execute(self, rep: dict[str, Any]) -> None:
        sc = self.sc
        home, hermes_home = write_chaos_home(self.root, self.srv.base_url, **sc.cfg)
        work = self.root / "work"
        work.mkdir()
        spec = self.root / "spec.json"
        spec.write_text(json.dumps({
            "session_id": self.session_id, "turns": [self.fault_msg, self.probe_msg]}), encoding="utf-8")
        env = hermetic_env(home, hermes_home, self.tag)
        env["TERMINAL_CWD"] = str(work)
        stderr = open(self.root / "driver.stderr", "wb")
        self.proc = subprocess.Popen(
            [python_exe(), "-m", "tests.e2e.core.chaos._agent_driver", str(spec)],
            cwd=str(work), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
        )
        threading.Thread(target=self._reader, args=(self.proc.stdout,), daemon=True).start()
        rep["stderr_path"] = str(self.root / "driver.stderr")

        if self._wait(lambda e: e["ev"] == "turn_start" and e["i"] == 0,
                      time.monotonic() + BOOT_DEADLINE_S) is None:
            rep["boot_failed"] = True
            return
        t0 = time.monotonic()
        deadline = t0 + TURN_DEADLINE_S
        if sc.interrupt_when is not None:
            trigger_deadline = t0 + TURN_DEADLINE_S
            while time.monotonic() < trigger_deadline and not sc.interrupt_when(self):
                if self._wait(lambda e: e["ev"] == "turn_end", time.monotonic() + 0.05):
                    break
            if sc.interrupt_when(self):
                rep["interrupt_at"] = time.monotonic() - t0
                rep["calls_at_interrupt"] = self.fault_calls()
                self.proc.stdin.write(b"interrupt\n")
                self.proc.stdin.flush()
                deadline = time.monotonic() + INTERRUPT_DEADLINE_S
        end = self._wait(lambda e: e["ev"] == "turn_end" and e["i"] == 0, deadline)
        rep["fault_calls"] = self.fault_calls()
        if end is None:
            rep["turn0"] = None
            rep["turn0_open_after"] = time.monotonic() - t0
            return
        rep["turn0"] = end[1]
        rep["turn0_elapsed"] = end[0] - t0
        if "interrupt_at" in rep:
            rep["after_interrupt"] = end[0] - t0 - rep["interrupt_at"]
        # the hung tool of the faulted turn must be gone before the next turn starts
        rep["tool_survivors_after_turn"] = describe_pids(wait_no_tagged(self.tag, PROCESS_REAP_S)) \
            if sc.id in {"tool_hang_past_timeout", "interrupt_hung_tool"} else []
        probe = self._wait(lambda e: e["ev"] == "turn_end" and e["i"] == 1,
                           time.monotonic() + PROBE_DEADLINE_S)
        rep["fault_calls_final"] = self.fault_calls()
        rep["turn1"] = probe[1] if probe else None
        closed = self._wait(lambda e: e["ev"] == "closed", time.monotonic() + EXIT_DEADLINE_S)
        rep["closed"] = closed is not None
        try:
            rep["exit_code"] = self.proc.wait(timeout=EXIT_DEADLINE_S)
        except subprocess.TimeoutExpired:
            rep["exit_code"] = None
        rep["orphans"] = describe_pids(wait_no_tagged(self.tag, PROCESS_REAP_S))
        log = hermes_home / "logs" / "agent.log"
        if rep["orphans"] and log.exists():
            rep["agent_log_tail"] = log.read_text(errors="replace")[-6000:]
        # Stale-kill timeline: tells a PROBE refused by the cross-turn breaker apart from a slow probe.
        rep["stale_log"] = [ln[:110] for ln in (log.read_text(errors="replace").splitlines() if log.exists() else [])
                            if "stale" in ln.lower() and ("WARNING" in ln or "ERROR" in ln)][-12:]
        db = hermes_home / "state.db"
        rep["persisted"] = persisted_messages(db, self.session_id) if db.exists() else None
        rep["integrity"] = integrity_ok(db) if db.exists() else "missing"
        probes = self.probe_requests()
        rep["probe_request"] = probes[-1] if probes else None
        rep["probe_count"] = len(probes)
        mains = [r["body"] for r in list(self.srv.requests) if r["kind"] == "main"]
        rep["last_request"] = mains[-1] if mains else None
        rep["max_request_bytes"] = max((len(json.dumps(b)) for b in mains), default=0)


# ── module fixture: run the selected scenarios concurrently ──────────────────


@pytest.fixture(scope="module")
def runs(request, tmp_path_factory) -> dict[str, Future]:
    selected = [
        item.callspec.params["scenario_id"] for item in request.session.items
        if item.module is request.module and hasattr(item, "callspec")
        and "scenario_id" in item.callspec.params
    ]
    pool = ThreadPoolExecutor(max_workers=SCENARIO_WORKERS, thread_name_prefix="chaos")
    futures = {
        sid: pool.submit(Run(BY_ID[sid], tmp_path_factory.mktemp(sid)).execute)
        for sid in dict.fromkeys(selected)
    }
    yield futures
    pool.shutdown(wait=True, cancel_futures=True)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content)


@pytest.mark.parametrize("scenario_id", list(BY_ID))
def test_agent_turn_liveness(scenario_id: str, runs: dict[str, Future]) -> None:
    sc = BY_ID[scenario_id]
    rep = runs[scenario_id].result(timeout=BOOT_DEADLINE_S + 3 * TURN_DEADLINE_S + 120)
    tail = Path(rep["stderr_path"]).read_text(errors="replace")[-3000:] if rep.get("stderr_path") else ""
    where = f"[{scenario_id}] stderr tail:\n{tail}"

    assert not rep.get("boot_failed"), f"driver never started the turn. {where}"

    # 1. bounded termination
    turn0 = rep["turn0"]
    if "interrupt_at" in rep:
        assert turn0 is not None, (
            f"turn still running {INTERRUPT_DEADLINE_S}s after agent.interrupt() "
            f"(calls={rep['fault_calls']}). {where}")
    else:
        assert sc.interrupt_when is None, f"interrupt trigger never became true. {where}"
        assert turn0 is not None, (
            f"faulted turn still running after {TURN_DEADLINE_S}s (calls={rep['fault_calls']}). {where}")
    final = turn0["final"]
    if sc.expect == "interrupted":
        # the CLI renders its own interruption notice; the turn must say it was interrupted
        assert turn0["interrupted"], f"interrupt not reflected in the result: {turn0}"
    else:
        assert final.strip(), f"turn ended with nothing surfaced to the user: {turn0}"
    if sc.expect == "answer":
        want = sc.answer.format(nonce=rep_nonce(rep), trickle=_trickle_text(rep_nonce(rep)))
        assert want in final, f"expected the answer {want!r}, got {final[:300]!r}"

    # 2. bounded provider calls
    calls = rep["fault_calls"]
    if sc.exact_calls is not None:
        assert calls == sc.exact_calls, f"healthy stream was retried: {calls} calls"
    assert 1 <= calls <= sc.call_bound, f"{calls} provider calls for one turn (bound {sc.call_bound})"
    if "interrupt_at" in rep:
        extra = rep["fault_calls_final"] - rep["calls_at_interrupt"]
        assert extra <= 2, f"the loop kept calling the provider after interrupt: +{extra}"

    # 3. reusability: the same agent answers the next message, with the fault in its history
    assert rep["turn1"] is not None, f"PROBE turn did not finish within {PROBE_DEADLINE_S}s. {where}"
    if sc.probe == "breaker":
        assert rep["turn1"]["failed"] and rep["turn1"]["final"].strip(), f"breaker refusal not surfaced: {rep['turn1']}"
        assert rep["probe_request"] is None, "a tripped stale breaker still billed the provider"
        sent = rep["last_request"]["messages"]
    else:
        assert f"alive {rep_nonce(rep)}" in rep["turn1"]["final"], (
            f"PROBE not answered: {rep['turn1']}\nfault calls {rep['fault_calls']}, probe requests "
            f"{rep['probe_count']}, stale log:\n" + "\n".join(rep.get("stale_log", [])))
        assert rep["probe_request"] is not None, "PROBE never reached the provider"
        sent = rep["probe_request"]["messages"]
        assert any(f"[[chaos:{scenario_id}]]" in _text(m.get("content")) for m in sent if m.get("role") == "user"), \
            "the faulted user turn vanished from the history sent with the next message"

    # 4. history integrity, persisted and sent
    persisted = rep["persisted"]
    assert persisted, "nothing persisted to state.db"
    assert unanswered_tool_calls(persisted) == [], "state.db has tool_calls without results"
    assert unanswered_tool_calls(sent) == [], "request sent to the model has tool_calls without results"
    assert rep["integrity"] == "ok"
    if sc.tool_tokens:
        for label, msgs in (("state.db", persisted), ("request", sent)):
            results = tool_results_by_id(msgs)
            calls_by_id = {
                c["id"]: c["function"]["arguments"]
                for m in msgs if m.get("role") == "assistant"
                for c in (json.loads(m["tool_calls"]) if isinstance(m.get("tool_calls"), str)
                          else m.get("tool_calls") or [])
            }
            assert len(calls_by_id) == PARALLEL_CALLS, f"{label}: {len(calls_by_id)} calls"
            for cid, args in calls_by_id.items():
                token = re.search(r"tok[0-9a-f]+n\d", args)
                token = token.group(0) if token else args
                assert token in results.get(cid, ""), f"{label}: {cid} ({token}) got {results.get(cid, '')[:160]!r}"
    if sc.background:
        started = [m.get("content") or "" for m in persisted if m.get("role") == "tool"]
        assert any('"pid"' in c and "proc_" in c for c in started), f"background job never started: {started}"
    if sc.huge:
        biggest = max((len(m.get("content") or "") for m in persisted if m.get("role") == "tool"), default=0)
        assert biggest <= MAX_PERSISTED_TOOL_CHARS, f"{biggest} chars of tool output persisted"
        assert rep["max_request_bytes"] <= MAX_REQUEST_BYTES, f"{rep['max_request_bytes']} byte request"

    # 5. process hygiene
    assert rep["tool_survivors_after_turn"] == [], f"hung tool outlived its turn: {rep['tool_survivors_after_turn']}"
    assert rep["closed"], f"agent.close() did not return. {where}"
    assert rep["exit_code"] is not None, f"driver did not exit {EXIT_DEADLINE_S}s after close(). {where}"
    assert rep["orphans"] == [], (
        f"processes outlived the session: {rep['orphans']}\nagent.log tail:\n{rep.get('agent_log_tail', '')}")


def rep_nonce(rep: dict[str, Any]) -> str:
    return rep["nonce"]


def test_retry_bound_matches_config() -> None:
    """Guard the oracle itself: the per-turn call bound must stay a small multiple of the
    configured retries, or a storm could hide under it."""
    assert MAX_PROVIDER_CALLS_PER_FAILED_TURN <= 8 * API_MAX_RETRIES
