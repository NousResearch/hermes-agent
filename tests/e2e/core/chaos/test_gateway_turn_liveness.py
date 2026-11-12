"""C8 chaos: agent-turn liveness through the REAL messaging gateway (``python -m gateway.run``).

The gateway process is the shipped entry point: it discovers a user platform plugin
(``_gateway_fake_platform``, an in-memory adapter dialled over loopback), connects it,
installs the runner's own message handler, and runs every inbound message through the real
``GatewayRunner._handle_message`` -> ``_handle_message_with_agent`` -> ``AIAgent`` turn with
the real turn lease, session store, busy-input policy, /stop path, state.db and SIGTERM
drain. Only the LLM provider (``tests/fakes/fake_llm_provider``) and the chat platform are
fake. Each scenario gets its own gateway so process hygiene is judged per fault; all
scenarios run concurrently against one scripted provider and are asserted per test.

After every fault the same invariants hold:

* the faulted turn ends (the adapter's processing-complete hook fires for that message)
  within TURN_DEADLINE_S — or within INTERRUPT_DEADLINE_S of /stop, a busy follow-up or
  SIGTERM when every configured timeout is LONG_TIMEOUT_S — and the user was sent
  something (answer or surfaced error) before it ended;
* provider main calls for the faulted turn stay <= MAX_PROVIDER_CALLS_PER_FAILED_TURN;
* the gateway event loop keeps ticking (100 ms heartbeat inside the gateway process);
* the SAME chat then accepts a new message and gets exactly one scripted answer, with the
  faulted turn in its history (turn-lease leak oracle, #104303);
* every tool_call has its own result in state.db and in what is sent back to the model
  (8-way parallel batch: each id carries ITS token, #93251);
* hung tool processes are reaped; SIGTERM makes the gateway exit by itself; no process
  carrying the scenario tag survives; state.db passes integrity_check.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.chaos._gateway_harness import SHUTDOWN_DEADLINE_S, Event, GatewayProc
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
    integrity_ok,
    kill_tagged,
    new_tag,
    tagged_pids,
    tool_results_by_id,
    unanswered_tool_calls,
    wait_no_tagged,
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
    pytest.mark.skipif(sys.platform != "linux", reason="orphan scan reads /proc; POSIX gateway signals"),
]

# The adapter ticks every 100 ms on the gateway loop. Healthy gaps are ~0.11 s, worst seen
# 0.6 s at load average ~100 (10x margin -> 6 s). A turn run synchronously on the loop
# freezes it for the whole faulted turn (>= 22 s for a provider hang with the chaos timeouts,
# LONG_TIMEOUT_S in the interrupt scenarios), so the bound still catches it.
HEARTBEAT_MAX_GAP_S = 6.0
PARALLEL_CALLS = 8
HUNG_COMMAND = "sleep 3600"
SCENARIO_WORKERS = 6
CHAT = "chaos-chat"
OTHER_CHAT = "chaos-chat-b"

_MARK = re.compile(r"\[\[chaos:(?P<fault>[a-z0-9_]+):(?P<nonce>[0-9a-f]+)\]\]")
_PROBE = re.compile(r"\[\[probe:(?P<nonce>[0-9a-f]+):(?P<n>\d+)\]\]")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _text_of(msg.get("content"))
    return ""


def _token(nonce: str, i: int) -> str:
    return f"tok{nonce}n{i}"


def _alive(nonce: str, n: int) -> str:
    return f"alive {nonce} {n}"


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
    last_user = _last_user_text(messages)
    if probe := _PROBE.search(last_user):
        return Text(_alive(probe["nonce"], int(probe["n"])))
    if mark := _MARK.search(last_user):
        return FAULTS[mark["fault"]](messages, mark["nonce"])
    return Text("unscripted")


def _main_requests_for(srv: FakeLLMServer, needle: str) -> list[dict[str, Any]]:
    return [
        r["body"] for r in list(srv.requests)
        if r["kind"] == "main" and needle in _last_user_text(r["body"].get("messages") or [])
    ]


# ── scenario matrix ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario:
    id: str
    fault: str
    model: str = PLAIN_MODEL
    long_timeouts: bool = False
    # How the faulted turn is ended: None = by the gateway's own timeouts/retry budget;
    # "stop" = /stop in the same chat; "busy" = a second user message (busy-input path);
    # "sigterm" = the gateway process is terminated mid-turn.
    ender: str | None = None
    # When the ender fires: "provider" = the provider holds the request; "tool" = the hung
    # tool process is running.
    when: str = "provider"
    busy_mode: str | None = None
    reaps_hung_tool: bool = False
    # A stalled turn makes >= HERMES_STREAM_STALE_GIVEUP stale attempts, and the cross-turn
    # breaker (#58962) then refuses the session's next turn WITHOUT calling the provider
    # until /new or a provider swap. Allowed only here, and only as a bounded, surfaced
    # refusal with zero provider calls; /new must then make the chat answer again.
    may_trip_stale_breaker: bool = False


SCENARIOS = [
    Scenario("provider_hang", "hang"),
    # Reasoning slug: the explicit stale_timeout_seconds must beat its reasoning floor.
    Scenario("provider_stall_reasoning_model", "stall", model=REASONING_MODEL, may_trip_stale_breaker=True),
    Scenario("provider_500_forever", "err500"),
    Scenario("provider_429_forever", "err429"),
    Scenario("provider_drop_mid_stream", "drop"),
    Scenario("parallel_8_terminal_calls", "parallel"),
    Scenario("tool_hang_past_timeout", "toolhang", reaps_hung_tool=True),
    Scenario("stop_during_provider_hang", "hang", long_timeouts=True, ender="stop"),
    Scenario("stop_during_hung_tool", "toolhang_long", long_timeouts=True, ender="stop", when="tool",
             reaps_hung_tool=True),
    # busy_input_mode interrupt: the follow-up must preempt a turn that would hang 600 s.
    Scenario("busy_interrupt_during_provider_hang", "hang", long_timeouts=True, ender="busy",
             busy_mode="interrupt"),
    # busy_input_mode queue: the follow-up waits for the faulted turn to fail, then is answered.
    Scenario("busy_queue_during_provider_hang", "hang", ender="busy", busy_mode="queue"),
    Scenario("sigterm_during_provider_hang", "hang", long_timeouts=True, ender="sigterm"),
]


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


def _session_with(state_db: Path, needle: str) -> str | None:
    conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=10)
    try:
        row = conn.execute(
            "SELECT session_id FROM messages WHERE role = 'user' AND content LIKE ? ORDER BY id DESC LIMIT 1",
            (f"%{needle}%",)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _persisted(state_db: Path, session_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 AND role IN ('user', 'assistant', 'tool') ORDER BY id",
            (session_id,)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        m: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["tool_call_id"]:
            m["tool_call_id"] = r["tool_call_id"]
        if r["tool_calls"]:
            m["tool_calls"] = r["tool_calls"]
        out.append(m)
    return out


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


def _is_send(chat: str, needle: str | None = None) -> Callable[[Event], bool]:
    return lambda e: e.ev == "send" and e.data.get("chat") == chat and (
        needle is None or needle in (e.data.get("text") or ""))


def _is_complete(msg_id: str) -> Callable[[Event], bool]:
    return lambda e: e.ev == "complete" and e.data.get("id") == msg_id


def _settle(gw: GatewayProc, msg_id: str, since: int, what: str) -> Event:
    """The message's processing finished AND the adapter's session guard is released.

    ``since`` must be at/after the message's answer: a message that arrives while startup
    restore is still running is parked (an early processing-complete) and replayed later."""
    done = gw.wait_for(_is_complete(msg_id), TURN_DEADLINE_S, what, since=since)
    assert done is not None, f"{what}: processing never completed{gw.log_tail()}"
    idle = gw.wait_for(lambda e: e.ev == "hb" and not e.data.get("busy"), TURN_DEADLINE_S, what,
                       since=done.idx)
    assert idle is not None, f"{what}: session guard never released{gw.log_tail()}"
    return done


def _probe(gw: GatewayProc, nonce: str, n: int, chat: str, deadline: float, what: str) -> tuple[str, int, float]:
    start = gw.mark()
    t0 = time.monotonic()
    msg_id = gw.send_user(f"are you there [[probe:{nonce}:{n}]]", chat=chat)
    got = gw.wait_for(_is_send(chat, _alive(nonce, n)), deadline, what, since=start)
    assert got is not None, (
        f"{what}: no answer within {deadline}s; sends={gw.sends_since(start)}{gw.log_tail()}")
    return msg_id, got.idx, time.monotonic() - t0


def _follow_up_through_breaker(gw: GatewayProc, srv: FakeLLMServer, scn: Scenario, nonce: str,
                               stats: dict[str, Any]) -> tuple[int, bool]:
    """Follow-up for a scenario that may trip the stale breaker. Returns (probe n that was
    answered, whether that probe ran in the faulted session)."""
    start = gw.mark()
    t0 = time.monotonic()
    msg_id = gw.send_user(f"are you there [[probe:{nonce}:2]]", chat=CHAT)
    done = gw.wait_for(_is_complete(msg_id), TURN_DEADLINE_S, "follow-up", since=start)
    assert done is not None, f"{scn.id}: follow-up never completed{gw.log_tail()}"
    stats["probe_s"] = round(time.monotonic() - t0, 2)
    _settle(gw, msg_id, start, f"{scn.id}: follow-up")
    if gw.wait_for(_is_send(CHAT, _alive(nonce, 2)), 0.1, "answer", since=start):
        return 2, True
    # Refused: it must be a surfaced, provider-free refusal (not a silent drop, not a storm).
    assert gw.sends_since(start), f"{scn.id}: follow-up refused silently"
    assert not _main_requests_for(srv, f"[[probe:{nonce}:2]]"), (
        f"{scn.id}: follow-up neither answered nor short-circuited")
    stats["stale_breaker_tripped"] = True
    reset_start = gw.mark()
    reset_id = gw.send_user("/new", chat=CHAT)
    _settle(gw, reset_id, reset_start, f"{scn.id}: /new")
    probe_id, probe_answer, probe_s = _probe(gw, nonce, 3, CHAT, TURN_DEADLINE_S, f"{scn.id}: probe after /new")
    stats["probe_after_new_s"] = round(probe_s, 2)
    _settle(gw, probe_id, probe_answer, f"{scn.id}: probe after /new")
    return 3, False


def run_scenario(scn: Scenario, srv: FakeLLMServer, root: Path) -> dict[str, Any]:
    nonce = uuid.uuid4().hex[:10]
    marker = f"[[chaos:{scn.fault}:{nonce}]]"
    cfg: dict[str, Any] = {"model": scn.model}
    if scn.long_timeouts:
        cfg.update(request_timeout=LONG_TIMEOUT_S, stale_timeout=LONG_TIMEOUT_S)
    extra = f"display:\n  busy_input_mode: {scn.busy_mode}\n" if scn.busy_mode else ""
    tag = new_tag()
    gw = GatewayProc(root, srv.base_url, tag, cfg=cfg, extra_config=extra)
    stats: dict[str, Any] = {"scenario": scn.id}
    try:
        t_boot = time.monotonic()
        gw.start()
        # Warm-up turn: the first inbound after boot waits for startup restore and builds the
        # agent cold; the scenario starts only once that turn is answered and settled.
        warm_mark = gw.mark()
        warm_id, warm_answer, _ = _probe(gw, nonce, 0, CHAT, TURN_DEADLINE_S, f"{scn.id}: warm-up")
        _settle(gw, warm_id, warm_answer, f"{scn.id}: warm-up")
        stats["boot_s"] = round(time.monotonic() - t_boot, 1)

        # ── the faulted turn ──
        start = gw.mark()
        hb_from = start
        t0 = time.monotonic()
        fault_id = gw.send_user(f"please work {marker}", chat=CHAT)
        deadline = TURN_DEADLINE_S
        if scn.ender:
            if scn.when == "tool":
                assert _wait_until(lambda: _hung_tool_pids(tag), TURN_DEADLINE_S), (
                    f"{scn.id}: hung tool process never spawned{gw.log_tail()}")
            else:
                assert _wait_until(lambda: _main_requests_for(srv, marker), TURN_DEADLINE_S), (
                    f"{scn.id}: provider never saw the turn{gw.log_tail()}")
            # Let the fault sit so the heartbeat samples the wedged gateway.
            gw.wait_for(lambda e: e.ev == "hb", TURN_DEADLINE_S, "hb", since=gw.mark())
            gw.wait_for(lambda e: e.ev == "hb", TURN_DEADLINE_S, "hb", since=gw.mark())
            if scn.ender == "stop":
                # A wedged chat must not wedge the gateway: another chat is served meanwhile.
                _, _, other_s = _probe(gw, nonce, 7, OTHER_CHAT, TURN_DEADLINE_S,
                                       f"{scn.id}: other chat while wedged")
                stats["other_chat_s"] = round(other_s, 2)
                t0, deadline = time.monotonic(), INTERRUPT_DEADLINE_S
                gw.send_user("/stop", chat=CHAT)
            elif scn.ender == "busy":
                busy_deadline = INTERRUPT_DEADLINE_S if scn.long_timeouts else TURN_DEADLINE_S
                t_busy = time.monotonic()
                _, _, busy_s = _probe(gw, nonce, 1, CHAT, busy_deadline,
                                      f"{scn.id}: follow-up sent while the turn hangs")
                stats["busy_answer_s"] = round(busy_s, 2)
                t0 = t_busy
                deadline = busy_deadline
            elif scn.ender == "sigterm":
                gw.stop()
                stats["shutdown_s"] = gw.shutdown_s and round(gw.shutdown_s, 2)
                assert gw.shutdown_s is not None, (
                    f"{scn.id}: gateway ignored SIGTERM for {SHUTDOWN_DEADLINE_S}s during a hung turn"
                    f"{gw.log_tail()}")
                assert gw.exit_code is not None and gw.exit_code >= 0, (
                    f"{scn.id}: gateway died by signal {gw.exit_code}{gw.log_tail()}")

        if scn.ender != "sigterm":
            done = gw.wait_for(_is_complete(fault_id), max(0.0, deadline - (time.monotonic() - t0)),
                               "fault turn", since=start)
            turn_s = time.monotonic() - t0
            assert done is not None, (
                f"{scn.id}: faulted turn still running after {deadline}s; "
                f"sends={gw.sends_since(start)}{gw.log_tail()}")
            stats["turn_s"] = round(turn_s, 2)
            surfaced = [e for e in gw.events_between(start, done.idx) if _is_send(CHAT)(e)]
            assert surfaced, f"{scn.id}: the turn ended without telling the user anything{gw.log_tail()}"
            _settle(gw, fault_id, start, f"{scn.id}: faulted turn")
            if scn.fault in ("parallel", "toolhang"):
                assert gw.wait_for(_is_send(CHAT, f"tools settled {nonce}"), 0.1, "answer", since=start), (
                    f"{scn.id}: final answer after the tool batch never delivered: {gw.sends_since(start)}")

        calls = len(_main_requests_for(srv, marker))
        stats["provider_calls"] = calls
        assert 1 <= calls <= MAX_PROVIDER_CALLS_PER_FAILED_TURN, f"{scn.id}: {calls} provider calls"

        if scn.reaps_hung_tool:
            assert _wait_until(lambda: not _hung_tool_pids(tag), PROCESS_REAP_S), (
                f"{scn.id}: hung tool survived its turn: {describe_pids(_hung_tool_pids(tag))}")

        state_db = gw.state_db
        if scn.ender != "sigterm":
            # ── the same chat must take and answer a new message, exactly once ──
            n = 2
            history_kept = True
            if scn.may_trip_stale_breaker:
                n, history_kept = _follow_up_through_breaker(gw, srv, scn, nonce, stats)
            else:
                probe_id, probe_answer, probe_s = _probe(gw, nonce, n, CHAT, TURN_DEADLINE_S,
                                                         f"{scn.id}: follow-up after the fault")
                stats["probe_s"] = round(probe_s, 2)
                _settle(gw, probe_id, probe_answer, f"{scn.id}: follow-up")
            answers = [t for t in gw.sends_since(warm_mark) if _alive(nonce, n) in t]
            assert len(answers) == 1, f"{scn.id}: follow-up answered {len(answers)} times"

            gap = max((float(e.data.get("max_gap") or 0) for e in gw.events_between(hb_from) if e.ev == "hb"),
                      default=0.0)
            stats["heartbeat_max_gap_s"] = gap
            assert gap < HEARTBEAT_MAX_GAP_S, f"{scn.id}: gateway event loop froze for {gap}s"

            probe_reqs = _main_requests_for(srv, f"[[probe:{nonce}:{n}]]")
            assert probe_reqs, f"{scn.id}: follow-up never reached the provider"
            sent = probe_reqs[-1]["messages"]
            if history_kept:
                assert any(m.get("role") == "user" and marker in _text_of(m.get("content")) for m in sent), (
                    f"{scn.id}: follow-up ran without the faulted turn in its history (not the same session)")
            assert unanswered_tool_calls(sent) == [], f"{scn.id}: model was sent unanswered tool_calls"

            session_id = _session_with(state_db, marker)
            assert session_id, f"{scn.id}: faulted user message never persisted"
            persisted = _persisted(state_db, session_id)
            assert unanswered_tool_calls(persisted) == [], f"{scn.id}: state.db has unanswered tool_calls"
            if scn.fault == "parallel":
                _assert_parallel_results(persisted, nonce, "state.db")
                _assert_parallel_results(sent, nonce, "provider request")

            # ── SIGTERM: the gateway leaves on its own and takes every child with it ──
            gw.stop()
            stats["shutdown_s"] = gw.shutdown_s and round(gw.shutdown_s, 2)
            assert gw.shutdown_s is not None, f"{scn.id}: gateway ignored SIGTERM{gw.log_tail()}"
            assert gw.exit_code is not None and gw.exit_code >= 0, (
                f"{scn.id}: gateway died by signal {gw.exit_code}{gw.log_tail()}")
        else:
            session_id = _session_with(state_db, marker)
            if session_id:
                assert unanswered_tool_calls(_persisted(state_db, session_id)) == []

        survivors = wait_no_tagged(tag)
        assert survivors == [], f"{scn.id}: orphans after exit: {describe_pids(survivors)}"
        assert integrity_ok(state_db) == "ok"
        return stats
    finally:
        gw.stop()
        kill_tagged(tag)


# ── pytest wiring: all scenarios run concurrently (one gateway each), asserted per test ──


@pytest.fixture(scope="module")
def scenario_futures(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    # Only the scenarios selected for this session (-k) are started.
    selected = {
        item.callspec.params["scn"].id for item in request.session.items
        if getattr(item, "callspec", None) is not None and "scn" in item.callspec.params
    }
    with FakeLLMServer(responder) as srv, ThreadPoolExecutor(
            max_workers=SCENARIO_WORKERS, thread_name_prefix="chaos-gw") as pool:
        futures: dict[str, Future] = {
            scn.id: pool.submit(run_scenario, scn, srv, tmp_path_factory.mktemp(scn.id))
            for scn in SCENARIOS if scn.id in selected
        }
        yield futures
        for fut in futures.values():
            fut.cancel()


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_gateway_turn_stays_live_under_fault(scn: Scenario, scenario_futures) -> None:
    stats = scenario_futures[scn.id].result(timeout=len(SCENARIOS) * (TURN_DEADLINE_S * 2 + 60))
    print(json.dumps(stats))
