"""Kanban worker SIGKILLed mid-run, through the real dispatcher CLI and real ``hermes chat -q`` workers.

One scenario is driven once per module (real processes end to end, the model is the recording fake):

1. attempt 1 answers in plain text and exits rc=0 without a terminal board call (a protocol
   violation: its log ends with the CLI exit summary and the ``[kanban-worker-exit] rc=0`` trailer);
2. the next tick reaps it and spawns attempt 2, which heartbeats and is then SIGKILLed while its
   provider request hangs (no trailer, no exit summary of its own);
3. a ``--max 0`` tick reaps the corpse, an operator ``kanban claim --ttl 1`` opens a fresh run,
   and once that claim expires a normal tick reclaims it and spawns attempt 3, which completes.

Every verdict is read from kanban.db (tasks / task_runs / task_events) and from the provider's
request log (how many model calls each attempt was billed).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

import pytest

from tests.e2e.core.kanban._helpers import Board, pid_alive, wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer, Hang, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="SIGKILL + /proc liveness"),
    # Workers are grandchildren (spawned by the dispatcher child, then reparented); the only
    # signals sent are SIGKILLs to PIDs this module read from its own scratch board.
    pytest.mark.live_system_guard_bypass,
]

# The scripted history spends two breaker ticks (the SIGKILL and the expired operator claim); a
# higher limit keeps the card retryable whichever way those deaths are booked.
TICK = ("--failure-limit", "5")
ATTEMPT_ONE_PROSE = "ATTEMPT_ONE_PROSE the card could not be finished in this run."
ATTEMPT_TWO_MARK = "ATTEMPT_TWO_MARK heartbeat sent, about to call the provider again."

# Red on current main for a tracked, open bug. Strict: the test FAILS the moment the bug is fixed,
# forcing the entry out so the contract is enforced again.
KNOWN: dict[str, str] = {
    "test_fresh_claim_does_not_inherit_previous_heartbeat":
        "#119155 a fresh claim keeps the previous run's last_heartbeat_at",
    "test_sigkilled_attempt_is_booked_as_a_crash_of_its_own":
        "#121255 a SIGKILLed worker is booked with the previous attempt's rc=0 trailer",
    "test_crash_diagnostic_comes_from_the_crashed_attempt":
        "#119618 crash diagnostic carries an older attempt's output",
}


class KnownGap(AssertionError):
    """The tracked bug's own assertion. Only this is xfailed; any harness failure stays red."""


def known(name: str):
    return pytest.mark.xfail(strict=True, raises=KnownGap, reason=KNOWN[name])


class Director:
    """Scripts the fake model per worker attempt (workers run strictly one after another)."""

    def __init__(self) -> None:
        self.attempt = 0
        self.billed: dict[int, int] = {}
        self.hanging = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, rec: dict):
        msgs = rec["body"]["messages"]
        with self._lock:
            if not any(m.get("role") in ("assistant", "tool") for m in msgs):
                self.attempt += 1
            n = self.attempt
            self.billed[n] = self.billed.get(n, 0) + 1
        last = msgs[-1].get("role")
        if n == 1:
            return Text(ATTEMPT_ONE_PROSE)
        if n == 2:
            if last == "tool":
                self.hanging.set()
                return Hang(300)
            return ToolCall("kanban_heartbeat", {"note": "attempt two alive"}, text=ATTEMPT_TWO_MARK)
        return Text("attempt three done") if last == "tool" else ToolCall(
            "kanban_complete", {"summary": "ATTEMPT_THREE_DONE"})


@dataclass
class Scenario:
    board: Board
    tid: str
    director: Director
    w2: int
    hb_before_kill: int
    after_claim: dict = field(default_factory=dict)
    claim_run: dict = field(default_factory=dict)
    ticks_after_done: list = field(default_factory=list)


def _drive(board: Board, director: Director) -> Scenario:
    tid = board.create("sigkill chaos card")
    board.dispatch(*TICK)
    w1 = board.task(tid)["worker_pid"]
    board.wait_worker_exit(tid, w1)
    board.dispatch(*TICK)  # reaps attempt 1 (protocol violation), spawns attempt 2
    w2 = int(board.task(tid)["worker_pid"])
    assert w2 != w1, board.diag(tid)
    wait_until(director.hanging.is_set, 90, f"attempt 2 to hang on its provider call\n{board.diag(tid)}")
    hb = board.task(tid)["last_heartbeat_at"]
    assert hb, f"attempt 2 never heartbeat\n{board.diag(tid)}"
    os.kill(w2, signal.SIGKILL)
    wait_until(lambda: not pid_alive(w2), 15, "SIGKILLed worker to disappear")
    # The fresh claim must start strictly after attempt 2's last heartbeat second.
    wait_until(lambda: int(time.time()) > int(hb) + 1, 5, "clock to pass the last heartbeat")
    board.dispatch(*TICK, "--max", "0")  # reap only
    assert board.task(tid)["status"] == "ready", board.diag(tid)
    board.cli("claim", tid, "--ttl", "1")
    sc = Scenario(board, tid, director, w2, int(hb))
    sc.after_claim = board.task(tid)
    sc.claim_run = board.runs(tid)[-1]
    wait_until(lambda: int(time.time()) > int(sc.after_claim["claim_expires"]), 5, "operator claim to expire")
    board.dispatch(*TICK)  # reclaims the expired claim, spawns attempt 3
    w3 = board.task(tid)["worker_pid"]
    assert w3, board.diag(tid)
    board.wait_worker_exit(tid, int(w3))
    wait_until(lambda: board.task(tid)["status"] == "done", 30, f"card done\n{board.diag(tid)}")
    sc.ticks_after_done = [board.dispatch(*TICK) for _ in range(2)]
    return sc


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory):
    director = Director()
    with FakeLLMServer(director) as srv:
        board = Board(tmp_path_factory.mktemp("kanban-sigkill"), srv.base_url)
        try:
            yield _drive(board, director)
        finally:
            board.kill_workers()


def _run_of_attempt(sc: Scenario, n: int) -> dict:
    runs = sc.board.runs(sc.tid)
    assert len(runs) >= n, sc.board.diag(sc.tid)
    return runs[n - 1]


def test_sigkilled_worker_is_reclaimed_and_the_retry_completes_once(scenario: Scenario) -> None:
    sc, b = scenario, scenario.board
    runs = b.runs(sc.tid)
    outcomes = [r["outcome"] for r in runs]
    # attempt 1 (violation), attempt 2 (SIGKILL), operator claim (expired), attempt 3.
    assert outcomes == ["crashed", "crashed", "reclaimed", "completed"], b.diag(sc.tid)
    assert runs[1]["worker_pid"] == sc.w2 and runs[1]["ended_at"], b.diag(sc.tid)
    task = b.task(sc.tid)
    assert task["status"] == "done" and task["worker_pid"] is None and task["claim_lock"] is None
    assert [r for r in runs if r["outcome"] == "completed"][0]["summary"] == "ATTEMPT_THREE_DONE"
    # Billing: every attempt paid only for its own turns, nothing ran after the card closed.
    billed = sc.director.billed
    assert sorted(billed) == [1, 2, 3] and billed[2] == 2 and billed[3] == 2, billed
    assert all(not t["spawned"] for t in sc.ticks_after_done), sc.ticks_after_done
    assert len(b.events(sc.tid, "completed")) == 1


@known("test_fresh_claim_does_not_inherit_previous_heartbeat")
def test_fresh_claim_does_not_inherit_previous_heartbeat(scenario: Scenario) -> None:
    sc = scenario
    run = sc.claim_run
    assert run["outcome"] is None and run["status"] == "running", sc.board.diag(sc.tid)
    assert sc.after_claim["current_run_id"] == run["id"]
    hb = sc.after_claim["last_heartbeat_at"]
    if hb is not None and int(hb) < int(run["started_at"]):
        raise KnownGap(
            f"fresh run {run['id']} (started {run['started_at']}) carries last_heartbeat_at={hb}, "
            f"attempt 2's value {sc.hb_before_kill}")


@known("test_sigkilled_attempt_is_booked_as_a_crash_of_its_own")
def test_sigkilled_attempt_is_booked_as_a_crash_of_its_own(scenario: Scenario) -> None:
    sc = scenario
    run2 = _run_of_attempt(sc, 2)
    assert run2["worker_pid"] == sc.w2
    kinds = [(e["kind"], e["payload"]) for e in sc.board.events(sc.tid) if e["run_id"] == run2["id"]]
    booked = [(k, p) for k, p in kinds if k in ("crashed", "protocol_violation", "rate_limited")]
    assert len(booked) == 1, kinds
    kind, payload = booked[0]
    if kind != "crashed" or (payload or {}).get("exit_code") == 0:
        raise KnownGap(f"SIGKILLed attempt booked as {kind} {payload}")


@known("test_crash_diagnostic_comes_from_the_crashed_attempt")
def test_crash_diagnostic_comes_from_the_crashed_attempt(scenario: Scenario) -> None:
    sc = scenario
    run1, run2 = _run_of_attempt(sc, 1), _run_of_attempt(sc, 2)
    # Vacuity guard: attempt 1's own diagnostic does carry its prose.
    assert "ATTEMPT_ONE_PROSE" in (run1["error"] or ""), run1
    if "ATTEMPT_ONE_PROSE" in (run2["error"] or ""):
        raise KnownGap(f"attempt 2's crash diagnostic quotes attempt 1: {run2['error'][-300:]!r}")
