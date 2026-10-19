"""Dispatcher SIGKILLed mid-tick and restarted: the board keeps every real task row (#119003 class).

A real ``hermes kanban dispatch`` process is SIGKILLed at three points of a tick — right after it
claimed a card (before the spawn), right after it spawned a worker, and immediately after launch —
then plain ticks run until the board drains. Workers are real ``hermes chat -q`` processes (they
survive the dispatcher: own session) and the model is the recording fake.

Invariants, read from kanban.db and the provider's request log:

* the ``tasks`` table holds exactly the created ids with their titles and bodies — no row
  destroyed, replaced or invented (no status-word placeholder id);
* every card ends ``done`` with exactly one ``completed`` run and one ``completed`` event;
* each card's ``kanban_complete`` was billed exactly once (no duplicate worker ran a card) — also
  when the kill lands after the spawn but before the dispatcher recorded the worker's pid;
* no card ever had two runs open at the same time.

A second scenario gives a live worker a claim TTL shorter than its provider call: the dispatcher
must extend that claim on every expired tick, never reclaim the card and spawn a duplicate.
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter

import pytest

from tests.e2e.core.kanban._helpers import PY, Board, pid_alive, wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="SIGKILL + /proc liveness"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs this board's reparented workers
]

N_CARDS = 5
_TASK_RE = re.compile(r"work kanban task (t_[0-9a-f]+)")


class Completer:
    """Every worker completes its own card; billing is counted per card id."""

    def __init__(self) -> None:
        self.completes: Counter[str] = Counter()
        self._lock = threading.Lock()

    def __call__(self, rec: dict):
        msgs = rec["body"]["messages"]
        if msgs[-1].get("role") == "tool":
            return Text("card closed")
        tid = next((m for m in (_TASK_RE.search(str(x.get("content"))) for x in msgs) if m), None)
        assert tid, "worker prompt did not name its task"
        with self._lock:
            self.completes[tid.group(1)] += 1
        return ToolCall("kanban_complete", {"summary": f"done {tid.group(1)}"})


def _event_count(board: Board, kind: str) -> int:
    if not board.db_path.exists():
        return 0
    return board._q("SELECT COUNT(*) AS n FROM task_events WHERE kind = ?", (kind,))[0]["n"]


def _kill_dispatcher_after(board: Board, kind: str | None) -> None:
    """Launch one real dispatch tick and SIGKILL it once a new ``kind`` event lands (None: at once)."""
    before = _event_count(board, kind) if kind else 0
    proc = subprocess.Popen(
        [PY, "-m", "hermes_cli.main", "kanban", "dispatch", "--json"], cwd=str(board.root),
        env=board.env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if kind:
            wait_until(lambda: _event_count(board, kind) > before or proc.poll() is not None, 60,
                       f"dispatcher to write a {kind} event", interval=0.01)
    finally:
        proc.send_signal(signal.SIGKILL)  # windows-footgun: ok — Linux-gated (module skips off Linux)
        proc.wait(timeout=30)
    for row in board.tasks():
        if row["worker_pid"]:
            board.spawned_pids.add(int(row["worker_pid"]))


def _overlapping_runs(runs: list[dict]) -> list[tuple[int, int]]:
    spans = [(r["id"], r["started_at"], r["ended_at"] or time.time()) for r in runs]
    return [(a[0], b[0]) for i, a in enumerate(spans) for b in spans[i + 1:]
            if a[1] < b[2] and b[1] < a[2] and a[2] > a[1] and b[2] > b[1]]


def test_dispatcher_sigkill_mid_tick_never_destroys_or_duplicates_cards(tmp_path) -> None:
    completer = Completer()
    with FakeLLMServer(completer) as srv:
        board = Board(tmp_path, srv.base_url, env_extra={"HERMES_KANBAN_CLAIM_TTL_SECONDS": "3"})
        try:
            created = {}
            for i in range(N_CARDS):
                title, body = f"chaos card {i}", f"body of chaos card {i}: keep me intact"
                created[board.create(title, "--body", body)] = (title, body)
            for kind in ("claimed", "spawned", None):
                _kill_dispatcher_after(board, kind)
            # Plain restarts until the board drains (stranded claims expire after the 3 s TTL).
            def drained() -> bool:
                if all(board.task(t)["status"] == "done" for t in created):
                    return True
                board.dispatch()
                return False
            wait_until(drained, 150, "every card to finish after the dispatcher restarts", interval=0.5)
            for pid in list(board.spawned_pids):
                wait_until(lambda p=pid: not pid_alive(p), 60, f"worker {pid} to exit")

            rows = board.tasks()
            assert {r["id"]: (r["title"], r["body"]) for r in rows} == created, rows
            for tid in created:
                runs = board.runs(tid)
                done = [r for r in runs if r["outcome"] == "completed"]
                assert len(done) == 1 and done[0]["summary"] == f"done {tid}", board.diag(tid)
                assert len(board.events(tid, "completed")) == 1, board.diag(tid)
                assert not _overlapping_runs(runs), board.diag(tid)
            assert dict(completer.completes) == {t: 1 for t in created}, (
                completer.completes, [board.diag(t) for t, n in completer.completes.items() if n != 1])
            # Vacuity guard: the claim-then-kill round really stranded a claim that the restarted
            # dispatcher had to recover (a kill that always landed between ticks proves nothing).
            outcomes = Counter(r["outcome"] for t in created for r in board.runs(t))
            assert outcomes["reclaimed"] >= 1, outcomes
        finally:
            board.kill_workers()


class SlowModel:
    """The worker's first call hangs on the provider (no chunk, no tool, so no heartbeat) until
    ``release``; every first call is counted so a duplicate worker on the card shows up as a bill."""

    def __init__(self) -> None:
        self.first_calls = 0
        self.hanging = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, rec: dict):
        if rec["body"]["messages"][-1].get("role") == "tool":
            return Text("slow card closed")
        with self._lock:
            self.first_calls += 1
        self.hanging.set()
        self.release.wait(120)
        return ToolCall("kanban_complete", {"summary": "slow model finally answered"})


def test_ttl_expiry_extends_a_live_hung_workers_claim_instead_of_respawning(tmp_path) -> None:
    """A claim TTL shorter than one provider call: every tick past expiry must extend the live
    worker's claim (``claim_extended``), never reclaim it and spawn a second worker beside it."""
    ttl = 2
    model = SlowModel()
    with FakeLLMServer(model) as srv:
        board = Board(tmp_path, srv.base_url, env_extra={"HERMES_KANBAN_CLAIM_TTL_SECONDS": str(ttl)})
        try:
            tid = board.create("slow model card")
            board.dispatch()
            pid = int(board.task(tid)["worker_pid"])
            wait_until(model.hanging.is_set, 90, f"worker to hang on its provider call\n{board.diag(tid)}")
            first_expiry = int(board.task(tid)["claim_expires"])
            # Tick until two TTL windows have passed and two ticks saw the claim expired, or the
            # dispatcher gave the card away (then the assertions below name what went wrong).
            def settled() -> bool:
                board.dispatch()
                if board.events(tid, "reclaimed") or len(board.events(tid, "spawned")) > 1:
                    return True
                return (int(time.time()) > first_expiry + 2 * ttl
                        and len(board.events(tid, "claim_extended")) >= 2)
            try:
                wait_until(settled, 60, "two ticks past the claim TTL", interval=0.2)
            except AssertionError as exc:
                raise AssertionError(f"{exc}\n{board.diag(tid)}") from None
            assert not board.events(tid, "reclaimed"), f"live worker's claim reclaimed\n{board.diag(tid)}"
            task = board.task(tid)
            assert pid_alive(pid) and task["worker_pid"] == pid, board.diag(tid)
            assert task["status"] == "running" and int(task["claim_expires"]) > first_expiry, board.diag(tid)
            assert [e["payload"]["pid"] for e in board.events(tid, "spawned")] == [pid], board.diag(tid)
            assert all(e["payload"]["worker_pid"] == pid for e in board.events(tid, "claim_extended"))
            assert model.first_calls == 1, f"a second worker billed the card\n{board.diag(tid)}"
            model.release.set()
            board.wait_worker_exit(tid, pid)
            wait_until(lambda: board.task(tid)["status"] == "done", 30, f"card done\n{board.diag(tid)}")
            assert [r["outcome"] for r in board.runs(tid)] == ["completed"], board.diag(tid)
        finally:
            model.release.set()
            board.kill_workers()
