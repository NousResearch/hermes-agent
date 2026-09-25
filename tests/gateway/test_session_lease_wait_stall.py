"""Regression for #121030: a bounded, live cross-process session turn lease wait is not a stuck agent.

A gateway turn that blocks in `SessionDB.acquire_session_turn_lease` because another Hermes process
holds the `session_turn_leases` row for the same durable session is *queued behind a live holder*,
not idle. `AIAgent.get_activity_summary()` is the only progress source both gateway stall paths
read:

* `gateway.run._watch_gateway_turn_inactivity` (per-turn inactivity watchdog; `agent.gateway_timeout`)
  abandons the turn with a hard interrupt and dumps wedged-turn stacks;
* `gateway.run_watchers.GatewaySessionWatchersMixin._check_session_stalls` (session stall watchdog;
  `agent.session_stall_timeout`) messages the user "I seem to be stuck ... /stop or /new".

The durable turn only starts *after* admission (`DurableTurnLease.start` stamps
`_touch_activity("starting new turn")`), so during the wait the summary advertises whatever stamped
the clock last: `last_activity=initializing` / `provenance=unknown`. The wait must publish itself
as known progress for its whole duration, and neither stall path may classify a live wait as stuck
(`/new` abandons a session another process is still writing to).

Real mechanism, compressed clock. The lease is the real SQLite row, held by a real second process;
the wait is the real `acquire_session_turn_lease` poll loop reached through the real
`admit_durable_turn_lease` admission; both watchdogs are the real functions. Only durations shrink
(`NOTICE_INTERVAL_S` and the thresholds below) so the regression runs in seconds instead of the
15-30 minutes in the report, while keeping the production ordering notice << threshold < wait budget.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from agent.session_activity import ActivityProvenance
from gateway.run import GatewayRunner, _watch_gateway_turn_inactivity
from hermes_state import SessionDB
from run_agent import AIAgent

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "durable-gateway-session-121030"
SESSION_KEY = "agent:main:telegram:dm:121030"

# Compressed clock. Production values in the comments; ordering and shape are preserved, and the
# observation window outlives every threshold so a wait can only pass by *keeping* progress visible.
NOTICE_INTERVAL_S = 0.5  # production: wait_notice_interval_seconds default 15.0
STALL_TIMEOUT_S = 2.0  # production: agent.session_stall_timeout default 300.0
INACTIVITY_TIMEOUT_S = 2.0  # production: agent.gateway_timeout default 1800.0
LEASE_WAIT_BUDGET_S = 20.0  # production: agent.turn_facade_lease.LEASE_WAIT_SECONDS 1800.0
IDLE_AT_WAIT_ENTRY_S = 30.0  # the agent's last stamp predates the wait (cached gateway agent)
OBSERVE_S = 3.0  # > every threshold: the wait must keep publishing progress for this window

# A real second process holding the real durability row. It prints HELD with its own holder string
# (`pid=<pid>:turn=...:platform=cli`, the shape `admit_durable_turn_lease` builds) and keeps that
# pid alive until it is told to release, so the waiter cannot reclaim the row as a dead holder.
_HOLDER_SOURCE = """
import os, sys
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(Path(sys.argv[1]))
session_id = sys.argv[2]
holder = "pid={}:turn=holder:platform=cli".format(os.getpid())
if not db.acquire_session_turn_lease(session_id, holder, ttl_seconds=600.0, wait_seconds=0.1):
    print("HOLDER-FAILED", flush=True)
    raise SystemExit(3)
print("HELD", holder, flush=True)
while True:
    line = sys.stdin.readline()
    if not line:
        break
    if line.strip() == "release":
        db.release_session_turn_lease(session_id, holder)
        print("RELEASED", flush=True)
        break
db.close()
"""


class _SessionLeaseHolder:
    """A live second process holding `session_turn_leases` for the session."""

    def __init__(self, db_path: Path, session_id: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_SOURCE, str(db_path), session_id],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = (self.proc.stdout.readline() or "").strip()
        if not line.startswith("HELD"):
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            self.close()
            raise AssertionError(f"lease holder process failed to start: {line!r} {stderr!r}")
        self.holder = line.split()[1]
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write("release\n")
                self.proc.stdin.flush()
                if self.proc.stdout is not None:
                    self.proc.stdout.readline()  # RELEASED
        finally:
            self.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def _compress_lease_wait_clock(monkeypatch) -> None:
    """Shrink only the wait's own sleep durations; the acquire mechanism is untouched.

    `acquire_session_turn_lease` is a real poll loop over the real row with a 1s poll and a 15s
    user-facing notice cadence. The production clock ratio (15s notices vs 300s/900s/1800s
    thresholds) is preserved at a test-sized scale so the wait's progress signal is observable.
    """
    real_acquire = SessionDB.acquire_session_turn_lease

    def acquire(
        self, session_id, holder, *, ttl_seconds=300.0, wait_seconds=1800.0,
        poll_interval_seconds=1.0, on_wait=None, wait_notice_interval_seconds=15.0,
        should_abort=None, acquire_patience_s=0.5,
    ):
        return real_acquire(
            self, session_id, holder, ttl_seconds=ttl_seconds, wait_seconds=wait_seconds,
            poll_interval_seconds=0.05, on_wait=on_wait,
            wait_notice_interval_seconds=NOTICE_INTERVAL_S, should_abort=should_abort,
            acquire_patience_s=acquire_patience_s,
        )

    monkeypatch.setattr(SessionDB, "acquire_session_turn_lease", acquire)


def _gateway_agent(db: SessionDB, status_callback) -> AIAgent:
    """An `AIAgent` stand-in whose activity clock, status plumbing and admission are production code.

    Only `__init__` is bypassed. Every method the lease wait and both stall paths read is the real
    implementation, and the pre-turn activity stamp mirrors `agent/agent_init.py`
    (`_last_activity_desc = "initializing"`, `provenance = unknown`).
    """
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = SESSION_ID
    agent.platform = "telegram"
    agent.model = "test-model"
    agent.log_prefix = ""
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._conversation_root_id = lambda: SESSION_ID
    agent._vprint = lambda *args, **kwargs: None
    agent.status_callback = status_callback
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    # Interrupt plumbing the inactivity reaper reaches through request_hard_interrupt.
    agent._hard_interrupt_requested = threading.Event()
    agent._pending_redirect_lock = threading.Lock()
    agent._tool_interrupt_reason = None
    agent._last_activity_ts = time.time() - IDLE_AT_WAIT_ENTRY_S
    agent._last_activity_desc = "initializing"
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    agent._turn_liveness_activity_generation = 0
    # ``get_activity_summary`` folds the iteration/tool counters into its snapshot.
    agent._current_tool = None
    agent._api_call_count = 0
    agent.max_iterations = sys.maxsize
    agent.iteration_budget = SimpleNamespace(used=0, max_total=0)
    for name in ("get_activity_summary", "_touch_activity", "_reset_activity_labels_after_turn"):
        setattr(agent, name, getattr(AIAgent, name).__get__(agent, AIAgent))
    return agent


class _LeaseWaitTurn:
    """A live holder process plus a real gateway-shaped turn waiting behind it."""

    def __init__(self, db_path, db, holder, agent, turn_thread, outcome, statuses, body_snapshot):
        self.db_path = db_path
        self.db = db
        self.holder = holder
        self.agent = agent
        self.turn_thread = turn_thread
        self.outcome = outcome
        self.statuses = statuses
        self.body_snapshot = body_snapshot

    def lease_row_holder(self) -> Optional[str]:
        """The holder currently owning the real row, read through a separate connection."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT holder FROM session_turn_leases LIMIT 1").fetchone()
        finally:
            conn.close()
        return None if row is None else row[0]

    def wait_statuses(self) -> List[str]:
        return [
            str(text) for _kind, text in self.statuses
            if text and "waiting" in str(text).lower()
        ]

    def finish(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Release the holder and stop the turn. Never raises: a failing test keeps its own error."""
        self.holder.release()
        self.turn_thread.join(timeout=timeout)
        return self.outcome.get("result")

    def close(self) -> None:
        if not self.turn_thread.is_alive():
            self.db.close()


def _start_lease_wait(monkeypatch, tmp_path) -> _LeaseWaitTurn:
    """Start (1) a real second process holding the lease and (2) a turn waiting on that same session."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session(SESSION_ID, source="telegram")
    db.append_messages_batch(
        SESSION_ID,
        [
            {"role": "user", "content": "first gateway message"},
            {"role": "assistant", "content": "first gateway reply"},
        ],
    )
    holder = _SessionLeaseHolder(db_path, SESSION_ID)

    statuses: List[Tuple[str, Any]] = []
    wait_entered = threading.Event()

    def status_callback(kind, text=None):
        statuses.append((kind, text))
        if text and "waiting" in str(text).lower():
            wait_entered.set()

    agent = _gateway_agent(db, status_callback)
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", LEASE_WAIT_BUDGET_S)
    _compress_lease_wait_clock(monkeypatch)

    body_snapshot: Dict[str, Any] = {}

    def _admitted_turn_body(_agent, _message, _system, history, *_args, **_kwargs):
        body_snapshot.update(_agent.get_activity_summary())
        _agent._touch_activity("api call 1 completed")
        return {"final_response": "done", "messages": list(history), "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", _admitted_turn_body)

    outcome: Dict[str, Any] = {}

    def _run_turn():
        try:
            outcome["result"] = AIAgent.run_conversation(
                agent, "follow-up from the gateway", conversation_history=[]
            )
        except BaseException as exc:  # surfaced by the tests that need a result, never swallowed
            outcome["error"] = exc

    turn_thread = threading.Thread(target=_run_turn, name="regression-gateway-turn", daemon=True)
    turn_thread.start()
    if not wait_entered.wait(timeout=15.0):
        holder.release()
        turn_thread.join(timeout=15.0)
        db.close()
        raise AssertionError(
            "harness error: the turn never entered the durable session turn lease wait"
        )
    return _LeaseWaitTurn(db_path, db, holder, agent, turn_thread, outcome, statuses, body_snapshot)


class _StallAdapter:
    def __init__(self) -> None:
        self._pending_messages: Dict[str, Any] = {}
        self.sent: List[Dict[str, Any]] = []

    async def send(self, chat_id, content, metadata=None):
        from gateway.platforms.base import SendResult

        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True)


def _stall_runner(agent) -> Tuple[GatewayRunner, _StallAdapter]:
    """Bare runner wired for `_check_session_stalls`: one adapter with one pending inbound."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    adapter = _StallAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {"fake": adapter}
    runner._profile_adapters = {}
    runner._running_agents = {SESSION_KEY: agent}
    runner._running_agents_ts = {}
    runner._queued_events = {}
    runner._session_stall_notified = {}
    runner._thread_metadata_for_source = lambda source, *args, **kwargs: {
        "thread_id": getattr(source, "thread_id", None)
    }
    adapter._pending_messages[SESSION_KEY] = SimpleNamespace(
        text="follow-up",
        source=SessionSource(chat_id="chat-121030", platform=Platform.TELEGRAM),
        timestamp=time.time(),
    )
    return runner, adapter


def test_live_lease_wait_publishes_itself_as_progress(tmp_path, monkeypatch):
    """The single progress source both stall paths read must show the wait as known progress."""
    turn = _start_lease_wait(monkeypatch, tmp_path)
    try:
        assert turn.lease_row_holder() == turn.holder.holder
        time.sleep(OBSERVE_S)
        assert turn.turn_thread.is_alive(), (
            "harness error: the turn left the lease wait before the observation window ended"
        )
        assert turn.lease_row_holder() == turn.holder.holder
        summary = turn.agent.get_activity_summary()
    finally:
        turn.finish()
        turn.close()

    idle = summary.get("seconds_since_activity")
    assert idle is not None
    assert idle < STALL_TIMEOUT_S, (
        "a live cross-process session turn lease wait must keep publishing progress, but idle grew "
        f"to {idle}s while the holder was alive (session stall timeout {STALL_TIMEOUT_S}s, "
        f"inactivity timeout {INACTIVITY_TIMEOUT_S}s)"
    )
    described = " ".join(
        str(summary.get(key) or "")
        for key in ("last_activity_description", "last_activity_provenance")
    ).lower()
    assert any(marker in described for marker in ("lease", "busy", "wait")), (
        "the lease wait must be visible as known state instead of the pre-turn stamp: "
        f"description={summary.get('last_activity_description')!r} "
        f"provenance={summary.get('last_activity_provenance')!r}"
    )


def test_inactivity_watcher_does_not_abandon_a_live_lease_wait(tmp_path, monkeypatch):
    """The per-turn inactivity watchdog must not classify a live bounded lease wait as a stuck turn."""
    turn = _start_lease_wait(monkeypatch, tmp_path)
    worker_done = threading.Event()
    timeout_fired = threading.Event()

    def _watch():
        _watch_gateway_turn_inactivity(
            agent_holder=[turn.agent],
            task_id=SESSION_ID,
            process_baseline=frozenset(),
            timeout=INACTIVITY_TIMEOUT_S,
            worker_done=worker_done,
            timeout_fired=timeout_fired,
            cleanup_lock=threading.Lock(),
            poll_interval=0.05,
            is_still_current=lambda: True,
        )

    watcher = threading.Thread(target=_watch, name="regression-turn-watchdog", daemon=True)
    watcher.start()
    try:
        time.sleep(OBSERVE_S)
        abandoned = timeout_fired.is_set()
        idle = turn.agent.get_activity_summary().get("seconds_since_activity")
    finally:
        worker_done.set()
        watcher.join(timeout=10.0)
        turn.finish()
        turn.close()

    assert not abandoned, (
        "the per-turn inactivity watchdog abandoned a live session turn lease wait after "
        f"{INACTIVITY_TIMEOUT_S}s and hard-interrupted the turn (idle={idle}s)"
    )


def test_session_stall_watchdog_does_not_call_a_live_lease_wait_stuck(tmp_path, monkeypatch):
    """The session stall watchdog must not tell the user a live lease wait is a stuck agent."""
    turn = _start_lease_wait(monkeypatch, tmp_path)
    runner, adapter = _stall_runner(turn.agent)
    try:
        asyncio.run(runner._check_session_stalls(STALL_TIMEOUT_S))
    finally:
        turn.finish()
        turn.close()

    delivered = [str(message["content"]) for message in adapter.sent]
    stuck = [text for text in delivered if "stuck" in text.lower() or "/new" in text]
    assert not stuck, (
        "the session stall watchdog reported a stuck agent while another Hermes process still holds "
        f"the session lease: {stuck}"
    )
    for text in delivered:
        lowered = text.lower()
        assert any(marker in lowered for marker in ("lease", "busy", "hermes process")), (
            f"a stall notice delivered during a live lease wait must name the holder state: {text!r}"
        )


def test_turn_resumes_after_the_holder_releases_without_leftover_wait_state(tmp_path, monkeypatch):
    """Once the holder releases, admission proceeds normally and the wait leaves nothing behind."""
    turn = _start_lease_wait(monkeypatch, tmp_path)
    try:
        result = turn.finish()
        assert result is not None, f"the waiting turn raised instead of returning: {turn.outcome}"
        assert turn.outcome.get("error") is None
        assert result.get("failed") is not True and result.get("interrupted") is not True
        assert result["final_response"] == "done"
        assert turn.lease_row_holder() is None, "the waited turn left a session turn lease row behind"
        assert getattr(turn.agent, "_active_session_turn_lease_holder", None) is None

        # The admitted turn's own progress replaced the wait: the clock is fresh and neither label
        # still advertises a lease wait (a wait marker left behind would re-arm both stall paths).
        body = turn.body_snapshot
        assert body.get("seconds_since_activity") is not None
        assert body["seconds_since_activity"] < STALL_TIMEOUT_S
        leftover = " ".join(
            str(body.get(key) or "")
            for key in ("last_activity_description", "last_activity_provenance")
        ).lower()
        assert not any(marker in leftover for marker in ("lease", "wait", "busy")), (
            f"the admitted turn still advertises the lease wait as its activity state: {leftover!r}"
        )

        # Normal admission resumes: the second turn acquires immediately, with no wait notice.
        statuses_before = list(turn.statuses)
        second = AIAgent.run_conversation(turn.agent, "second message", conversation_history=[])
        new_statuses = [str(text) for _kind, text in turn.statuses[len(statuses_before):] if text]
        assert [text for text in new_statuses if "waiting" in text.lower()] == []
        assert second["final_response"] == "done"
        assert turn.lease_row_holder() is None
        assert turn.outcome.get("error") is None
    finally:
        turn.finish()
        turn.close()
