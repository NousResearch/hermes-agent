"""A host/transport refusal is a retryable incompletion, never a card failure.

Under host memory/IO pressure the gateway's event loop stalls and the worker's websocket drops
mid-submit. The submitter then exits non-zero having written ``Kanban owner refused: …``, and
``_exit_code_kind`` reads that as a plain ``nonzero_exit`` crash: the card's
``consecutive_failures`` climbs, the breaker auto-blocks, and the retry re-derives work the dead
attempt had already pushed. Measured on one host in one hour: 15 such deaths across four profiles
and three boards, 20 cards moved to ``blocked``.

Three distinct host signatures reach the board through the same exit 1, and only one of them is
safe to resend immediately, so they need opposite retry policies:

* the owner was never reached (websocket open failed or timed out, no ready owner) — nothing of
  the card ran, so a prompt retry duplicates nothing;
* the owner admitted the claim (its ``Session: <id>`` was printed) and the transport then died —
  the turn may have completed and may already have pushed, so reconcile that session, do not redo;
* the owner answered but its admission write never settled the board lock, leaving the receipt
  ``unknown`` — the attempt stays retained by the owner, so this too is reconcile, never resend.

The invariant these tests guard: a refusal produced by the *host* is booked on the infrastructure
lane the spawn refusal (#114720) already uses — recorded, retryable, never counted, never tripped —
while a refusal the *owner answered* (an RPC error frame: lock mismatch, wrong run) stays a counted
card failure, so a genuinely poisoned card still reaches the breaker.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_transport as kbt
from hermes_cli import kanban_worker_client as kwc
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, GatewayUnavailableError
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

# The codes as literals, so a rename in the module is a breaking change here rather than a silent
# re-aliasing. ``man(3)``: 71 = EX_OSERR (an internal system error — the host's, not the task's),
# 70 = EX_IOERR (a non-recoverable input/output error: the operation may have completed anyway).
HOST_REFUSAL = 71
TURN_UNKNOWN = 70

SESSION_A = "local-" + "a" * 64


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _refused_worker(
    conn, tid: str, pid: int, rc: int, refusal: str, *, session_id: str = "",
) -> None:
    """Claim ``tid`` for a worker that exited ``rc`` after the host refused it.

    Booked the way a real burst books it: the submitter is not a child of this dispatcher process,
    so its exit reaches the sweep through the trailer it wrote in the task log — and only an
    admitted submit ever printed the ``Session:`` line the refusal tells the operator to resume.
    """
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Session: {session_id}"] if session_id else []
    lines += [refusal, f"{KANBAN_WORKER_EXIT_TRAILER}{rc}"]
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _latest(conn, tid: str):
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    return run, kb.get_task(conn, tid)


def _event_kinds(conn, tid: str) -> list[str]:
    return [row[0] for row in conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)
    ).fetchall()]


# ---------------------------------------------------------------- dispatcher accounting


def test_never_submitted_refusal_counts_nothing_and_never_blocks(kanban_home):
    """A worker the owner never reached exits 71 (EX_OSERR): recorded as an infrastructure
    incompletion, ``consecutive_failures`` untouched, card back to ``ready`` — and the breaker
    cannot trip on it even at ``failure_limit=1``, because a host burst is not the card's fault."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        for i in range(3):
            _refused_worker(
                conn, tid, 71000 + i, HOST_REFUSAL,
                "Kanban owner refused: Gateway connection failed; no local fallback",
            )
            # The crash rate a reviewer reads must not carry the host's own transport: this return
            # is crashed-only, so an infrastructure burst would silently inflate it (and did).
            assert kbd.detect_crashed_workers(conn) == []
            kb.recompute_ready(conn, failure_limit=1)

        run, task = _latest(conn, tid)
        assert task.status == "ready", task.status
        assert task.consecutive_failures == 0
        assert run["outcome"] == "transport_refused"
        meta = kb._json_dict(run["metadata"])
        assert meta["infrastructure"] is True
        assert meta["reconcile_required"] is False
        # Board history that says "crashed" is exactly what made this class read as card failure.
        assert "crashed" not in _event_kinds(conn, tid)


def test_transport_codes_are_distinct_from_the_reserved_worker_codes():
    """Contract between the two exit-code tables: the submitter's refusal codes must not collide
    with the ones the worker CLI already exits with, or one lane would silently book the other."""
    reserved = {
        kb.KANBAN_RATE_LIMIT_EXIT_CODE, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
    }
    assert {kbt.KANBAN_HOST_REFUSAL_EXIT_CODE, kbt.KANBAN_TURN_UNKNOWN_EXIT_CODE}.isdisjoint(reserved)
    assert kbt.KANBAN_HOST_REFUSAL_EXIT_CODE != kbt.KANBAN_TURN_UNKNOWN_EXIT_CODE


def test_post_admission_refusal_is_reconciled_not_resent(kanban_home):
    """The turn may have completed, so the retry must RESUME the session instead of redoing
    possibly-pushed work: distinct code (70 EX_IOERR), ``reconcile_required``, the session id in
    durable metadata, and the respawn guard spaces the retry."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _refused_worker(
            conn, tid, 72001, TURN_UNKNOWN,
            "Kanban owner refused: Gateway disconnected; turn outcome is unknown. Resume the "
            "printed session ID to check whether it completed or was interrupted; do not resend "
            "the work.",
            session_id=SESSION_A,
        )

        kbd.detect_crashed_workers(conn)

        run, task = _latest(conn, tid)
        assert task.consecutive_failures == 0
        assert run["outcome"] == "transport_refused"
        meta = kb._json_dict(run["metadata"])
        assert meta["reconcile_required"] is True
        assert meta["session_id"] == SESSION_A
        # The retry worker reads ``last_failure_error`` as context for its own attempt: it has to
        # say "do not redo" and name the session to resume, or the fleet re-does pushed work.
        assert "do not redo" in task.last_failure_error.lower()
        assert SESSION_A in task.last_failure_error
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"


def test_never_submitted_refusal_retries_promptly(kanban_home):
    """Nothing was submitted, so a retry cannot run the claim twice — the cooldown that spaces
    reconcile-required refusals must NOT hold this one back."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _refused_worker(
            conn, tid, 73001, HOST_REFUSAL,
            "Kanban owner refused: Gateway connection failed; no local fallback",
        )
        kbd.detect_crashed_workers(conn)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_managed_interpreter_code_does_not_mask_the_submitter_refusal(kanban_home):
    """The burst's dominant shape, and the reason a code-only fix would have missed it.

    A managed worker's authority is the ``worker_result`` event the owner writes for its bound
    interpreter (``managed_run_exit_code``). When the transport dies after admission the
    interpreter exits 1 and that code outranks the submitter's trailer, so the refusal is visible
    only in the trailer channel. A refusal code seen in ANY channel books the infrastructure lane:
    no card's own work can produce 70 or 71."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _refused_worker(
            conn, tid, 74001, TURN_UNKNOWN,
            "Kanban owner refused: Gateway disconnected; turn outcome is unknown",
            session_id=SESSION_A,
        )
        # The owner's own record of this run's exit: rc 1, the value that on base outranks the
        # trailer and is booked as a counted crash.
        run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
        kb._append_event(conn, tid, "worker_result",
                         {"pid": 74001, "claim_lock": kb.get_task(conn, tid).claim_lock,
                          "exit_code": 1, "last_output": ""}, run_id=run_id)
        conn.commit()

        kbd.detect_crashed_workers(conn)

        run, task = _latest(conn, tid)
        assert task.consecutive_failures == 0, task.last_failure_error
        assert run["outcome"] == "transport_refused"
        assert kb._json_dict(run["metadata"])["reconcile_required"] is True


def test_owner_answered_refusal_still_counts_as_a_card_failure(kanban_home):
    """Negative control against laundering: an owner that ANSWERED and refused (an RPC error frame,
    not an ambiguous transport loss) is a real problem with the claim. rc 1 stays a counted crash so
    a genuinely poisoned card still reaches the breaker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _refused_worker(conn, tid, 75001, 1, "Kanban owner refused: claim lock does not match")
        kbd.detect_crashed_workers(conn)
        _, task = _latest(conn, tid)
        assert task.consecutive_failures == 1
        assert "crashed" in _event_kinds(conn, tid)


def test_burst_stamps_host_pressure_and_names_capacity(kanban_home, caplog):
    """The signature of capacity rather than of cards: several workers refused inside one tick. The
    refusal payload carries the host's own numbers (PSI, load, memory, swap) and the log names
    memory/IO pressure, so the next operator reads a cause instead of "Gateway disconnected"."""
    pressure = {
        "psi_mem_full_avg60": 4.47, "psi_io_full_avg60": 18.15, "load1": 27.08, "load15": 19.4,
        "mem_available_kib": 20_971_520, "swap_used_kib": 7_340_032,
    }
    with kbc.connect() as conn:
        tids = []
        for i, pid in enumerate((76001, 76002, 76003)):
            tid = kb.create_task(conn, title=f"t{i}", assignee="a")
            tids.append(tid)
            _refused_worker(
                conn, tid, pid, TURN_UNKNOWN,
                "Kanban owner refused: Gateway disconnected; turn outcome is unknown",
                session_id=SESSION_A,
            )
        original = kbd._host_pressure_sample
        kbd._host_pressure_sample = lambda: dict(pressure)
        try:
            with caplog.at_level("WARNING"):
                kbd.detect_crashed_workers(conn)
        finally:
            kbd._host_pressure_sample = original

        for tid in tids:
            run = conn.execute(
                "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            assert kb._json_dict(run["metadata"])["host_pressure"] == pressure

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "pressure" in text.lower()
        assert "3" in text


# ------------------------------------------------------------------ submitter exit codes


class _FakeEvents:
    async def get(self):
        await asyncio.sleep(0)


class _FakeClient:
    """Answers ``kanban.run`` and then dies, or fails before anything was submitted."""

    def __init__(self, *, fail_at: str = "", error: Optional[Exception] = None,
                 receipt_status: str = "terminal"):
        self.fail_at, self.error, self.receipt_status = fail_at, error, receipt_status
        self.events = _FakeEvents()
        self.calls: list[str] = []

    async def rpc(self, method, **kw):
        self.calls.append(method)
        if method == self.fail_at and self.error is not None:
            raise self.error
        if method == "kanban.run":
            return {"session_id": SESSION_A,
                    "receipt": {"status": self.receipt_status, "admission_id": "adm-1"}}
        return {"status": self.receipt_status, "admission_id": "adm-1"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _submit_client(monkeypatch, client, tmp_path) -> int:
    """Drive the submitter through its real entry point and return the code it exits with."""
    for key in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_DB", "HERMES_TUI_GATEWAY_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "b")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:1")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setattr(kwc, "connect_gateway", lambda: client, raising=False)
    import hermes_cli.gateway_client as gc
    monkeypatch.setattr(gc, "connect_gateway", lambda: client)
    monkeypatch.setattr(kwc, "admission_answer_budget", lambda: 5.0, raising=False)
    return kwc._submit()


def test_connect_failure_exits_host_refusal_code(kanban_home, tmp_path, monkeypatch):
    """Never reached the owner: the submitter exits 71 so the retry is prompt and uncounted."""
    client = _FakeClient(fail_at="kanban.run",
                         error=GatewayClientError("Gateway connection failed; no local fallback"))
    assert _submit_client(monkeypatch, client, tmp_path) == HOST_REFUSAL


def test_no_ready_owner_exits_host_refusal_code(kanban_home, tmp_path, monkeypatch):
    """A gateway still draining past the submitter's wait deadline is the same class: nothing was
    submitted, so the card keeps its whole retry budget."""
    client = _FakeClient(fail_at="kanban.run",
                         error=GatewayUnavailableError("Gateway owner is draining", "draining"))
    monkeypatch.setattr(kwc, "OWNER_WAIT_S", -1.0, raising=False)
    assert _submit_client(monkeypatch, client, tmp_path) == HOST_REFUSAL


def test_disconnect_after_admission_exits_turn_unknown_code(kanban_home, tmp_path, monkeypatch):
    """Admitted, then the socket died: exit 70, so the retry reconciles the session."""
    client = _FakeClient(fail_at="prompt.receipt",
                         error=GatewayClientError("Gateway disconnected; turn outcome is unknown."),
                         receipt_status="running")
    assert _submit_client(monkeypatch, client, tmp_path) == TURN_UNKNOWN


def test_unknown_admission_receipt_exits_turn_unknown_code(kanban_home, tmp_path, monkeypatch):
    """The owner answered but its admission write never settled the board lock: the attempt is
    retained by the owner, so it is reconciled (70), never resent as a plain crash (1)."""
    assert _submit_client(monkeypatch, _FakeClient(receipt_status="unknown"), tmp_path) == TURN_UNKNOWN


def test_owner_rpc_refusal_stays_a_counted_exit(kanban_home, tmp_path, monkeypatch):
    """The submitter must not upgrade an owner's own refusal into an infrastructure code."""
    client = _FakeClient(fail_at="kanban.run",
                         error=GatewayRPCError("claim lock does not match", code=-32001))
    assert _submit_client(monkeypatch, client, tmp_path) == 1


def test_admission_answer_timeout_is_reconciled_not_resent(kanban_home, tmp_path, monkeypatch):
    """The third signature PM measured: ``kanban.run`` was SENT and never answered because the
    owner's admission write could not settle the board lock. The owner may still admit after this
    submitter gives up, so this is the reconcile class (70) — claiming "nothing ran" (71) here
    would let the retry run a claim the owner is already working on."""
    client = _FakeClient(fail_at="kanban.run", error=TimeoutError("admission answer budget"))
    assert _submit_client(monkeypatch, client, tmp_path) == TURN_UNKNOWN


def test_submitter_bug_stays_a_counted_exit(kanban_home, tmp_path, monkeypatch):
    """Anti-laundering control for the new codes: an error that is not a transport refusal at all
    (the submitter's own bug) must keep its counted exit, or every failure would join the
    never-counted lane and the breaker would stop seeing real poison."""
    client = _FakeClient(fail_at="kanban.run", error=ValueError("unexpected internal failure"))
    assert _submit_client(monkeypatch, client, tmp_path) == 1


@pytest.mark.platforms("linux")
def test_host_pressure_sample_reads_the_host_and_never_raises():
    """The numbers recorded on a burst have to come from the real kernel interface, not a stub:
    load average, PSI stall and memory/swap on Linux. A sampler that raised would take the reclaim
    sweep down with it, which is why the contract is best-effort by construction."""
    sample = kbt.host_pressure_sample()
    assert sample["load1"] >= 0.0
    assert sample["psi_memory_full_avg60"] >= 0.0
    assert sample["mem_available_kib"] > 0
