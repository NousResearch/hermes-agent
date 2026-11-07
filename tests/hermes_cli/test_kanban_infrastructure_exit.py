"""A host-side stall must not spend the card's retry budget (#139612).

Two verdicts leave the submitter with one exit code (``KANBAN_INFRASTRUCTURE_EXIT_CODE``, 74 —
the sentinel chosen for the host-failure family):

* the owner never answered: silent past ``OWNER_WAIT_S``, never admitted or silent after
  admitting;
* the receipt settled ``unknown`` and the board cannot reconcile a ``worker_result`` for it.

The dispatcher books either death through the infrastructure seam of #114720: the run keeps
its ``infrastructure`` marker (the outage stays visible in the run history), but
``consecutive_failures`` is untouched, the breaker never trips, and the respawn guard spaces
the re-probes.

Reconcile: a receipt of ``unknown`` reads this run's ``worker_result`` first — gated by the
claim lock AND the ``worker_bound`` pid, the same invariant that let it be written — and
returns its REAL exit code. Missing, foreign or mismatched answers keep the old raise: an
ambiguous receipt never synthesizes success.

Every board below is a real kanban DB in a tmp HERMES_HOME: the reconcile and the reclaim
bookkeeping run against real rows, not stubs.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client as kwc
from hermes_cli.kanban_db import KANBAN_INFRASTRUCTURE_EXIT_CODE


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a freshly initialised kanban DB; returns its path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", raising=False)
    kb.init_db()
    return kb.kanban_db_path()


def _connect(db_path):
    """A board connection with row access by name — what every ``kanban_db`` reader assumes."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _claimed(db_path, *, title="card"):
    """A task claimed for this run: ``running``, an open run, a host-prefixed claim lock."""
    with _connect(db_path) as conn:
        task_id = kb.create_task(conn, title=title, assignee="a")
    with _connect(db_path) as conn:
        claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id
    return claimed


def _bind(db_path, task, *, pid, result_pid=None, exit_code=None, last_output="",
          claim_lock=None):
    """Write the events a real bound worker leaves: ``worker_bound`` always, ``worker_result``
    only when the attempt is supposed to HAVE a result. ``result_pid``/``claim_lock`` let a
    test plant another attempt's (or another interpreter's) result instead of ours."""
    with _connect(db_path) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "worker_bound",
                         {"pid": pid, "claim_lock": task.claim_lock},
                         run_id=task.current_run_id)
        if exit_code is not None:
            kb._append_event(conn, task.id, "worker_result",
                             {"pid": pid if result_pid is None else result_pid,
                              "claim_lock": task.claim_lock if claim_lock is None else claim_lock,
                              "exit_code": exit_code, "last_output": last_output},
                             run_id=task.current_run_id)


def _params(task, db_path):
    return {"board": "default", "task_id": task.id, "run_id": task.current_run_id,
            "claim_lock": task.claim_lock}


async def _no_sleep(_seconds):
    return None


# --------------------------------------------------------------------------- the registry


def test_the_sentinel_is_the_ex_ioerr_code():
    """74 is BSD EX_IOERR, the code chosen for the host-failure family so the fleet does not
    grow two competing sentinels; it must stay distinct from 75 (EX_TEMPFAIL) and 78
    (EX_CONFIG)."""
    assert KANBAN_INFRASTRUCTURE_EXIT_CODE == 74
    assert len({KANBAN_INFRASTRUCTURE_EXIT_CODE, kb.KANBAN_RATE_LIMIT_EXIT_CODE,
                kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE}) == 3


def test_exit_code_kind_classifies_infrastructure_apart():
    assert kbd._exit_code_kind(KANBAN_INFRASTRUCTURE_EXIT_CODE) == ("infrastructure_failure", 74)
    # The neighbours are untouched: the quota wall, the config wall, a plain failure.
    assert kbd._exit_code_kind(75) == ("rate_limited", 75)
    assert kbd._exit_code_kind(78) == ("terminal_provider", 78)
    assert kbd._exit_code_kind(1) == ("nonzero_exit", 1)


# --------------------------------------------------------------------------- reconcile (unknown receipt)


@pytest.mark.asyncio
async def test_unknown_receipt_with_a_matching_result_reports_the_real_outcome(board):
    """The receipt says ``unknown`` but the attempt DID write its result under this claim, by
    this run's bound interpreter: the real exit code is the verdict, no raise."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=0, last_output="all good")

    code = kwc._exit_code({"status": "unknown"}, _params(task, board), str(board))
    assert code == 0
    assert kbd._exit_code_kind(code) == ("clean_exit", 0)  # booked as the success it is


@pytest.mark.asyncio
async def test_unknown_receipt_with_a_nonzero_result_returns_that_failure(board):
    """Reconcile is not a pardon: a real failed exit still comes back as a real failure."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=3, last_output="boom")

    code = kwc._exit_code({"status": "unknown"}, _params(task, board), str(board))
    assert code == 3
    assert kbd._exit_code_kind(code)[0] == "nonzero_exit"


@pytest.mark.asyncio
async def test_unknown_receipt_without_any_result_raises_owner_unavailable(board):
    """No ``worker_result`` at all -> fail closed: raise, and the submitter exits 74 so the
    death is booked through the infrastructure seam. Success is never synthesized."""
    task = _claimed(board)
    _bind(board, task, pid=4242)  # bound, but no result was ever written

    with pytest.raises(kwc.OwnerUnavailableError, match="execution unknown"):
        kwc._exit_code({"status": "unknown"}, _params(task, board), str(board))


@pytest.mark.asyncio
async def test_a_result_from_another_claim_is_not_reported(board):
    """The reconcile gate's other direction: a ``worker_result`` carrying a DIFFERENT claim
    lock is another attempt's outcome — refused, old raise stands."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=0, claim_lock="somebody-elses-claim")

    with pytest.raises(kwc.OwnerUnavailableError, match="execution unknown"):
        kwc._exit_code({"status": "unknown"}, _params(task, board), str(board))


@pytest.mark.asyncio
async def test_a_result_from_another_interpreter_is_not_reported(board):
    """And the pid direction: a result written by an interpreter the run never BOUND is not
    this attempt's outcome either."""
    task = _claimed(board)
    _bind(board, task, pid=4242, result_pid=9999, exit_code=0)

    with pytest.raises(kwc.OwnerUnavailableError, match="execution unknown"):
        kwc._exit_code({"status": "unknown"}, _params(task, board), str(board))


def test_worker_result_for_claim_reads_a_real_board(board):
    """The reader helper itself, against real rows: ours returns the dict, the gates return
    None."""
    task = _claimed(board)
    params = _params(task, board)
    assert session_kanban.worker_result_for_claim(str(board), params) is None
    _bind(board, task, pid=4242, exit_code=0, last_output="done")
    got = session_kanban.worker_result_for_claim(str(board), params)
    assert got is not None and got["exit_code"] == 0


# --------------------------------------------------------------------------- the submitter's exit


@pytest.mark.asyncio
async def test_owner_silent_past_the_wait_is_an_infrastructure_death(board, monkeypatch):
    """A transient (draining) owner that never answers: the give-up raise is the new
    OwnerUnavailableError, not the old generic client error, and the never-admitted case says
    so."""
    from hermes_cli import gateway_client

    @asynccontextmanager
    async def _draining():
        unavailable = gateway_client.GatewayUnavailableError
        raise unavailable("Gateway draining: update_paused", "draining")
        yield  # unreachable: the raise above is what makes this an async generator

    monkeypatch.setattr(gateway_client, "connect_gateway", _draining)
    monkeypatch.setattr(kwc, "OWNER_WAIT_S", 0.0)  # the wait is spent at once
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(kwc.OwnerUnavailableError, match="silent for"):
        await kwc.run(_params(_claimed(board), board), str(board))


def test_a_host_death_exits_74_with_its_trailer(board, monkeypatch, capfd):
    """The whole submitter contract end-to-end: env in, exit code out. A receipt that stays
    ``unknown`` with nothing to reconcile makes ``main()`` return 74 (not the counted 1 the
    ValueError clause used to book) and the trailer in the log records the same code."""
    task = _claimed(board)
    _bind(board, task, pid=4242)  # bound, no result
    env = {"HERMES_KANBAN_BOARD": "default", "HERMES_KANBAN_TASK": task.id,
           "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
           "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock, "HERMES_KANBAN_DB": str(board)}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    from hermes_cli import gateway_client
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

    class _StubClient:
        events = None

        def __init__(self):
            import asyncio
            self.events = asyncio.Queue()

        async def rpc(self, method, **params):
            if method == "kanban.run":
                return {"session_id": "s1",
                        "receipt": {"status": "unknown", "admission_id": "a1"}}
            return {"status": "unknown", "admission_id": "a1"}

    @asynccontextmanager
    async def _connect_stub():
        yield _StubClient()

    monkeypatch.setattr(gateway_client, "connect_gateway", _connect_stub)
    monkeypatch.setattr(kwc.asyncio, "sleep", _no_sleep)

    assert kwc.main() == KANBAN_INFRASTRUCTURE_EXIT_CODE
    captured = capfd.readouterr()
    assert f"{KANBAN_WORKER_EXIT_TRAILER}74" in captured.err
    assert "Kanban host unavailable" in captured.err


# --------------------------------------------------------------------------- the dispatcher's booking


def _dead_worker_pid() -> int:
    proc = subprocess.Popen(["sleep", "0.05"])
    proc.wait()
    return proc.pid


def _park_as_dead_worker(conn, task, pid):
    """Leave the claim exactly as a crashed submitter leaves it: ``running``, a dead pid, no
    fingerprint (legacy row -> liveness is a plain ``/proc`` existence check)."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (pid, task.id))


def _register_exit(pid, code):
    """Feed the reap registry the way a reaped child would: the raw wait status of ``code``."""
    kbd._recent_worker_exits[pid] = (kbd._wait_status_from_returncode(code), 10**9)


def test_infrastructure_exit_releases_the_card_without_counting_a_failure(board):
    """The headline: exit 74 -> released, ``consecutive_failures`` untouched, NOT blocked, no
    ``gave_up`` anywhere — while the run and the event keep the ``infrastructure`` marker so
    the outage stays visible."""
    task = _claimed(board)
    pid = _dead_worker_pid()
    conn = _connect(board)
    _park_as_dead_worker(conn, task, pid)
    _register_exit(pid, KANBAN_INFRASTRUCTURE_EXIT_CODE)

    kbd.detect_crashed_workers(conn)

    row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone())
    assert row["consecutive_failures"] == 0
    assert row["status"] == "ready"  # released, never left running
    run = dict(conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                            (task.id,)).fetchone())
    assert run["outcome"] == "crashed" and run["ended_at"]
    meta = json.loads(run["metadata"])
    assert meta["infrastructure"] is True
    assert meta["exit_kind"] == "infrastructure_failure" and meta["exit_code"] == 74
    assert "requeued without counting a failure" in (row["last_failure_error"] or "")
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task.id,))]
    assert "gave_up" not in kinds and "blocked" not in kinds


def test_plain_nonzero_exit_still_counts(board):
    """The control: a worker that died with any other code pays like before — exit 1 books a
    counted crash."""
    task = _claimed(board)
    pid = _dead_worker_pid()
    conn = _connect(board)
    _park_as_dead_worker(conn, task, pid)
    _register_exit(pid, 1)

    crashed = kbd.detect_crashed_workers(conn)

    assert crashed == [task.id]
    row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone())
    assert row["consecutive_failures"] == 1
    meta = json.loads(dict(conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task.id,)).fetchone())["metadata"])
    assert not meta.get("infrastructure")


def test_repeated_infrastructure_deaths_never_trip_the_breaker(board):
    """Three systemic identical deaths in one tick trip a plain crash's breaker at 1
    (``failure_limit=1``); the infrastructure seam must beat that: counter untouched, no
    auto-block, no ``gave_up``."""
    task = _claimed(board)
    conn = _connect(board)
    deads = []
    for i in range(3):
        dead = kbd._DeadWorker(
            "infrastructure_failure", KANBAN_INFRASTRUCTURE_EXIT_CODE,
            f"pid {i} exited 74: the owner never completed this run (host-side stall) — "
            "requeued without counting a failure",
            "crashed", {"pid": i, "claimer": "x"},
            infrastructure_failure=True,
        )
        deads.append((task.id, i, "claim", dead))

    auto_blocked = kbd._account_crashes(conn, deads)

    assert auto_blocked == []
    row = dict(conn.execute("SELECT consecutive_failures, status FROM tasks WHERE id = ?",
                            (task.id,)).fetchone())
    assert row["consecutive_failures"] == 0
    assert row["status"] == "running"  # _account_crashes alone never blocks
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (task.id,)).fetchall()]
    assert "gave_up" not in kinds


def test_infrastructure_death_takes_the_cooldown_then_retries(board):
    """The guard spaces the next spawn while the outage is fresh — and lets the card through
    once the cooldown elapsed. The outage retries forever, spaced, like the #114720 refusal."""
    import time as _time
    task = _claimed(board)
    pid = _dead_worker_pid()
    conn = _connect(board)
    _park_as_dead_worker(conn, task, pid)
    _register_exit(pid, KANBAN_INFRASTRUCTURE_EXIT_CODE)
    kbd.detect_crashed_workers(conn)

    assert kbd.check_respawn_guard(conn, task.id) == "infrastructure_cooldown"

    # Age the run well past the default cooldown: the card becomes spawnable again, never blocked.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE task_id = ? AND ended_at IS NOT NULL",
            (int(_time.time()) - 10_000, task.id))
    assert kbd.check_respawn_guard(conn, task.id) is None


def test_plain_crash_does_not_get_the_infrastructure_cooldown(board):
    """The cooldown's other direction: a counted crash (no ``infrastructure`` marker) is NOT
    spaced by it — it still burns its budget tick by tick like before."""
    task = _claimed(board)
    pid = _dead_worker_pid()
    conn = _connect(board)
    _park_as_dead_worker(conn, task, pid)
    _register_exit(pid, 1)
    kbd.detect_crashed_workers(conn)

    assert kbd.check_respawn_guard(conn, task.id) is None
