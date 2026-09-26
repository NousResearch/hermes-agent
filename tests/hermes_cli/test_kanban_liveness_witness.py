"""One liveness witness, three answers.

The 2026-09-26 false-reclaim tick declared six live workers dead inside ONE dispatch tick and spawned
a duplicate beside each (``EVIDENCE-blast-radius.txt``). Two probes answered "dead" where the honest
answer was "cannot tell": a secondary ``ps`` that exited non-zero, and a start-time read that returned
nothing (``EVIDENCE-reproduce-defect.txt``). The recorded start also moved by ~9 s for a live process
because the macOS boot reference it was built from was adjusted mid-flight
(``EVIDENCE-reference-change.txt``).

These tests pin the contract that follows from that: a liveness read is ``alive`` / ``dead`` /
``unknown``, only PROOF of death releases a claim or authorizes a signal, and an unprovable read holds
the claim. They also pin the run fence that rides on the verdict, and the recovery that records a
reclaim honestly instead of leaving a crash as the attempt history's last word.
"""

import json
import os
import signal
import subprocess
import time

import pytest

from gateway import status as gw_status
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid=None, started_at, max_runtime=None) -> str:
    """A ``running`` card claimed by a host-local worker whose pid/fingerprint we control."""
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid if pid is not None else os.getpid())
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
            (started_at, old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = "
            "(SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


# The two fingerprints a deployed board carries that can no longer be compared: the pre-witness shape
# (a bare start tick) and the macOS shape whose boot-witness component is empty. Neither can be
# salvaged, because for both of them "the value differs" is indistinguishable from "the boot reference
# was adjusted" — so both must read UNKNOWN, never DEAD.
UNCOMPARABLE_FINGERPRINTS = ["179041285600", "|1790412856", 179041285600]


def test_pid_liveness_answers_three_ways():
    """``dead`` needs proof; a probe that could not read the process is not proof."""
    assert kbd._pid_liveness(os.getpid()) == kbd.WORKER_ALIVE
    assert kbd._pid_liveness(9_999_999) == kbd.WORKER_DEAD
    assert kbd._pid_liveness(None) == kbd.WORKER_DEAD
    assert kbd._pid_liveness(0) == kbd.WORKER_DEAD


def test_secondary_probe_exit_status_is_unknown_never_death(monkeypatch):
    """BRANCH B of the reproduction: ``ps`` exits non-zero for a PID that ``kill(0)`` says is live.

    ``ps`` also exits 1 for a PID it cannot see under load, so the only honest answer is ``unknown``;
    reading it as death is what let one tick reclaim six live cards.
    """
    fake = subprocess.CompletedProcess(args=["ps"], returncode=1, stdout="")
    monkeypatch.setattr(kbd.subprocess, "run", lambda *a, **k: fake)
    assert kbd._pid_liveness(os.getpid()) == kbd.WORKER_UNKNOWN

    # A worker whose identity DOES match stays alive: the witness/start agreement is proof on its own,
    # and a probe that could not read the process must not overrule it.
    real = kbd._process_fingerprint(os.getpid())
    assert kbd._worker_liveness(os.getpid(), real) == kbd.WORKER_ALIVE

    # An identity that cannot be compared is HELD rather than released beside the live process.
    assert kbd._worker_liveness(os.getpid(), "179041285600") == kbd.WORKER_UNKNOWN
    assert kbd._worker_not_dead(os.getpid(), "179041285600") is True

    # A foreign boot witness of the SAME form is proof of its own: that worker is not ours.
    monkeypatch.setattr(gw_status, "host_boot_witness", lambda: "bootsession:THIS-HOST")
    assert kbd._worker_liveness(os.getpid(), "bootsession:SOME-OTHER-BOOT|1") == kbd.WORKER_DEAD

    # A witness that came back in a DIFFERENT form is not proof of a reboot: the probe that yields it
    # (`sysctl` above, under the same `ps` outage) changed its answer's SHAPE, not the boot. Reading
    # that as "different boot" would declare every live row dead in one tick — the incident itself.
    monkeypatch.setattr(gw_status, "host_boot_witness", lambda: "boottime:179041285600")
    assert kbd._worker_liveness(os.getpid(), "bootsession:SOME-OTHER-BOOT|1") == kbd.WORKER_UNKNOWN


@pytest.mark.platforms("macos")
def test_boot_reference_shift_does_not_move_the_fingerprint(monkeypatch):
    """The reproduction's ~9 s drift: psutil imported before the boot reference was adjusted.

    ``psutil._psosx.adjust_proc_create_time`` shifts ``create_time()`` by ``|INIT_BOOT_TIME - boot|``
    for the rest of that process's life, so two readers of the SAME live process disagree. The
    boot-relative accessor undoes that shift, and the fingerprint it produces is identical either way.
    """
    import importlib

    psosx = importlib.import_module("psutil._psosx")
    original = psosx.INIT_BOOT_TIME
    assert original, "psutil exposes no boot reference to shift"
    before = gw_status.get_process_uptime_start_time(os.getpid())
    assert before is not None
    try:
        psosx.INIT_BOOT_TIME = original + 9
        shifted = gw_status.get_process_uptime_start_time(os.getpid())
    finally:
        psosx.INIT_BOOT_TIME = original
    assert shifted == before
    assert abs(shifted - before) <= gw_status.START_TIME_DRIFT_TOLERANCE


def test_start_time_read_failure_is_unknown_and_holds_the_claim():
    """BRANCH A of the reproduction: the start-time read fails after a good fingerprint was recorded.

    The live worker is not "recycled" — nothing was read, which is not a mismatch. Every path that
    could release or signal it must hold instead.
    """
    assert kbd._worker_liveness(os.getpid(), None) == kbd.WORKER_ALIVE  # legacy NULL row: existence
    real = kbd._process_fingerprint(os.getpid())
    assert real is not None

    original = gw_status._get_process_start_time
    try:
        gw_status._get_process_start_time = lambda pid: None
        assert kbd._worker_liveness(os.getpid(), real) == kbd.WORKER_UNKNOWN
        assert kbd._worker_alive(os.getpid(), real) is False
        assert kbd._worker_not_dead(os.getpid(), real) is True
    finally:
        gw_status._get_process_start_time = original


@pytest.mark.parametrize("fingerprint", UNCOMPARABLE_FINGERPRINTS)
def test_an_incomparable_fingerprint_is_held_never_declared_dead(board, fingerprint):
    """The deploy-time consequence of the format change, pinned deliberately.

    Every row running at deploy time carries one of these shapes and is therefore HELD until it turns
    over. That is the safe direction: the alternative is declaring live work dead, which is the defect.
    """
    conn = board
    tid = _claimed_running(conn, started_at=fingerprint, max_runtime=1)
    assert kbd._worker_liveness(os.getpid(), fingerprint) == kbd.WORKER_UNKNOWN
    assert kbd._worker_alive(os.getpid(), fingerprint) is False
    assert kbd._worker_not_dead(os.getpid(), fingerprint) is True

    killed = []
    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"


def test_one_tick_reclaims_no_live_worker_and_spawns_no_duplicate(board):
    """The board-chain scenario: six cards, one dispatch tick, every worker alive.

    Six reclaims in a single tick is the false-reclaim signature; a correct kernel reclaims NONE of
    them and leaves each claim standing so the dispatcher cannot spawn a duplicate beside it.
    """
    conn = board
    tids = [_claimed_running(conn, started_at=fingerprint, max_runtime=1)
            for fingerprint in (UNCOMPARABLE_FINGERPRINTS * 2)]
    assert len(tids) == 6

    assert kbd.detect_crashed_workers(conn) == []
    assert kb.release_stale_claims(conn) == 0
    assert kbd.reconcile_orphaned_running(conn) == []
    for tid in tids:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).worker_pid == os.getpid()


def test_orphan_reconcile_defers_an_unprovable_worker_and_records_why(board):
    """An orphan whose identity cannot be read must NOT be requeued, and the deferral must be
    recorded — that record is what a later recovery reads before it deletes anything."""
    conn = board
    tid = _claimed_running(conn, started_at="|1790412856")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock = NULL, claim_expires = NULL WHERE id = ?", (tid,))

    assert kbd.reconcile_orphaned_running(conn) == []
    assert kb.get_task(conn, tid).status == "running"
    events = [e for e in kb.list_events(conn, tid) if e.kind == "reclaim_deferred"]
    assert events, "the deferral must be recorded, not silent"
    payload = events[-1].payload
    assert payload["reason"] == "orphaned_running_worker_alive"
    assert payload["liveness"] == kbd.WORKER_UNKNOWN


def test_recovery_reconciliation_needs_a_disowned_premise(board):
    """History a worker recorded is never rewritten; only a proven-infrastructure abandonment
    is reconciled, and the reconciliation attributes nothing to the worker."""
    conn = board
    worker_owned = _claimed_running(conn, started_at=kbd._process_fingerprint(os.getpid()))
    assert kb._synthesize_empty_completion_run(conn, worker_owned) is None
    assert len(kb.list_runs(conn, worker_owned) if hasattr(kb, "list_runs") else
               conn.execute("SELECT id FROM task_runs WHERE task_id = ?", (worker_owned,)).fetchall()) == 1

    disowned = _claimed_running(conn, started_at="|1790412856")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET outcome = 'reclaimed', status = 'reclaimed', metadata = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            ('{"reason": "orphaned_running"}', disowned),
        )
        conn.execute("UPDATE tasks SET worker_pid = NULL, claim_lock = NULL WHERE id = ?", (disowned,))
    run_id = kb._synthesize_empty_completion_run(conn, disowned)
    assert run_id is not None
    row = conn.execute("SELECT outcome, summary, metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["outcome"] == "completed"
    assert row["summary"] in (None, ""), "the recovery attributes no prose to the worker"
    assert "'reason': 'orphaned_running'" not in (row["metadata"] or ""), "must not look disowned itself"



# The A/B subset: these read only API that exists on the BASE commit, so this file can be run
# against the base blob set (worktree) to show the failure each one fixes. The tests above are the
# contract; these are the ones that go red without the change.
def _failing_ps(monkeypatch):
    """The secondary macOS probe exits non-zero under load -- the live evidence's BRANCH B."""
    real_run = subprocess.run

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "ps":
            return subprocess.CompletedProcess(args=list(cmd), returncode=1, stdout="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(kbd.subprocess, "run", _run)


@pytest.mark.platforms("macos")
def test_the_start_time_accessor_does_not_move_with_the_boot_reference():
    """BRANCH A at the source: the 9 s offsets in the evidence are psutil's own adjustment."""
    psosx = pytest.importorskip("psutil._psosx")
    before = gw_status.get_process_start_time(os.getpid())
    assert before is not None
    original = psosx.INIT_BOOT_TIME
    try:
        psosx.INIT_BOOT_TIME = original + 9  # psutil imported before the reference moved by 9 s
        after = gw_status.get_process_start_time(os.getpid())
        shifted = gw_status.get_process_uptime_start_time(os.getpid())
    finally:
        psosx.INIT_BOOT_TIME = original
    assert after is not None and shifted is not None
    # The absolute accessor is the one that moves (its callers keep needing epoch time; the fix leaves
    # its semantics alone) — that movement is the incident, and it is what the pair below replaces.
    assert abs(before - after) > gw_status.START_TIME_DRIFT_TOLERANCE
    assert shifted == gw_status.get_process_uptime_start_time(os.getpid())


@pytest.mark.platforms("macos")
def test_a_failed_secondary_probe_holds_the_claim(board, monkeypatch):
    """BRANCH B end to end: a `ps` that fails must not release a live worker's claim."""
    conn = board
    tid = _claimed_running(conn, pid=os.getpid(), started_at=kbd._process_fingerprint(os.getpid()))
    _failing_ps(monkeypatch)
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    row = conn.execute("SELECT worker_pid FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_pid"] == os.getpid()


@pytest.mark.platforms("macos")
def test_the_reclaim_tick_does_not_take_six_live_workers(board, monkeypatch):
    """The board-chain signature of 2026-09-26: six live cards declared dead in ONE tick."""
    conn = board
    ids = [
        _claimed_running(conn, pid=os.getpid(), started_at=kbd._process_fingerprint(os.getpid()))
        for _ in range(6)
    ]
    _failing_ps(monkeypatch)
    assert kbd.detect_crashed_workers(conn) == []
    assert [kb.get_task(conn, t).status for t in ids] == ["running"] * 6


def test_a_legacy_fingerprint_does_not_release_the_claim(board):
    """The accepted deploy consequence: an old-shape row is HELD, never declared dead."""
    conn = board
    tid = _claimed_running(conn, pid=os.getpid(), started_at=1)
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
