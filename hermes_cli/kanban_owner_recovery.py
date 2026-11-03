"""Read-only canonical admission fences for dispatcher reclaim paths."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
import psutil


def _owner_receipt(conn, task_id):
    """The parsed ``owner_admitted`` receipt of the task's CURRENT run, or None.

    The receipt is the owner's pointer to its own admission ledger. No receipt means the run
    never reached an owner (a plain dispatcher worker): nothing about it is ambiguous.
    """
    marker = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind='owner_admitted' ORDER BY e.id DESC LIMIT 1",
        (task_id,)).fetchone()
    if marker is None:
        return None
    try:
        receipt = json.loads(marker['payload'])
    except (ValueError, TypeError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def _admission_row(receipt):
    """``(status, outcome)`` from the owner ledger ``receipt`` names, or None when unreadable.

    None is "the owner said nothing we can read" — never "the owner said the attempt ended".
    """
    if not receipt:
        return None
    try:
        with closing(sqlite3.connect(Path(receipt['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            row = owner.execute('SELECT status,outcome FROM session_admissions WHERE target_session_id=? AND request_id=?',
                                (receipt['session_id'], receipt['request_id'])).fetchone()
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
        return None
    return (row[0], row[1]) if row is not None else None


def owner_reclaim_paused(conn, task_id):
    """Missing/ambiguous owner receipts never authorize another attempt."""
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return False
    try:
        row = _admission_row(receipt)
        if row is not None and row[0] == 'terminal':
            # Interrupted still pauses the generic reclaim paths; owner_attempt_provably_over decides
            # when nothing can still be running it.
            return row[1] == 'interrupted'
        if row is None or row[0] in {'queued', 'unknown'}:
            return True
        from hermes_cli.process_identity import worker_birth
        return worker_birth(psutil.Process(receipt['pid'])) != receipt['birth']
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return True


def _latest_run_event(conn, task_id, kind):
    """The newest ``kind`` payload on the task's CURRENT run, or ``{}``.

    A payload that cannot be read is not a payload that says the attempt is over, so
    callers treat ``{}`` as "no evidence" rather than as an absence.
    """
    row = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind=? ORDER BY e.id DESC LIMIT 1",
        (task_id, kind)).fetchone()
    try:
        payload = json.loads(row['payload']) if row else {}
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _host_local_lock(lock) -> bool:
    """True when ``lock`` was issued by this host — the only pids we can judge."""
    from hermes_cli import kanban_db as kb
    return bool(lock) and str(lock).startswith(kb._host_prefix())


def bound_interpreter_gone(conn, task_id):
    """True when the owner bound an interpreter to the task's current run and it is dead."""
    bound = _latest_run_event(conn, task_id, 'worker_bound')
    if not bound:
        return False
    from hermes_cli.kanban_db_dispatch import _worker_alive
    return not _worker_alive(bound.get('pid'), bound.get('started_at'))


def _owner_process_gone(receipt) -> bool:
    """True when the receipt's own pid is gone, or its number now belongs to another process.

    The identity rule ``owner_reclaim_paused`` already applies: a live pid whose ``birth``
    differs is a recycled number, not the owner. A zombie keeps its birth until reaped but has
    already exited, so liveness goes through the dispatcher's zombie-aware ``_pid_alive``. A live pid we cannot compare (no ``birth``
    recorded) and an unreadable one are never proof of death.
    """
    try:
        pid = int(receipt['pid'])
    except (KeyError, TypeError, ValueError):
        return False
    from hermes_cli import kanban_db as kb
    if not kb._pid_alive(pid):
        return True  # absent, or an exited zombie still waiting to be reaped
    try:
        from hermes_cli.process_identity import worker_birth
        current = worker_birth(psutil.Process(pid))
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False
    birth = receipt.get('birth')
    return birth is not None and current != birth


def _run_worker_pids(conn, task_row, task_id):
    """Every worker pid recorded for the task's current run, with its spawn fingerprint."""
    recorded = [(task_row['worker_pid'], task_row['worker_started_at'])]
    for kind in ('spawned', 'worker_bound'):
        payload = _latest_run_event(conn, task_id, kind)
        recorded.append((payload.get('pid'), payload.get('started_at')))
    return [(int(pid), fingerprint) for pid, fingerprint in recorded if pid]


def owner_attempt_provably_over(conn, task_id) -> bool:
    """True when an unresolved owner receipt cannot still be executing this attempt.

    ``owner_reclaim_paused`` is about effects: an admission that may still be in flight is
    never repeated. It must not also mean the run keeps its claim forever. A paused run whose
    owner died, or whose owner settled the admission after losing its worker, may never have
    written a ``worker_bound`` event, so ``bound_interpreter_gone`` cannot speak for it. This
    asks the question that matters — can anything still be executing this attempt? — from
    evidence the board already keeps:

      * the run never bound an interpreter (a bound run is the crash sweep's to book);
      * every worker pid recorded for the run (``tasks.worker_pid``, its ``spawned`` and
        ``worker_bound`` events) is dead or recycled;
      * and the owner's own admission is SETTLED (``status='terminal'``) with at least one
        recorded pid, or the host-local claim has EXPIRED with no heartbeat inside
        ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` and the admission is settled or (pids
        recorded) the owner's process is gone.

    A settled admission outranks the owner's liveness: the owner is usually the long-lived
    gateway, alive by design, and it settles a lost managed worker's admission without
    extending the claim.

    Missing evidence is never proof: no receipt, a live pid whose identity cannot be
    compared, no recorded pid inside the claim's TTL, or an unreadable clock all return False
    and the pause holds.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _worker_alive
    if _latest_run_event(conn, task_id, 'worker_bound'):
        return False  # a bound run is the crash sweep's to book (bound_interpreter_gone)
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return False  # no receipt: this pause did not come from an owner attempt
    row = conn.execute(
        "SELECT claim_lock, claim_expires, worker_pid, worker_started_at, last_heartbeat_at "
        "FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or not _host_local_lock(row['claim_lock']):
        return False
    pids = _run_worker_pids(conn, row, task_id)
    if any(_worker_alive(pid, fingerprint) for pid, fingerprint in pids):
        return False
    admission = _admission_row(receipt)
    settled = admission is not None and admission[0] == 'terminal'
    if settled and pids:
        return True
    now = int(time.time())
    expires = kb._to_epoch(row['claim_expires'])
    if expires is None or expires >= now:
        return False
    heartbeat = kb._to_epoch(row['last_heartbeat_at'])
    if heartbeat is not None and now - heartbeat <= kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS:
        return False
    # A lapsed claim: a settled admission is proof; an unsettled one needs recorded pids and a dead owner.
    return settled or (bool(pids) and _owner_process_gone(receipt))


def owner_attempt_note(conn, task_id, lost=None) -> str:
    """One clause naming why the owner says the attempt ended, or ``""``.

    The reclaim paths put this in the run error so the board says what happened instead of
    a bare ``stale_lock=``: a managed worker that died (its owner recorded ``worker_lost``,
    passed as ``lost`` when the caller already read it) is a different fault from a worker
    that never started.
    """
    if lost is None:
        lost = _latest_run_event(conn, task_id, 'worker_lost')
    reason = str(lost.get('reason') or '')
    if reason:
        return f"managed worker lost: {reason[:80]}"
    receipt = _owner_receipt(conn, task_id)
    if receipt is None:
        return ""
    admission = _admission_row(receipt)
    if admission is not None and admission[0] == 'terminal':
        return f"owner admission settled {admission[1] or 'terminal'}"
    return ""


def reclaim_error(conn, task_id, claim_lock) -> str:
    """A stale-lock reclaim's run error, naming how the attempt ended when the owner recorded it."""
    note = owner_attempt_note(conn, task_id)
    return f"stale_lock={claim_lock}" + (f" [{note}]" if note else "")


def apply_attempt_note(conn, task_id, pid, dead) -> None:
    """Fold the owner's own record of how the attempt ended into ONE dead worker's booking.

    A bound interpreter whose ``worker_result`` says exit 0 after its managed execution was
    lost reads as a protocol violation. The owner's ``worker_lost`` evidence outranks it: the
    turn never finished, so it books as a crash and charges the breaker, and the
    protocol-violation streak (which reads the persisted payload) does not count it.
    """
    lost = _latest_run_event(conn, task_id, "worker_lost")
    note = owner_attempt_note(conn, task_id, lost)
    if not note:
        return
    dead.event_payload["attempt_note"] = note
    if lost:
        dead.event_payload["worker_lost"] = lost
        if dead.protocol_violation:
            from hermes_cli.kanban_db_dispatch import _PROTOCOL_VIOLATION_ERROR
            dead.protocol_violation = False
            dead.event_payload.pop("protocol_violation", None)
            dead.event_payload["exit_kind"] = dead.kind
            dead.event_kind = "crashed"
            # Keep the worker's last output that _classify_dead_worker appended.
            dead.error_text = (f"pid {pid} exited cleanly after its managed worker was lost."
                               + dead.error_text.removeprefix(_PROTOCOL_VIOLATION_ERROR))
    # Before the worker's own output: the stored error is capped, and the output may be long.
    from hermes_cli.kanban_db_dispatch import WORKER_OUTPUT_SEP
    head, sep, output = dead.error_text.partition(WORKER_OUTPUT_SEP)
    dead.error_text = f"{head} [{note}]{sep}{output}"


def owner_reclaim_blocked(conn, task_id, *, crash_sweep=False) -> bool:
    """True when a reclaim path must leave this task alone.

    The pause protects the effects of an attempt that may still be in flight, and stops
    holding the claim as soon as nothing can still be executing it. A run whose bound
    interpreter is dead stays the crash sweep's alone (``crash_sweep=True``), so it is booked
    ``crashed`` and re-queued once rather than ``reclaimed`` by whichever path ticks first. A
    run that never bound one is released by any path once ``owner_attempt_provably_over``:
    the shape that used to be unreclaimable forever, because ``bound_interpreter_gone`` was the
    only override and it needs the ``worker_bound`` event that run never wrote.
    """
    if not owner_reclaim_paused(conn, task_id):
        return False
    if crash_sweep and bound_interpreter_gone(conn, task_id):
        return False
    return not owner_attempt_provably_over(conn, task_id)


def owner_reclaim_blocked_after_kill(conn, task_id, *, signalled: bool) -> bool:
    """The under-lock re-check for a reclaim path, after its termination step.

    When this host just signalled the run's worker (``signalled``), killing it takes its bound
    interpreter down too (``_watch_submitter``) and the owner settles the admission ``unknown``:
    a pause this path caused itself, which would leave a timed-out run for the crash sweep to
    misbook ``crashed``. Only an interpreter still LIVE (one that bound after the first look)
    keeps the claim. Without a local signal (a foreign-host claim) the bound interpreter's pid
    means nothing here, so a "dead" reading is no proof and the plain gate applies.
    """
    return owner_reclaim_blocked(conn, task_id, crash_sweep=signalled)


def managed_run_exit_code(conn, task_id, worker_pid, claim_lock):
    """The exit code a managed run's authority recorded for the task's CURRENT run, else None.

    Managed interpreters are children of the authority, not the dispatcher, so their result is
    read from the run's ``worker_result`` event (it survives a dispatcher restart and cannot be
    reaped here). It speaks for the run only under the same claim and from a pid of this run:
    the dispatcher's own worker, or the owner-side interpreter ``worker_bound`` recorded under
    that claim — ``session_kanban._record_worker_result`` writes the interpreter's pid, never the
    spawned submitter's ``tasks.worker_pid``."""
    def latest(kind):
        row = conn.execute(
            "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
            "WHERE t.id=? AND e.task_id=t.id AND e.kind=? ORDER BY e.id DESC LIMIT 1", (task_id, kind)).fetchone()
        try:
            payload = json.loads(row['payload']) if row else {}
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    result, bound = latest('worker_result'), latest('worker_bound')
    pids = {worker_pid, *([bound.get('pid')] if bound.get('claim_lock') == claim_lock else [])}
    return result.get('exit_code') if result.get('claim_lock') == claim_lock and result.get('pid') in pids else None
