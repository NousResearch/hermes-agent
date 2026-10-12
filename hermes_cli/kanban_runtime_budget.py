"""Runtime budget for kanban workers: how long an attempt may run before the
dispatcher terminates and re-queues it (the "re-approach").

Split out of ``kanban_db_dispatch`` (a facade the file-length ratchet may not
grow) because it is a self-contained policy: given an explicit cap, a worker
estimate or an operator default, decide whether this attempt has overrun.

``enforce_max_runtime`` historically required the card to carry an explicit
``max_runtime_seconds`` (``--max-runtime``), which no card sets by default, so a
running worker had no runtime bound before the 4 h/1 h stale window. Precedence:

1. the card's explicit ``max_runtime_seconds`` (unchanged);
2. the worker's own estimate (``kanban_heartbeat(expected_runtime_seconds=...)``),
   enforced with :data:`ESTIMATE_GRACE_FACTOR` headroom — an estimate is a guess
   and a nearly-done attempt should not be killed on a good one; and
3. ``kanban.default_max_runtime_seconds`` (0 = off), so a card that was never
   estimated is still bounded when an operator opts in.

Host-local only: the dispatcher signals a PID whose spawn fingerprint it can
verify, never a recycled one. Every hermes import is lazy so this module can be
imported from the dispatch facade without an import cycle.
"""
from __future__ import annotations

import contextlib
import signal
import sqlite3
import time
from math import ceil
from typing import Any, Optional

# Multiplier on a worker's self-estimate before the attempt is treated as
# overrun: an estimate is a guess, and a nearly-done attempt should not be
# killed for a good one. 1.5 gives 50% headroom.
ESTIMATE_GRACE_FACTOR = 1.5

# Dispatcher-side cap for cards with neither an explicit ``max_runtime_seconds``
# nor a worker estimate. 0 = off (backward compatible: the 4 h/1 h stale path
# stays the only wall-clock bound). Operators opt in with
# ``kanban.default_max_runtime_seconds`` in config.yaml.
DEFAULT_MAX_RUNTIME_SECONDS = 0


def default_max_runtime_seconds() -> int:
    """Dispatcher default cap from config.yaml; 0 = off. Never raises.

    A broken or unreadable config must not stop the dispatcher, so this is one of
    the deliberate catch-all boundaries: it degrades to "off" rather than taking
    the tick down.
    """
    try:
        from hermes_cli.config import load_config
        kanban = (load_config() or {}).get("kanban") or {}
        return max(0, int(kanban.get("default_max_runtime_seconds") or 0))
    except Exception:  # health: allow BLE001 -- an unreadable config must not stop the dispatcher
        return DEFAULT_MAX_RUNTIME_SECONDS


def effective_runtime_limit(
    explicit: Any, estimate: Any, default_limit: int,
) -> tuple[Optional[int], Optional[str]]:
    """``(limit, source)`` for one attempt: explicit > estimate*grace > default.

    ``(None, None)`` means unbounded — leave the worker alone. Source is carried
    into the ``timed_out`` payload so an operator can tell which budget fired.
    """
    from hermes_cli import kanban_db as kb

    if explicit is not None:
        return int(explicit), "max_runtime"
    est = kb._opt_int(estimate)
    if est is not None and est > 0:
        return int(ceil(est * ESTIMATE_GRACE_FACTOR)), "estimate"
    if default_limit and default_limit > 0:
        return int(default_limit), "default"
    return None, None


def enforce_max_runtime(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """Terminate workers whose runtime budget has elapsed; returns their ids.

    SIGTERM, short grace, then SIGKILL. Emits ``timed_out`` (payload names the
    limit, its source and what was estimated) and restores the task's source
    phase so the next tick re-spawns the same kind of worker — the re-approach —
    unless the circuit breaker already gave up, leaving it blocked.
    ``signal_fn`` is a test hook.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = kb._host_prefix()
    default_limit = default_max_runtime_seconds()

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.estimated_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt: ``tasks.started_at`` records the FIRST start,
        # so retries must be measured from the active task_runs row.
        elapsed = now - int(row["active_started_at"])
        limit, limit_source = effective_runtime_limit(
            row["max_runtime_seconds"], row["estimated_runtime_seconds"], default_limit)
        if limit is None or elapsed < limit:
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        started_at = kb._row_get(row, "worker_started_at")
        if started_at == kbd.UNVERIFIED_WORKER_FINGERPRINT and kb._pid_alive(pid):
            # Fingerprint capture failed at spawn: we cannot prove this live PID is our worker, so
            # it is neither signalled nor released beside (duplicate). It is reclaimed once it exits.
            kb._log.warning("kanban: task %s worker pid %s exceeded max runtime but has no verified "
                            "identity; not signalled", tid, pid)
            continue
        # SIGTERM then SIGKILL after 5 s grace; workers wanting a cleaner
        # shutdown install their own SIGTERM handler. A recycled PID (fingerprint
        # mismatch) is never signalled: the worker is already gone.
        killed = False
        kill = kbd._kill_fn(signal_fn)
        if kill is not None and not (kb._pid_alive(pid) and kbd._pid_recycled(pid, started_at)):
            with contextlib.suppress(ProcessLookupError, OSError):
                kill(pid, signal.SIGTERM)
            # Short polling wait — no time.sleep on the write txn.
            kbd._poll_worker_exit(pid, started_at)
            if kbd._worker_alive(pid, started_at):
                killed = kbd._sigkill(kill, pid)

        error = f"elapsed {int(elapsed)}s > limit {limit}s ({limit_source})"
        with kb.write_txn(conn):
            retry_status = kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": limit,
                    "limit_source": limit_source,
                    "estimated_runtime_seconds": kb._opt_int(row["estimated_runtime_seconds"]),
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = kb._end_run(
                    conn, tid, outcome="timed_out", status="timed_out",
                    error=error, metadata=payload,
                )
                kb._append_event(conn, tid, "timed_out", payload, run_id=run_id)
                timed_out.append(tid)
        # Outside the write_txn above because ``_record_task_failure`` opens its
        # own. If the breaker trips this flips the task to ``blocked`` and emits
        # ``gave_up`` on top of the ``timed_out`` already emitted.
        if cur.rowcount == 1:
            kbd._record_task_failure(
                conn, tid,
                error=error,
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "sigkill": killed, "retry_status": retry_status},
            )
    return timed_out
