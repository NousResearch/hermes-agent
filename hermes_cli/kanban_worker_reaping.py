"""Reaping for kanban workers that outlived their run.

Two sweeps, run in this order by the dispatch reclaim phase:

* :func:`reap_terminal_workers` ends a worker *process* that outlived its closed
  run (issue #111791), keyed on the run row's retained pid + spawn fingerprint;
* :func:`reap_orphaned_worker_scopes` stops the transient systemd *scope* a
  finished run's descendants still hold (#70716) — the case where the worker
  itself is long dead, so no pid-keyed sweep can see the leak.

Split out of :mod:`hermes_cli.kanban_db_dispatch` (which is at its ``FILE_LINES``
cap) along the topic the two sweeps share. Helpers that stay in the dispatcher are
reached late-bound, the same import-cycle break the dispatcher itself uses.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Optional

from tools.process_registry import _stop_systemd_unit
from tools.process_registry import systemd_user_bus_env
from tools.process_registry_scopes import parse_kanban_worker_scope
from tools.process_registry_scopes import WORKER_SCOPE_PREFIX

# A healthy worker is still alive for a while after kanban_complete /
# kanban_request_review returns (final assistant turn, session persistence), so
# a run's retained worker is only reaped once ended_at is at least this old
# (two default dispatch ticks).
TERMINAL_WORKER_REAP_GRACE_SECONDS = 120

# Spawn fingerprint persisted when /proc probing failed: identity is unknown, so
# the pid must never be signalled on its own.
UNVERIFIED_WORKER_FINGERPRINT = "unverified"


def reap_terminal_workers(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """End host-local workers that outlived their run (issue #111791) — a worker
    that called ``kanban_complete`` and then hung keeps its ``state.db`` sidecar
    fds open and no ``running``-only sweep can see it once ``tasks.worker_pid`` is
    cleared. Keys on the closed ``task_runs`` row's retained pid + spawn
    fingerprint: a legacy row (NULL fingerprint) or a recycled PID is never
    signalled; a pid that is simply gone just has its evidence cleared. A run
    that ended less than ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago is left
    alone so a worker still finalising after its own transition is not killed.
    One row's failure (signal, /proc probe) is logged and skips only that row.
    Returns the task ids whose worker was terminated."""
    rows = conn.execute(
        "SELECT id, task_id, worker_pid, worker_started_at, claim_lock FROM task_runs "
        "WHERE ended_at IS NOT NULL AND ended_at <= ? "
        "AND worker_pid IS NOT NULL AND worker_started_at IS NOT NULL",
        (int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS,),
    ).fetchall()
    host_prefix = _kb._host_prefix()
    reaped: list[str] = []
    for row in rows:
        try:
            _reap_terminal_worker_row(conn, row, host_prefix, signal_fn, reaped)
        except Exception:
            _kb._log.debug(
                "kanban dispatch: terminal worker reap failed for run %s (task %s)",
                row["id"], row["task_id"], exc_info=True,
            )
    return reaped


def _reap_terminal_worker_row(conn, row, host_prefix: str, signal_fn, reaped: list[str]) -> None:
    # Late-bound: both live in the dispatcher, which imports this module.
    from hermes_cli.kanban_db_dispatch import _terminate_reclaimed_worker, _worker_alive

    pid, fingerprint = int(row["worker_pid"]), row["worker_started_at"]
    if pid == os.getpid() or not str(row["claim_lock"] or "").startswith(host_prefix):
        return
    if fingerprint == UNVERIFIED_WORKER_FINGERPRINT and _kb._pid_alive(pid):
        return  # unproven identity: never signalled; its evidence is cleared once the pid is gone
    alive = _worker_alive(pid, fingerprint)
    termination = None
    if alive:
        termination = _terminate_reclaimed_worker(
            pid, row["claim_lock"], signal_fn=signal_fn, started_at=fingerprint)
        if not termination["terminated"]:
            return  # still alive: try again next tick
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL "
            "WHERE id = ? AND worker_pid = ? AND worker_started_at = ?",
            (row["id"], pid, fingerprint),
        )
        if alive:
            _kb._append_event(
                conn, row["task_id"], "terminal_worker_reaped",
                {"pid": pid, "worker_started_at": fingerprint, **(termination or {})}, run_id=row["id"],
            )
    if alive:
        reaped.append(row["task_id"])


def _worker_scope_units() -> set[str]:
    """Names of the live ``hermes-worker-*`` transient scopes on this user's manager.

    Enumerated with ``--no-legend --plain`` (both supported by systemd >= 206, well
    below this tree's floor) so no locale- or width-dependent column parsing is needed.
    An empty set means "could not enumerate" for *every* caller — the probe being
    unavailable is indistinguishable from no scopes existing, and either way there is
    nothing this sweep is permitted to touch.
    """
    binary = shutil.which("systemctl")
    if binary is None:
        return set()
    try:
        result = subprocess.run(
            [binary, "--user", "list-units", "--type=scope", "--all", "--no-legend", "--plain"],
            capture_output=True, timeout=15, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", check=False,
            env=systemd_user_bus_env(),
        )
    except (OSError, subprocess.SubprocessError):
        # No usable user manager (container, no bus, timeout): nothing to enumerate,
        # and nothing this sweep may touch. Log the traceback so a real breakage
        # is diagnosable rather than silently reading as "no scopes".
        _kb._log.debug("kanban dispatch: worker-scope enumeration failed", exc_info=True)
        return set()
    if result.returncode != 0:
        return set()
    names = set()
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if parts and parts[0].startswith(WORKER_SCOPE_PREFIX):
            names.add(parts[0])
    return names


def reap_orphaned_worker_scopes(
    conn: sqlite3.Connection, *, now: Optional[int] = None, stop_fn=None,
) -> list[str]:
    """Stop the worker scopes of *finished* runs that still hold live descendants (#70716).

    A kanban worker runs in its own transient ``systemd-run --user --scope`` so a
    gateway restart cannot kill it (``_restart_safe_worker_argv``). That scope is
    launched ``--collect``, which removes it only once **every** process in its
    cgroup has exited. A worker that leaves a background helper running — a mock
    server, a crawler, a ``serve_forever()`` child — therefore keeps its scope
    ``active`` forever: killing the worker PID cannot reap the child, so the child
    outlives the run indefinitely, holding its port and RSS, and the next attempt at
    the same card dies on ``EADDRINUSE`` looking like broken test infrastructure
    rather than an orphan (``t_eca0aab6`` / ``t_dcfc99ec``).

    :func:`reap_terminal_workers` cannot help: it keys on the closed ``task_runs``
    row's retained pid, and that pid *is* the dead worker — which is exactly why the
    row's evidence looks clean while the scope is still occupied. So this sweep
    deliberately does **not** decide liveness from PIDs. It inverts the question: it
    enumerates the live ``hermes-worker-*`` scopes and stops the ones whose run names
    **no** currently-open row. That set is authoritative for "a dead run's scope", and
    a worker that is mid-run, mid-``kanban_complete``, or between claim and
    ``worker_registered`` always has an open row holding its scope.

    Three further guards keep this from ever touching a live run:

    * only names matching :data:`WORKER_SCOPE_PREFIX` + a 12-hex task id +
      ``-run-<int>.scope`` are considered;
    * ``_kb._current_run_id`` for that task must be *None* — a run id that merely
      differs is a **replaced** worker, not a dead one (run 1797 was superseded by 1800
      at 01:56:51 while its scope was still occupied);
    * an unreadable board row (``OSError`` from a corrupt/locked DB) is treated as
      "live" — never stop a scope on a guess.

    A scope that vanishes between the listing and the stop is a no-op (``_stop_systemd_unit``
    reports it as stopped), so the pass is idempotent and self-clearing. Returns the
    scope names stopped. Best-effort throughout: every failure is logged and skips
    only its own scope.
    """
    if not sys.platform.startswith("linux"):
        return []
    units = _worker_scope_units()
    if not units:
        return []
    if now is None:
        now = int(time.time())
    if stop_fn is None:
        stop_fn = _stop_systemd_unit

    reaped: list[str] = []
    for unit in sorted(units):
        try:
            parsed = parse_kanban_worker_scope(unit)
            if parsed is None:
                continue
            task_id, run_id = parsed
            # A worker whose run is still closing owns its scope and is not yet
            # reaped (same grace as reap_terminal_workers, same reason: the final
            # assistant turn and session persistence run after kanban_complete).
            row = conn.execute(
                "SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is not None and _kb._row_get(row, "ended_at") is not None:
                if now - int(_kb._row_get(row, "ended_at")) < TERMINAL_WORKER_REAP_GRACE_SECONDS:
                    continue
            if _kb._current_run_id(conn, task_id) is not None:
                continue  # live run, or a run that superseded this one
            if not stop_fn(unit):
                continue
            reaped.append(unit)
            _kb._log.warning(
                "kanban dispatch: reaped leaked descendant(s) in %s (task %s run %s, run closed)",
                unit, task_id, run_id,
            )
            with contextlib.suppress(Exception):
                _kb._append_event(
                    conn, task_id, "worker_scope_reaped",
                    {"scope": unit, "run_id": run_id}, run_id=run_id,
                )
        except Exception:  # health: allow BLE001 -- per-scope boundary: one unreadable or unstoppable unit must not abort the sweep; the traceback is logged
            _kb._log.debug(
                "kanban dispatch: worker-scope reap failed for %s", unit, exc_info=True,
            )
    return reaped


# Late-bound origin namespace (see the module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from ``kanban_db_dispatch``,
# which imports this module.
from hermes_cli import kanban_db as _kb
