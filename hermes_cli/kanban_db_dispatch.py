"""Dispatcher: crash/stale/orphan detection, failure accounting and the respawn circuit breaker, memory-aware concurrency caps, the one-shot ``dispatch_once`` pass, worker spawning (``_default_spawn``), worker-log rotation and the long-lived ``run_daemon`` loop.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING

# KENSEI CUSTOM (fork re-anchor): datetime (daily spawn counter), hashlib
# (deterministic review sampling), and module logger for re-anchored machinery.
from datetime import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402

_log = logging.getLogger(__name__)

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

# KENSEI CUSTOM (fork re-anchor): best-effort activity-ledger import — the
# ledger must never break dispatch.
try:
    from hermes_cli.profile_activity_ledger import record_event_if_enabled
except Exception:  # pragma: no cover
    def record_event_if_enabled(**_kw):
        return None

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


# After this many consecutive non-success attempts on a task/profile the
# dispatcher parks the task in ``blocked`` with a reason — prevents retry storms.
DEFAULT_FAILURE_LIMIT = 2

# KENSEI CUSTOM (fork re-anchor): fork's alias name for the same breaker default.
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT

# Worker log files larger than this at spawn time are rotated.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and make a terminal board call (kanban_block/kanban_complete/kanban_request_review)
# before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# A healthy worker is still alive for a while after kanban_complete /
# kanban_request_review returns (final assistant turn, session persistence), so
# a run's retained worker is only reaped once ended_at is at least this old
# (two default dispatch ticks).
TERMINAL_WORKER_REAP_GRACE_SECONDS = 120

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|auth\w*|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Cooldown after a rate-limited (quota-wall) requeue before re-spawning. Without
# it the task would re-spawn on the very next tick and bounce off the same quota
# wall, burning a worker slot every tick for hours. Overridable via
# ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass.

    ``kanban.default_assignee`` applied this tick before spawning (#27145). Surfaces the auto-assignment to
    telemetry / CLI / dashboard so the operator can see when the dispatcher is acting on the fallback rule
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is ``(task_id, assignee,
    current_running_count)``. NOT an operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so telemetry / dashboards can show "this
    profile is busy" vs
    the board's dispatch lock (issue #35240). A losing dispatcher does no DB writes this tick — the lock
    holder is making progress on the same board. This is the steady-state signal that a single-writer guard
    is
    """

    reclaimed: int = 0
    promoted: int = 0
    reconciled_orphans: list[str] = field(default_factory=list)
    """``running`` cards requeued by :func:`reconcile_orphaned_running` (broken
    claim bookkeeping, dead/gone worker)."""
    reaped_terminal_workers: list[str] = field(default_factory=list)
    """Task ids whose worker outlived its closed run and was terminated by
    :func:`reap_terminal_workers`."""
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids with no assignee at all — operator-actionable (usually a
    misfiled task waiting for routing)."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Unassigned task ids that had ``kanban.default_assignee`` applied this
    tick before spawning, so telemetry/CLI/dashboard can show the dispatcher
    acting on the fallback rule rather than explicit assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids whose assignee names a control-plane lane (e.g. a Claude
    Code terminal like ``orion-cc``), not a Hermes profile. Expected steady-state
    on multi-lane setups, NOT operator-actionable; tracked apart so health
    telemetry can tell "stuck" from "correctly idle"."""
    skipped_review_nonspawnable: list[tuple[str, str]] = field(default_factory=list)
    """``(task_id, unresolvable_reviewer)`` pairs: review-column tasks whose
    reviewer is not a spawnable profile (e.g. the ``sdlc-review`` skill name
    passed as ``reviewer``). Operator-actionable — unlike the ready lane,
    nothing ever pulls a review task, so it starves silently without this
    signal."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """``(task_id, assignee, current_running_count)`` deferred because the
    assignee is at ``kanban.max_in_progress_per_profile``. Picked up on a later
    tick; separate bucket so dashboards show "profile busy" vs "stuck"."""
    # KENSEI CUSTOM (fork re-anchor): ``(task_id, from_stage, to_stage)`` triples for
    # tasks that advanced through the feature pipeline (gate passed).
    pipeline_advanced: list[tuple[str, str, str]] = field(default_factory=list)
    # KENSEI CUSTOM (fork re-anchor): task ids rejected before claim/spawn because
    # dispatcher-visible invariants failed (e.g. forced skills missing).
    dispatcher_rejected: list[str] = field(default_factory=list)
    # KENSEI CUSTOM (fork re-anchor): True when the daily spawn budget was exhausted
    # this tick (cost-governance hard stop, P2-1).
    budget_exhausted: bool = False
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed for no heartbeat within ``dispatch_stale_timeout_seconds``."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """``(task_id, reason)`` skipped by the respawn guard: ``"blocker_auth"``
    (quota/auth error — also auto-blocked), ``"recent_success"`` (completed run
    within guard window), ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released to ``ready`` WITHOUT counting
    a failure — a long quota window must never trip the circuit breaker."""
    skipped_locked: bool = False
    """True when another process held the board's dispatch lock: this tick did
    no DB writes; the lock holder is making progress on the same board."""
    memory_pressure: Optional[str] = None
    """Memory pressure that restricted this tick: ``"critical"`` (no new
    workers), ``"elevated"`` (at most one), ``None`` (no restriction).
    Reclaim/promotion bookkeeping still ran; deferred tasks stay queued."""


def describe_suppression(results: Iterable[Optional["DispatchResult"]]) -> str:
    """One line naming why the tick(s) held ready work back, or ``""``.

    ``active_pr=1, recent_success=2, rate_limited=1, skipped_locked=1,
    memory_pressure=critical`` — the respawn-guard reasons counted per task
    plus the tick-level holds. Feeds the "dispatcher stuck" warnings of the
    CLI daemon and the embedded gateway dispatcher, which otherwise report a
    bare zero-spawn count while ``hermes kanban tail`` is the only place the
    guard reason is written (#111910).
    """
    counts: dict[str, int] = {}
    pressure: Optional[str] = None
    for res in results:
        if res is None:
            continue
        for _task_id, reason in res.respawn_guarded:
            counts[reason] = counts.get(reason, 0) + 1
        if res.rate_limited:
            counts["rate_limited"] = counts.get("rate_limited", 0) + len(res.rate_limited)
        if res.skipped_locked:
            counts["skipped_locked"] = counts.get("skipped_locked", 0) + 1
        if res.memory_pressure:
            pressure = res.memory_pressure
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    if pressure:
        parts.append(f"memory_pressure={pressure}")
    return ", ".join(parts)


# Bounded registry of recently-reaped worker exits, filled by the reap loop in
# ``dispatch_once`` and read by ``detect_crashed_workers`` to classify a dead-pid
# task. Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``; raw status kept so
# both WIFEXITED/WEXITSTATUS and WIFSIGNALED can be consulted. Trimmed by age
# plus a total size cap. Process-local by nature (``waitpid`` only reaps our own
# children): a per-tick ``hermes kanban dispatch`` process finds it empty, so
# ``_classify_dead_worker_exit`` falls back to the exit trailer the worker
# leaves in its own log (``KANBAN_WORKER_EXIT_TRAILER``).
_RECENT_WORKER_EXIT_TTL_SECONDS = 600
_RECENT_WORKER_EXITS_MAX = 4096
_recent_worker_exits: "dict[int, tuple[int, float]]" = {}

# Windows has no ``waitpid(-1)``: a child's exit code is only recoverable
# through a live handle, so ``_default_spawn`` parks each worker's ``Popen``
# here (Windows only) and ``reap_worker_zombies`` polls it. Entry: ``pid -> Popen``.
_live_worker_procs: "dict[int, subprocess.Popen]" = {}


def _wait_status_from_returncode(returncode: int) -> int:
    """Encode a ``Popen.returncode`` in the wait-status layout the registry stores."""
    return (int(returncode) & 0xFF) << 8


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped child's exit status; duplicate pids overwrite (latest wins)."""
    if not pid or pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[int(pid)] = (int(raw_status), now)
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """``(kind, code)`` for a reaped worker PID: ``clean_exit`` (rc 0 while
    still ``running`` = protocol violation), ``rate_limited``
    (``KANBAN_RATE_LIMIT_EXIT_CODE``, never counts as a failure),
    ``nonzero_exit``, ``signaled`` (``code`` is the signal), ``unknown`` (pid
    not in the reap registry; ``code`` None)."""
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    # Bit-level POSIX wait-status decode instead of os.WIFEXITED/WEXITSTATUS/
    # WIFSIGNALED/WTERMSIG: those helpers do not exist on Windows, where the
    # registry is fed by reap_worker_zombies' Popen poll. Low 7 bits = signal
    # (0 = normal exit, 0x7F = stopped), bits 8-15 = exit code.
    raw = int(raw)
    signal_number = raw & 0x7F
    if signal_number == 0:
        return _exit_code_kind((raw >> 8) & 0xFF)
    if signal_number != 0x7F:
        return ("signaled", signal_number)
    return ("unknown", None)


def _exit_code_kind(code: int) -> "tuple[str, int]":
    """``(kind, code)`` for a worker's exit code, however it was observed."""
    if code == 0:
        return ("clean_exit", 0)
    if code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE:
        return ("rate_limited", code)
    if code == _kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE:
        return ("terminal_provider", code)
    return ("nonzero_exit", code)


_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)


def _worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Exit code from the trailer the worker CLI wrote to its own log; None when absent.

    The durable twin of ``_recent_worker_exits``: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running this sweep ever reaped the worker. Last trailer
    wins — the log is append-mode across re-runs.
    """
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return None
    matches = _EXIT_TRAILER_RE.findall(raw or "")
    return int(matches[-1]) if matches else None


def reap_worker_zombies() -> "list[int]":
    """Reap exited workers without blocking; returns reaped PIDs. POSIX reaps
    every child via ``waitpid(-1)``; Windows polls the ``Popen`` handles
    parked by ``_default_spawn`` (the only way to learn a child's exit code
    there), so the rate-limit sentinel exit is classified on both hosts."""
    reaped: "list[int]" = []
    if _kb._IS_WINDOWS:
        for pid, proc in list(_live_worker_procs.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            _record_worker_exit(pid, _wait_status_from_returncode(returncode))
            _live_worker_procs.pop(pid, None)
            reaped.append(pid)
        return reaped
    try:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            _record_worker_exit(pid, status)
            reaped.append(pid)
    except Exception:
        pass
    return reaped


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Uses ``gateway.status._pid_exists`` (OpenProcess on Windows, ``os.kill(pid, 0)``
    on POSIX). **DO NOT** call ``os.kill(pid, 0)`` directly on Windows — there
    ``sig=0`` is ``CTRL_C_EVENT`` broadcast to the console group, potentially
    killing unrelated processes.

    Zombies (exited, not yet reaped) still pass the existence check, so a
    worker would look "alive" forever between exit and reap. Linux: peek at
    ``/proc/<pid>/status`` and treat ``State: Z`` as dead; macOS: ask ``ps``
    for the BSD ``stat`` field and treat ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True


# ``worker_started_at`` value for a spawn whose fingerprint could not be captured. Distinct from the
# NULL legacy row (pre-fingerprint spawn): such a worker is held (its claim is never released beside
# the live PID) but NEVER signalled — missing process identity is refusal, not permission (#99558).
UNVERIFIED_WORKER_FINGERPRINT = "unverified"


def _process_fingerprint(pid: int) -> Optional[str]:
    """Restart-stable identity of a live process: ``"<instantiation epoch>|<start time>"``. The start
    time alone (``/proc/<pid>/stat`` field 22 on Linux) is clock ticks since THIS boot, so a row that
    survives a reboot could match an unrelated process with the same PID and the same tick value;
    ``gateway.drain_control.current_instantiation_epoch`` (``boot_id`` + PID-1 start) changes on every
    reboot / container recreate, so the composed value never survives one. ``None`` when unreadable."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time
    start = get_process_start_time(int(pid))
    if start is None:
        return None
    return f"{current_instantiation_epoch()}|{start}"


def _worker_alive(pid: Optional[int], started_at) -> bool:
    """True when ``pid`` is live AND is still the worker we spawned. ``started_at`` is the fingerprint
    recorded by ``_set_worker_pid``; after a reboot (or any PID recycle) an unrelated process can own
    the number, so bare existence is never enough to extend a claim or to signal. A legacy row without
    a fingerprint keeps the existence answer: killing it is the pre-fingerprint behaviour and the row is
    rewritten with a fingerprint on its next spawn. An UNVERIFIED spawn also keeps the existence answer
    (a claim is never released beside a possibly-live worker) but ``_terminate_reclaimed_worker``
    refuses to signal it."""
    if not _kb._pid_alive(pid):
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    return not _pid_recycled(pid, started_at)


def _pid_recycled(pid: Optional[int], started_at) -> bool:
    """True when a live ``pid`` is NOT the process fingerprinted at spawn (or the fingerprint can no
    longer be read). Signalling it would hit a stranger. ``None`` fingerprint = legacy row, never
    recycled; the UNVERIFIED marker is always foreign. An integer fingerprint (rows written before the
    boot witness was added) compares the start time only."""
    if started_at is None or not pid:
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    if isinstance(started_at, str) and "|" in started_at:
        return _process_fingerprint(int(pid)) != started_at
    from gateway.status import _start_times_agree, get_process_start_time
    current = get_process_start_time(int(pid))
    if current is None:
        return True
    try:
        return not _start_times_agree(current, started_at)
    except (TypeError, ValueError):
        return True


def _kill_fn(signal_fn) -> Optional[Callable[[int, int], None]]:
    """``signal_fn`` test hook, else ``os.kill`` when the platform has one."""
    if signal_fn is not None:
        return signal_fn
    return os.kill if hasattr(os, "kill") else None


def _poll_worker_exit(pid: int, started_at: Optional[int] = None) -> bool:
    """Poll ~5 s (10 x 0.5 s) for ``pid`` to die; True once it is gone."""
    for _ in range(10):
        if not _worker_alive(pid, started_at):
            return True
        time.sleep(0.5)
    return False


def _sigkill(kill, pid: int) -> bool:
    """Best-effort SIGKILL; True when the signal was delivered."""
    try:
        # signal.SIGKILL doesn't exist on Windows; SIGTERM maps to TerminateProcess.
        kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
        return True
    except (ProcessLookupError, OSError):
        return False


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    started_at=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths. ``started_at`` is the spawn-time
    fingerprint: when the live process no longer matches it, the PID was recycled and nothing is
    signalled — the worker is gone, which is what the reclaim wanted (``terminated`` = True). An
    UNVERIFIED spawn (fingerprint capture failed) that is still live is never signalled either, but
    it is reported as surviving (``signal_refused``) so the reclaim holds the claim instead of
    spawning a duplicate beside it."""
    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info
    if not str(claim_lock).startswith(_kb._host_prefix()):
        return info
    info["host_local"] = True

    kill = _kill_fn(signal_fn)
    if kill is None:
        return info
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        # Never signal by bare number: a dead PID is "gone" (reclaim proceeds), a live one is held.
        info["signal_refused"] = True
        info["terminated"] = not _kb._pid_alive(pid)
        return info
    if _kb._pid_alive(pid) and _pid_recycled(pid, started_at):
        info["terminated"] = True
        info["pid_recycled"] = True
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Already gone = successful termination. Leaving terminated=False would
        # make the reclaim guard misread a dead worker as alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    if _poll_worker_exit(pid, started_at):
        info["terminated"] = True
        return info
    if _worker_alive(pid, started_at):
        if not _sigkill(kill, pid):
            return info
        info["sigkill"] = True
    info["terminated"] = not _worker_alive(pid, started_at)
    return info


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
                {"pid": pid, "worker_started_at": fingerprint, **termination}, run_id=row["id"],
            )
    if alive:
        reaped.append(row["task_id"])


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming then would release the claim and spawn a second worker while the
    first still runs — the duplication loop. Only host-local workers we actually
    signalled count; a non-local lock or no-op attempt (no ``os.kill``) must fall
    through to the normal release path since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("host_local")
        and (termination.get("termination_attempted") or termination.get("signal_refused"))
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records ``reclaim_deferred``.
    The next tick retries the kill; not spawning a duplicate is what lets the
    throttled worker finally die.
    """
    grace = now + _kb.RECLAIM_DEFER_GRACE_SECONDS
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?", (grace, run_id))
        payload = {"reason": reason, "claim_lock": claim_lock, "claim_expires_now": grace}
        payload.update(termination)
        _kb._append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def _reap_done_workers(conn: sqlite3.Connection) -> list[int]:
    """Kill workers whose tasks reached a terminal status but whose PID is still alive.

    When a worker completes its task (kanban_complete → done/archived) and calls
    sys.exit(), atexit cleanup can deadlock on shared resources (state.db WAL,
    kanban.db lock files, log FDs) under concurrent access — the worker stays
    alive indefinitely in futex_wait_queue, holding swap pages but doing zero
    work.  The dispatcher's zombie reaper only handles PID-is-dead (true zombie)
    cases and never sees these.

    This function finds tasks in 'done' / 'archived' status whose ``worker_pid``
    is still alive on this host and sends SIGKILL.  Once dead, the normal zombie
    reaper handles the waitpid on the next tick.  The ``worker_pid`` column is
    cleared so ``detect_crashed_workers`` doesn't try to reclaim an already-done
    task on a subsequent tick.
    """
    reaped: list[int] = []
    import signal
    try:
        rows = conn.execute(
            "SELECT id, worker_pid FROM tasks "
            "WHERE status IN ('done', 'archived') AND worker_pid IS NOT NULL"
        ).fetchall()
    except Exception:
        return reaped

    for row in rows:
        pid = row["worker_pid"]
        if not pid or pid <= 0:
            continue
        if not _pid_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            reaped.append(pid)
        except (ProcessLookupError, PermissionError):
            pass  # already gone or no permission
        try:
            conn.execute(
                "UPDATE tasks SET worker_pid = NULL WHERE id = ?", (row["id"],)
            )
        except Exception:
            pass
    return reaped


def _clear_stale_ready_claims(conn: sqlite3.Connection) -> int:
    """Clear orphaned claim_lock / worker_pid fields from ready tasks.

    When a task is manually reset from running→ready (e.g. after DB
    recovery, operator intervention, or a kill -9 on stuck workers),
    the claim_lock and worker_pid columns retain their old values.
    The dispatcher interprets a non-NULL claim_lock as an active claim
    and skips the task — it stays stuck in 'ready' forever.

    This runs at the start of every dispatch tick so the board self-
    heals without operator intervention.  Returns the number of tasks
    cleaned.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, started_at = NULL "
            "WHERE status = 'ready' AND claim_lock IS NOT NULL"
        )
        return cur.rowcount


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record a ``heartbeat`` event + touch ``last_heartbeat_at``.

    Liveness signal orthogonal to the PID check: a worker whose forked child
    (train loop, crawl) is stuck can still have a live Python process.
    Returns False if the task is not running or its claim expired.
    """
    now = int(time.time())
    with _kb.write_txn(conn):
        sql = "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running'"
        params: tuple = (now, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params += (int(expected_run_id),)
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _kb._current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute("UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?", (now, run_id))
        _kb._append_event(
            conn, task_id, "heartbeat",
            {"note": note} if note else None,
            run_id=run_id,
        )
    return True


def enforce_max_runtime(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    SIGTERM, short grace, then SIGKILL. Emits ``timed_out`` and restores the
    task's source phase so the next tick re-spawns the same kind of worker —
    unless the circuit breaker already gave up, leaving it blocked. Host-local
    only (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a test hook.
    """
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = _kb._host_prefix()

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
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
        limit = int(row["max_runtime_seconds"])
        if elapsed < limit:
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        started_at = _kb._row_get(row, "worker_started_at")
        if started_at == UNVERIFIED_WORKER_FINGERPRINT and _kb._pid_alive(pid):
            # Fingerprint capture failed at spawn: we cannot prove this live PID is our worker, so
            # it is neither signalled nor released beside (duplicate). It is reclaimed once it exits.
            _kb._log.warning("kanban: task %s worker pid %s exceeded max runtime but has no verified "
                             "identity; not signalled", tid, pid)
            continue
        # SIGTERM then SIGKILL after 5 s grace; workers wanting a cleaner
        # shutdown install their own SIGTERM handler. A recycled PID (fingerprint
        # mismatch) is never signalled: the worker is already gone.
        killed = False
        kill = _kill_fn(signal_fn)
        if kill is not None and not (_kb._pid_alive(pid) and _pid_recycled(pid, started_at)):
            with contextlib.suppress(ProcessLookupError, OSError):
                kill(pid, signal.SIGTERM)
            # Short polling wait — no time.sleep on the write txn.
            _poll_worker_exit(pid, started_at)
            if _worker_alive(pid, started_at):
                killed = _sigkill(kill, pid)

        error = f"elapsed {int(elapsed)}s > limit {limit}s"
        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
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
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = _kb._end_run(
                    conn, tid, outcome="timed_out", status="timed_out",
                    error=error, metadata=payload,
                )
                _kb._append_event(conn, tid, "timed_out", payload, run_id=run_id)
                # KENSEI CUSTOM (fork re-anchor): loop-diagnostics for the
                # terminal timeout failure (run already closed above).
                _attach_loop_diagnosis(
                    conn, tid,
                    run_id=run_id,
                    outcome="timed_out",
                    error=f"elapsed {int(elapsed)}s > limit {limit}s",
                )
                timed_out.append(tid)
        # Outside the write_txn above because ``_record_task_failure`` opens its
        # own. If the breaker trips this flips the task to ``blocked`` and emits
        # ``gave_up`` on top of the ``timed_out`` already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=error,
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "sigkill": killed, "retry_status": retry_status},
            )
    return timed_out


# A running task with no heartbeat for this long is inactive regardless of
# ``dispatch_stale_timeout_seconds`` (spec: ">4h started + no commits in 1h").
_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks with no heartbeat progress; returns their ids.

    Stale = running longer than ``stale_timeout_seconds`` (active run's
    ``started_at``, else ``tasks.started_at``) AND ``last_heartbeat_at`` NULL or
    older than ``_STALE_HEARTBEAT_GAP_SECONDS``. Task returns to its source
    phase, run closes ``outcome='stale'``, a live host-local worker is killed.
    ``0`` disables the check; ``signal_fn`` is a test hook. Deliberately NOT
    counted via ``_record_task_failure``: an absent heartbeat is not a worker
    failure, and counting it would let long-running tasks trip the breaker.
    """
    if stale_timeout_seconds <= 0:
        return []

    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        if row["active_started_at"] is None:
            continue
        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _STALE_HEARTBEAT_GAP_SECONDS:
            continue

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        termination = _kb._terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn, started_at=_kb._row_get(row, "worker_started_at"))

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": _kb._opt_int(last_hb),
                "heartbeat_age_seconds": _kb._opt_int(hb_age),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _kb._end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _kb._append_event(conn, tid, "stale", payload, run_id=run_id)
            reclaimed.append(tid)

    return reclaimed


def claim_pipeline_task(
    conn,
    task_id,
    *,
    ttl_seconds=None,
    claimer=None,
):
    """Atomically transition a pipeline-stage task to ``running``.

    Pipeline tasks live in stage-specific statuses (``research``,
    ``prd``, ``spec``, ``council``) rather than ``ready``.  A worker
    spawned on a gate-failure needs to move the task to ``running``
    so it can work on the artifact, then return it to its original
    stage on completion.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed or is not in a pipeline stage.

    Stores the originating stage in the claim event payload so
    ``complete_pipeline_task`` knows which status to restore.
    """
    import time
    from hermes_cli.feature_pipeline import PIPELINE_STAGES
    # All helpers are module-level in this file — no circular import.

    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)

    _pipeline_statuses = tuple(PIPELINE_STAGES)

    # Read the current pipeline stage BEFORE the CAS, so we can
    # record it in the claim event.  The status column will be
    # `running` after the CAS — we want the originating stage.
    origin_row = conn.execute(
        "SELECT status, pipeline_stage FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    origin_stage = (
        origin_row["pipeline_stage"] if origin_row else None
    ) or (origin_row["status"] if origin_row else None)

    with write_txn(conn):
        cur = conn.execute(
            f"""\
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status IN ({','.join('?' * len(_pipeline_statuses))})
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id, *_pipeline_statuses),
        )
        if cur.rowcount != 1:
            return None

        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_cur = conn.execute(
            """\
            INSERT INTO task_runs (
                task_id, profile, step_key, status,
                claim_lock, claim_expires, max_runtime_seconds,
                started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id,
             "source_status": "pipeline",
             "pipeline_stage": origin_stage},
            run_id=run_id,
        )
        return get_task(conn, task_id)


def complete_pipeline_task(
    conn,
    task_id,
    *,
    result=None,
    summary=None,
    metadata=None,
):
    """Return a pipeline-stage worker to its original stage.

    After a pipeline worker finishes writing its artifact (e.g.
    research-brief.md), the task MUST go back to its pipeline stage
    — NOT ``done`` — so the gate re-checks on the next dispatcher
    tick and advances naturally.

    Reads the originating ``pipeline_stage`` from the most recent
    ``claimed`` event whose ``source_status`` = ``"pipeline"``.
    Falls back to the current ``tasks.pipeline_stage`` column.
    """
    import time

    now = int(time.time())
    row = conn.execute(
        "SELECT id, pipeline_stage FROM tasks WHERE id = ? AND status = 'running'",
        (task_id,),
    ).fetchone()
    if not row:
        return False

    # Recover the originating stage from the latest pipeline claim event.
    origin = conn.execute(
        """SELECT json_extract(payload, '$.pipeline_stage')
             FROM task_events
            WHERE task_id = ?
              AND kind = 'claimed'
              AND json_extract(payload, '$.source_status') = 'pipeline'
            ORDER BY id DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    origin_stage = origin[0] if origin else None

    # Belt-and-braces: fall back to the column if the event is missing.
    stage = origin_stage or row["pipeline_stage"]
    if not stage:
        stage = "research"

    with write_txn(conn):
        conn.execute(
            """UPDATE tasks
                 SET status        = ?,
                     pipeline_stage = ?,
                     claim_lock    = NULL,
                     claim_expires = NULL,
                     worker_pid    = NULL,
                     result        = ?,
                     completed_at  = ?
               WHERE id = ?
                 AND status = 'running'""",
            (stage, stage, result, now, task_id),
        )
        _append_event(
            conn, task_id, "completed",
            {"result_len": len(result) if result else 0,
             "summary": summary or None,
             "source_status": "pipeline",
             "returned_to_stage": stage,
             "pipeline_stage": stage},
        )
    return True


def clear_stale_pipeline_claims(conn: sqlite3.Connection) -> int:
    """Clear stale claim locks on pipeline-stage tasks.

    ``release_stale_claims`` only looks at ``status='running'``, but
    pipeline tasks sit in stage statuses (research, prd, spec, etc.)
    with ``claim_lock`` still set after a worker crash.  The pipeline
    dispatch query at line ~7684 requires ``claim_lock IS NULL``, so
    those tasks become permanently invisible to the dispatcher.

    This function targets pipeline-stage tasks whose claim has expired
    and clears the lock fields so they re-enter the gate-check loop.
    """
    try:
        from hermes_cli.feature_pipeline import PIPELINE_STAGES
    except ImportError:
        return 0
    _pipeline_statuses = tuple(PIPELINE_STAGES)
    if not _pipeline_statuses:
        return 0
    now = int(time.time())
    _placeholders = ",".join("?" * len(_pipeline_statuses))
    rows = conn.execute(
        f"SELECT id FROM tasks "
        f"WHERE status IN ({_placeholders}) "
        f"  AND claim_lock IS NOT NULL "
        f"  AND claim_expires IS NOT NULL "
        f"  AND claim_expires < ?",
        (*_pipeline_statuses, now),
    ).fetchall()
    if not rows:
        return 0
    cleared = 0
    with write_txn(conn):
        for r in rows:
            conn.execute(
                "UPDATE tasks SET claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL WHERE id = ?",
                (r["id"],),
            )
            _append_event(
                conn, r["id"], "claim_stale_cleared",
                {"reason": "pipeline claim expired, clearing for gate re-entry"},
            )
            cleared += 1
    return cleared


def reconcile_orphaned_running(conn: sqlite3.Connection) -> list[str]:
    """Requeue ``running`` cards with broken claim bookkeeping; returns their ids.

    A task ``running`` with NULL ``claim_lock``/``claim_expires`` (crash
    mid-claim, manual SQL, DB restore) is a zombie forever: ``release_stale_claims``
    needs ``claim_expires``, ``detect_crashed_workers`` needs a host-local lock +
    pid, ``detect_stale_running`` is off by default. Orphans go back to ``ready``
    with a comment, leaked run closed, ``reconciled`` event; a row with a live
    host-local PID is deferred so no duplicate spawns beside it.
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid, worker_started_at FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _worker_alive(pid, _kb._row_get(row, "worker_started_at")):
            # Never requeue beside a live process. Retry next tick.
            _kb._log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with _kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": _kb._opt_int(row["claim_expires"]),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _kb._end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            _kb._insert_comment(
                conn, tid, "dispatcher",
                "reconciliation: card was 'running' with no valid claim "
                "(dead/gone worker) — requeued to ready",
                now,
            )
            _kb._append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _kb._log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message (strip PIDs, timestamps) so same-root-cause errors group."""
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


# ~96% of "clean exit without a terminal tool call" tasks complete on a later
# run, so a protocol violation gets a bounded retry before the breaker trips.
# The budget is a violation-only STREAK (``_protocol_violation_streak``),
# independent of ``consecutive_failures``: other failure kinds neither consume
# nor extend it. Per-task ``max_retries`` overrides it.
_PROTOCOL_VIOLATION_FAILURE_LIMIT = 3

# Closed runs to walk when counting the streak; it trips at a handful anyway.
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks closed runs newest-first (including the one ``detect_crashed_workers``
    just closed). ``rate_limited`` runs are neutral and skipped (a quota wall
    says nothing about the task); any other closed run breaks the streak, so
    the budget counts ONLY protocol violations. Violations are recognized by the
    ``protocol_violation`` run-metadata marker, with the error text as fallback
    for runs recorded before the marker existed.
    """
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        if outcome == "crashed" and (
            _kb._json_dict(row["metadata"]).get("protocol_violation")
            or "protocol violation" in (row["error"] or "")
        ):
            streak += 1
            continue
        break
    return streak


_PROTOCOL_VIOLATION_ERROR = (
    # Worker subprocess returned 0 but its task is still ``running`` in the DB — it exited without calling
    # ``kanban_complete`` / ``kanban_block`` / ``kanban_request_review``. Overwhelmingly the work itself succeeded and only the
    # paperwork was skipped, so a retry usually completes; the corrective sentence below is surfaced to the
    # retry worker via the prior-attempt error in ``build_worker_context`` (guidance approach from #61817).
    # Keep this short: ``_record_task_failure`` caps the stored error at 500 chars and the worker's own
    # last output (``_worker_final_output``, up to 400 chars) is appended after it — a longer preamble
    # truncates away the worker's explanation, which is the part the board and the retry worker need.
    "worker exited cleanly (rc=0) without kanban_complete, kanban_block "
    "or kanban_request_review — protocol violation. "
    "If the prior run already did the work, verify it and "
    "report it via kanban_complete (or kanban_request_review); "
    "a run without a terminal kanban call counts as failed no "
    "matter what it did."
)


_EXIT_SUMMARY_MARKER = "Resume this session with:"
# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")
_LOG_NOISE_PREFIXES = ("session_id:", "Query:", "Initializing agent")


def _worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's last printed text, for the board diagnostic.

    A ``chat -q`` worker's stdout/stderr are redirected to its per-task log
    (``_default_spawn``), so when it exits without a terminal board call the
    reason is usually sitting there: the model's own explanation of why it could
    not comply (#88603), or the rendered provider error (#46593). The reap used to
    discard it in favour of a canned message on every retry. Trims the CLI exit
    summary, rule lines and the ``session_id:`` trailer; returns "" (never raises)
    on a missing/empty log.

    ``board`` must come from the dispatching tick: ambient current-board resolution
    is wrong for every board but the one the dispatcher thread happens to call
    "current", so the log would silently not be found.
    """
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return ""
    if not raw:
        return ""
    raw = _EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_EXIT_SUMMARY_MARKER)
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_LOG_NOISE_PREFIXES):
            lines.append(ln)
    return " ".join(lines)[-400:]


@dataclass
class _DeadWorker:
    """How ``detect_crashed_workers`` should book one dead worker."""

    kind: str
    code: Optional[int]
    error_text: str
    event_kind: str
    event_payload: dict
    protocol_violation: bool = False
    rate_limited: bool = False
    terminal_provider: bool = False
    """``KANBAN_TERMINAL_PROVIDER_EXIT_CODE``: the provider rejected the worker's
    credential/model — trips the breaker on this first occurrence."""

    @property
    def run_outcome(self) -> str:
        # A rate-limited requeue is recorded as ``rate_limited`` so board history
        # doesn't show a phantom crash for a quota wall.
        return "rate_limited" if self.rate_limited else "crashed"


def _classify_dead_worker(
    pid: int, claimer: Optional[str], *, task_id: Optional[str] = None, board: Optional[str] = None,
) -> _DeadWorker:
    """Map a dead worker's reaped exit status to its reclaim bookkeeping.

    A clean exit or a crash carries the worker's own last output (``worker_output``
    in the event payload, appended to the error text) so the board and the retry
    worker see WHY instead of a bare label; a rate-limited requeue does not need it.
    """
    dead = _classify_dead_worker_exit(pid, claimer, task_id=task_id, board=board)
    if task_id and not dead.rate_limited:
        worker_output = _worker_final_output(task_id, board=board)
        if worker_output:
            dead.error_text += f" Worker's last output: {worker_output!r}"
            dead.event_payload["worker_output"] = worker_output
    return dead


def _classify_dead_worker_exit(
    pid: int,
    claimer: Optional[str],
    *,
    task_id: Optional[str] = None,
    board: Optional[str] = None,
) -> _DeadWorker:
    """Exit status -> reclaim bookkeeping, before the worker's own words are folded in.

    The reap registry only knows children of THIS process; a per-tick dispatcher
    reads the exit trailer the worker left in its log instead, so the same death
    gets the same booking (protocol violation / rate-limit requeue / crash) as
    under the gateway-embedded dispatcher. A worker that never reached its exit
    epilogue (killed, OOM) leaves no trailer and stays a plain crash.
    """
    kind, code = _classify_worker_exit(pid)
    if kind == "unknown" and task_id:
        logged = _worker_log_exit_code(task_id, board=board)
        if logged is not None:
            kind, code = _exit_code_kind(logged)
    if kind == "clean_exit":
        # rc=0 while still ``running``: usually the work succeeded and only the
        # paperwork was skipped; the corrective sentence reaches the retry
        # worker via ``build_worker_context``.
        return _DeadWorker(
            kind, code, _PROTOCOL_VIOLATION_ERROR, "protocol_violation",
            # ``protocol_violation`` is the durable marker for
            # _protocol_violation_streak: _end_run copies this payload into the
            # run metadata.
            {"pid": pid, "claimer": claimer, "exit_code": code, "protocol_violation": True},
            protocol_violation=True,
        )
    if kind == "rate_limited":
        # Quota wall — NOT a task failure. Release to the source phase and do
        # NOT count a failure so a long quota window can't trip the breaker.
        return _DeadWorker(
            kind, code,
            f"pid {pid} exited rate-limited (quota wall) — requeued without counting a failure",
            "rate_limited",
            {"pid": pid, "claimer": claimer, "exit_code": code},
            rate_limited=True,
        )
    if kind == "terminal_provider":
        # The worker classified its own provider failure as unhealable (credential
        # revoked, model gone): every further spawn would hit the same wall, so
        # ``_account_crashes`` trips the breaker now instead of after ``failure_limit``.
        return _DeadWorker(
            kind, code,
            f"pid {pid} exited on a terminal provider error (exit {code}): the provider rejected "
            "this profile's credential or model — fix the configuration, then unblock.",
            "crashed",
            {"pid": pid, "claimer": claimer, "exit_kind": kind, "exit_code": code, "terminal_provider": True},
            terminal_provider=True,
        )
    if kind == "nonzero_exit":
        error_text = f"pid {pid} exited with code {code}"
    elif kind == "signaled":
        error_text = f"pid {pid} killed by signal {code}"
    else:
        error_text = f"pid {pid} not alive"
    event_payload = {"pid": pid, "claimer": claimer}
    if code is not None and kind != "unknown":
        event_payload["exit_kind"] = kind
        event_payload["exit_code"] = code
    return _DeadWorker(kind, code, error_text, "crashed", event_payload)


@dataclass
class _CrashSweep:
    """Everything ``detect_crashed_workers`` collects inside its reclaim txn."""

    crashed: list[str] = field(default_factory=list)
    rate_limited: list[str] = field(default_factory=list)
    # ``(task_id, pid, claimer, dead_worker)``: accounted after the txn via
    # ``_record_task_failure`` (needs its own write_txn).
    crash_details: list[tuple[str, int, str, _DeadWorker]] = field(default_factory=list)
    # KENSEI: ``(task_id, run_id, outcome, error_text)`` for reclaimed crashes. Attached
    # after the reclaim txn commits: the diagnosis integration emits its own
    # event and opens its own write txn, so it must not run inside this one.
    diagnosis_requests: list[tuple[str, Optional[int], str, str]] = field(default_factory=list)
    # Worker-exit observer payloads, fired only after every reclaim/accounting
    # txn has committed.
    exited_hook_payloads: list[dict] = field(default_factory=list)


def _reclaim_dead_workers(conn: sqlite3.Connection, board: Optional[str] = None) -> _CrashSweep:
    """Release every host-local ``running`` task whose worker PID is dead."""
    sweep = _CrashSweep()
    with _kb.write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, worker_started_at, claim_lock, started_at, assignee "
            "FROM tasks "
            "WHERE status = 'running' AND worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = _kb._host_prefix()
        for row in rows:
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Launch-window grace so a freshly-spawned worker isn't reclaimed
            # before its PID is visible on /proc.
            started_at = _kb._row_get(row, "started_at")
            if started_at is not None and time.time() - started_at < _kb._resolve_crash_grace_seconds():
                continue
            if _worker_alive(row["worker_pid"], _kb._row_get(row, "worker_started_at")):
                continue

            pid = int(row["worker_pid"])
            dead = _classify_dead_worker(pid, row["claim_lock"], task_id=row["id"], board=board)
            retry_status = _kb._retry_status_for_run(conn, row["id"])
            dead.event_payload["retry_status"] = retry_status
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue
            run_id = _kb._end_run(
                conn, row["id"],
                outcome=dead.run_outcome, status=dead.run_outcome,
                error=dead.error_text,
                metadata=dict(dead.event_payload),
            )
            _kb._append_event(conn, row["id"], dead.event_kind, dead.event_payload, run_id=run_id)
            sweep.exited_hook_payloads.append({
                "task_id": row["id"],
                "assignee": row["assignee"],
                "run_id": run_id,
                "worker_pid": pid,
                "exit_kind": dead.kind,
                "exit_code": dead.code,
                "outcome": dead.run_outcome,
                "retry_status": retry_status,
            })
            if dead.rate_limited or dead.protocol_violation:
                # Stamp last_failure_error WITHOUT touching ``consecutive_failures``:
                # a rate-limited requeue must show ``check_respawn_guard`` a quota
                # blocker; a below-budget protocol violation never reaches
                # ``_record_task_failure`` (which stamps this column), yet the
                # board UI and retry worker need the corrective message.
                conn.execute(
                    "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                    (dead.error_text[:500], row["id"]),
                )
            if dead.rate_limited:
                sweep.rate_limited.append(row["id"])
            else:
                sweep.crashed.append(row["id"])
                sweep.crash_details.append((row["id"], pid, row["claim_lock"], dead))
                # KENSEI loop-diagnostics: this run was closed two statements above, so
                # pass its run_id explicitly. Queued for after the txn — the
                # integration emits an event and opens its own write txn.
                sweep.diagnosis_requests.append(
                    (row["id"], run_id, dead.run_outcome, dead.error_text)
                )
    return sweep


def _account_crashes(conn: sqlite3.Connection, crash_details: list) -> list[str]:
    """Count each crash against the breaker; returns the task ids it tripped.

    Protocol violations get a BOUNDED violation-only budget independent of
    ``consecutive_failures`` (per-task ``max_retries`` takes precedence);
    systemic same-error crashes (>= 3 identical fingerprints this tick) and
    terminal provider errors (credential revoked, model gone — a retry cannot
    heal them) trip immediately.
    """
    auto_blocked: list[str] = []
    fp_counts: dict[str, int] = {}
    for _, _, _, dead in crash_details:
        fp = _error_fingerprint(dead.error_text)
        fp_counts[fp] = fp_counts.get(fp, 0) + 1
    for tid, pid, claimer, dead in crash_details:
        error_text = dead.error_text
        if dead.protocol_violation:
            streak = _protocol_violation_streak(conn, tid)
            trow = conn.execute("SELECT max_retries FROM tasks WHERE id = ?", (tid,)).fetchone()
            if trow is None:
                continue  # task deleted mid-loop
            task_override = _kb._row_get(trow, "max_retries")
            violation_limit = (
                int(task_override) if task_override is not None else _PROTOCOL_VIOLATION_FAILURE_LIMIT
            )
            if streak < violation_limit:
                # Below budget: already back at ``ready`` with the error stamped.
                # No ``_record_task_failure`` — must not consume the unified budget.
                continue
            # ``force_trip``: the decision (incl. per-task ``max_retries``) was
            # already made against the violation streak above.
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=violation_limit,
                force_trip=True,
                release_claim=False,
                end_run=False,
                event_payload_extra={
                    "pid": pid,
                    "claimer": claimer,
                    "protocol_violations": streak,
                    "protocol_violation_limit": violation_limit,
                },
            )
        elif dead.terminal_provider:
            # A retry cannot heal a revoked credential or a missing model, so
            # the whole ``failure_limit`` budget would be spent on identical
            # failures. ``force_trip`` blocks now, sticky: ``recompute_ready``
            # must not auto-resume it before the operator fixes the provider.
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                force_trip=True,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer, "terminal_provider": True},
            )
        else:
            is_systemic = fp_counts.get(_error_fingerprint(error_text), 0) >= 3
            extra = {"pid": pid, "claimer": claimer}
            if is_systemic:
                # Trips at 1, below any ``failure_limit``: hold it for an operator.
                extra["sticky"] = True
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if is_systemic else None,
                release_claim=False,
                end_run=False,
                event_payload_extra=extra,
            )
        if tripped:
            auto_blocked.append(tid)
    return auto_blocked


def detect_crashed_workers(conn: sqlite3.Connection, board: Optional[str] = None) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Restores the source phase immediately (no waiting for the claim TTL), for
    tasks claimed by *this host* only — other hosts' PIDs are meaningless.
    Clean exit while ``running`` is a protocol violation with a bounded
    violation-only retry budget; ``KANBAN_RATE_LIMIT_EXIT_CODE`` is a quota
    wall, released WITHOUT counting a failure and surfaced via the
    ``_last_rate_limited`` attribute (the return stays crashed-only).
    """
    sweep = _reclaim_dead_workers(conn, board=board)
    # Loop-diagnostics for reclaimed crashes. Runs here, not inside the reclaim
    # txn: the integration emits a ``diagnosis`` event and opens its own write
    # txn, and it must never mask the crash. The reaper closed each run before
    # returning, so the run_id is passed explicitly.
    for _tid, _rid, _outcome, _err in sweep.diagnosis_requests:
        _attach_loop_diagnosis(
            conn, _tid,
            run_id=_rid,
            outcome=_outcome,
            error=_err,
        )
    # Outside the main txn: account each crash and maybe trip the breaker.
    auto_blocked = _account_crashes(conn, sweep.crash_details) if sweep.crash_details else []
    # Side-channel attributes keep the public ``list[str]`` return stable;
    # ``dispatch_once`` reads them to populate ``DispatchResult``. Rate-limited
    # requeues did NOT count a failure and are NOT crashes.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    detect_crashed_workers._last_rate_limited = sweep.rate_limited  # type: ignore[attr-defined]
    # Fired only now, after the reclaim txn AND breaker accounting have
    # committed, so subscribers always observe fully durable board state.
    if sweep.exited_hook_payloads and _kb._kanban_observer_consumed("on_kanban_worker_exited"):
        _board = _kb.get_current_board()
        for hook_fields in sweep.exited_hook_payloads:
            hook_fields = dict(hook_fields)
            _kb._fire_kanban_lifecycle_hook(
                # Kanban worker-lifecycle, task-mutation, and dispatcher-tick observers (RFC #58548,
                # accepted as the design basis in the #64231 batch disposition; on_kanban_dispatch_tick is
                # the re-port of PR #56066). All five are observers only: return values are ignored, and
                # every fire site is fully best-effort, so a broken callback can never break dispatch or a
                # task mutation. Cost rule: every call site short-circuits on has_hook(), so when nothing
                # subscribes no payload is built and the hot paths (each dispatcher tick, each task write)
                # pay one dict probe. WHICH PROCESS: worker spawn/exit/stale-claim and the dispatch tick
                # fire in the DISPATCHER process (gateway-embedded dispatcher or ``hermes kanban
                # dispatch``); on_kanban_task_updated fires in whichever process committed the mutation
                # (CLI, worker, or the gateway-embedded dashboard API). Common kwargs (task-scoped hooks):
                # task_id: str, profile_name: str, board: str | None, assignee: str | None, run_id: int |
                # None. on_kanban_worker_spawned fires after ``spawn_fn`` returns AND the worker PID (when
                # one was reported) is durably persisted, per the RFC timing contract; like
                # kanban_task_claimed it runs inside the board's dispatch lock, so callbacks must stay fast.
                # Adds: worker_pid: int | None, workspace_path: str. Privacy: workspace_path is a filesystem
                # path and may reveal project layout or usernames.
                "on_kanban_worker_exited",
                hook_fields.pop("task_id"),
                board=_board,
                **hook_fields,
            )
    return sweep.crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    force_trip: bool = False,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
    infrastructure: bool = False,
) -> bool:
    """Record a non-success outcome (spawn_failed / crashed / timed_out)
    and maybe trip the circuit breaker.

    Unified replacement for the old spawn-only ``_record_spawn_failure``.
    Every path that ends a task with a non-success outcome funnels
    through here so the ``consecutive_failures`` counter and the
    auto-block threshold stay consistent.

    Returns True when the task was auto-blocked (counter reached
    ``failure_limit``), False when it was just updated in place.

    Modes:

    * ``release_claim=True, end_run=True`` — spawn-failure path.
      Caller has a running task with an open run; this transitions
      it back to its source phase (or ``blocked`` when the breaker trips),
      releases the claim, and closes the run with ``outcome=<outcome>``.

    * ``release_claim=False, end_run=False`` — timeout/crash path.
      Caller has ALREADY restored the task's source phase and closed the
      run with the appropriate outcome. This just increments the
      counter; if the breaker trips, the task is re-transitioned
      into ``blocked`` and a ``gave_up`` event is emitted.

    ``event_payload_extra`` merges into the ``gave_up`` event payload
    when the breaker trips, so callers can include outcome-specific
    context (e.g. pid on crash, elapsed on timeout).

    Resolution order for the effective threshold:
      1. per-task ``max_retries`` if set (nothing else overrides)
      2. caller-supplied ``failure_limit`` (gateway passes the config
         value from ``kanban.failure_limit``; tests pass fixed values)
      3. ``DEFAULT_FAILURE_LIMIT``

    ``force_trip=True`` trips the breaker unconditionally, skipping the
    counter-vs-threshold comparison (the resolution order above is then
    only reported in the ``gave_up`` payload, not re-evaluated). Callers
    use it when they have already applied their own bounded-retry policy
    — e.g. the clean-exit protocol-violation streak in
    ``detect_crashed_workers``, which resolves the per-task
    ``max_retries`` override against the violation streak itself. The
    failure is still counted into ``consecutive_failures`` and the
    ``gave_up`` payload is stamped sticky so ``recompute_ready`` cannot
    promote the card in the same tick.

    ``infrastructure=True``: the host refused the spawn (no restart-safe scope,
    #114720) — nothing about the card ran, so the run and event are recorded
    with ``infrastructure: true`` but ``consecutive_failures`` is left alone and
    the breaker never trips; the card stays retryable and
    :func:`check_respawn_guard` spaces the retries.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    error = error[:500]
    blocked = False
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        retry_status = (
            _kb._retry_status_for_run(conn, task_id, row["current_run_id"])
            if release_claim
            else ("review" if row["status"] == "review" else "ready")
        )
        failures = int(row["consecutive_failures"]) + (0 if infrastructure else 1)

        # Per-task override wins over both caller-supplied and default
        # thresholds. None (the common case) falls through.
        task_override = (
            row["max_retries"] if "max_retries" in row.keys() else None
        )
        if task_override is not None:
            effective_limit = int(task_override)
            limit_source = "task"
        else:
            effective_limit = int(failure_limit)
            limit_source = "dispatcher"

        if infrastructure or not (force_trip or failures >= effective_limit):
            # Below threshold — or an infrastructure refusal that must never
            # trip the breaker (nothing about the card ran; #114720).
            if release_claim:
                # Spawn path: transition running → ready + clear claim.
                # For pipeline tasks, return to the pipeline stage so the
                # next dispatcher tick re-checks the gate — the worker may
                # have produced the artifact before timing out.
                _pipeline_stage = _kb._task_pipeline_stage(conn, task_id)
                if _pipeline_stage:
                    conn.execute(
                        "UPDATE tasks SET status = ?, pipeline_stage = ?, "
                        "claim_lock = NULL, claim_expires = NULL, "
                        "worker_pid = NULL, "
                        "consecutive_failures = ?, last_failure_error = ? "
                        "WHERE id = ? AND status = 'running'",
                        (_pipeline_stage, _pipeline_stage,
                         failures, error, task_id),
                    )
                else:
                    # Restore the claimed source phase + clear claim (upstream).
                    conn.execute(
                        "UPDATE tasks SET status = ?, claim_lock = NULL, "
                        "claim_expires = NULL, worker_pid = NULL, "
                        "consecutive_failures = ?, last_failure_error = ? "
                        "WHERE id = ? AND status = 'running'",
                        (retry_status, failures, error, task_id),
                    )
            else:
                # Timeout/crash path: caller already restored the source phase.
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error, task_id),
                )
            if end_run:
                # Spawn path: close the open run with outcome.
                detail = {"failures": failures, "retry_status": retry_status}
                if infrastructure:
                    detail["infrastructure"] = True
                run_id = _kb._end_run(
                    conn, task_id, outcome=outcome, status=outcome, error=error, metadata=detail,
                )
                _kb._append_event(conn, task_id, outcome, {"error": error, **detail}, run_id=run_id)
                # Loop-diagnostics: attach the failure report for this
                # terminal attempt failure (run closed above).
                if run_id is not None:
                    _attach_loop_diagnosis(
                        conn, task_id,
                        run_id=run_id,
                        outcome=outcome,
                        error=error,
                    )
            # Timeout/crash path: the run was closed by the reaper before this
            # call, so the diagnosis is attached by the reaper itself (see
            # ``detect_crashed_workers``) — never here, where it would re-open
            # an already-closed run.
            return False

        # Trip the breaker.
        if release_claim:
            # Spawn path: still running, also clear claim state.
            conn.execute(
                "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "consecutive_failures = ?, last_failure_error = ? "
                "WHERE id = ? AND status IN ('running', 'ready', 'triage', 'review')",
                (failures, error, task_id),
            )
        else:
            # Timeout/crash path: source phase already restored with claim
            # cleared; just flip to blocked + update counter fields.
            conn.execute(
                "UPDATE tasks SET status = 'blocked', "
                "consecutive_failures = ?, last_failure_error = ? "
                "WHERE id = ? AND status IN ('ready', 'review', 'running')",
                (failures, error, task_id),
            )
        run_id = None
        if end_run:
            # Only the spawn path has an open run to close.
            run_id = _kb._end_run(
                conn, task_id,
                outcome="gave_up", status="gave_up",
                error=error,
                metadata={
                    "failures": failures,
                    "trigger_outcome": outcome,
                    "effective_limit": effective_limit,
                    "limit_source": limit_source,
                    "retry_status": retry_status,
                },
            )
        payload = {
            "failures": failures,
            "effective_limit": effective_limit,
            "limit_source": limit_source,
            "error": error,
            "trigger_outcome": outcome,
            "retry_status": retry_status,
        }
        if force_trip:
            # The caller applied its own bounded policy, so the counter cannot
            # judge this block: ``recompute_ready`` holds it for an operator.
            payload["sticky"] = True
        if event_payload_extra:
            payload.update(event_payload_extra)
        _kb._append_event(
            conn, task_id, "gave_up", payload, run_id=run_id,
        )
        # Loop-diagnostics: the run was closed above (spawn path). The
        # gave_up event carries the final outcome; attach the failure
        # report so the block reason / operator view has the root cause.
        if end_run:
            _attach_loop_diagnosis(
                conn, task_id,
                run_id=run_id,
                outcome="gave_up",
                error=error,
            )
        blocked = True
    return blocked


def _record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    failure_limit: int = None,
) -> bool:
    return _record_task_failure(
        conn, task_id, error,
        outcome="spawn_failed",
        failure_limit=failure_limit,
        release_claim=True,
        end_run=True,
    )


def _attach_loop_diagnosis(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    outcome: Optional[str] = None,
    error: Optional[str] = None,
    failed_action_id: Optional[str] = None,
    board: Optional[str] = None,
    force: bool = False,
) -> None:
    """Best-effort loop-diagnostics attachment on a terminal attempt failure.

    Delegates to ``loop_diagnostics_integration.attach_failure_diagnosis``
    with a guard so a diagnosis failure can NEVER mask the worker error or
    break the failure path. The integration module itself never raises, but
    this wrapper also catches import errors (e.g. the observability module
    being pruned) so the failure lifecycle is byte-identical to today when
    the feature is unavailable.

    When ``run_id`` is omitted it is resolved from the task's active run
    before the run is closed (the caller should pass it explicitly when the
    run has already been closed — e.g. the crash/timeout reaper paths which
    close the run before accounting).
    """
    if not task_id:
        return
    try:
        from hermes_cli.observability.loop_diagnostics_integration import (
            attach_failure_diagnosis,
        )

        if run_id is None:
            run_id = _current_run_id(conn, task_id)
        attach_failure_diagnosis(
            conn,
            task_id,
            run_id=run_id,
            outcome=outcome,
            error=error,
            failed_action_id=failed_action_id,
            board=board,
            force=force,
        )
    except Exception as exc:
        _log.debug(
            "loop-diagnostics: attach failed for %s run %s (%s)",
            task_id, run_id, exc,
        )


def _set_worker_pid(conn: sqlite3.Connection, task_id: str, pid: int) -> None:
    """Record the spawned child's pid + its restart-stable fingerprint (``_process_fingerprint``), and
    emit a ``spawned`` event carrying them. The fingerprint is what lets every later liveness/kill
    decision tell OUR worker from a process that recycled the PID after a reboot. A failed capture is
    persisted as ``UNVERIFIED_WORKER_FINGERPRINT``, never NULL: NULL is the legacy pre-fingerprint row
    whose bare-PID kill authority a new spawn must not inherit."""
    started_at = _process_fingerprint(int(pid)) or UNVERIFIED_WORKER_FINGERPRINT
    with _kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                     (int(pid), started_at, task_id))
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute("UPDATE task_runs SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                         (int(pid), started_at, run_id))
        _kb._append_event(conn, task_id, "spawned", {"pid": int(pid), "started_at": started_at}, run_id=run_id)


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on success. NOT called on spawn success: a
    spawn proves the worker could start, not that the run will succeed, so
    timeouts and crashes must accumulate across spawn boundaries.
    """
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review row before any claim attempt. Priority order:
    ``"infrastructure_cooldown"`` (latest run is a ``spawn_failed`` the host
    refused — no restart-safe scope — within the cooldown; never counted),
    ``"rate_limit_cooldown"`` (latest run ``rate_limited`` within the cooldown;
    checked BEFORE ``blocker_auth`` because the requeue stamps a quota-flavored
    ``last_failure_error`` that would otherwise park the task forever — that
    path never increments ``consecutive_failures``), ``"blocker_auth"``
    (quota/auth pattern; the breaker still trips eventually), then for the
    ready lane only ``"recent_success"`` (completed run within the window, unless
    a re-queue event arrived after it — a deliberate re-run) and ``"active_pr"``
    (PR URL in a recent comment; re-spawning risks a duplicate PR — unless a
    handoff event followed the comment: the named profile must work on that
    PR). The review lane skips the last two: they are the *inputs* to a review
    handoff. Stale / dead claim locks are NOT a guard reason — the reclaim
    passes own those.
    """
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown — see docstring for why this precedes blocker_auth.
    #    LATEST run only: a newer crash/completion supersedes the rate-limit run.
    #    An infrastructure spawn refusal (#114720) shares the cooldown: the host
    #    condition is not the card's, so it retries forever, spaced, and never
    #    reaches the breaker.
    rl_cooldown = _kb._resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if latest_run is not None and latest_run["outcome"] == "spawn_failed":
        if rl_cooldown > 0 and _kb._json_dict(latest_run["metadata"]).get("infrastructure"):
            ended_at = latest_run["ended_at"]
            if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
                return "infrastructure_cooldown"
    if latest_run is not None and latest_run["outcome"] == "rate_limited":
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, skipping blocker_auth so
            # the stamped rate-limit text doesn't re-trap the task.
            return None
        ended_at = latest_run["ended_at"]
        if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — return early so blocker_auth doesn't catch the
        # stamped rate-limit text; this path intentionally retries forever
        # (spaced by the cooldown) until quota returns or a real run supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.
    err = row["last_failure_error"]
    if err and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR URL
    # are the canonical *inputs* to a review handoff, not duplicate-work signals.
    if lane == "review":
        return None

    # 3. Completed run within guard window. Exception: an explicit re-queue
    #    AFTER that success (done→ready drag, re-promotion, unblock, reclaim) is
    #    a deliberate "run it again" — otherwise a manual done→ready would sit
    #    silently held until the window elapses.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    recent_completed = conn.execute(
        "SELECT ended_at FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id, cutoff),
    ).fetchone()
    if recent_completed:
        completed_at = int(recent_completed["ended_at"] or 0)
        requeued_after = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND created_at >= ? "
            # KENSEI CUSTOM (fork re-anchor): 'operator_repair' also counts as a
            # deliberate re-queue request.
            "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed', 'operator_repair') "
            "LIMIT 1",
            (task_id, completed_at),
        ).fetchone()
        if not requeued_after:
            return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    #    Exception: a handoff AFTER the newest PR comment (operator reassign,
    #    reviewer changes_requested, review reopen) names the profile that must
    #    now work on THAT PR — a closer or the implementer finishing it, not a
    #    duplicate implementation (#111910). A crash/reclaim is not a handoff,
    #    so the worker that opened the PR is still not re-spawned against it.
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    for c in conn.execute(
        "SELECT body, created_at FROM task_comments "
        "WHERE task_id = ? AND created_at >= ? ORDER BY created_at DESC",
        (task_id, pr_cutoff),
    ).fetchall():
        if not (c["body"] and _RESPAWN_GUARD_PR_URL_RE.search(c["body"])):
            continue
        events = conn.execute(
            # Strictly after: a same-second tie stays guarded (fail closed).
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND created_at > ? "
            "AND kind IN ('assigned', 'changes_requested', 'review_reopened')",
            (task_id, int(c["created_at"] or 0)),
        ).fetchall()
        if any(_is_handoff_event(e["kind"], e["payload"]) for e in events):
            return None
        return "active_pr"

    return None


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed."""
    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def _profile_exists_fn() -> Optional[Callable[[str], bool]]:
    """``hermes_cli.profiles.profile_exists``, or ``None`` when it cannot be
    imported (local import avoids a cycle; callers fall back to trusting the
    assignee).

    When ``kanban.dispatch_profiles`` is set (#110995) the returned predicate
    additionally requires the assignee to be listed, fail-closed — so a card
    assigned to ``default`` is only claimable by homes that opted into it.
    Foreign assignees land in the existing ``skipped_nonspawnable`` bucket.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return profile_exists

    def _gated(name: str) -> bool:
        try:
            canon = normalize_profile_name(name)
        except ValueError:
            return False
        return canon in allowlist and bool(profile_exists(name))

    return _gated


def _dispatch_profile_allowlist(normalize_profile_name) -> Optional[frozenset]:
    """Per-home claim allowlist ``kanban.dispatch_profiles`` (#110995).

    On a shared board (one ``kanban.db`` mounted across several Hermes homes),
    every home's ``profile_exists`` returns True for ``default`` — the root
    profile every home has — so a card assigned to ``default`` is claimable by
    every home's dispatcher. A home opts out of foreign claims by declaring
    which assignees it may claim::

        kanban:
          dispatch_profiles: ["sage", "researcher"]   # or "sage,researcher"

    Returns ``None`` only when the key is absent from the user config (upstream
    behavior: any existing profile is claimable). A present value is
    fail-closed: an empty list, ``null`` or a bare ``dispatch_profiles:`` claims
    nothing. The user layer is read without the ``DEFAULT_CONFIG`` merge (whose
    ``None`` placeholder would make the key look present in every home), and a
    config read that raises also claims nothing — a corrupt config on a shared
    board must never widen this home's claim scope silently (#113620).
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban", {})
    except Exception as exc:
        _kb._log.warning(
            "kanban: could not read kanban.dispatch_profiles (%s: %s) — "
            "this home claims no cards until the config is readable",
            type(exc).__name__, exc,
        )
        return frozenset()
    if not isinstance(kanban, Mapping) or "dispatch_profiles" not in kanban:
        return None
    raw = kanban["dispatch_profiles"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        _kb._log.warning(
            "kanban: kanban.dispatch_profiles is present but empty — this home "
            "claims no cards; omit the key to allow any existing profile"
        )
        return frozenset()
    names = [str(n) for n in raw] if isinstance(raw, (list, tuple)) else str(raw).split(",")
    allowed = set()
    for n in names:
        try:
            allowed.add(normalize_profile_name(n))
        except ValueError:
            continue
    return frozenset(allowed)


def dispatch_profile_allowlist_summary() -> str:
    """Human-readable resolution of ``kanban.dispatch_profiles`` for this home.

    Surfaced by ``hermes kanban diagnostics`` so an operator on a shared board
    can see what a home believes it may claim (#113620): ``any`` (key absent),
    the sorted allowed names, or ``none (fail-closed: ...)``.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name
    except Exception as exc:
        return f"none (fail-closed: profiles unavailable: {exc})"
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return "any"
    if allowlist:
        return ", ".join(sorted(allowlist))
    return ("none (fail-closed: kanban.dispatch_profiles is present but names no valid "
            "profile, or the config could not be read — omit the key to allow any)")


def _has_spawnable(conn: sqlite3.Connection, status: str) -> bool:
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = ? AND assignee IS NOT NULL AND claim_lock IS NULL",
        (status,),
    ).fetchall()
    if not rows:
        return False
    profile_exists = _profile_exists_fn()
    if profile_exists is None:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return True
    return any(profile_exists(row["assignee"]) for row in rows)


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one ready+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Used by the gateway- and CLI-embedded dispatchers' health telemetry to
    decide whether ``0 spawned`` is a "stuck" condition (real spawnable
    work waiting) or a "correctly idle" condition (only control-plane
    lanes like ``orion-cc`` / ``orion-research`` waiting on terminals
    that pull tasks via ``claim_task`` directly).

    Falls back to "any ready+assigned" if ``profile_exists`` is not
    importable (e.g. partial install) — preserves the old behavior so
    the warning still fires in degraded environments.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    for row in rows:
        if _is_profile_spawnable(row["assignee"]):
            return True
    return False


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one review+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Mirror of :func:`has_spawnable_ready` for the review column —
    used by the health telemetry to decide whether the dispatcher
    should have spawned a review agent.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'review' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    for row in rows:
        if _is_profile_spawnable(row["assignee"]):
            return True
    return False


def review_dispatch_enabled() -> bool:
    """Whether review tasks dispatch automatically. Default true (Hermes ships
    ``sdlc-review``); operators disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("kanban", {}).get("review_dispatch", True))
    except Exception:
        return True


# Memory-aware dispatch guard: an uncapped board once OOM'd a 1 GiB host. Two
# safeguards — a memory-DERIVED default cap when none is configured
# (``resolve_max_in_progress``) and a live memory-PRESSURE guard inside the
# tick (``_memory_pressure_level``) because a static cap can't see other
# tenants. Both fail open: non-Linux / read error → no cap / "unknown".

# Assumed per-worker footprint for the derived cap; deliberately conservative
# so the cap errs toward fewer workers on small VMs.
MEMORY_GUARD_MB_PER_WORKER = 512

# Derived default bounds: never below 2 (smallest VM must still progress),
# never above 8 (more fan-out must be explicit in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2
DERIVED_MAX_IN_PROGRESS_CEILING = 8


def _system_memory_sample() -> dict:
    """Best-effort system memory snapshot (KiB values), ``{}`` when unknown.

    Local import keeps ``kanban_db`` importable without the gateway package.
    Module-level indirection is also the test seam — conftest patches this to
    ``{}`` so results don't depend on the CI runner's live memory.
    """
    try:
        from gateway.lifecycle_ledger import sample_memory
        return sample_memory() or {}
    except Exception:
        return {}


def derive_default_max_in_progress(sample: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """Memory-derived default for ``kanban.max_in_progress`` when unset:
    ``clamp(MemTotal / MEMORY_GUARD_MB_PER_WORKER, FLOOR, CEILING)``. Returns
    ``None`` (no cap) when total memory is unknown, so macOS/Windows dev
    machines are unaffected.
    """
    if sample is None:
        sample = _system_memory_sample()
    total_kib = sample.get("mem_total_kib")
    if isinstance(total_kib, bool) or not isinstance(total_kib, int) or total_kib <= 0:
        return None
    workers = (total_kib // 1024) // MEMORY_GUARD_MB_PER_WORKER
    return max(DERIVED_MAX_IN_PROGRESS_FLOOR, min(workers, DERIVED_MAX_IN_PROGRESS_CEILING))


def resolve_max_in_progress(configured: Optional[int]) -> Optional[int]:
    """Effective global concurrency cap: explicit config wins, else the
    memory-derived default. All config-parsing callers route through this so
    both paths agree.
    """
    if configured is not None:
        return configured
    return derive_default_max_in_progress()


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Shared so every dispatch entry point agrees on "explicitly configured": a
    positive integer wins, anything else falls through to the derived default.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("max_in_progress")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Number of tasks in ``status='running'``.

    Used by the multi-board sweep to count OTHER boards' workers against the
    host-level budget — the memory-derived cap bounds the machine, not the
    board. Fails open to 0 so a broken board doesn't brick dispatch on healthy ones.
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    Caps bound the HOST, but each board's tick only sees its own DB; without
    this a derived cap of N gets multiplied by the number of active boards.
    Boards are matched by resolved DB path, so ``HERMES_KANBAN_DB`` (pins every
    board to one file) yields 0. Fails open per board.
    """
    try:
        current_path = str(_kb.kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or _kb.DEFAULT_BOARD
        try:
            path = _kb.kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = _kbc.connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                with contextlib.suppress(Exception):
                    other.close()
        except Exception:
            continue
    return total


def _memory_pressure_level(sample: Optional[Mapping[str, Any]] = None) -> str:
    """Classify system memory pressure: ok/elevated/critical/unknown.

    Reuses :func:`gateway.memory_status.classify_pressure` so "critical" matches
    the dashboard banner and lifecycle-ledger OOM heuristics. ``unknown``
    (non-Linux, read failure) imposes no restriction — never brick dispatch
    where /proc is unavailable.
    """
    if sample is None:
        sample = _system_memory_sample()
    if not sample:
        return "unknown"
    try:
        from gateway.memory_status import classify_pressure
        return classify_pressure(sample.get("mem_available_kib"), sample.get("mem_total_kib"))
    except Exception:
        return "unknown"


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    max_spawn_per_tick: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Thin wrapper around :func:`_dispatch_once_locked`. It acquires a
    non-blocking, board-scoped dispatch lock (issue #35240) so that two
    dispatchers pointed at the same ``kanban.db`` — e.g. the service-
    managed gateway and a shell-spawned orphan that escaped the service
    cgroup — can never run a reclaim/spawn/write tick concurrently and
    race on WAL frames. The losing dispatcher returns an empty
    ``DispatchResult`` with ``skipped_locked=True`` and does no DB writes;
    the holder is already making progress on the same board.

    The lock is keyed off the board's resolved DB path, so unrelated
    boards tick in parallel. See :func:`_dispatch_tick_lock` for the
    cross-process / cross-platform mechanics.
    """
    try:
        db_path = _kb.kanban_db_path(board=board)
    except Exception:
        # Path resolution should never fail, but if it somehow does we
        # must not lose the tick — fall through to an unguarded dispatch
        # rather than dropping work.

        result = _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            max_spawn_per_tick=max_spawn_per_tick,
            reconcile_orphans=reconcile_orphans,
        )
        _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
        return result
    with _kbc._dispatch_tick_lock(db_path) as held:
        if not held:
            result = DispatchResult(skipped_locked=True)
        else:
            result = _dispatch_once_locked(
                conn,
                spawn_fn=spawn_fn,
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
                board=board,
                default_assignee=default_assignee,
                max_in_progress_per_profile=max_in_progress_per_profile,
                max_spawn_per_tick=max_spawn_per_tick,
                reconcile_orphans=reconcile_orphans,
            )
            # Still under the dispatch lock: run the periodic PASSIVE WAL
            # checkpoint (see _maybe_checkpoint_wal; the -wal file size is
            # bounded by journal_size_limit on the writer's natural reset).
            _kbc._maybe_checkpoint_wal(conn, db_path)
    # The dispatch lock has been released here. Fire the tick observer
    # strictly OUTSIDE the single-writer critical section (#56066 sweeper
    # finding / #64231 disposition): a slow subscriber must never extend
    # the lock hold and stall a sibling dispatcher's tick.
    _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
    return result


def _call_spawn_fn(spawn_fn, task: Task, workspace: str, board: Optional[str]) -> Optional[int]:
    """Back-compat: older spawn_fn signatures (and test stubs) accept only
    ``(task, workspace)``; pass ``board`` only when the callable supports it."""
    import inspect
    try:
        sig = inspect.signature(spawn_fn)
        if "board" in sig.parameters:
            return spawn_fn(task, workspace, board=board)
        return spawn_fn(task, workspace)
    except (TypeError, ValueError):
        return spawn_fn(task, workspace)


def _dispatch_lane_task(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    assignee: str,
    result: "DispatchResult",
    *,
    lane: str,
    dry_run: bool,
    ttl_seconds: Optional[int],
    board: Optional[str],
    failure_limit: int,
    spawn_fn,
    per_profile_cap: Optional[int],
    per_profile_running: dict[str, int],
) -> bool:
    """Guard, claim, resolve the workspace and spawn one ready/review row.
    Returns True when a spawn slot was consumed (real or ``dry_run``); every
    skip is recorded on ``result``.
    """
    task_id = row["id"]
    # Non-profile assignees (control-plane lanes that pull via ``claim_task``)
    # would fail ``hermes -p <assignee>`` at startup and loop ready→crash→ready
    # forever. Bucketed apart from skipped_unassigned: the operator cannot fix
    # it by assigning a profile, and health telemetry suppresses "stuck" for it.
    profile_exists = _profile_exists_fn()
    if profile_exists is not None and not profile_exists(assignee):
        result.skipped_nonspawnable.append(task_id)
        return False
    # Per-profile cap: one profile's local model / API quota / browser pool
    # must not be overwhelmed by a fan-out even with global headroom.
    if per_profile_cap is not None:
        current = per_profile_running.get(assignee, 0)
        if current >= per_profile_cap:
            result.skipped_per_profile_capped.append((task_id, assignee, current))
            return False
    guard_reason = check_respawn_guard(conn, task_id, lane=lane)
    if guard_reason is not None:
        result.respawn_guarded.append((task_id, guard_reason))
        # Event so ``hermes kanban tail`` shows why the task looks stuck.
        # Honour kanban.default_assignee: when the dispatcher hits an unassigned ready task and an
        # operator-configured fallback exists, persist the assignment and proceed. This removes the
        # dashboard footgun where a task created without an assignee parks in 'ready' forever even though
        # the operator's intent ("default") was perfectly clear (#27145). Mutating the row (not just the
        # in-memory view) keeps diagnostics and the board state consistent: the task is now legitimately
        # owned by ``kanban.default_assignee``, not "unassigned but secretly routed".
        if not dry_run:
            with _kb.write_txn(conn):
                _kb._append_event(conn, task_id, "respawn_guarded", {"reason": guard_reason})
        return False

    def _count_spawn(name: str) -> None:
        # Later rows in this tick respect the per-profile cap; subsequent
        # ticks re-query from the DB.
        if per_profile_cap is not None and name:
            per_profile_running[name] = per_profile_running.get(name, 0) + 1

    if dry_run:
        result.spawned.append((task_id, assignee, ""))
        _count_spawn(assignee)
        return True
    claim = _kb.claim_review_task if lane == "review" else _kb.claim_task
    claimed = claim(conn, task_id, ttl_seconds=ttl_seconds)
    if claimed is None:
        return False
    try:
        resolved_branch_name = None
        if claimed.workspace_kind == "worktree":
            workspace, resolved_branch_name = _kbw._resolve_worktree_workspace(claimed, board=board)
        else:
            workspace = _kbw.resolve_workspace(claimed, board=board)
    except Exception as exc:
        if _record_task_failure(
            conn, claimed.id, f"workspace: {exc}",
            outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
        ):
            result.auto_blocked.append(claimed.id)
        return False
    _kbw.set_workspace_path(conn, claimed.id, str(workspace))
    if claimed.workspace_kind == "worktree":
        _kbw.set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
    _kbw._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
    if lane == "review":
        # Force-load sdlc-review; the kanban lifecycle is already in every
        # worker's system prompt via KANBAN_GUIDANCE.
        claimed.skills = list(dict.fromkeys([*(claimed.skills or []), "sdlc-review"]))
    try:
        pid = _call_spawn_fn(spawn_fn if spawn_fn is not None else _default_spawn, claimed, str(workspace), board)
        if pid:
            _set_worker_pid(conn, claimed.id, int(pid))
        # Fires AFTER the PID (when reported) is durably persisted. Best-effort.
        _kb._fire_worker_spawned_hook(conn, claimed, str(workspace), pid, board=board)
        # consecutive_failures is deliberately NOT reset here: resetting on
        # spawn would let a task that keeps timing out loop forever. Cleared
        # only on successful completion (complete_task).
        result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
        _count_spawn(claimed.assignee)
        return True
    except Exception as exc:
        from tools.process_registry import RestartSafeScopeUnavailable

        # The host refused the spawn (no restart-safe scope): nothing about the
        # card ran, so it must not spend the card's retry budget (#114720).
        infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
        if infrastructure:
            _kb._log.warning("kanban dispatcher: spawn of %s deferred, host cannot place the worker: %s", claimed.id, exc)
        if _record_task_failure(
            conn, claimed.id, str(exc),
            outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
            infrastructure=infrastructure,
        ):
            result.auto_blocked.append(claimed.id)
        return False


def _apply_default_assignee(
    conn: sqlite3.Connection, task_id: str, assignee: str, *, dry_run: bool,
) -> bool:
    """Persist ``kanban.default_assignee`` on an unassigned ready row.

    Mutating the row keeps board state honest: the task is legitimately owned
    by the default, not "unassigned but secretly routed". ``dry_run`` reports
    without writing. Returns False when the write failed.
    """
    if dry_run:
        return True
    try:
        with _kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (assignee, task_id),
            )
            _kb._append_event(
                conn, task_id, "assigned",
                {"assignee": assignee, "source": "kanban.default_assignee"},
            )
    except Exception:
        _kb._log.debug(
            "kanban dispatch: failed to apply default_assignee=%r to task %s",
            assignee, task_id, exc_info=True,
        )
        return False
    return True


def _run_reclaim_phase(
    conn: sqlite3.Connection,
    result: DispatchResult,
    *,
    stale_timeout_seconds: int,
    failure_limit: int,
    reconcile_orphans: bool,
    board: Optional[str] = None,
) -> None:
    """Reclaim stale/orphaned/crashed/timed-out running tasks, then promote."""
    reap_worker_zombies()
    result.reaped_terminal_workers = reap_terminal_workers(conn)
    result.reclaimed = _kb.release_stale_claims(conn, failure_limit=failure_limit)
    if reconcile_orphans:
        result.reconciled_orphans = reconcile_orphaned_running(conn)
    result.stale = detect_stale_running(conn, stale_timeout_seconds=stale_timeout_seconds)
    result.crashed = detect_crashed_workers(conn, board=board)
    # Side-channel attributes (see detect_crashed_workers); rate-limited tasks
    # went back to ``ready`` and the respawn guard defers them until quota clears.
    result.auto_blocked.extend(getattr(detect_crashed_workers, "_last_auto_blocked", []))
    result.rate_limited.extend(getattr(detect_crashed_workers, "_last_rate_limited", []))
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = _kb.recompute_ready(conn, failure_limit=failure_limit)


def _tick_spawn_budget(
    conn: sqlite3.Connection,
    result: DispatchResult,
    *,
    max_spawn: Optional[int],
    max_in_progress: Optional[int],
    board: Optional[str],
) -> tuple[bool, Optional[int]]:
    """``(may_spawn, spawn_budget)`` for this tick; ``budget None`` = uncapped.

    ``max_spawn`` is a live per-board concurrency cap (running + this tick's
    spawns), not a per-tick budget — a per-tick reading would grow concurrency
    by N every tick. ``max_in_progress`` is a HOST-level cap: running workers on
    every other board count against the same budget, else N boards multiply the
    cap by N — exactly the fan-out the memory-derived default exists to prevent.
    """
    # Count already-running tasks so max_spawn enforces concurrency, not a
    # per-tick budget: "running" tasks stay running until the worker makes a terminal
    # board call (kanban_complete/kanban_block/kanban_request_review) or the TTL reclaims them.
    running_count = 0
    spawn_budget: Optional[int] = None
    if max_spawn is not None or max_in_progress is not None:
        running_count = count_running_tasks(conn)

    # Both ready and review loops consume from the same budget.
    if max_spawn is not None:
        if running_count >= max_spawn:
            return False, None
        spawn_budget = max_spawn - running_count

    if max_in_progress is not None:
        total_running = running_count + count_running_tasks_other_boards(board)
        if total_running >= max_in_progress:
            return False, None
        remaining = max_in_progress - total_running
        if spawn_budget is None or spawn_budget > remaining:
            spawn_budget = remaining

    # Memory-pressure guard: a static cap can't see the host's actual state.
    # critical -> spawn nothing this tick; elevated -> at most one new worker.
    # Reclaim/promotion already ran, so bookkeeping stays live; deferred tasks
    # wait for a later tick. "unknown" imposes no restriction.
    pressure = _memory_pressure_level()
    if pressure == "critical":
        result.memory_pressure = pressure
        _kb._log.warning(
            "kanban dispatch: system memory pressure is critical; "
            "spawning no new workers this tick (deferred, not dropped)"
        )
        return False, None
    if pressure == "elevated":
        result.memory_pressure = pressure
        if spawn_budget is None or spawn_budget > 1:
            _kb._log.warning(
                "kanban dispatch: system memory pressure is elevated; "
                "limiting to at most 1 new worker this tick"
            )
            spawn_budget = 1
    return True, spawn_budget


def _lane_rows(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    """Unclaimed rows of one lane in dispatch order."""
    return conn.execute(
        "SELECT id, assignee FROM tasks "
        f"WHERE status = '{status}' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()


def _any_spawnable_review(
    conn: sqlite3.Connection,
    review_rows: list[sqlite3.Row],
    *,
    per_profile_cap: Optional[int] = None,
    per_profile_running: Optional[dict[str, int]] = None,
) -> bool:
    """Mirror review dispatch gates before reserving ready-lane capacity.

    Unavailable profile metadata retains the historic fail-open behavior. A
    review row that :func:`_dispatch_lane_task` would refuse this tick — its
    assignee already at the per-profile cap, or respawn-guarded — cannot
    consume the reservation, so it must not withhold capacity from an
    otherwise ready task (one such row would pin ``ready_budget`` to 0).
    """
    if not review_rows:
        return False
    profile_exists = _profile_exists_fn()
    running = per_profile_running or {}
    for row in review_rows:
        assignee = row["assignee"]
        if not assignee:
            continue
        if profile_exists is not None and not profile_exists(assignee):
            continue
        if per_profile_cap is not None and running.get(assignee, 0) >= per_profile_cap:
            continue
        if check_respawn_guard(conn, row["id"], lane="review") is None:
            return True
    return False


def _resolve_default_assignee(default_assignee: Optional[str]) -> Optional[str]:
    """``kanban.default_assignee`` when it names a real profile this home may
    claim (``kanban.dispatch_profiles`` gated, same predicate as the spawn
    gate). Otherwise ``None`` so an unassigned shared-board card is never
    written to. When the profiles module isn't importable trust the
    operator's config: the downstream check still buckets a missing profile
    as nonspawnable."""
    name = (default_assignee or "").strip() or None
    if name:
        profile_exists = _profile_exists_fn()
        if profile_exists is not None and not profile_exists(name):
            return None
    return name


# The dispatch lock has been released here. Fire the tick observer strictly OUTSIDE the single-writer
# critical section (#56066 sweeper finding / #64231 disposition): a slow subscriber must never extend the
# lock hold and stall a sibling dispatcher's tick.
def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    max_spawn_per_tick: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """One dispatcher tick: reclaim stale/crashed running tasks, promote
    todo -> ready, then atomically claim each spawnable ready/review row and
    call ``spawn_fn(task, workspace_path, board) -> Optional[int]``, recording
    the PID so later ticks catch crashes before the TTL. Cap semantics:
    :func:`_tick_spawn_budget`."""
    result = DispatchResult()
    _run_reclaim_phase(
        conn, result, stale_timeout_seconds=stale_timeout_seconds,
        failure_limit=failure_limit, reconcile_orphans=reconcile_orphans, board=board,
    )
    may_spawn, spawn_budget = _tick_spawn_budget(
        conn, result, max_spawn=max_spawn, max_in_progress=max_in_progress, board=board,
    )
    if not may_spawn:
        return result

    ready_rows = _lane_rows(conn, "ready")
    # Review rows are enumerated up front so the budget split can see whether
    # review work exists at all.
    review_rows = _lane_rows(conn, "review") if review_dispatch_enabled() else []
    # Per-profile concurrency cap (#21582): when set, track how many workers
    # each assignee already has in flight, and refuse to spawn when this
    # would push that assignee past the cap. Prevents fan-out workloads from
    # melting a single profile's local model / API quota / browser pool while
    # leaving other profiles idle. Tasks blocked this way go to
    # skipped_per_profile_capped (not skipped_unassigned — the
    # operator-actionable signal is different: "this profile is busy, try
    # again later" not "this needs routing"). Resolved BEFORE the review
    # reservation so the reservation can see which review rows the lane loop
    # would refuse this tick.
    _per_profile_cap = max_in_progress_per_profile if (
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    _per_profile_running: dict[str, int] = {}
    if _per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "GROUP BY assignee"
        ):
            _per_profile_running[prow["assignee"]] = int(prow["n"])
    # Review-lane reservation: the ready loop runs first and would otherwise
    # consume the ENTIRE shared budget, starving reviews under a sustained ready
    # backlog. When spawnable review work exists and there is any budget, hold
    # one slot back. The review rows are checked against the same gates the
    # review lane would apply this tick (per-profile cap, respawn guard), so a
    # review row that cannot spawn does not pin ready_budget to 0.
    ready_budget = spawn_budget
    if spawn_budget is not None and spawn_budget > 0 and _any_spawnable_review(
        conn, review_rows,
        per_profile_cap=_per_profile_cap, per_profile_running=_per_profile_running,
    ):
        ready_budget = max(spawn_budget - 1, 0)
    spawned = 0
    # Per-tick spawn budget (kanban.max_spawn_per_tick): caps the number of
    # NEW starts made during this single dispatch tick, across the ready,
    # review, and pipeline lanes. Distinct from max_spawn (live concurrency)
    # and max_in_progress (concurrency ceiling). ``_tick_started`` counts every
    # start committed this tick — in BOTH dry_run and real paths — so a
    # --dry-run report matches a real dispatch. Non-positive / non-int values
    # coerce to None (no per-tick cap), preserving legacy behaviour.
    _per_tick_cap = (
        max_spawn_per_tick
        if isinstance(max_spawn_per_tick, int)
        and not isinstance(max_spawn_per_tick, bool)
        and max_spawn_per_tick > 0
        else None
    )
    _tick_started = 0
    # Normalize default_assignee once: empty/whitespace string → None so the
    # rest of the loop can use ``if default_assignee:`` as a single check.
    # We also resolve profile_exists once here for the same reason.
    _default_assignee = (default_assignee or "").strip() or None
    _default_assignee_resolved = False
    if _default_assignee:
        try:
            from hermes_cli.profiles import profile_exists as _pe
            _default_assignee_resolved = bool(_pe(_default_assignee))
        except Exception:
            # Profiles module not importable (test stubs, exotic envs).
            # Trust the operator's config and try the assignment; the
            # downstream profile_exists check on the assigned row will
            # bucket it as nonspawnable if the profile genuinely isn't
            # there, with the existing diagnostic.
            _default_assignee_resolved = True
    # OOM-aware spawn backpressure: skip new spawns when system free RAM
    # drops below the configured threshold. The threshold defaults to
    # 512 MB if not set in config (kanban.min_free_ram_mb). This prevents
    # the dispatcher from compounding memory pressure when the VPS is
    # already under memory stress (#Audit-H4a).
    try:
        from hermes_cli.config import get_kanban_config
        _kanban_cfg = get_kanban_config()
    except Exception:
        _kanban_cfg = {}
    try:
        from gateway.memory_monitor import get_system_free_ram_mb
        _min_free_ram_mb = _kanban_cfg.get("min_free_ram_mb", 512)
        if _min_free_ram_mb and _min_free_ram_mb > 0:
            _free_mb = get_system_free_ram_mb()
            if _free_mb is not None and _free_mb < _min_free_ram_mb:
                _log.warning(
                    "dispatch_once: free RAM %dMB below threshold %dMB — "
                    "skipping spawns this tick",
                    _free_mb, _min_free_ram_mb,
                )
                return result
    except Exception:
        _log.debug(
            "dispatch_once: OOM backpressure probe failed — "
            "skipping check (fails open)",
            exc_info=True,
        )
    # Daily spawn budget (P2-1 cost governance): skip further spawns when
    # today's cumulative agent spawns have reached the configured ceiling.
    # The counter is global (all boards) and resets at UTC midnight. Set
    # ``kanban.daily_spawn_budget`` to 0 to disable (unlimited).
    _daily_budget = int(_kanban_cfg.get("daily_spawn_budget", 0) or 0)
    if _daily_budget > 0:
        _daily_count = _get_daily_spawn_count(conn)
        _daily_remaining = _daily_budget - _daily_count
        if _daily_remaining <= 0:
            _log.warning(
                "dispatch_once: daily spawn budget exhausted (%d/%d) — "
                "skipping all spawns this tick",
                _daily_count, _daily_budget,
            )
            result.budget_exhausted = True
            return result
    else:
        _daily_remaining = -1  # unlimited

    # Same-profile stagger: when consecutive ready tasks share an assignee,
    # add 1-5s random delay between spawns to prevent PID race conditions on
    # shared profile state (temp files, lock files, state.db). (#t_b3aa7761)
    _last_stagger_assignee: Optional[str] = None
    for row in ready_rows:
        if ready_budget is not None and spawned >= ready_budget:
            break
        if _per_tick_cap is not None and _tick_started >= _per_tick_cap:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee: when the dispatcher hits an
            # unassigned ready task and an operator-configured fallback
            # exists, persist the assignment and proceed. This removes the
            # dashboard footgun where a task created without an assignee
            # parks in 'ready' forever even though the operator's intent
            # ("default") was perfectly clear (#27145). Mutating the row
            # (not just the in-memory view) keeps diagnostics and the
            # board state consistent: the task is now legitimately owned
            # by ``kanban.default_assignee``, not "unassigned but secretly
            # routed".
            if _default_assignee and _default_assignee_resolved:
                # Dry-run: show what WOULD happen (auto-assign + spawn) without
                # mutating the DB. Real run: mutate the row + emit the
                # 'assigned' event so the board state matches what just happened.
                if not dry_run:
                    try:
                        with _kb.write_txn(conn):
                            conn.execute(
                                "UPDATE tasks SET assignee = ? WHERE id = ? "
                                "AND (assignee IS NULL OR assignee = '')",
                                (_default_assignee, row["id"]),
                            )
                            _kb._append_event(
                                conn, row["id"], "assigned",
                                {
                                    "assignee": _default_assignee,
                                    "source": "kanban.default_assignee",
                                },
                            )
                    except Exception:
                        _log.debug(
                            "kanban dispatch: failed to apply default_assignee=%r "
                            "to task %s",
                            _default_assignee, row["id"], exc_info=True,
                        )
                        result.skipped_unassigned.append(row["id"])
                        continue
                row_assignee = _default_assignee
                result.auto_assigned_default.append(row["id"])
            else:
                result.skipped_unassigned.append(row["id"])
                continue
        # Skip ready tasks whose assignee is not a real Hermes profile.
        # `_default_spawn` invokes ``hermes -p <assignee>`` which fails
        # with "Profile 'X' does not exist" when the assignee names a
        # control-plane lane (e.g. an interactive Claude Code terminal
        # like ``orion-cc`` / ``orion-research``) rather than a Hermes
        # profile. Those task lanes are pulled by terminals via
        # ``claim_task`` directly and should NEVER auto-spawn — the
        # subprocess would crash on startup, get reaped as a zombie,
        # the task would loop back to ``ready`` on next tick, and we'd
        # burn CPU forever (#kanban-dispatcher-crash-loop 2026-05-05).
        if not _is_profile_spawnable(row_assignee):
            # Bucket separately from skipped_unassigned: the operator
            # cannot fix this by assigning a profile (the assignee IS the
            # intended owner — a lead or a terminal lane). Health telemetry
            # uses this distinction to suppress spurious "stuck" warnings on
            # multi-lane setups where the ready queue is steadily full
            # of human-pulled work.
            result.skipped_nonspawnable.append(row["id"])
            continue
        # Pre-spawn gate: forced-skills visibility check.
        # A task that requires skills the assignee profile cannot see
        # will fail at startup — reject it now rather than wasting a
        # worker spawn cycle and hitting the failure breaker.
        #
        # SKILL REQUEST FALLBACK: Before blocking, attempt to grant each
        # missing skill via the skill broker (task-scoped grant). This
        # implements the design pattern where profiles have a defined skill
        # list and borrow skills on-demand from the library. The grant is
        # logged in the profile activity ledger for Denji review.
        ready_task = _kb.get_task(conn, row["id"])
        forced_skills = list(ready_task.skills or []) if ready_task else []
        missing_forced_skills = _missing_worker_forced_skills(
            row["assignee"], forced_skills,
        )
        if missing_forced_skills:
            # Attempt skill grants before blocking
            still_missing = []
            for skill_name in missing_forced_skills:
                try:
                    from tools.skill_grants import grant_skill
                    grant_result = grant_skill(
                        profile=row["assignee"],
                        skill=skill_name,
                        task_id=row["id"],
                        reason=f"Auto-grant for forced skill on task {row['id']}",
                    )
                    if grant_result.get("granted"):
                        _log.info(
                            "dispatch_once: auto-granted skill '%s' to profile '%s' for task %s",
                            skill_name, row["assignee"], row["id"],
                        )
                    else:
                        still_missing.append(skill_name)
                        _log.warning(
                            "dispatch_once: skill grant denied for '%s' on profile '%s': %s",
                            skill_name, row["assignee"], grant_result.get("reason", "unknown"),
                        )
                except Exception as exc:
                    still_missing.append(skill_name)
                    _log.warning(
                        "dispatch_once: skill grant failed for '%s' on profile '%s': %s",
                        skill_name, row["assignee"], exc,
                    )
            if still_missing:
                result.dispatcher_rejected.append(row["id"])
                if not dry_run:
                    if _block_missing_forced_skills(
                        conn, row["id"], row["assignee"], still_missing,
                        forced_skills=forced_skills,
                    ):
                        result.auto_blocked.append(row["id"])
                continue
        # Per-profile concurrency cap (#21582): even if there's global
        # headroom, refuse to spawn for an assignee that's already at
        # its in-flight cap. Prevents one profile's local model / API
        # quota / browser pool from being overwhelmed by a fan-out
        # while the global max_in_progress / max_spawn caps still allow
        # work on OTHER profiles.
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row_assignee, 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row_assignee, current)
                )
                continue
        # Respawn guard: refuse to re-spawn when useful work is already
        # in-flight/recent, or when the last failure is a deterministic
        # blocker (quota / auth). The guard defers the spawn this tick so
        # the task gets a chance to clear (rate limits often reset in
        # seconds-to-minutes); the existing consecutive_failures counter
        # still trips the auto-block circuit breaker after failure_limit
        # consecutive failures, so a persistent auth error eventually
        # blocks via the normal path rather than on first occurrence.
        guard_reason = check_respawn_guard(conn, row["id"])
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            # Emit an event so operators can see why the task was
            # skipped when reading `hermes kanban tail` — without
            # this the task appears stuck in ready with no diagnosis.
            if not dry_run:
                with _kb.write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
            continue
        if not dry_run:
            task_for_skills = _kb.get_task(conn, row["id"])
            missing_skills = _missing_worker_forced_skills(
                row["assignee"], task_for_skills.skills if task_for_skills else None
            )
            if missing_skills:
                if _block_missing_forced_skills(
                    conn, row["id"], row["assignee"], missing_skills,
                    forced_skills=task_for_skills.skills if task_for_skills else None,
                ):
                    result.auto_blocked.append(row["id"])
                continue
        if dry_run:
            result.spawned.append((row["id"], row_assignee, ""))
            _tick_started += 1
            spawned += 1
            # Increment per-profile counter even in dry_run so the cap
            # check sees the would-be spawn on subsequent iterations.
            # Without this, dry_run reports every task as spawnable and
            # under-reports the capped subset (#21582).
            if _per_profile_cap is not None and row_assignee:
                _per_profile_running[row_assignee] = (
                    _per_profile_running.get(row_assignee, 0) + 1
                )
            continue
        claimed = _kb.claim_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _kbw._resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = _kbw.resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        _kb.set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            _kb.set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _kb._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            pid = _spawn_with_board(_spawn, claimed, str(workspace), board=board)
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): fires AFTER spawn_fn
            # returned and the PID (when reported) is durably persisted,
            # per the RFC timing contract. Best-effort — can never break
            # the dispatch loop.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            # NOTE: we intentionally do NOT reset consecutive_failures
            # here. A successful spawn proves the worker can start but
            # doesn't prove the run will succeed. Under unified
            # failure counting, resetting on spawn would let a task
            # that keeps timing out after spawn loop forever. The
            # counter is cleared only on successful completion (see
            # complete_task).
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            _tick_started += 1
            if _daily_budget > 0:
                _consume_daily_spawn(conn)

            # Same-profile stagger: if this task's assignee matches the
            # previous spawn's assignee, sleep 1-5s to spread profile state
            # access across the tick. Prevents race conditions when two
            # workers share temp files, lock files, and state.db. (#t_b3aa7761)
            if _last_stagger_assignee is not None and claimed.assignee == _last_stagger_assignee:
                delay = random.uniform(1.0, 5.0)
                _log.debug(
                    "kanban dispatch: staggering %s by %.1fs (same profile %s)",
                    claimed.id, delay, claimed.assignee,
                )
                time.sleep(delay)
            _last_stagger_assignee = claimed.assignee
            # Track the new in-flight count for this profile so later
            # iterations in this same tick respect the per-profile cap
            # (#21582). Subsequent ticks re-query from the DB.
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            from tools.process_registry import RestartSafeScopeUnavailable

            # The host refused the spawn (no restart-safe scope): nothing about
            # the card ran, so it must not spend the card's retry budget (#114720).
            infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
            if infrastructure:
                _log.warning(
                    "kanban dispatcher: spawn of %s deferred, host cannot place the worker: %s",
                    claimed.id, exc,
                )
            auto = _record_task_failure(
                conn, claimed.id, str(exc),
                outcome="spawn_failed", failure_limit=failure_limit,
                release_claim=True, end_run=True,
                infrastructure=infrastructure,
            )
            if auto:
                result.auto_blocked.append(claimed.id)

    # ---- review column dispatch ----
    # Review tasks are tasks that a worker moved to 'review' after
    # creating a PR.  The dispatcher spawns a review agent (loading
    # sdlc-review skill) that verifies the candidate and either approves
    # (→ done) or requests changes (→ ready/todo for the implementer).
    #
    # Same concurrency model as ready dispatch: review spawns count
    # against max_spawn alongside ready tasks, so the total number of
    # running workers stays bounded.
    # Auto-dispatch is enabled by default because Hermes bundles the
    # ``sdlc-review`` skill and reviewer workers can now approve, request
    # changes without block-loop accounting, or escalate a genuine blocker.
    # Human-only boards can disable it with ``kanban.review_dispatch``.
    #
    # ``review_rows`` was enumerated before the ready loop; when it is
    # non-empty the ready loop ran against ``ready_budget`` (one slot held
    # back) so this lane cannot be permanently starved by a sustained
    # ready backlog. The review loop itself still checks the FULL shared
    # ``spawn_budget`` — the reservation caps the ready lane, it does not
    # grant the review lane extra capacity.
    #
    # KENSEI CUSTOM — reviewer concurrency lane: review spawns are ALSO
    # governed by a separate ``max_review_spawn`` cap (default:
    # max(1, max_spawn // 2)) so review cannot starve work and vice versa.
    # The total fleet concurrency is ``max_spawn + max_review_spawn`` — a
    # conscious trade-off for pipeline safety. See Phase 2 P2-1.
    #
    # KENSEI CUSTOM — sticky-reviewer pin: tasks that re-enter review after
    # a rejection get their assignee swapped to the rejecting reviewer's
    # profile so the same reviewer (with context) picks up the follow-up
    # pass. No-op for tasks without a prior review_rejected.
    _pin_sticky_reviewers(conn)
    _review_spawned = 0
    _max_review_spawn = max(1, (max_spawn or 4) // 2)
    for row in review_rows:
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        if _review_spawned >= _max_review_spawn:
            break
        if _per_tick_cap is not None and _tick_started >= _per_tick_cap:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        if not _is_profile_spawnable(row["assignee"]):
            # Operator-actionable starvation (unlike the ready lane): nothing
            # ever pulls a review task, so an unresolvable reviewer name (e.g.
            # the ``sdlc-review`` skill passed as ``reviewer``) would sit here
            # forever with no signal. Surface it every tick so the dispatch
            # diagnostics/log output carries the task id and the unresolvable
            # profile name; routing the fix (reassign to a real reviewer) is
            # left to the operator — auto-reassignment was explicitly rejected
            # (Sahil, 2026-09-20) because silently swapping the reviewer
            # changes who signs off on the work.
            _log.warning(
                "REVIEW TASK STARVED: review task %s has reviewer '%s' which "
                "is not a spawnable profile (not a profile dir, in "
                "kanban.nonspawnable_profiles, or tier 3). It cannot be "
                "dispatched and nothing else pulls review tasks. Reassign it "
                "to a real reviewer profile.",
                row["id"], row["assignee"],
            )
            result.skipped_review_nonspawnable.append((row["id"], row["assignee"]))
            continue
        # Per-profile concurrency cap — mirrors the ready-lane check so a
        # fan-out of review tasks for the same reviewer profile is bounded.
        row_assignee = row["assignee"]
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row_assignee, 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row_assignee, current)
                )
                continue
        # Respawn guard (lane="review"): rate-limit cooldown and auth-blocker
        # still apply; recent_success and active_pr are skipped (upstream
        # commit a235d1917e — these are the *inputs* to a review handoff).
        guard_reason = check_respawn_guard(conn, row["id"], lane="review")
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            if not dry_run:
                with _kb.write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason, "lane": "review"},
                    )
            continue
        if dry_run:
            result.spawned.append((row["id"], row["assignee"] or "", ""))
            _tick_started += 1
            spawned += 1
            _review_spawned += 1
            # Increment per-profile counter even in dry_run so the cap
            # check sees the would-be spawn on subsequent iterations.
            if _per_profile_cap is not None and row["assignee"]:
                _per_profile_running[row["assignee"]] = (
                    _per_profile_running.get(row["assignee"], 0) + 1
                )
            if _daily_budget > 0:
                _consume_daily_spawn(conn)
            continue
        claimed = _kb.claim_review_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _kbw._resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = _kbw.resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        _kb.set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            _kb.set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _kb._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        # Force-load the sdlc-review skill for review agents — it carries
        # the review logic (AC verification, merge, etc.). The mandatory
        # kanban lifecycle is already injected into every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill the
        # review agent needs.
        claimed.skills = list(
            dict.fromkeys([*(claimed.skills or []), "sdlc-review"])
        )
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            pid = _spawn_with_board(_spawn, claimed, str(workspace), board=board)
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): same contract as the
            # ready-lane fire above — after spawn + PID persistence.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            _review_spawned += 1
            _tick_started += 1
            if _daily_budget > 0:
                _consume_daily_spawn(conn)

            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            from tools.process_registry import RestartSafeScopeUnavailable

            # Host refused the spawn (no restart-safe scope): #114720 — never
            # charges the card's retry budget.
            infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
            if infrastructure:
                _log.warning(
                    "kanban dispatcher: review spawn of %s deferred, host cannot place the worker: %s",
                    claimed.id, exc,
                )
            auto = _record_task_failure(
                conn, claimed.id, str(exc),
                outcome="spawn_failed", failure_limit=failure_limit,
                release_claim=True, end_run=True,
                infrastructure=infrastructure,
            )
            if auto:
                result.auto_blocked.append(claimed.id)

    # ---- feature pipeline dispatch ----
    # Pipeline tasks (research, prd, spec, council) follow a gated progression:
    #   triage → research → prd → spec → council (LLM deliberation)
    # Each stage has a gate function that validates the artifact before
    # promotion.  When the gate fails, the task stays in its current
    # status and the assigned lead continues working on the artifact.
    # Exception: council REVISE bounces back to spec (capped at max_revise_loops).
    # When the gate passes, the task advances to the next stage.
    #
    # Pipeline tasks are NOT dispatched to workers like ready tasks.
    # Instead, the gate check runs each tick.  If the gate fails and
    # the task has an assignee, we spawn the lead to continue working.
    # If the gate passes, we promote to the next stage.
    from hermes_cli.feature_pipeline import (
        PIPELINE_STAGES,
        GATE_FUNCTIONS,
        HUMAN_GATE_STAGES,
        get_next_stage,
        get_pipeline_mode,
        check_human_approved,
        time_in_stage_hours,
    )
    # Read the full pipeline stage list dynamically so the dispatcher stays
    # in sync with feature_pipeline.PIPELINE_STAGES (design doc §3).
    _PIPELINE_STATUSES = tuple(PIPELINE_STAGES)
    # Build a placeholder list for the IN (...) clause; the dispatcher loops
    # over rows but filters in Python where the IN list is large.
    _placeholders = ",".join("?" * len(_PIPELINE_STATUSES))
    # Clear stale claim locks on pipeline tasks so they re-enter the
    # gate-check loop.  release_stale_claims only looks at status='running',
    # but pipeline tasks sit in stage statuses with claim_lock still set
    # after a worker crash — making them permanently invisible to the
    # pipeline dispatch query below (which requires claim_lock IS NULL).
    clear_stale_pipeline_claims(conn)
    pipeline_rows = conn.execute(
        f"SELECT id, assignee, skills, pipeline_stage, pipeline_mode "
        f"FROM tasks WHERE status IN ({_placeholders}) "
        f"AND claim_lock IS NULL "
        f"ORDER BY priority DESC, created_at ASC",
        _PIPELINE_STATUSES,
    ).fetchall()
    for row in pipeline_rows:
        # Gate checks, council launches, and human-gate handling are
        # near-zero-cost (file stat + regex).  Only the spawn-on-failure
        # path respects spawn pool capacity — the gate check itself fires
        # unconditionally.  This prevents pipeline tasks from stalling
        # silently when the ready-task pool is saturated.
        stage = row["pipeline_stage"]
        mode = get_pipeline_mode(dict(row))
        if stage not in GATE_FUNCTIONS and stage not in HUMAN_GATE_STAGES:
            # Pass-through stages (execute, pr+qa) — auto-advance when the
            # next stage is ready. Lead-driven; no artifact gate.
            next_stage = get_next_stage(stage, mode)
            if next_stage is None:
                # End of pipeline
                if not dry_run:
                    with _kb.write_txn(conn):
                        conn.execute(
                            "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                            (stage, "todo", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "pipeline_complete",
                            {"stage": stage, "mode": mode},
                        )
                result.pipeline_advanced.append((row["id"], stage, "todo"))
                continue
            if not dry_run:
                with _kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                        (next_stage, next_stage, row["id"]),
                    )
                    _append_event(
                        conn, row["id"], "pipeline_advanced",
                        {"from_stage": stage, "to_stage": next_stage, "mode": mode},
                    )
            result.pipeline_advanced.append((row["id"], stage, next_stage))
            continue
        if stage in HUMAN_GATE_STAGES:
            # Human gate (sign_off, final_sign_off) — check events table.
            # Gate passes when Sahil approves via CLI/Discord. No lead spawn;
            # tasks wait passively. Stale-nudge after configurable idle hours
            # (throttled to once per stale window so a stuck gate cannot spam).
            approved = check_human_approved(conn, row["id"], stage)
            if approved:
                next_stage = get_next_stage(stage, mode)
                if not dry_run:
                    with _kb.write_txn(conn):
                        if next_stage:
                            conn.execute(
                                "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                                (next_stage, next_stage, row["id"]),
                            )
                            _append_event(
                                conn, row["id"], "pipeline_advanced",
                                {"from_stage": stage, "to_stage": next_stage,
                                 "approved_by": "human", "mode": mode},
                            )
                        else:
                            conn.execute(
                                "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                                (stage, "todo", row["id"]),
                            )
                            _append_event(
                                conn, row["id"], "pipeline_complete",
                                {"stage": stage, "approved_by": "human", "mode": mode},
                            )
                result.pipeline_advanced.append((row["id"], stage, next_stage or "todo"))
            else:
                hours = time_in_stage_hours(conn, row["id"], stage)
                stale_hours = _get_sign_off_timeout_hours()
                if hours > stale_hours and not dry_run:
                    last_nudge = _hours_since_last_event(
                        conn, row["id"], "human_gate_stale_nudge", stage
                    )
                    if last_nudge is None or last_nudge >= stale_hours:
                        with _kb.write_txn(conn):
                            _append_event(
                                conn, row["id"], "human_gate_stale_nudge",
                                {"stage": stage, "hours_idle": round(hours, 1)},
                            )
                _log.debug("Human gate %s waiting for approval: %s (%.1f hrs)",
                             stage, row["id"], hours)
            continue
        if stage not in GATE_FUNCTIONS:
            # Unknown stage — skip
            continue
        gate_fn = GATE_FUNCTIONS[stage]
        # Council deliberation is expensive (multi-LLM). Run it in the
        # background, never inside the dispatcher tick: when the verdict is
        # missing the gate returns COUNCIL_PENDING and we launch/await the
        # background run here instead of blocking.
        if stage == "council":
            if _maybe_launch_council(conn, row["id"], dry_run=dry_run):
                continue
        # Determine artifact directory.
        # FIX 2026-08-13 (t_9df6f54b): the previous code used
        # ``os.environ["HERMES_HOME"]`` here. When the dispatcher runs
        # embedded in a PROFILE gateway (e.g. sirvir,
        # HERMES_HOME=~/.hermes/profiles/sirvir) it looked for artifacts
        # under the profile home, while pipeline workers write them to the
        # SHARED root (~/.hermes/feature-artifacts/). Result: the gate
        # failed on every tick ("Missing research-brief.md") even though
        # the artifact existed — infinite re-claim loop on research.
        # kanban_home() resolves the shared root across profile HERMES_HOME
        # exactly like the kanban board paths do.
        artifact_base = os.path.join(
            str(_kb.kanban_home()),
            "feature-artifacts",
        )
        artifact_dir = os.path.join(artifact_base, row["id"])
        gate_result = gate_fn(artifact_dir)
        runtime_result = _validate_pipeline_runtime_state(
            conn, row["id"], stage, artifact_dir,
        )
        if runtime_result:
            # Canonical task state outranks a self-authored artifact.  While
            # children are still running, wait passively instead of spawning
            # a parent worker every dispatcher tick.
            if runtime_result.startswith("Waiting for child tasks:"):
                gate_result = runtime_result
            elif gate_result is None:
                gate_result = runtime_result
        if gate_result is None:
            # Gate passed — promote to next stage.
            # Audit is special: PASS/CONDITIONAL passes the gate, but
            # CONDITIONAL also auto-creates a follow-up task so the issues
            # are tracked (design doc §3 [11]).
            if stage == "audit":
                try:
                    from hermes_cli.feature_pipeline import get_audit_verdict
                    audit_verdict = get_audit_verdict(artifact_dir)
                except Exception:
                    audit_verdict = None
                if audit_verdict == "CONDITIONAL" and not dry_run:
                    new_id = _create_audit_followup_task(
                        conn, row["id"], "CONDITIONAL",
                        summary="(see audit-report.md)",
                    )
                    if new_id:
                        _append_event(
                            conn, row["id"], "audit_followup_created",
                            {"followup_id": new_id},
                        )
                    _record_denji_review_signal(
                        conn, row["id"], signal_type="audit_conditional",
                        followup_id=new_id,
                    )
            elif stage == "decompose" and not dry_run:
                try:
                    _create_decompose_child_tasks(
                        conn, row["id"], artifact_dir,
                    )
                except Exception as exc:
                    with _kb.write_txn(conn):
                        _append_event(
                            conn,
                            row["id"],
                            "gate_failed",
                            {
                                "stage": stage,
                                "reason": f"Child task materialisation failed: {exc}",
                            },
                        )
                    _log.exception(
                        "Decomposition materialisation failed for %s", row["id"]
                    )
                    continue
            next_stage = get_next_stage(stage, mode)
            if not dry_run:
                with _kb.write_txn(conn):
                    if next_stage:
                        conn.execute(
                            "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                            (next_stage, next_stage, row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "pipeline_advanced",
                            {"from_stage": stage, "to_stage": next_stage, "mode": mode},
                        )
                        if stage == "audit":
                            _record_denji_review_signal(
                                conn, row["id"],
                                signal_type="audit_passed",
                                verdict=get_audit_verdict(artifact_dir) or "PASS",
                            )
                    else:
                        # End of pipeline — gate passed, task completes
                        conn.execute(
                            "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                            (stage, "todo", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "pipeline_complete",
                            {"stage": stage, "mode": mode},
                        )
            result.pipeline_advanced.append((row["id"], stage, next_stage or "todo"))
        elif stage == "council":
            # Council REVISE — bounce back to spec with loop tracking.
            # Don't spawn a lead; the spec author continues working.
            if not dry_run:
                with _kb.write_txn(conn):
                    revise_count = _get_council_revise_count(conn, row["id"], "council")
                    max_loops = _get_max_revise_loops()
                    if revise_count >= max_loops:
                        conn.execute(
                            "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                            ("blocked", "council", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "gate_failed",
                            {"stage": stage, "reason": gate_result,
                             "escalated": True,
                             "revise_count": revise_count,
                             "max_loops": max_loops},
                        )
                    else:
                        conn.execute(
                            "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                            ("spec", "spec", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "gate_failed",
                            {"stage": stage, "reason": gate_result,
                             "bounced_to": "spec",
                             "revise_count": revise_count + 1},
                        )
                        _record_council_revise(conn, row["id"], "council")
                        # Delete the stale council verdict so a fresh
                        # deliberation runs when the task re-enters council
                        # after the spec is revised.  Without this, re-entry
                        # reads the old REVISE verdict and bounces immediately
                        # without re-deliberating.
                        _clear_council_verdict(artifact_dir, row["id"])
            result.pipeline_advanced.append((row["id"], stage, "spec"))
        elif stage == "audit":
            # Audit BLOCKED — bounce to spec (capped). Same loop cap policy
            # as council REVISE — the spec author fixes the blockers, and at
            # max_revise_loops we escalate to operator.
            if not dry_run:
                with _kb.write_txn(conn):
                    revise_count = _get_council_revise_count(conn, row["id"], "audit")
                    max_loops = _get_max_revise_loops()
                    if revise_count >= max_loops:
                        conn.execute(
                            "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                            ("blocked", "audit", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "gate_failed",
                            {"stage": stage, "reason": gate_result,
                             "escalated": True,
                             "revise_count": revise_count,
                             "max_loops": max_loops},
                        )
                    else:
                        conn.execute(
                            "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                            ("spec", "spec", row["id"]),
                        )
                        _append_event(
                            conn, row["id"], "gate_failed",
                            {"stage": stage, "reason": gate_result,
                             "bounced_to": "spec",
                             "revise_count": revise_count + 1},
                        )
                        _record_council_revise(conn, row["id"], "audit")
            result.pipeline_advanced.append((row["id"], stage, "spec"))
        else:
            # Gate failed — record event then dispatch lead to continue working
            if not dry_run:
                with _kb.write_txn(conn):
                    _append_event(
                        conn, row["id"], "gate_failed",
                        {"stage": stage, "reason": gate_result},
                    )
            if gate_result.startswith("Waiting for child tasks:"):
                continue
            if not row["assignee"]:
                result.skipped_unassigned.append(row["id"])
                continue
            assignee = row["assignee"]
            _stage_owner = _get_stage_owner(stage)
            # Always reassign to the configured stage owner for this stage
            # when the current assignee doesn't match.  The old assignee from
            # a previous stage may still be spawnable (e.g. kensei-review
            # carried over from the prd stage) but is the wrong profile for
            # the current stage's artifact (e.g. spec → octacon-frontend).
            if _stage_owner and _stage_owner != assignee:
                if not dry_run:
                    conn.execute(
                        "UPDATE tasks SET assignee = ? WHERE id = ?",
                        (_stage_owner, row["id"]),
                    )
                    _append_event(
                        conn, row["id"], "assigned",
                        {"assignee": _stage_owner,
                         "reason": f"stage owner ({_stage_owner}) for stage '{stage}'"},
                    )
                assignee = _stage_owner
            if not _is_profile_spawnable(assignee):
                # Stage owner is also nonspawnable — this is a config error.
                # Surface loudly: a misconfigured stage_owners value silently
                # starves every task at this pipeline stage (no alert before).
                _log.error(
                    "PIPELINE DISPATCH BLOCKED: task %s stage '%s' owner "
                    "'%s' is not spawnable (in kanban.nonspawnable_profiles "
                    "or no profile dir). Fix pipeline.stage_owners in "
                    "config.yaml. Task will not progress until corrected.",
                    row["id"], stage, assignee,
                )
                result.skipped_nonspawnable.append(row["id"])
                continue
            # Per-tick spawn budget: skip this pipeline spawn (not the whole
            # loop — cheap gate checks/advances for other rows still run) when
            # the tick's start budget is exhausted. The gate_failed event above
            # already persisted, so the task is retried on the next tick.
            if _per_tick_cap is not None and _tick_started >= _per_tick_cap:
                continue
            if dry_run:
                result.spawned.append((row["id"], row["assignee"], ""))
                _tick_started += 1
                continue
            claimed = claim_pipeline_task(conn, row["id"], ttl_seconds=ttl_seconds)
            if claimed is None:
                continue
            try:
                workspace = _kbw.resolve_workspace(claimed, board=board)
            except Exception as exc:
                auto = _record_spawn_failure(
                    conn, claimed.id, f"workspace: {exc}",
                    failure_limit=failure_limit,
                )
                if auto:
                    result.auto_blocked.append(claimed.id)
                continue
            _kb.set_workspace_path(conn, claimed.id, str(workspace))
            _kb._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
            _spawn = spawn_fn if spawn_fn is not None else _default_spawn
            try:
                pid = _spawn_with_board(_spawn, claimed, str(workspace), board=board)
                if pid:
                    _set_worker_pid(conn, claimed.id, int(pid))
                # Record pipeline spawn for Denji's frequency tracking
                _record_pipeline_spawn(
                    conn, claimed.id, stage=stage, assignee=claimed.assignee or "",
                )
                result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
                spawned += 1
                _tick_started += 1
                if _daily_budget > 0:
                    _consume_daily_spawn(conn)

            except Exception as exc:
                from tools.process_registry import RestartSafeScopeUnavailable

                # Host refused the spawn (no restart-safe scope): #114720.
                infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
                if infrastructure:
                    _log.warning(
                        "kanban dispatcher: pipeline spawn of %s deferred, host cannot place the worker: %s",
                        claimed.id, exc,
                    )
                auto = _record_task_failure(
                    conn, claimed.id, str(exc),
                    outcome="spawn_failed", failure_limit=failure_limit,
                    release_claim=True, end_run=True,
                    infrastructure=infrastructure,
                )
                if auto:
                    result.auto_blocked.append(claimed.id)

    return result


_MAX_REVISE_LOOPS_HARD_CAP = 4


def _get_max_revise_loops() -> int:
    """Return max_revise_loops, preferring council.* then legacy pipeline.*.

    Hard-clamped to ``_MAX_REVISE_LOOPS_HARD_CAP`` (4): a configured value
    above 4 is ignored so no task can exceed the maximum council revision
    cycles.  Valid lower configured values are respected.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        loops = cfg.get("council", {}).get("max_revise_loops")
        if loops is None:
            loops = cfg.get("pipeline", {}).get("max_revise_loops", 4)
        value = int(loops) if loops is not None else 4
    except Exception:
        value = 4
    return min(value, _MAX_REVISE_LOOPS_HARD_CAP)


def _get_stage_owner(stage: str) -> str | None:
    """Return the configured stage owner for *stage* from config.yaml.

    Reads ``pipeline.stage_owners`` map (e.g. ``research: remii``,
    ``spec: octacon``).  Returns None when no owner is configured for the
    stage or the config is unreadable.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        owners = cfg.get("pipeline", {}).get("stage_owners", {})
        return owners.get(stage)
    except Exception:
        return None


def _is_profile_spawnable(name: str) -> bool:
    """Return True if *name* is eligible for kanban worker spawn.

    A profile must BOTH: (a) have a directory on disk, AND (b) NOT be
    listed in ``kanban.nonspawnable_profiles``.  This blocks lead profiles
    (remii, octacon, quan, etc.) from being spawned while allowing their
    specialist sub-profiles (remii-deep, quan-code, etc.).

    Orchestrator-only names (no profile directory) are blocked by
    ``profile_exists()`` already — this function adds the second layer
    for profiles that DO have directories but are not workers.

    Fails CLOSED: if spawnability cannot be determined (profiles import
    failure, or config load failure), the profile is treated as
    non-spawnable. A task that cannot be assigned simply waits in
    ``ready`` (recoverable), which is safer than dispatching to a
    profile whose spawnability could not be verified.
    """
    try:
        from hermes_cli.profiles import profile_exists
    except Exception as exc:
        _log.error(
            "_is_profile_spawnable(%s): could not import hermes_cli.profiles "
            "(%s); treating as non-spawnable (fail-closed)", name, exc,
        )
        return False
    if not profile_exists(name):
        return False
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        blocked = cfg.get("kanban", {}).get("nonspawnable_profiles", [])
        if name in blocked:
            return False
    except Exception as exc:
        _log.error(
            "_is_profile_spawnable(%s): could not load config (%s); "
            "treating as non-spawnable (fail-closed)", name, exc,
        )
        return False
    # Tier gate: Tier-3 (dormant/specialized) profiles are never spawnable.
    # They require explicit Sahil approval and runtime proof before activation.
    # Fails open for profile-level config reads (tier is advisory);
    # fails closed for the root nonspawnable list above.
    try:
        from hermes_cli.config import read_user_config_raw
        from hermes_cli.profiles import get_profile_dir
        _profile_dir = get_profile_dir(name)
        _config_path = _profile_dir / "config.yaml"
        if _config_path.is_file():
            _profile_cfg = read_user_config_raw(_config_path)
            _tier = _profile_cfg.get("tier")
            if _tier is not None:
                try:
                    if int(_tier) == 3:
                        _log.info(
                            "_is_profile_spawnable(%s): tier 3 profile — "
                            "not spawnable", name,
                        )
                        return False
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        _log.warning(
            "_is_profile_spawnable(%s): tier check skipped (%s)", name, exc,
        )
    return True


def _hours_since_last_event(
    conn: sqlite3.Connection, task_id: str, kind: str, stage: str,
) -> Optional[float]:
    """Hours since the most recent event of ``kind`` for ``stage``.

    Returns None if no such event exists. Used to throttle repeat nudges.
    """
    row = conn.execute(
        "SELECT created_at FROM task_events "
        "WHERE task_id = ? AND kind = ? "
        "AND json_extract(payload, '$.stage') = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (task_id, kind, stage),
    ).fetchone()
    if not row or row[0] is None:
        return None
    # created_at is an epoch-seconds integer (see _append_event).
    try:
        created = int(row[0])
    except (TypeError, ValueError):
        return None
    return max(0.0, (time.time() - created) / 3600.0)


def _get_sign_off_timeout_hours() -> int:
    """Return the sign-off stale timeout from config, default 48 hours."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return int(cfg.get("pipeline", {}).get("sign_off_timeout_hours", 48))
    except Exception:
        return 48


def _revise_event_kind(loop_kind: str) -> str:
    """Event kind used to track a given revise loop.

    Council REVISE and audit BLOCKED have independent caps (design doc §3),
    so each gets its own event kind and counter.
    """
    return "audit_revise" if loop_kind == "audit" else "council_revise"


def _get_council_revise_count(
    conn: sqlite3.Connection, task_id: str, loop_kind: str = "council"
) -> int:
    """Count revise loops of ``loop_kind`` ("council" or "audit") for a task."""
    row = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, _revise_event_kind(loop_kind)),
    ).fetchone()
    return row[0] if row else 0


def _record_council_revise(
    conn: sqlite3.Connection, task_id: str, loop_kind: str = "council"
) -> None:
    """Record a revise event for loop tracking (epoch created_at via _append_event)."""
    kind = _revise_event_kind(loop_kind)
    _append_event(conn, task_id, kind, {"stage": loop_kind})


def _council_artifact_dir(task_id: str) -> str:
    # FIX 2026-08-13 (t_9df6f54b): anchored to the SHARED kanban root via
    # kanban_home(), not the dispatcher gateway's HERMES_HOME — a profile
    # gateway (sirvir) would otherwise write/read the council verdict in
    # its own profile home and never see the pipeline's verdict.
    base = os.path.join(
        str(kanban_home()),
        "feature-artifacts",
    )
    return os.path.join(base, task_id)


def _write_fallback_council_verdict(artifact_dir: str, task_id: str, error: str) -> None:
    """Write a REVISE verdict when the council fails irrecoverably.

    Keeps a failed deliberation bounded: the gate bounces the task to spec
    (capped by max_revise_loops) instead of relaunching the council forever.
    """
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        md_path = os.path.join(artifact_dir, "council-verdict.md")
        json_path = os.path.join(artifact_dir, "council-verdict.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Council Verdict — {task_id}\n\n"
                f"**Verdict: REVISE**\n\n"
                f"## Issues\n\n"
                f"- **[CRITICAL]** Council deliberation failed: {error}\n\n"
                f"## Chairman Rationale\n\n"
                f"Deliberation could not complete; manual review required.\n"
            )
        import json as _json
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump({
                "verdict": "REVISE",
                "issues": [{"severity": "critical",
                            "description": f"Council deliberation failed: {error}"}],
                "dissents": [],
                "chairman_rationale": "Deliberation could not complete; manual review required.",
                "tokens_used": 0,
                "elapsed_seconds": 0.0,
                "critique_count": 0,
                "critique_verdicts": [],
            }, f, indent=2)
    except OSError:
        _log.exception("Could not write fallback council verdict for %s", task_id)


def _maybe_launch_council(
    conn: sqlite3.Connection, task_id: str, *, dry_run: bool = False
) -> bool:
    """Launch (or await) the council deliberation off the dispatcher thread.

    Returns True if the dispatcher should skip this task this tick (verdict
    not ready yet), False if a verdict exists and the gate should evaluate it.

    Independently enforces the council revision cap before launching: if the
    task's ``council_revise`` count has already reached the effective cap, the
    council is NOT relaunched (even after a manual state reset or stale
    ``council_running`` marker deletion).  The task is atomically blocked at
    council and a single idempotent ``council_revision_cap_reached`` event is
    appended.
    """
    artifact_dir = _council_artifact_dir(task_id)
    if os.path.exists(os.path.join(artifact_dir, "council-verdict.md")):
        return False  # verdict ready — let the gate parse it

    # Independent revision-cap guard: a task that has already exhausted its
    # council revisions must never relaunch, regardless of verdict/marker
    # state.  This closes the manual-reset / stale-marker bypass.
    revise_count = _get_council_revise_count(conn, task_id, "council")
    cap = _get_max_revise_loops()
    if revise_count >= cap:
        if not dry_run:
            with write_txn(conn):
                # Always restore the canonical blocked state, even when an
                # operator manually resets the task after the cap event was
                # first recorded.  Event emission itself remains idempotent.
                conn.execute(
                    "UPDATE tasks SET status = ?, pipeline_stage = ? "
                    "WHERE id = ?",
                    ("blocked", "council", task_id),
                )
                already = conn.execute(
                    "SELECT 1 FROM task_events WHERE task_id = ? "
                    "AND kind = 'council_revision_cap_reached' LIMIT 1",
                    (task_id,),
                ).fetchone()
                if not already:
                    _append_event(
                        conn, task_id, "council_revision_cap_reached",
                        {
                            "count": revise_count,
                            "cap": cap,
                            "reason": (
                                "council revision cap reached; refusing to "
                                "relaunch council"
                            ),
                        },
                    )
        return True

    if dry_run:
        return True

    # Resolve the deliberation timeout so a crashed run can be relaunched.
    try:
        from hermes_cli.config import get_council_config
        timeout_s = int(get_council_config().timeout_seconds)
    except Exception:
        timeout_s = 600
    relaunch_after_h = (timeout_s / 3600.0) + 0.25  # timeout + 15min buffer

    last_run = _hours_since_last_event(conn, task_id, "council_running", "council")
    if last_run is not None and last_run < relaunch_after_h:
        return True  # already deliberating

    with write_txn(conn):
        _append_event(conn, task_id, "council_running", {"stage": "council"})

    import threading

    def _worker() -> None:
        try:
            from hermes_cli.council import deliberate
            deliberate(task_id, artifact_dir)
        except Exception as exc:  # noqa: BLE001 — bound the failure to a verdict
            _log.exception("Council deliberation failed for %s", task_id)
            _write_fallback_council_verdict(artifact_dir, task_id, str(exc)[:300])

    threading.Thread(target=_worker, name=f"council-{task_id}", daemon=True).start()
    _log.info("Council deliberation launched (background) for %s", task_id)
    return True


def _clear_council_verdict(artifact_dir: str, task_id: str) -> None:
    """Delete the council verdict file so a fresh deliberation runs on re-entry."""
    verdict_path = os.path.join(artifact_dir, "council-verdict.md")
    try:
        os.remove(verdict_path)
        _log.info("Council verdict cleared for %s (bouncing to spec)", task_id)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _log.warning("Could not clear council verdict for %s: %s", task_id, exc)


def _record_pipeline_spawn(
    conn: sqlite3.Connection, task_id: str, *, stage: str, assignee: str
) -> None:
    """Record a pipeline_spawn event for Denji's spawn-frequency tracking.

    Fires each time the dispatcher claims a task in a pipeline stage and
    spawns the lead to continue working. Denji's review-cycle scripts
    group on ``assignee`` and surface recurring spawn patterns to be
    promoted to persistent profiles (D6 in the design doc).
    """
    _append_event(conn, task_id, "pipeline_spawn",
                  {"stage": stage, "assignee": assignee or ""})


def _record_denji_review_signal(
    conn: sqlite3.Connection, task_id: str, *, signal_type: str, **details
) -> None:
    """Emit a denji_review_signal event.

    This is the consumer wiring for the existing ``denji_review_signal: True``
    flag on completion events. Denji's review-cycle scripts scan for these
    events to generate audit follow-up reviews (Phase D, #15 in the design).
    """
    _append_event(conn, task_id, "denji_review_signal",
                  {"signal_type": signal_type, **details})


def _get_spawn_frequency_threshold() -> int:
    """Return the spawn-frequency threshold for Denji promotion proposals.

    When a single (assignee, stage) pair accumulates this many pipeline_spawn
    events in the rolling 7-day window, Denji surfaces a promotion proposal.
    Default 8 — about once per workday.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        n = cfg.get("pipeline", {}).get("spawn_frequency_threshold", 8)
        return int(n) if n is not None else 8
    except Exception:
        return 8


def get_spawn_frequency(
    conn: sqlite3.Connection, *, days: int = 7
) -> list[dict]:
    """Aggregate pipeline_spawn events by (assignee, stage) for Denji.

    Returns a list of dicts:
        [{"assignee": str, "stage": str, "spawn_count": int, "tasks": [str,...]}]

    Sorted by spawn_count desc. A row hitting the configured threshold
    (default 8) is the trigger for Denji to file a promotion proposal.
    """
    import json as _json
    # created_at is epoch-seconds (int); compare against an epoch cutoff.
    cutoff = int(time.time()) - int(days) * 86400
    rows = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE kind = 'pipeline_spawn' AND created_at >= ?",
        (cutoff,),
    ).fetchall()
    agg: dict[tuple[str, str], dict] = {}
    for row in rows:
        try:
            data = _json.loads(row[0])
        except (TypeError, ValueError):
            continue
        assignee = data.get("assignee", "")
        stage = data.get("stage", "")
        key = (assignee, stage)
        entry = agg.setdefault(key, {
            "assignee": assignee, "stage": stage,
            "spawn_count": 0, "tasks": set(),
        })
        entry["spawn_count"] += 1
        task_id = data.get("task_id", "")
        if task_id:
            entry["tasks"].add(task_id)
    # Materialise the sets for JSON-friendly output and sort by count desc
    out = []
    for entry in agg.values():
        out.append({
            "assignee": entry["assignee"],
            "stage": entry["stage"],
            "spawn_count": entry["spawn_count"],
            "tasks": sorted(entry["tasks"]),
        })
    out.sort(key=lambda r: (-r["spawn_count"], r["assignee"], r["stage"]))
    return out


def build_denji_report(conn: sqlite3.Connection, *, days: int = 7) -> dict:
    """Consume the pipeline governance signals into one report for Denji.

    This is the consumer side of the Denji wiring (design doc build #15):
    spawn-frequency promotion proposals, audit review signals, and express
    bypass-records over the rolling window. Denji's review-cycle cron calls
    this (via ``hermes feature denji-report``) instead of the signals sitting
    unread in the events table.
    """
    import json as _json
    cutoff = int(time.time()) - int(days) * 86400

    spawn = get_spawn_frequency(conn, days=days)
    threshold = _get_spawn_frequency_threshold()
    promotion_proposals = [
        {**r, "threshold": threshold}
        for r in spawn if r["spawn_count"] >= threshold
    ]

    def _load(kind: str) -> list[dict]:
        rows = conn.execute(
            "SELECT task_id, payload, created_at FROM task_events "
            "WHERE kind = ? AND created_at >= ? ORDER BY created_at DESC",
            (kind, cutoff),
        ).fetchall()
        items = []
        for r in rows:
            try:
                data = _json.loads(r[1]) if r[1] else {}
            except (TypeError, ValueError):
                data = {}
            items.append({"task_id": r[0], "created_at": r[2], **data})
        return items

    review_signals = _load("denji_review_signal")
    signal_counts: dict[str, int] = {}
    for s in review_signals:
        signal_counts[s.get("signal_type", "unknown")] = (
            signal_counts.get(s.get("signal_type", "unknown"), 0) + 1
        )
    bypasses = _load("bypass_record")

    return {
        "window_days": days,
        "spawn_frequency": spawn,
        "promotion_proposals": promotion_proposals,
        "review_signals": review_signals,
        "review_signal_counts": signal_counts,
        "bypass_records": bypasses,
        "bypass_count": len(bypasses),
    }


def _create_audit_followup_task(
    conn: sqlite3.Connection, parent_id: str, audit_verdict: str,
    *, summary: str = "",
) -> Optional[str]:
    """Create a follow-up task tracking audit CONDITIONAL issues.

    Only used when the audit gate returns CONDITIONAL — the gate still
    passes, but the conditional issues are tracked as a child task so
    they're not lost. Returns the new task id, or None on failure.
    """
    try:
        # Read parent for context
        parent = conn.execute(
            "SELECT id, title FROM tasks WHERE id = ?", (parent_id,)
        ).fetchone()
        if not parent:
            return None
        title = f"[audit-followup] {parent[1]}"
        body_lines = [
            "## Problem",
            f"Audit returned CONDITIONAL for parent task {parent_id}.",
            "Track and resolve the conditional issues surfaced by the audit.",
            "",
            "## Success Criteria",
            "- All CONDITIONAL issues from audit-report.md are addressed",
            "- Tests still pass after the fixes",
            "- New commit / PR linked back to the parent task",
            "",
            "## Audit Summary",
            summary or "(see audit-report.md in parent artifacts)",
            "",
            "## Verdict",
            f"**{audit_verdict}**",
        ]
        body = "\n".join(body_lines)
        # The follow-up is written on the same connection as the parent, so
        # it lands on the parent's board automatically (boards are separate
        # DB files). board=None resolves the current board's default_workdir.
        new_id = create_task(
            conn,
            title=title,
            body=body,
            assignee="octacon",
            tier="fast",
            board=None,
        )
        # Link the follow-up to the parent
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, new_id),
        )
        return new_id
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception(
            "Failed to create audit follow-up task for %s: %s", parent_id, exc
        )
        return None


def _decompose_children_event(
    conn: sqlite3.Connection, parent_id: str,
) -> list[dict[str, str]]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'decompose_children_created' ORDER BY id DESC LIMIT 1",
        (parent_id,),
    ).fetchone()
    if not row:
        return []
    try:
        payload = json.loads(row[0])
        children = payload.get("children", [])
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(children, list):
        return []
    return [child for child in children if isinstance(child, dict)]


def _create_decompose_child_tasks(
    conn: sqlite3.Connection, parent_id: str, artifact_dir: str,
) -> list[dict[str, str]]:
    """Materialise the validated decomposition manifest as one task DAG.

    The operation is idempotent and all task/link rows are written under one
    transaction.  Markdown is never parsed into executable state.
    """
    existing = _decompose_children_event(conn, parent_id)
    if existing:
        missing = [
            child.get("task_id", "") for child in existing
            if not child.get("task_id") or get_task(conn, child["task_id"]) is None
        ]
        if missing:
            raise RuntimeError(
                "decomposition event references missing task rows: "
                + ", ".join(missing)
            )
        return existing

    raw_parent = conn.execute(
        "SELECT id, title, tier, project_id FROM tasks WHERE id = ?", (parent_id,)
    ).fetchone()
    if not raw_parent:
        raise ValueError(f"unknown parent task {parent_id}")

    from hermes_cli.feature_pipeline import (
        _validate_decompose_manifest,
        load_decompose_manifest,
    )

    manifest = load_decompose_manifest(artifact_dir)
    validation_error = _validate_decompose_manifest(manifest)
    if validation_error:
        raise ValueError(validation_error)
    if manifest["parent_task_id"] != parent_id:
        raise ValueError(
            "decompose-tasks.json parent_task_id does not match the pipeline task"
        )

    parent_title = raw_parent[1]
    parent_tier = raw_parent[2]
    parent_project_id = raw_parent[3]
    pending = {task["key"]: task for task in manifest["tasks"]}
    ids_by_key: dict[str, str] = {}
    children: list[dict[str, str]] = []

    with write_txn(conn, allow_nested=True):
        while pending:
            progressed = False
            for key, task in list(pending.items()):
                dependencies = task["dependencies"]
                if any(dep not in ids_by_key for dep in dependencies):
                    continue
                body = task["body"].rstrip() + (
                    f"\n\n## Pipeline Context\nParent feature: {parent_id} "
                    f"({parent_title}).\nTask key: {key}.\n"
                    f"Shared artifacts: {artifact_dir}.\n"
                )
                owner = task["owner"].strip()
                if not _is_profile_spawnable(owner):
                    raise ValueError(
                        f"decompose task {key} owner is not spawnable: {owner}"
                    )
                task_id = create_task(
                    conn,
                    title=task["title"],
                    body=body,
                    assignee=owner,
                    created_by="feature-pipeline",
                    workspace_kind=task.get("workspace_kind", "scratch"),
                    tier=parent_tier,
                    project_id=(
                        parent_project_id
                        if task.get("workspace_kind", "scratch") == "worktree"
                        else None
                    ),
                    project_source_task_id=(
                        parent_id
                        if task.get("workspace_kind", "scratch") == "worktree"
                        else None
                    ),
                    parents=[ids_by_key[dep] for dep in dependencies],
                    idempotency_key=f"decompose:{parent_id}:{key}",
                    skills=task.get("skills"),
                )
                ids_by_key[key] = task_id
                children.append(
                    {"key": key, "task_id": task_id, "role": task["role"]}
                )
                del pending[key]
                progressed = True
            if not progressed:
                raise RuntimeError("decomposition dependency graph could not be resolved")
        _append_event(
            conn,
            parent_id,
            "decompose_children_created",
            {"schema_version": 1, "children": children},
        )
    return children


def _validate_pipeline_runtime_state(
    conn: sqlite3.Connection,
    parent_id: str,
    stage: str,
    artifact_dir: str,
) -> Optional[str]:
    """Cross-check file evidence against canonical child-task state."""
    if stage not in {"execute", "pr+qa", "audit"}:
        return None
    children = _decompose_children_event(conn, parent_id)
    if not children:
        return "Missing materialised decomposition child graph"
    role = {"execute": "implementation", "pr+qa": "qa", "audit": "audit"}[stage]
    expected = [child for child in children if child.get("role") == role]
    if not expected:
        return f"Materialised decomposition has no {role} tasks"
    rows = {
        row["id"]: row["status"]
        for row in conn.execute(
            f"SELECT id, status FROM tasks WHERE id IN ({','.join('?' for _ in expected)})",
            [child["task_id"] for child in expected],
        ).fetchall()
    }
    waiting = [
        child["key"] for child in expected
        if rows.get(child["task_id"]) != "done"
    ]
    if waiting:
        return "Waiting for child tasks: " + ", ".join(waiting)

    if stage == "execute":
        path = os.path.join(artifact_dir, "execution-evidence.json")
        try:
            with open(path, encoding="utf-8") as f:
                evidence = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None  # artifact gate reports the precise file error
        if evidence.get("parent_task_id") != parent_id:
            return "execution-evidence.json parent_task_id does not match task"
        actual = {
            (item.get("key"), item.get("task_id"))
            for item in evidence.get("children", [])
            if isinstance(item, dict)
        }
        wanted = {(item["key"], item["task_id"]) for item in expected}
        if actual != wanted:
            return "execution-evidence.json does not exactly cover implementation children"
    elif stage == "pr+qa":
        path = os.path.join(artifact_dir, "pr-qa-evidence.json")
        try:
            with open(path, encoding="utf-8") as f:
                evidence = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None  # artifact gate reports the precise file error
        if evidence.get("parent_task_id") != parent_id:
            return "pr-qa-evidence.json parent_task_id does not match task"
    return None


def _record_bypass_record(
    conn: sqlite3.Connection, task_id: str, *,
    skipped_stages: list[str], launched_by: str, mode: str,
) -> None:
    """Record an express-path bypass-record event for Denji review.

    Express launches skip PRD, Council, and Tech Review. Each launch
    writes a ``bypass_record`` event with the skipped stages, the
    launcher, and the timestamp. Denji samples these for governance
    review (design doc §4a).
    """
    _append_event(conn, task_id, "bypass_record", {
        "skipped_stages": skipped_stages,
        "launched_by": launched_by,
        "mode": mode,
    })


def _get_daily_spawn_count(conn: sqlite3.Connection) -> int:
    """Return cumulative agent spawns for today (UTC). 0 if no row yet."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT count FROM daily_spawn_counter WHERE date_utc = ?", (today,),
    ).fetchone()
    return int(row["count"]) if row else 0


def _consume_daily_spawn(conn: sqlite3.Connection) -> None:
    """Increment today's spawn counter. Creates the row if it does not exist.

    Uses INSERT … ON CONFLICT so the row is auto-created on first spawn
    of the day.  Called after a successful worker spawn so the budget is
    only consumed by real spawns, not dry-runs or skipped tasks.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    now = int(time.time())
    conn.execute(
        "INSERT INTO daily_spawn_counter (date_utc, count, last_tick) "
        "VALUES (?, 1, ?) "
        "ON CONFLICT(date_utc) DO UPDATE SET count = count + 1, last_tick = ?",
        (today, now, now),
    )
    conn.commit()


def _count_events(conn: sqlite3.Connection, task_id: str, kind: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, kind),
    ).fetchone()
    return int(row["n"]) if row else 0


def _should_review(
    conn: sqlite3.Connection,
    task_tier: str,
    task_id: str,
    *,
    kanban_cfg: Optional[dict] = None,
) -> bool:
    """Decide whether a completed task should enter review based on its tier.

    When ``kanban.tiered_review`` is enabled:
      * ``full`` tier → always review (mandatory).
      * ``fast`` tier → sampled review (1 in N + all failures).
      * Unclassified / NULL tier → no review (system automation).

    When ``tiered_review`` is disabled (or unset), the legacy WS-4
    "review everything" behaviour applies: all ``full`` and ``fast``
    tier tasks go to review.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import get_kanban_config
            kanban_cfg = get_kanban_config()
        except Exception:
            kanban_cfg = {}

    tiered_enabled = bool(kanban_cfg.get("tiered_review", False))
    tier = (task_tier or "").lower().strip()

    if tier not in ("full", "fast"):
        return False

    if tier == "full":
        return True  # mandatory review

    # Fast tier — sampled review.
    # Always review tasks whose last run failed (non-completed).
    last_outcome_row = conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    last_failed = (
        last_outcome_row is not None
        and last_outcome_row["outcome"] != "completed"
    )
    if last_failed:
        return True

    if not tiered_enabled:
        # Legacy WS-4: review everything (fast tier just gets sampled by default)
        return True

    # Sample 1 in N fast-tier tasks.
    sample_rate = max(1, int(kanban_cfg.get("review_sample_rate", 5) or 5))
    # Deterministic sampling by task_id so the same task always gets the same
    # decision across dispatcher ticks AND across gateway restarts. Built-in
    # hash() is per-process salted (PYTHONHASHSEED), so it would flip the
    # decision after a restart; sha256 is stable.
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    sample_bucket = int(digest, 16) % sample_rate
    return sample_bucket == 0


def _kanban_setting(name: str, default: int) -> int:
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("kanban", {})
    except Exception:
        return default
    value = cfg.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pin_sticky_reviewers(conn: sqlite3.Connection) -> int:
    """Reassign unclaimed review-column tasks to their preferred reviewer.

    Called from ``dispatch_once`` right before the review-row sweep.
    Bounded scan: at most one UPDATE per review task with a recorded
    rejection. Tasks without a prior rejection are no-ops.

    R-2: only pin when the preferred reviewer is spawnable; otherwise the
    task would sit in review with a non-spawnable assignee and the
    dispatcher would skip it forever.
    """
    rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'review' AND claim_lock IS NULL"
    ).fetchall()
    pinned = 0
    for row in rows:
        preferred = preferred_reviewer_profile(conn, row["id"])
        if not preferred:
            continue
        if not _is_profile_spawnable(preferred):
            # Sticky reviewer is not spawnable; don't pin to avoid strand
            continue
        if row["assignee"] == preferred:
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET assignee = ? "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL",
                (preferred, row["id"]),
            )
            if cur.rowcount == 1:
                _append_event(
                    conn, row["id"], "assigned",
                    {"profile": preferred, "via": "sticky_reviewer"},
                )
                pinned += 1
    return pinned


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    max_bytes = _positive_int(kanban_cfg.get("worker_log_rotate_bytes"), DEFAULT_LOG_ROTATE_BYTES, minimum=1)
    backup_count = _positive_int(kanban_cfg.get("worker_log_backup_count"), DEFAULT_LOG_BACKUP_COUNT, minimum=0)
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(backup_count, DEFAULT_LOG_BACKUP_COUNT, minimum=0)
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Interpreter-bound Hermes CLI invocation (``hermes_cli.main`` is the
    console-script target — there is no top-level ``hermes`` package)."""
    return [sys.executable, "-m", "hermes_cli.main"]


def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _kb._IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    return [command + ext for ext in raw.split(";") if ext]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    On Windows ``shutil.which`` may search the current directory before PATH
    for bare names — unsafe for a dispatcher. Only explicit PATH entries are
    considered; empty / ``.`` entries are skipped.
    """
    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and (_kb._IS_WINDOWS or os.access(candidate, os.X_OK)):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """argv for a resolved Hermes executable path. Windows batch shims
    (``.cmd``/``.bat``) are unsafe as argv[0] because the argument vector
    includes task-derived values; prefer the module form."""
    if _kb._IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv for ``Popen``: ``$HERMES_BIN``
    (path-like -> absolute; bare names keep PATH semantics, never a
    same-directory file), then the running interpreter's ``sys.executable -m
    hermes_cli.main`` (exactly this install; also covers shim-less cron,
    systemd ``User=``, launchd), then ``which("hermes")`` (Windows: safe PATH
    search, batch shims fall back to the module form) only when ``hermes_cli``
    is not importable. The module argv must win over PATH: a PATH-first lookup
    lets an attacker-planted ``hermes`` shadow the running install (#111569).
    Mirrors ``gateway.run._resolve_hermes_bin``; local because ``hermes_cli``
    sits below ``gateway`` in the dependency order.
    """
    import importlib.util
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    try:
        if importlib.util.find_spec("hermes_cli") is not None:
            return _module_hermes_argv()
    except Exception:
        pass

    hermes_bin = _safe_which_no_cwd("hermes") if _kb._IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def validate_forced_skills_visible(forced_skills: list[str], profile_home: str) -> list[str]:
    """Public API: return forced skill names not visible under a profile home.

    Mirrors the visibility check used by the dispatcher pre-spawn gate.
    Used by SDK consumers and tests; does not resolve profile env vars.
    """
    roots = [Path(profile_home) / "skills"]
    roots.extend(_profile_external_skill_dirs(Path(profile_home)))

    missing: list[str] = []
    seen: set[str] = set()
    for raw in forced_skills:
        skill_name = str(raw or "").strip()
        if not skill_name or skill_name in seen:
            continue
        seen.add(skill_name)
        if not _skill_visible_in_search_dirs(skill_name, roots):
            missing.append(skill_name)
    return missing


def _worker_skill_visible_in_home(skill_name: str, profile_home: Path) -> bool:
    search_dirs: list[Path] = []
    local_skills = profile_home / "skills"
    if local_skills.is_dir():
        search_dirs.append(local_skills)
    search_dirs.extend(_profile_external_skill_dirs(profile_home))
    return _skill_visible_in_search_dirs(skill_name, search_dirs)


def _worker_skill_enabled_in_home(skill_name: str, profile_home: Path) -> bool:
    """True if ``skill_name`` is in the profile's ``skills.enabled_skills``
    allowlist (or ``always_skills``, which is implicitly enabled).

    Mirrors the child CLI's allowlist gate (``tools.skills_tool`` →
    ``agent.skill_utils.get_enabled_skill_names``). When the profile has NO
    ``enabled_skills`` key configured, access is unrestricted (back-compat),
    so the skill is considered enabled. A configured-but-empty allowlist
    denies everything except ``always_skills``. Fails OPEN on config-read
    errors (the visibility check still guards the hard-fail path).
    """
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return True
    try:
        from agent.skill_utils import yaml_load
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(parsed, dict):
        return True
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return True
    if "enabled_skills" not in skills_cfg:
        return True  # no allowlist configured → unrestricted
    enabled = set()
    for key in ("enabled_skills", "always_skills"):
        raw = skills_cfg.get(key)
        if isinstance(raw, str):
            enabled.add(raw.strip())
        elif isinstance(raw, list):
            enabled.update(str(x).strip() for x in raw if str(x).strip())
    return skill_name in enabled


def _missing_worker_forced_skills(profile_name: str, skills: Optional[Iterable[Any]]) -> list[str]:
    """Return forced skills that would make the child CLI abort at startup.

    A forced skill is "missing" if it is either (a) not visible under the
    profile's skills tree (the child cannot find it at all) or (b) visible
    but NOT in the profile's ``enabled_skills`` allowlist (the child's
    allowlist gate blocks the load, which hard-fails the worker when every
    requested skill is blocked). Both conditions make the child CLI abort
    with ``ValueError: Unknown skill(s)`` — the pre-spawn gate must reject
    them so the task is blocked with a ``forced_skill_rejected`` event the
    skill-reroute cron can catch, instead of silently crash-looping.
    """
    requested: list[str] = []
    seen: set[str] = set()
    for raw in skills or []:
        name = str(raw or "").strip()
        if not name or name == "kanban-worker" or name in seen:
            continue
        seen.add(name)
        requested.append(name)
    if not requested:
        return []

    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
        profile_arg = normalize_profile_name(profile_name)
        profile_home = Path(resolve_profile_env(profile_arg))
    except Exception:
        return []

    return [
        name for name in requested
        if not _worker_skill_visible_in_home(name, profile_home)
        or not _worker_skill_enabled_in_home(name, profile_home)
    ]


def _block_missing_forced_skills(
    conn: sqlite3.Connection,
    task_id: str,
    assignee: str,
    missing: list[str],
    forced_skills: Optional[list[str]] = None,
) -> bool:
    missing_display = ", ".join(missing)
    reason = (
        f"forced skill(s) not visible to assignee profile '{assignee}': "
        f"{missing_display}. Install/copy the skill into that profile or "
        "remove it from task.skills before dispatch."
    )
    blocked = block_task(conn, task_id, reason=reason)
    if blocked:
        with write_txn(conn):
            _append_event(
                conn,
                task_id,
                "forced_skill_rejected",
                {
                    "reason": "missing_forced_skills",
                    "assignee": assignee,
                    "missing_skills": list(missing),
                    "forced_skills": list(forced_skills) if forced_skills else None,
                },
            )
    return blocked


def _kanban_worker_skill_available(hermes_home: Optional[str]) -> bool:
    """True if the bundled ``kanban-worker`` skill resolves for the home the
    spawned worker will run under.

    The dispatcher injects ``--skills kanban-worker`` into every worker. When
    the worker activates a profile (``hermes -p <name>``), its ``SKILLS_DIR``
    becomes ``<profile_home>/skills`` — which on many profiles does NOT contain
    the bundled skill (it ships in the *default* root home, not every
    profile-scoped skills dir). Preloading a missing skill is fatal at CLI
    startup (``ValueError: Unknown skill(s): kanban-worker``), aborting the
    worker before the agent loop runs.

    For profiles that have ``kanban-worker`` in their ``always_skills`` config,
    we skip the ``--skills`` flag entirely — the profile loads it naturally.
    This avoids the ``--skills`` resolution bug on sub-profile workers.
    """
    base = Path(hermes_home) if hermes_home else (Path.home() / ".hermes")
    # If profile has kanban-worker in always_skills, skip the flag
    # -- always_skills resolution works where --skills flag fails.
    config_path = base / "config.yaml"
    if config_path.exists():
        try:
            from agent.skill_utils import yaml_load
            cfg = yaml_load(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                always = cfg.get("skills", {}).get("always_skills", [])
                if isinstance(always, list) and "kanban-worker" in always:
                    return False  # profile loads it, no --skills needed
        except Exception:
            pass
    return _worker_skill_visible_in_home("kanban-worker", base)


def _profile_external_skill_dirs(profile_home: Path) -> list[Path]:
    """Return ``skills.external_dirs`` as the child profile will resolve them."""
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return []
    try:
        from agent.skill_utils import yaml_load
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return []
    raw_dirs = skills_cfg.get("external_dirs")
    if not raw_dirs:
        return []
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    local_skills = (profile_home / "skills").resolve()
    seen: set[Path] = set()
    result: list[Path] = []
    for entry in raw_dirs:
        entry_s = str(entry or "").strip()
        if not entry_s:
            continue
        expanded = os.path.expandvars(entry_s.replace("~", str(Path.home()), 1))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = profile_home / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate == local_skills or candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _skill_visible_in_search_dirs(skill_name: str, search_dirs: Iterable[Path]) -> bool:
    """Mirror the local-skill lookup strategies used by ``skill_view``."""
    name = (skill_name or "").strip()
    if not name:
        return True

    local_category_name: Optional[str] = None
    if ":" in name:
        namespace, _, bare = name.partition(":")
        if namespace and bare:
            local_category_name = f"{namespace}/{bare}"

    try:
        from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files
    except Exception:
        is_excluded_skill_path = lambda path: False  # type: ignore[assignment]
        iter_skill_index_files = None  # type: ignore[assignment]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        direct_path = search_dir / name
        if direct_path.is_dir() and (direct_path / "SKILL.md").is_file():
            return True
        if direct_path.with_suffix(".md").is_file():
            return True
        if local_category_name:
            categorized_path = search_dir / local_category_name
            if categorized_path.is_dir() and (categorized_path / "SKILL.md").is_file():
                return True
            if categorized_path.with_suffix(".md").is_file():
                return True
        try:
            skill_files = (
                iter_skill_index_files(search_dir, "SKILL.md")
                if iter_skill_index_files is not None
                else search_dir.rglob("SKILL.md")
            )
            for skill_md in skill_files:
                if is_excluded_skill_path(skill_md):
                    continue
                if skill_md.parent.name == name and skill_md.is_file():
                    return True
            for found_md in search_dir.rglob(f"{name}.md"):
                if is_excluded_skill_path(found_md):
                    continue
                if found_md.name != "SKILL.md" and found_md.is_file():
                    return True
        except OSError:
            continue
    return False


def _profile_external_skill_dirs(profile_home: Path) -> list[Path]:
    """Return ``skills.external_dirs`` as the child profile will resolve them."""
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return []
    try:
        from agent.skill_utils import yaml_load
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return []
    raw_dirs = skills_cfg.get("external_dirs")
    if not raw_dirs:
        return []
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    local_skills = (profile_home / "skills").resolve()
    seen: set[Path] = set()
    result: list[Path] = []
    for entry in raw_dirs:
        entry_s = str(entry or "").strip()
        if not entry_s:
            continue
        expanded = os.path.expandvars(entry_s.replace("~", str(Path.home()), 1))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = profile_home / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate == local_skills or candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _skill_visible_in_search_dirs(skill_name: str, search_dirs: Iterable[Path]) -> bool:
    """Mirror the local-skill lookup strategies used by ``skill_view``."""
    name = (skill_name or "").strip()
    if not name:
        return True

    local_category_name: Optional[str] = None
    if ":" in name:
        namespace, _, bare = name.partition(":")
        if namespace and bare:
            local_category_name = f"{namespace}/{bare}"

    try:
        from agent.skill_utils import is_excluded_skill_path, iter_skill_index_files
    except Exception:
        is_excluded_skill_path = lambda path: False  # type: ignore[assignment]
        iter_skill_index_files = None  # type: ignore[assignment]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        direct_path = search_dir / name
        if direct_path.is_dir() and (direct_path / "SKILL.md").is_file():
            return True
        if direct_path.with_suffix(".md").is_file():
            return True
        if local_category_name:
            categorized_path = search_dir / local_category_name
            if categorized_path.is_dir() and (categorized_path / "SKILL.md").is_file():
                return True
            if categorized_path.with_suffix(".md").is_file():
                return True
        try:
            skill_files = (
                iter_skill_index_files(search_dir, "SKILL.md")
                if iter_skill_index_files is not None
                else search_dir.rglob("SKILL.md")
            )
            for skill_md in skill_files:
                if is_excluded_skill_path(skill_md):
                    continue
                if skill_md.parent.name == name and skill_md.is_file():
                    return True
            for found_md in search_dir.rglob(f"{name}.md"):
                if is_excluded_skill_path(found_md):
                    continue
                if found_md.name != "SKILL.md" and found_md.is_file():
                    return True
        except OSError:
            continue
    return False


def _worker_skill_visible_in_home(skill_name: str, profile_home: Path) -> bool:
    search_dirs: list[Path] = []
    local_skills = profile_home / "skills"
    if local_skills.is_dir():
        search_dirs.append(local_skills)
    search_dirs.extend(_profile_external_skill_dirs(profile_home))
    return _skill_visible_in_search_dirs(skill_name, search_dirs)


def _worker_skill_enabled_in_home(skill_name: str, profile_home: Path) -> bool:
    """True if ``skill_name`` is in the profile's ``skills.enabled_skills``
    allowlist (or ``always_skills``, which is implicitly enabled).

    Mirrors the child CLI's allowlist gate (``tools.skills_tool`` →
    ``agent.skill_utils.get_enabled_skill_names``). When the profile has NO
    ``enabled_skills`` key configured, access is unrestricted (back-compat),
    so the skill is considered enabled. A configured-but-empty allowlist
    denies everything except ``always_skills``. Fails OPEN on config-read
    errors (the visibility check still guards the hard-fail path).
    """
    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        return True
    try:
        from agent.skill_utils import yaml_load
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(parsed, dict):
        return True
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return True
    if "enabled_skills" not in skills_cfg:
        return True  # no allowlist configured → unrestricted
    enabled = set()
    for key in ("enabled_skills", "always_skills"):
        raw = skills_cfg.get(key)
        if isinstance(raw, str):
            enabled.add(raw.strip())
        elif isinstance(raw, list):
            enabled.update(str(x).strip() for x in raw if str(x).strip())
    return skill_name in enabled


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    When ``max_runtime_seconds`` exceeds the terminal tool's default timeout,
    raise only the child's default so a long command isn't killed by the
    generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Resolved at dispatch time and passed as an explicit ``--toolsets`` pin so
    worker startup cannot fall back to a stale root/active-profile config or a
    profile whose top-level ``toolsets`` is only the kanban orchestrator
    surface. ``model_tools`` still appends the task-scoped kanban lifecycle
    tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope)
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        # Toolset availability probes read credentials (``get_secret``); under multiplex an
        # unscoped read raises and the pin was silently dropped for every worker.
        secret_token = (
            set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
            if is_multiplex_active() else None)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _kb._log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort: the durable gate is ``state_meta`` in
    ``retag_kanban_worker_sessions``; the in-process set avoids reopening
    state.db on every spawn. A tick must never fail because a session DB was
    busy or missing.
    """
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state_registry import acquire, release_or_close

        # Inside the gateway the dispatcher shares the process's registry handle; a bare
        # SessionDB() here was one more writer connection on the same state.db (#100896).
        db = acquire()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            release_or_close(db)
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _kb._log.debug("kanban worker: legacy session retag skipped (%s)", exc)


_spawn_board_cache: dict[int, tuple[bool, object]] = {}


def _spawn_with_board(
    spawn_fn, task, workspace: str, *, board: Optional[str] = None
) -> Optional[int]:
    """Call ``spawn_fn(task, workspace, board=board)`` if the callable
    accepts ``board``, else ``spawn_fn(task, workspace)``.

    Introspects the signature once and caches the result per callable so the
    three dispatch sites don't each pay ``inspect.signature`` on every spawn.
    """
    fn_id = id(spawn_fn)
    cached = _spawn_board_cache.get(fn_id)
    if cached is not None:
        return spawn_fn(task, workspace, board=board) if cached[0] else spawn_fn(task, workspace)
    import inspect
    try:
        accepts_board = "board" in inspect.signature(spawn_fn).parameters
    except (TypeError, ValueError):
        accepts_board = False
    _spawn_board_cache[fn_id] = (accepts_board, spawn_fn)
    if accepts_board:
        return spawn_fn(task, workspace, board=board)
    return spawn_fn(task, workspace)


def _worker_argv(task: Task, profile_arg: str, hermes_home: Optional[str]) -> list[str]:
    """Build the ``hermes -p <profile> --cli ... chat -q ...`` worker command."""
    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
        # A worker must NEVER boot the interactive TUI: its no-TTY bail-out
        # exits 0 without doing the task → "protocol violation" every attempt.
        "--cli",
        # Workers run under a profile-scoped HERMES_HOME and so see that
        # profile's shell-hook allowlist; pass --accept-hooks explicitly so
        # configured hooks still register.
        "--accept-hooks",
    ]
    # One `--skills X` pair per name: easier to read in `ps` and avoids quoting
    # ambiguity if a skill name contains unusual chars.
    for sk in task.skills or ():
        if sk:
            cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too so the worker resolves the model against the
        # intended backend (model X with provider Y is the classic board-stall).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Independent of the model override — a task can run the profile's own
    # model at a different depth.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _resolve_worker_cli_toolsets(hermes_home)
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend(["chat", "-q", f"work kanban task {task.id}"])
    # goal_mode rides the same `-q` path: cli.py runs the judge loop there too, so the
    # worker log keeps its live tool feed (forcing -Q blanked it).
    return cmd


def _open_worker_log(task: Task, board: Optional[str]):
    """Append-mode per-task log (a re-run on unblock appends, never overwrites),
    rotated first. Anchored at the board root (not the shared kanban root) so
    `hermes kanban log` reads its own file and boards sharing task ids don't
    collide."""
    log_dir = _kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)
    return open(log_path, "ab")


def _restart_safe_worker_argv(task: Task, command: list[str]) -> list[str]:
    """Wrap a systemd-hosted dispatcher's worker in the shared restart-safe scope.

    Kanban workers are long-lived agentic runs that outlive the dispatcher
    tick, so they never take cron's degraded mode under the managed gateway:
    ``require_restart_safe_scope=True`` makes the helper raise
    ``RestartSafeScopeUnavailable`` there (an infrastructure spawn failure the
    dispatcher does not charge to the card). Under any other systemd unit
    (``Type=oneshot`` dispatch timers, #113612) ``outlives_parent=True`` gets the
    worker its own scope so the unit's cgroup teardown cannot kill it.
    """
    from tools.process_registry import restart_safe_gateway_child_argv

    if task.current_run_id is None:
        # Outside managed systemd this is harmless, but a managed dispatch must
        # never mint an untraceable worker.  Check topology through the shared
        # helper first, using a placeholder suffix that cannot be launched.
        dispatch = restart_safe_gateway_child_argv(
            command,
            unit_suffix=f"kanban-{task.id}-run-missing",
            require_restart_safe_scope=True,
            outlives_parent=True,
        )
        if dispatch.mode != "in_process":
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return command

    return restart_safe_gateway_child_argv(
        command,
        unit_suffix=f"kanban-{task.id}-run-{task.current_run_id}",
        require_restart_safe_scope=True,
        outlives_parent=True,
    ).argv


def _default_spawn(task: Task, workspace: str, *, board: Optional[str] = None) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    Returns the child's PID so the dispatcher can detect crashes before the
    claim TTL expires; completion is still observed via the worker's own
    ``complete`` / ``block`` transitions. ``board`` pins the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root to the
    board the task was claimed from, so workers cannot see other boards.
    """
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    profile_arg = normalize_profile_name(task.assignee)

    # KENSEI CUSTOM (fork re-anchor): defense-in-depth forced-skills check before
    # spawning. The primary gate is the pre-spawn check in dispatch_once; this
    # catches direct _default_spawn callers (review spawns, stubs, manual paths).
    all_forced_skills = list(task.skills or []) + (
        ["kanban-worker"] if _kanban_worker_skill_available(None) else []
    )
    _missing = _missing_worker_forced_skills(profile_arg, all_forced_skills)
    if _missing:
        raise RuntimeError(
            f"Forced skill(s) not visible under profile '{profile_arg}': "
            f"{', '.join(_missing)}. The pre-spawn gate in dispatch_once "
            f"should have blocked this task before reaching _default_spawn."
        )

    from agent.secret_scope import (
        build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope)
    from tools.environments.local import build_subprocess_env, strip_launch_profile_env

    try:
        profile_home = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        profile_home = None

    multiplex_active = is_multiplex_active()
    # build_subprocess_env's secret scrub resolves terminal.env_passthrough vars
    # through get_secret(), which raises UnscopedSecretError with no profile scope
    # installed while multiplexing is on — mirrors _resolve_worker_cli_toolsets's
    # own scope-then-read ordering a few functions up in this module.
    secret_token = (
        set_secret_scope(build_profile_secret_scope(Path(profile_home)))
        if multiplex_active and profile_home else None)
    # KENSEI COMBINE: the orchestrator's own HERMES_KANBAN_TASK (a worker spawning
    # sub-workers) must not make the dispatcher child look like a delegate
    # descendant — delegated_child_subprocess_env scrubs on env-TASK presence.
    # Strip dispatcher identity from the base so the scrub never fires here; the
    # child's scope is granted explicitly below.
    import os as _os
    _base_env = {k: v for k, v in _os.environ.items() if k not in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE", "HERMES_DELEGATED_CHILD_CONTEXT")}
    try:
        env = build_subprocess_env(
            base=_base_env,
            scrub_secrets=multiplex_active,
            inherit_profile_home=True,
        )
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
    # delegated_child_subprocess_env consults the spawner's os.environ; with a
    # worker-parent that carries HERMES_KANBAN_TASK it mislabels this dispatcher
    # child as a delegate descendant and injects the marker. Undo it: the
    # dispatcher child's scope is granted explicitly below.
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    # The dispatcher is detached from every conversation; its worker must never
    # inherit routing mirrored by a previous gateway turn.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml:
    # without it the child's get_hermes_home() falls back to the DEFAULT
    # profile root because `hermes -p` applies its override before
    # hermes_constants is imported.
    if profile_home:
        env["HERMES_HOME"] = profile_home
        # A multiplexer dispatching for another profile must not hand it the launch
        # profile's .env settings / TERMINAL_* policy — a standalone dispatcher never would.
        strip_launch_profile_env(env, profile_home)
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Tag the session `kanban` so session-browsing surfaces filter it out by
    # source instead of rendering one sidebar row per attempt.
    env["HERMES_SESSION_SOURCE"] = "kanban"
    # TERMINAL_CWD takes precedence over process cwd in file_tools and
    # build_context_files_prompt; without it relative writes land in the gateway
    # user's home and workers load the gateway's AGENTS.md. file_tools rejects
    # relative / sentinel values, so only set a real absolute directory.
    # Pin TERMINAL_CWD to the task's workspace so the worker's file tools and context-file loader anchor on
    # the workspace, not whatever cwd the dispatching gateway happened to export. The worker subprocess is
    # already launched with cwd=workspace, but TERMINAL_CWD takes precedence over the process cwd in both
    # file_tools._resolve_base_dir (#41312 — relative write_file paths were landing in the gateway user's
    # home) and build_context_files_prompt (#34619 — workers loaded the dispatching gateway's AGENTS.md
    # instead of the task's). Setting it to the workspace fixes both: the workspace is where the task's work
    # actually happens.
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode (Ralph-style /goal judge loop in cli.py quiet-mode path).
    # Only set when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    for var in ("TERMINAL_TIMEOUT", "TERMINAL_MAX_FOREGROUND_TIMEOUT"):
        override = _worker_terminal_timeout_env(task.max_runtime_seconds, env.get(var))
        if override is not None:
            env[var] = override
    # Pin the board DB + workspaces root so the worker's kanban paths still
    # match after `hermes -p` rewrites HERMES_HOME (symlink / Docker layouts).
    env["HERMES_KANBAN_DB"] = str(_kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(_kb.workspaces_root(board=board))
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    # Board slug — defense-in-depth pin if a path is resolved without the
    # DB / workspaces env vars.
    env["HERMES_KANBAN_BOARD"] = _kb._normalize_board_slug(board) or _kb.get_current_board()
    # kanban_comment reads HERMES_PROFILE for its default author; `-p` alone
    # doesn't set the env var.
    env["HERMES_PROFILE"] = profile_arg
    # `--cli` is the highest-precedence TUI override; dropping HERMES_TUI covers
    # older hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    cmd = _worker_argv(task, profile_arg, env.get("HERMES_HOME"))
    # A worker spawned by a managed systemd gateway must leave the gateway's
    # cgroup before startup; otherwise restarting the service kills the worker
    # that is performing the handoff.
    cmd = _restart_safe_worker_argv(task, cmd)
    log_f = _open_worker_log(task, board)
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kb._IS_WINDOWS else 0,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # Intentionally NOT closing log_f: the child keeps writing after return;
    # the OS-level FD stays open in the child until it exits.
    if _kb._IS_WINDOWS:
        _live_worker_procs[proc.pid] = proc
    # KENSEI CUSTOM (fork re-anchor): activity-ledger record for the dispatch.
    record_event_if_enabled(
        source="kanban.dispatcher",
        actor_profile=os.environ.get("HERMES_PROFILE") or "dispatcher",
        target_profile=profile_arg,
        event_type="kanban.worker.dispatched",
        object_type="kanban_task",
        object_id=task.id,
        board=board,
        status_from=task.status,
        status_to="running",
        summary=f"Dispatched kanban task {task.id} to {profile_arg}",
        payload={
            "pid": proc.pid,
            "run_id": task.current_run_id,
            "skills": task.skills or [],
            "workspace": workspace,
        },
    )
    return proc.pid


# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds; exits cleanly on
    SIGINT / SIGTERM so it is systemd-friendly. ``stop_event`` and ``on_tick``
    are test hooks. Each tick resolves ``kanban.max_in_progress`` exactly like
    the gateway dispatcher and ``hermes kanban dispatch`` — the standalone
    daemon must not be the one uncapped entry point.
    """
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only on the main thread — tests call this inline from
    # worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, _handle)

    while not stop_event.is_set():
        try:
            # Re-resolved every tick (config load is mtime-cached) so operator
            # edits apply without a restart.
            max_in_progress = resolve_max_in_progress(configured_max_in_progress())
            with contextlib.closing(_kbc.connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                )
            if on_tick is not None:
                with contextlib.suppress(Exception):
                    on_tick(res)
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
from hermes_cli import kanban_db_connect as _kbc  # noqa: E402
from hermes_cli import kanban_db_workspace as _kbw  # noqa: E402

# KENSEI CUSTOM (fork re-anchor): best-effort activity-ledger import — the
# ledger must never break dispatch.
try:
    from hermes_cli.profile_activity_ledger import record_event_if_enabled
except Exception:  # pragma: no cover
    def record_event_if_enabled(**_kw):
        return None

# ---------------------------------------------------------------------------
# KENSEI CUSTOM (fork re-anchor): late-bound aliases so the re-anchored
# council/pipeline/forced-skill machinery can use the fork's original bare
# names (they resolve at call time via this namespace binding).
# ---------------------------------------------------------------------------
_append_event = _kb._append_event
_fire_dispatch_tick_hook = _kb._fire_dispatch_tick_hook
_fire_kanban_lifecycle_hook = _kb._fire_kanban_lifecycle_hook
_fire_worker_spawned_hook = _kb._fire_worker_spawned_hook
_kanban_observer_consumed = _kb._kanban_observer_consumed
_resolve_rate_limit_cooldown_seconds = _kb._resolve_rate_limit_cooldown_seconds
_resolve_crash_grace_seconds = _kb._resolve_crash_grace_seconds
_retry_status_for_run = _kb._retry_status_for_run
_end_run = _kb._end_run
_current_run_id = _kb._current_run_id
release_stale_claims = _kb.release_stale_claims
recompute_ready = _kb.recompute_ready
get_current_board = _kb.get_current_board
count_running_tasks_other_boards = count_running_tasks_other_boards  # local
kanban_db_path = _kb.kanban_db_path
Task = _kb.Task
kanban_home = _kb.kanban_home
preferred_reviewer_profile = _kb.preferred_reviewer_profile
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT  # fork alias name (KENSEI CUSTOM re-anchor)
_claimer_id = _kb._claimer_id
_resolve_claim_ttl_seconds = _kb._resolve_claim_ttl_seconds
block_task = _kb.block_task
create_task = _kb.create_task
get_task = _kb.get_task
write_txn = _kb.write_txn
_parents_satisfied = _kb._parents_satisfied
