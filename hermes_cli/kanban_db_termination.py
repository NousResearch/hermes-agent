"""Worker-termination machinery + verdict-authority attribution for the kanban board.

Topic module of the kanban_db family (moved out of the ``kanban_db`` /
``kanban_db_dispatch`` facades, which stand over their size ratchets — the same
sibling-offset shape as ``kanban_db_default_assignee`` / ``kanban_db_context``).
Owns: every way a run's worker process gets ended (reclaim terminate, terminal
reap sweep, transition-triggered kill), the process identities an attempt
recorded, and the D2/D3 attribution that says WHICH execution ended a review
run (incident 2026-11-16 PR#51 dual-reviewer collision; card t_58b75f7e).

Sibling names are read through their modules at CALL time (the late-bound
aliases imported LAST below), never rebound here — so existing patch seams
(``kanban_db_dispatch._terminate_reclaimed_worker``,
``kanban_db_dispatch.reap_terminal_workers``…) keep holding against the code
paths that call into this module.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sqlite3
import time
from typing import Any
from typing import Optional

# Rows whose ended_at is older than this are resolved without scanning their
# events: any worker that outlived their run that long has long since been dealt
# with by a previous tick (or by the reclaim paths), and the sweep must not
# rescan board history every tick now that evidence lives in events too.
TERMINAL_WORKER_REAP_MAX_AGE_SECONDS = 7 * 86400

# Event kinds that close a review execution: the moment a reviewer's authority
# ends, its worker process must stop being armed (kill-at-transition) and the
# audit trail must say WHO ended it (attribution), because a review-lane survivor
# keeps PR comment rights and a shell — order-of-events alone cannot tell a
# legitimate verdict-then-close from a self-override merge.
_REVIEW_TERMINAL_EVENT_KINDS = ("changes_requested", "completed")


def _signal_targets_and_verify(
    alive: list[tuple[int, Any]], targets: list[tuple[int, Any]], kill, group: int, info: dict,
) -> bool:
    """SIGTERM every live target (the group first when set), SIGKILL-escalate the
    survivors, then re-check EVERY recorded identity via /proc. Returns the
    ``terminated`` verdict; ``info`` records ``sigkill`` / ``survivors``. Never
    reports success beside a live worker — the caller retries next tick.
    """
    if group:
        with contextlib.suppress(OSError):
            # Negative pid = the whole group (workers are spawned start_new_session, so this
            # is the attempt's own isolated group — the dispatcher's group was excluded above).
            kill(-group, signal.SIGTERM)
    for t_pid, _fp in alive:
        try:
            kill(t_pid, signal.SIGTERM)
        except OSError:
            # Already gone = successful termination (Windows reports an absent pid as a bare OSError,
            # not ProcessLookupError). Leaving terminated=False would make the reclaim defer forever.
            pass

    remaining = [(t_pid, t_fp) for t_pid, t_fp in alive
                 if not _kbd._poll_worker_exit(t_pid, t_fp)]
    if remaining:
        if group:
            with contextlib.suppress(OSError):
                _kbd._sigkill(kill, -group)
        survivors: list[int] = []
        for t_pid, t_fp in remaining:
            if not _kbd._worker_alive(t_pid, t_fp):
                continue
            if _kbd._sigkill(kill, t_pid):
                info["sigkill"] = True
            if _kbd._worker_alive(t_pid, t_fp):
                survivors.append(t_pid)
        if survivors:
            info["survivors"] = survivors
            return False  # still alive: the caller retries; never report success beside a live worker
    return not any(_kbd._worker_alive(t_pid, t_fp) for t_pid, t_fp in targets)


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    started_at=None,
    extra_targets: Optional[list[tuple[int, Any]]] = None,
    kill_group: bool = False,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths. ``started_at`` is the spawn-time
    fingerprint: when the live process no longer matches it, the PID was recycled and nothing is
    signalled — the worker is gone, which is what the reclaim wanted (``terminated`` = True). An
    UNVERIFIED spawn (fingerprint capture failed) that is still live is never signalled either, but
    it is reported as surviving (``signal_refused``) so the reclaim holds the claim instead of
    spawning a duplicate beside it.

    ``extra_targets`` are (pid, fingerprint) pairs recorded for the same attempt — a dispatcher
    spawn is only half the story when the session binds elsewhere (``worker_bound``): the PR#51
    survivor merged 41 s after a reaper that SIGTERM'd only the spawn parent and recorded
    ``terminated: true``. ``terminated`` is therefore the answer for EVERY target, each re-checked
    via /proc liveness (never "the kill call didn't raise"); ``kill_group`` additionally signals a
    worker process group that is not our own (never our group — the dispatcher's children share
    it). A target equal to ``os.getpid()`` is never signalled: this runs inside gateway/CLI/pytest
    processes that must not self-terminate as a side effect of closing someone else's run."""
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

    kill = _kbd._kill_fn(signal_fn)
    if kill is None:
        return info
    if started_at == _kbd.UNVERIFIED_WORKER_FINGERPRINT:
        # Never signal by bare number: a dead PID is "gone" (reclaim proceeds), a live one is held.
        info["signal_refused"] = True
        info["terminated"] = not _kb._pid_alive(pid)
        return info
    targets: list[tuple[int, Any]] = [(int(pid), started_at)]
    targets += [(int(t_pid), t_fp) for t_pid, t_fp in (extra_targets or [])]
    targets = [(t_pid, t_fp) for t_pid, t_fp in targets
               if t_pid > 0 and t_pid != os.getpid()]
    if not targets:
        info["terminated"] = True  # only self / already-cleared targets: nothing to signal
        return info
    # Per-target identity filter: a recycled pid is evidence of a GONE worker (drop it, it must
    # not mask a live sibling target); an UNVERIFIED fingerprint is refusal, not permission —
    # the attempt stays armed until its identity can be proven.
    if any(fp == _kbd.UNVERIFIED_WORKER_FINGERPRINT for _p, fp in targets if _kb._pid_alive(_p)):
        info["signal_refused"] = True
        info["terminated"] = False
        return info
    recycled = any(_kb._pid_alive(t_pid) and _kbd._pid_recycled(t_pid, t_fp) for t_pid, t_fp in targets)
    targets = [(t_pid, t_fp) for t_pid, t_fp in targets
               if not (_kb._pid_alive(t_pid) and _kbd._pid_recycled(t_pid, t_fp))]
    if recycled:
        info["pid_recycled"] = True
    if not targets:
        info["terminated"] = True  # every recorded pid is gone or recycled: the attempt is over
        return info

    group = 0
    if kill_group and signal_fn is None and hasattr(os, "killpg"):
        with contextlib.suppress(OSError):
            group = os.getpgid(int(pid))
        if group in (0, os.getpgrp()):
            group = 0  # never signal our own process group: the dispatcher's children live in it

    alive = [(t_pid, t_fp) for t_pid, t_fp in targets if _kbd._worker_alive(t_pid, t_fp)]
    if not alive:
        info["terminated"] = True
        return info
    info["termination_attempted"] = True
    info["terminated"] = _signal_targets_and_verify(alive, targets, kill, group, info)
    return info


def _run_worker_targets(conn: sqlite3.Connection, run_id: int, row_pid, row_fp) -> list[tuple[int, Any]]:
    """Every process identity recorded for one attempt: the run row's spawn-pair plus the
    ``spawned`` / ``worker_bound`` events. The dispatcher spawn is only half the story when the
    session binds a separate interpreter (``worker_bound``) — PR#51's survivor merged 41 s after a
    reaper that signalled only the spawn parent. Event payloads carry their own spawn-time
    fingerprint; an event with no ``started_at`` is identity-less and never a kill target (missing
    process identity is refusal, not permission, #99558). Deduped by pid; the row pair wins so the
    claim/timeout fingerprint stays authoritative."""
    targets: dict[int, Any] = {}
    if row_pid:
        targets[int(row_pid)] = row_fp
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE run_id = ? AND kind IN ('spawned', 'worker_bound')",
        (run_id,),
    ).fetchall()
    for ev in rows:
        try:
            payload = _kb._json_dict(_kb._row_get(ev, "payload"))
            ev_pid = payload.get("pid")
            if not ev_pid:
                continue
            targets.setdefault(int(ev_pid), payload.get("started_at"))
        except (TypeError, ValueError):
            continue
    return list(targets.items())


def reap_terminal_workers(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """End host-local workers that outlived their run (issue #111791) — a worker
    that called ``kanban_complete`` and then hung keeps its ``state.db`` sidecar
    fds open and no ``running``-only sweep can see it once ``tasks.worker_pid`` is
    cleared. Keys on the closed ``task_runs`` row's retained pid + spawn
    fingerprint AND the ``spawned`` / ``worker_bound`` event identities beside it:
    a legacy row (NULL fingerprint) or a recycled PID is never signalled; a pid
    that is simply gone just has its evidence cleared. A run that ended less than
    ``kanban_db_dispatch.TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago is left alone so
    a worker still finalising after its own transition is not killed.
    ``terminated`` covers every target — evidence is cleared (``worker_reaped``)
    only when the whole attempt, bound session included, is verifiably gone. One
    row's failure (signal, /proc probe) is logged and skips only that row.
    Returns the task ids whose worker was terminated."""
    now = int(time.time())
    rows = conn.execute(
        "SELECT id, task_id, worker_pid, worker_started_at, claim_lock, ended_at FROM task_runs "
        "WHERE ended_at IS NOT NULL AND ended_at <= ? AND worker_reaped IS NULL "
        "AND ended_at >= ?",
        (now - _kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, now - TERMINAL_WORKER_REAP_MAX_AGE_SECONDS),
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
    if row["ended_at"] is None or not str(row["claim_lock"] or "").startswith(host_prefix):
        return
    if row["worker_started_at"] is None and row["worker_pid"] and _kb._pid_alive(row["worker_pid"]):
        return  # legacy row (no fingerprint): never signalled and its evidence stays (#99558)
    targets = _run_worker_targets(conn, int(row["id"]), row["worker_pid"], row["worker_started_at"])
    targets = [(pid, fp) for pid, fp in targets if pid != os.getpid()]
    if any(fp == _kbd.UNVERIFIED_WORKER_FINGERPRINT and _kb._pid_alive(pid) for pid, fp in targets):
        return  # unproven identity: never signalled; its evidence is cleared once the pid is gone
    alive = [(pid, fp) for pid, fp in targets if _kbd._worker_alive(pid, fp)]
    if not alive:
        # Nothing recorded for this attempt is live (or the row carried no evidence at all,
        # or its pids were recycled onto other processes): nothing to signal — resolve the
        # evidence (clearing a stale/recycled spawn pair, as the signal path does) so the
        # sweep never rescans board history. No event — nothing was signalled.
        with _kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL, "
                         "worker_reaped = ? WHERE id = ? AND worker_reaped IS NULL",
                         (int(time.time()), row["id"]))
        return
    termination = _kbd._terminate_reclaimed_worker(
        alive[0][0], row["claim_lock"], signal_fn=signal_fn, started_at=alive[0][1],
        extra_targets=alive[1:], kill_group=True,
    )
    if not termination["terminated"]:
        return  # still alive: try again next tick
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL, worker_reaped = ? "
            "WHERE id = ? AND worker_reaped IS NULL",
            (int(time.time()), row["id"]),
        )
        _kb._append_event(
            conn, row["task_id"], "terminal_worker_reaped",
            {
                "pid": termination["prev_pid"],
                "worker_started_at": row["worker_started_at"],
                "targets": [pid for pid, _fp in targets],
                **termination,
            },
            run_id=row["id"],
        )
    reaped.append(row["task_id"])


def kill_run_worker_now(conn: sqlite3.Connection, task_id: str, run_id: int) -> dict[str, Any]:
    """Terminate one run's worker processes at the moment a terminal transition commits (D2).

    The post-hoc reap (ended_at + grace + next tick) leaves a review worker fully armed — shell,
    gh CLI, comment rights — for the whole window after its authority ends; the PR#51 survivor
    used exactly that window to merge. This signals every identity recorded for the attempt
    (spawn pair + ``worker_bound``, never our own pid, never an unproven identity), SIGKILL-
    escalating, and reports honestly: ``terminated`` is true only when every target is verifiably
    gone. Best-effort: a signal/probe failure degrades to the next reap tick, never an exception
    into the transition."""
    info: dict[str, Any] = {"terminated": False, "targets": []}
    try:
        row = conn.execute(
            "SELECT id, claim_lock, worker_pid, worker_started_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return info
        info["targets"] = [
            (pid, fp) for pid, fp in _run_worker_targets(conn, int(run_id), row["worker_pid"], row["worker_started_at"])
            if pid != os.getpid() and _kbd._worker_alive(pid, fp)
        ]
        if not info["targets"]:
            info["terminated"] = True
            return info
        info.update(_kbd._terminate_reclaimed_worker(
            info["targets"][0][0], row["claim_lock"],
            started_at=info["targets"][0][1],
            extra_targets=info["targets"][1:], kill_group=True,
        ))
    except Exception:  # health: allow BLE001 -- a failed kill never breaks the transition; degrades to the reap tick; logs exc_info
        _kb._log.debug(
            "kanban dispatch: transition-triggered kill failed for run %s (task %s)",
            run_id, task_id, exc_info=True,
        )
    return info


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


def _retry_status_for_run(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int] = None,
) -> str:
    """``review`` when the run's ``claimed`` event says ``source_status=review``,
    else ``ready`` — one place, so crash/timeout/reclaim can't silently turn a
    reviewer run into an implementation run."""
    if run_id is None:
        run_id = _kb._current_run_id(conn, task_id)
    if run_id is None:
        return "ready"
    event = _kb._latest_event(conn, task_id, "claimed", run_id)
    payload = _kb._json_dict(_kb._row_get(event, "payload"))
    return "review" if payload.get("source_status") == "review" else "ready"


def _actor_attribution(conn: sqlite3.Connection, task_id: str) -> dict:
    """Identity of the session RECORDING a transition (D3): pid + live session id + kanban
    worker identity (board/task/run, which reconstructs the ``owner_admitted`` ``request_id``
    format). An interactive session is identified by pid + session id; a merge-order witness
    can then assert AUTHORITY per transition, not just order. Advisory: never blocks."""
    actor: dict[str, Any] = {"pid": os.getpid()}
    with contextlib.suppress(Exception):
        sid = os.environ.get("HERMES_SESSION_ID")
        if sid:
            actor["session_id"] = sid
    board = os.environ.get("HERMES_KANBAN_BOARD")
    run = os.environ.get("HERMES_KANBAN_RUN_ID")
    if run:
        actor["kanban_worker"] = {"task_id": task_id, "run_id": int(run)}
        if board:
            actor["kanban_worker"]["board"] = board
    return actor


def _record_ending_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    """The run this transition is about to close (snapshot BEFORE the transition's UPDATE
    wipes ``tasks.current_run_id``) — the supersede case: another session's transition ending
    a live worker's run can only be detected if the ending run is read first."""
    return _kb._current_run_id(conn, task_id)


def _after_review_transition(
    conn: sqlite3.Connection, task_id: str, *, run_id: Optional[int],
    ending_run_id: Optional[int], actor: Optional[dict], kind: str,
) -> None:
    """Post-commit tail of a review transition: collision witness + immediate
    termination of the ended run's worker.

    * ``review_claim_collision`` (Pepper's process half of the same incident, t_ea9f278a):
      one event row when this transition lands on a card whose review execution is live
      beside the actor's — the double-execution the board otherwise books under a run that
      never posted it, with no mechanical trace.
    * kill-at-transition (D2): when the transition ends a run that was claimed from the
      review column (``changes_requested`` always; ``completed`` when the card sat in
      ``review`` or superseded a review run), the ended run's worker is terminated NOW
      instead of at reap-grace + next tick. Self-transition of a non-review worker is the
      untouched healthy path (grace reap stays).
    """
    try:
        ending_run_review = run_id is not None and _kb._retry_status_for_run(conn, task_id, run_id) == "review"
        if ending_run_id is not None and ending_run_id != run_id:
            ending_run_review = ending_run_review or _kb._retry_status_for_run(conn, task_id, ending_run_id) == "review"
        ending_exec = _ending_review_execution(conn, task_id, run_id, ending_run_id, actor)
        collision = _review_claim_collision_info(conn, task_id, run_id, ending_run_id, actor, ending_exec)
        if collision is not None:
            with _kb.write_txn(conn):
                _kb._append_event(conn, task_id, "review_claim_collision", {
                    "kind": kind,
                    "run_id": run_id,
                    "ending_run_id": ending_run_id,
                    "actor": actor or {},
                    **collision,
                }, run_id=run_id)
        if not ending_run_review:
            return
        # Kill-at-transition (D2): terminate the ending execution's worker NOW — but only
        # when the transition was recorded by a session OTHER than the worker's own
        # (foreign close / supersede over a live reviewer). A worker recording its own
        # verdict keeps the untouched graceful path (grace reap owns the final kill).
        if ending_exec is None or ending_exec["self_execution"] or not ending_exec["live"]:
            return
        info = kill_run_worker_now(conn, task_id, int(ending_exec["run_id"]))
        if not info.get("terminated"):
            # Left armed until the next reap tick: record the gap loudly rather than
            # letting "handled" be assumed (the reap sweep retries on that tick).
            with _kb.write_txn(conn):
                _kb._append_event(conn, task_id, "worker_kill_deferred", {
                    "run_id": int(ending_exec["run_id"]), "kind": kind,
                    "targets": [t[0] for t in info.get("targets", [])],
                    "detail": {k: v for k, v in info.items() if k not in ("targets", "terminated")},
                }, run_id=int(ending_exec["run_id"]))
    except Exception:  # health: allow BLE001 -- the transition witness is advisory and never breaks a committed verdict; logs exc_info
        _kb._log.debug("kanban: post-transition termination witness failed for %s", task_id, exc_info=True)


def _ending_review_execution(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int],
    ending_run_id: Optional[int], actor: Optional[dict],
) -> Optional[dict]:
    """The execution this transition ended, as seen AFTER the transition committed.

    ``self_execution`` is a SESSION property, not a run-row property: in PR#51 the
    interactive session ended the live reviewer run through its own ``current_run_id``
    (run 474), which a run-equality test misreads as self-service. The run's own
    session is the one whose identities (spawn pair + ``worker_bound``) — or whose
    owner gateway — include the acting pid, or whose run id the actor carries in its
    worker env. ``owner_admitted``'s pid is the owner gateway (long-lived, hosts many
    sessions): consulted for IDENTITY ONLY, never a liveness signal nor a kill target
    — only attempt-scoped identities can be.
    """
    candidates = [x for x in (ending_run_id, run_id) if x is not None]
    for candidate in candidates:
        row = conn.execute(
            "SELECT id, profile, ended_at, worker_pid, worker_started_at FROM task_runs "
            "WHERE id = ? AND task_id = ?", (int(candidate), task_id),
        ).fetchone()
        if row is None or _kb._row_get(row, "ended_at") is None:
            continue  # never-claimed/synthesized path, or a run this transition did not end
        identities = _run_worker_targets(conn, int(candidate), None, None)
        owner_pid = None
        owner_ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'owner_admitted' ORDER BY id DESC LIMIT 1", (task_id, int(candidate)),
        ).fetchone()
        if owner_ev is not None:
            with contextlib.suppress(Exception):
                _op = _kb._json_dict(_kb._row_get(owner_ev, "payload")).get("pid")
                owner_pid = int(_op) if _op else None
        actor_pids = {actor["pid"]} if actor and actor.get("pid") else set()
        kw = (actor or {}).get("kanban_worker")
        self_execution = isinstance(kw, dict) and kw.get("run_id") == int(candidate)
        if not self_execution:
            self_execution = bool(actor_pids & ({pid for pid, _fp in identities}
                                                | ({owner_pid} if owner_pid else set())))
        live = [(pid, fp) for pid, fp in identities
                if pid != os.getpid() and _kbd._worker_alive(pid, fp) and not _kbd._pid_recycled(pid, fp)]
        return {"run_id": int(candidate), "profile": _kb._row_get(row, "profile"),
                "self_execution": self_execution, "live": live,
                "session_keys": {f"pid:{pid}" for pid, _fp in identities}}
    return None


def _review_claim_collision_info(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int],
    ending_run_id: Optional[int], actor: Optional[dict],
    ending_exec: Optional[dict],
) -> Optional[dict]:
    """Describe a live review execution beside the acting one; None when there is none.

    Two shapes (Pepper t_ea9f278a): (a) a review transition recorded while ANOTHER review
    run on the same card is still live (or its worker still lives after its row closed);
    (b) the ending run IS a live review execution this actor is not — an interactive
    session closing a dispatcher-spawned reviewer's run (exactly the PR#51 attribution
    failure, D3 shape). A worker recording its OWN verdict is never a collision."""
    actor_sessions: set = set()
    if actor:
        if actor.get("pid"):
            actor_sessions.add(f"pid:{actor['pid']}")
        if actor.get("session_id"):
            actor_sessions.add(f"session:{actor['session_id']}")
        kw = actor.get("kanban_worker")
        if isinstance(kw, dict) and kw.get("run_id") is not None:
            actor_sessions.add(f"run:{kw['run_id']}")
    live: list[dict] = []
    # Shape (b): the execution this transition just ended, if it was a live review
    # run held by a DIFFERENT session (PR#51: run 474's worker alive, verdict and
    # merge posted by another execution 41 s after its authority ended).
    if (ending_exec is not None and not ending_exec["self_execution"]
            and ending_exec["live"]
            and not (ending_exec["session_keys"] & actor_sessions)):
        live.append({"run_id": ending_exec["run_id"], "profile": ending_exec["profile"],
                     "pids": [pid for pid, _fp in ending_exec["live"]]})
    # Shape (a): any OTHER review run on the card still live beside the actor's.
    rows = conn.execute(
        "SELECT id, profile, ended_at FROM task_runs WHERE task_id = ? AND id IS NOT ? ORDER BY id DESC LIMIT 20",
        (task_id, run_id if run_id is not None else -1),
    ).fetchall()
    for r in rows:
        other = int(r["id"])
        if ending_exec is not None and other == ending_exec["run_id"]:
            continue  # already accounted for as the ending execution
        if _kb._retry_status_for_run(conn, task_id, other) != "review":
            continue
        identities = _run_worker_targets(conn, other, None, None)
        session_keys = {f"pid:{pid}" for pid, _fp in identities}
        run_live = (r["ended_at"] is None and any(_kbd._worker_alive(pid, fp) for pid, fp in identities)) or \
            (r["ended_at"] is not None and any(_kbd._worker_alive(pid, fp) and not _kbd._pid_recycled(pid, fp)
                                              for pid, fp in identities))
        if not run_live:
            continue
        if session_keys and actor_sessions and session_keys & actor_sessions:
            continue  # same execution under another run row: not a second executor
        live.append({"run_id": other, "profile": r["profile"],
                     "pids": [pid for pid, _fp in identities if _kbd._worker_alive(pid, _fp)]})
    return {"live_runs": live} if live else None


def _review_claim_collision_at_claim(
    conn: sqlite3.Connection, task_id: str, run_id: int, lock: str,
) -> None:
    """Mechanical trace for a review claim landing while ANOTHER run on the same
    card is still open (Pepper's case (b), t_ea9f278a).

    The incident showed what a silent double-execution costs: a card parked in
    ``review`` while the same identity is steered elsewhere leaves the board
    attributing later events to a run that never posted them (the D3 shape).
    Witness-only — one event row, no policy change, no steering refusal:
    arbitration belongs to the process layer (review-fan-in-collision-policy)."""
    try:
        rows = conn.execute(
            "SELECT id, profile, claim_lock, started_at FROM task_runs "
            "WHERE task_id = ? AND id != ? AND ended_at IS NULL ORDER BY id DESC LIMIT 10",
            (task_id, int(run_id)),
        ).fetchall()
        others = [
            {"run_id": int(_kb._row_get(r, "id")), "profile": _kb._row_get(r, "profile"),
             "claim_lock": _kb._row_get(r, "claim_lock"), "started_at": _kb._row_get(r, "started_at")}
            for r in rows
        ]
        if not others:
            return
        _kb._append_event(conn, task_id, "review_claim_collision", {
            "kind": "claim",
            "run_id": int(run_id),
            "claim_lock": lock,
            "actor": _actor_attribution(conn, task_id),
            "live_runs": others,
        }, run_id=int(run_id))
    except Exception:  # health: allow BLE001 -- the claim witness is advisory and never fails the claim; logs exc_info
        _kb._log.debug("kanban: review-claim collision witness failed for %s", task_id, exc_info=True)


def _completed_event_payload(
    result: Optional[str], event_summary: Optional[str], verified_cards: list[str], metadata: Any,
) -> dict:
    """``completed`` event payload: first summary line (400 chars) so gateway
    notifiers / dashboard WS render without a second round-trip; verified
    cards; and ``metadata["artifacts"]`` promoted so the notifier can upload
    them as native attachments without fetching the run row."""
    # Mirror CLI's _show_voice_status: include STT/TTS provider availability so the user can tell at a
    # glance *why* voice mode isn't working ("STT provider: MISSING ..." is the common case). ``record_key``
    # mirrors the configured ``voice.record_key`` so the TUI can both bind it (frontend
    # ``isVoiceToggleKey``) and display it in /voice status — previously the TUI hardcoded Ctrl+B and
    # ignored the config (#18994).
    payload: dict = {
        "result_len": len(result) if result else 0,
        "summary": _kb._first_line(event_summary, 400) or None,
    }
    if verified_cards:
        payload["verified_cards"] = verified_cards
    if isinstance(metadata, dict):
        cleaned = _kb._cleaned_artifact_paths(metadata)
        if cleaned:
            payload["artifacts"] = cleaned
    return payload


# Late-bound sibling namespaces (see module docstring): imported LAST so this
# module is fully populated before the facades import names from it, and its own
# cross-module reads all happen at call time.
from hermes_cli import kanban_db as _kb
from hermes_cli import kanban_db_dispatch as _kbd
