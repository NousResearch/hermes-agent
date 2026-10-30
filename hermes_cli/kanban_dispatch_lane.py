"""Task-lane dispatch implementation separated from the dispatcher facade."""

from __future__ import annotations

import sqlite3
from typing import Optional


def _record_nonspawnable_profile(conn, task_id, assignee, result, dry_run, profile_exists_fn, kb) -> bool:
    profile_exists = profile_exists_fn()
    if profile_exists is None or profile_exists(assignee):
        return False
    result.skipped_nonspawnable.append(task_id)
    # Persist one diagnostic per unchanged task state, avoiding an event each tick.
    if not dry_run:
        with kb.write_txn(conn):
            last = conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,),
            ).fetchone()
            payload = kb._json_or_null({"assignee": assignee})
            if last is None or last["kind"] != "skipped_nonspawnable" or last["payload"] != payload:
                kb._append_event(conn, task_id, "skipped_nonspawnable", {"assignee": assignee})
    return True


def dispatch_lane_task(
    conn: sqlite3.Connection, row: sqlite3.Row, assignee: str, result, *,
    lane: str, dry_run: bool, ttl_seconds: Optional[int], board: Optional[str],
    failure_limit: int, spawn_fn, per_profile_cap: Optional[int],
    per_profile_running: dict[str, int], deps: dict,
) -> bool:
    """Guard, claim, resolve workspace, and spawn one ready or review row."""
    _profile_exists_fn = deps["profile_exists_fn"]
    check_respawn_guard = deps["check_respawn_guard"]
    _local_gpu_dispatch_preflight = deps["local_gpu_dispatch_preflight"]
    _kb = deps["kb"]
    _kbw = deps["kbw"]
    _call_spawn_fn = deps["call_spawn_fn"]
    _default_spawn = deps["default_spawn"]
    _set_worker_pid = deps["set_worker_pid"]
    _record_task_failure = deps["record_task_failure"]
    task_id = row["id"]
    # Non-profile assignees (control-plane lanes that pull via ``claim_task``)
    # would fail ``hermes -p <assignee>`` at startup and loop ready→crash→ready
    # forever. Bucketed apart from skipped_unassigned: the operator cannot fix
    # it by assigning a profile, and health telemetry suppresses "stuck" for it.
    if _record_nonspawnable_profile(
        conn, task_id, assignee, result, dry_run, _profile_exists_fn, _kb,
    ):
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

    if not dry_run:
        admission_reason = _local_gpu_dispatch_preflight(row, assignee)
        if admission_reason:
            result.skipped_local_admission.append((task_id, admission_reason))
            with _kb.write_txn(conn):
                last = conn.execute(
                    "SELECT kind,payload FROM task_events WHERE task_id=? "
                    "ORDER BY created_at DESC,id DESC LIMIT 1", (task_id,),
                ).fetchone()
                payload = {"assignee": assignee, "reason": admission_reason}
                if (last is None or last["kind"] != "local_gpu_admission_refused"
                        or last["payload"] != _kb._json_or_null(payload)):
                    _kb._append_event(conn, task_id, "local_gpu_admission_refused", payload)
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
    # health: allow BLE001 -- all workspace resolution failures are persisted as a task failure.
    except Exception as exc:
        if _record_task_failure(
            conn, claimed.id, f"workspace: {exc}",
            outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
        ):
            result.auto_blocked.append(claimed.id)
        return False
    _kbw.set_workspace_path(conn, claimed.id, str(workspace))
    # Keep the in-memory Task handed to the worker-spawn boundary in sync with the
    # persisted run contract; scratch workspace rows start with workspace_path=NULL.
    claimed.workspace_path = str(workspace)
    if claimed.workspace_kind == "worktree":
        claimed.branch_name = resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}"
        _kbw.set_branch_name(conn, claimed.id, claimed.branch_name)
    _kbw._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
    if lane == "review":
        # Force-load sdlc-review; the kanban lifecycle is already in every
        # worker's system prompt via KANBAN_GUIDANCE.
        claimed.skills = list(dict.fromkeys([*(claimed.skills or []), "sdlc-review"]))
    try:
        pid = _call_spawn_fn(spawn_fn if spawn_fn is not None else _default_spawn, claimed, str(workspace), board)
        if pid:
            _set_worker_pid(conn, claimed.id, int(pid), run_id=claimed.current_run_id, claim_lock=claimed.claim_lock)
        # Fires AFTER the PID (when reported) is durably persisted. Best-effort.
        _kb._fire_worker_spawned_hook(conn, claimed, str(workspace), pid, board=board)
        # consecutive_failures is deliberately NOT reset here: resetting on
        # spawn would let a task that keeps timing out loop forever. Cleared
        # only on successful completion (complete_task).
        result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
        _count_spawn(claimed.assignee)
        return True
    # health: allow BLE001 -- spawn failures are classified and persisted for every backend.
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
