"""Kanban board watcher methods for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3).
These are the background-loop methods that subscribe to kanban boards, deliver
notifications/artifacts, and drive the multi-agent dispatcher. They use only
``self`` state, so they live on a mixin that ``GatewayRunner`` inherits — the
``self._kanban_*`` call sites resolve identically via the MRO, making this a
behavior-neutral move that lifts ~1,000 LOC out of run.py.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")



# ---------------------------------------------------------------------------
# M2-1: Smart gave_up detection — pure functions
# ---------------------------------------------------------------------------

def stderr_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Similarity ratio of the first 200 characters of two stderr strings.

    Edge cases:
    - Both empty/None → 1.0 (trivially identical).
    - One empty, other not → 0.0.
    - Non-string → coerced to ``str()``.
    """
    a = str(s1 or "")[:200]
    b = str(s2 or "")[:200]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def detect_repeated_verification_errors(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[dict]:
    """Detect 2 consecutive verification failures with same cmd + similar stderr.

    Returns a detection dict ``{"reason", "details"}`` when the last two
    ``verification_failed`` events share the same failed command and their
    stderr (first 200 chars) has >80% similarity.  Returns ``None``
    otherwise.
    """
    rows = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'verification_failed' "
        "ORDER BY created_at DESC, id DESC LIMIT 2",
        (task_id,),
    ).fetchall()
    if len(rows) < 2:
        return None
    events: list[dict] = []
    for r in rows:
        try:
            events.append(json.loads(r["payload"]) if r["payload"] else {})
        except Exception:
            events.append({})
    older, newer = events[1], events[0]  # events[0] is newest
    older_failures = older.get("failures") or []
    newer_failures = newer.get("failures") or []
    if not older_failures or not newer_failures:
        return None
    old_cmd = str(older_failures[0].get("cmd", ""))
    new_cmd = str(newer_failures[0].get("cmd", ""))
    if old_cmd != new_cmd:
        return None
    old_stderr = str(older_failures[0].get("stderr", ""))
    new_stderr = str(newer_failures[0].get("stderr", ""))
    sim = stderr_similarity(old_stderr, new_stderr)
    if sim > 0.8:
        return {
            "reason": (
                f"repeated_verification_error: same cmd '{old_cmd}' "
                f"with {sim:.0%} stderr similarity across consecutive failures"
            ),
            "details": {
                "cmd": old_cmd,
                "stderr_similarity": round(sim, 3),
                "older_stderr_head": old_stderr[:200],
                "newer_stderr_head": new_stderr[:200],
            },
        }
    return None


def detect_token_anomaly(
    conn: sqlite3.Connection, task_id: str, *,
    threshold: int = 50000,
) -> Optional[dict]:
    """Detect a single closed run whose token usage exceeds *threshold*.

    Reads ``task_runs.metadata`` (JSON) for ``input_tokens`` and
    ``output_tokens``.  Returns a detection dict when the total exceeds
    the threshold, ``None`` otherwise.
    """
    row = conn.execute(
        "SELECT id, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row:
        return None
    try:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
    except Exception:
        meta = {}
    input_tokens = int(meta.get("input_tokens", 0) or 0)
    output_tokens = int(meta.get("output_tokens", 0) or 0)
    total = input_tokens + output_tokens
    if total > threshold:
        return {
            "reason": (
                f"token_anomaly: {total} tokens "
                f"(input={input_tokens}, output={output_tokens}, "
                f"threshold={threshold})"
            ),
            "details": {
                "run_id": int(row["id"]),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total": total,
                "threshold": threshold,
            },
        }
    return None


def detect_no_output(
    conn: sqlite3.Connection, task_id: str, *,
    min_runs: int = 3,
) -> Optional[dict]:
    """Detect *min_runs* consecutive closed runs with no artifacts.

    Checks the ``metadata.artifacts`` field on each run.  Returns a
    detection dict when all of the last *min_runs* runs have empty
    artifacts, ``None`` otherwise.
    """
    rows = conn.execute(
        "SELECT id, metadata, outcome FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT ?",
        (task_id, min_runs),
    ).fetchall()
    if len(rows) < min_runs:
        return None
    empty_count = 0
    for r in rows:
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except Exception:
            meta = {}
        if not meta.get("artifacts"):
            empty_count += 1
    if empty_count >= min_runs:
        return {
            "reason": (
                f"no_output: {empty_count}/{len(rows)} consecutive runs "
                f"produced no artifacts"
            ),
            "details": {
                "empty_runs": empty_count,
                "total_runs_checked": len(rows),
                "min_required": min_runs,
            },
        }
    return None


def check_early_giveup(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    enable_repeated_error: bool = True,
    enable_token_anomaly: bool = True,
    enable_no_output: bool = True,
    token_threshold: int = 50000,
    no_output_min_runs: int = 3,
) -> Optional[dict]:
    """Check all smart gave_up conditions (M2-1).

    Each condition can be independently toggled so operators can disable
    false-positive-prone checks without touching the others.  Returns the
    first detection or ``None`` when no condition fires.
    """
    if enable_repeated_error:
        d = detect_repeated_verification_errors(conn, task_id)
        if d:
            return d
    if enable_token_anomaly:
        d = detect_token_anomaly(conn, task_id, threshold=token_threshold)
        if d:
            return d
    if enable_no_output:
        d = detect_no_output(conn, task_id, min_runs=no_output_min_runs)
        if d:
            return d
    return None


# ---------------------------------------------------------------------------
# M2-2: Board-level convergence detection — pure function
# ---------------------------------------------------------------------------

def compute_board_convergence(
    conn: sqlite3.Connection,
    *,
    blocked_ratio_threshold: float = 0.2,
    resolve_rate_threshold: float = 0.8,
    time_window_seconds: float = 600,
) -> dict:
    """Compute board-level convergence metrics (pure function, no side effects).

    Counts non-archived tasks by status, tallies ``verification_failed``
    and ``remediation_created`` events, and evaluates the convergence
    predicate::

        converged = AND(
            new_tasks_created == 0,
            blocked_ratio  < threshold,
            resolve_rate   > threshold,
            verification_failed == 0,
        )

    The ``verification_failed`` and ``remediation_created`` counts are
    bounded by ``time_window_seconds``: only events created within the
    last N seconds are counted (default 600s = 10 minutes).  This
    prevents stale historical events from permanently blocking the
    convergence predicate — without the window, a single legacy
    ``verification_failed`` row from days ago would keep the board
    from ever being declared converged.

    ``time_window_seconds=0`` collapses to ``cutoff = time.time()``,
    i.e. the predicate looks at events created strictly after "now"
    (effectively no qualifying events — useful as a "force converge"
    knob for tests).

    Returns a dict with all metrics + the boolean ``converged`` flag.
    """
    status_counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        status_counts[row["status"]] = int(row["n"])
    total = sum(status_counts.values())

    resolved = status_counts.get("done", 0)
    blocked = status_counts.get("blocked", 0)
    running = status_counts.get("running", 0)
    ready = status_counts.get("ready", 0)

    blocked_ratio = blocked / total if total > 0 else 0.0
    resolve_rate = resolved / total if total > 0 else 0.0

    # P0 fix: bound vf/remediation counts to a recent window so stale
    # historical events don't permanently block convergence.
    cutoff = time.time() - time_window_seconds

    vf_row = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events "
        "WHERE kind = 'verification_failed' AND created_at > ?",
        (cutoff,),
    ).fetchone()
    verification_failed = int(vf_row["n"]) if vf_row else 0

    rt_row = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events "
        "WHERE kind = 'remediation_created' AND created_at > ?",
        (cutoff,),
    ).fetchone()
    new_tasks_created = int(rt_row["n"]) if rt_row else 0

    converged = (
        total > 0
        and new_tasks_created == 0
        and blocked_ratio < blocked_ratio_threshold
        and resolve_rate > resolve_rate_threshold
        and verification_failed == 0
    )

    return {
        "total_tasks": total,
        "resolved": resolved,
        "blocked": blocked,
        "running": running,
        "ready": ready,
        "blocked_ratio": round(blocked_ratio, 4),
        "resolve_rate": round(resolve_rate, 4),
        "new_tasks_created": new_tasks_created,
        "verification_failed": verification_failed,
        "converged": converged,
        "time_window_seconds": time_window_seconds,
    }


# ---------------------------------------------------------------------------
# M2-3: task_loop_closed event writer
# ---------------------------------------------------------------------------

def record_task_loop_closed(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    metrics: dict,
    loop_depth: int = 0,
    duration_seconds: int = 0,
    task_loop_id: Optional[str] = None,
) -> None:
    """Write a ``task_loop_closed`` event with a structured payload (M2-3).

    Called from :class:`EpochEngine` when convergence is detected or the
    loop budget is exhausted.  The payload includes all convergence
    metrics plus loop-depth and duration for downstream learning.
    """
    from hermes_cli import kanban_db as _kb
    payload = {
        "task_loop_id": task_loop_id,
        "total_tasks": metrics.get("total_tasks", 0),
        "resolved": metrics.get("resolved", 0),
        "blocked": metrics.get("blocked", 0),
        "new_tasks_created": metrics.get("new_tasks_created", 0),
        "verification_failed": metrics.get("verification_failed", 0),
        "converged": metrics.get("converged", False),
        "loop_depth": loop_depth,
        "duration_seconds": duration_seconds,
    }
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "task_loop_closed", payload)


# ---------------------------------------------------------------------------
# M3-1: Notify parent tasks on child verification failure
# ---------------------------------------------------------------------------

def _build_failure_summary(failures: list[dict]) -> str:
    """Build a concise one-line summary from verification failure results.

    Caps at 3 commands and truncates each stderr head to 120 chars so the
    notification stays compact — the parent thread should not absorb the
    child's entire failure detail (that would bloat parent context).
    """
    if not failures:
        return "verification failed (no command details)"
    parts: list[str] = []
    for f in failures[:3]:
        cmd = str(f.get("cmd", "?"))
        exit_code = f.get("exit_code", "?")
        stderr_head = str(f.get("stderr", ""))[:120]
        parts.append(f"`{cmd}` exit={exit_code}: {stderr_head}")
    return "; ".join(parts)


def notify_parents_on_verification_failure(
    conn: sqlite3.Connection,
    kb_module,
    task_id: str,
    *,
    failures: list[dict],
    loop_depth: int,
) -> int:
    """Write a notification comment to each parent of *task_id* (M3-1).

    Called when a child task's verification fails.  Looks up parents via
    ``task_links``, builds a concise summary, and writes a comment to each
    parent task's thread.

    This function is **notification-only** — it does NOT modify parent
    task status.  Parent status is managed by the existing dependency
    mechanism (promotion, blocking, etc.).

    Returns the number of parents notified.

    Edge cases handled:
    - No parents → returns 0 (no-op).
    - Parent doesn't exist / is archived → skipped silently.
    - Cycle (A→B→A) → impossible: ``task_links`` enforces DAG invariant
      via ``_would_create_cycle``, so notification never loops.
    """
    parent_list = kb_module.parent_ids(conn, task_id)
    if not parent_list:
        return 0

    child_task = kb_module.get_task(conn, task_id)
    child_title = child_task.title if child_task else task_id

    failure_summary = _build_failure_summary(failures)
    notified = 0

    for parent_id in parent_list:
        parent_task = kb_module.get_task(conn, parent_id)
        if parent_task is None:
            continue
        if getattr(parent_task, "status", None) == "archived":
            continue

        body = (
            f"⚠️ 子任务 [{child_title}] 验证失败 "
            f"(loop {loop_depth + 1})\n"
            f"失败原因: {failure_summary}"
        )
        try:
            kb_module.add_comment(
                conn, parent_id,
                author="task-loop-system",
                body=body,
            )
            notified += 1
            logger.info(
                "M3-1: notified parent %s about child %s verification failure",
                parent_id, task_id,
            )
        except Exception as exc:
            logger.warning(
                "M3-1: failed to notify parent %s for child %s: %s",
                parent_id, task_id, exc,
            )

    return notified


class TaskLoopState:
    """Conceptual task-loop lifecycle phases.

    Documents the intended state machine for loop engineering.  M0
    implements IDLE → COLLECTING → INJECTING → (fire-and-forget back to
    IDLE).  WAITING / EVALUATING / CLOSED are reserved for M2
    convergence-detection work and are not yet exercised at runtime.
    """

    IDLE = "idle"              # No terminal events; waiting for work.
    COLLECTING = "collecting"  # Terminal events detected; cooldown window.
    INJECTING = "injecting"    # Synthesizing + dispatching orchestrator msg.
    WAITING = "waiting"        # Orchestrator processing the injection.
    EVALUATING = "evaluating"  # Checking convergence / new tasks / blocked.
    CLOSED = "closed"          # Loop budget exhausted; reset counters.


# Deprecated alias — earlier code referenced this as ``EpochState``.
EpochState = TaskLoopState


class TaskLoopEngine:
    """Orchestrator task-loop detection + message-injection engine.

    Extracted behavior-neutrally from ``GatewayKanbanWatchersMixin.
    _kanban_orchestrator_callback`` (was ~450 lines).  The engine detects
    when a board transitions from active work to idle (all terminal
    events processed, no ready tasks blocking), then injects a synthetic
    ``MessageEvent`` into the orchestrator profile's session so it can
    plan the next loop.

    Per-board tracking state that was previously class-level mutable
    defaults on the mixin (Bug #6) now lives here as instance attributes,
    so each ``GatewayRunner`` gets its own isolated engine.

    The conceptual state machine (:class:`TaskLoopState`) is documented but
    not yet enforced at runtime — the ``tick`` method performs the full
    IDLE → COLLECTING → INJECTING cycle in one pass.  M2 will split this
    into discrete transitions with convergence evaluation.
    """

    def __init__(self) -> None:
        # Per-board cooldown / anti-loop state.
        self._cooldowns: dict[str, float] = {}
        self._stale_counts: dict[str, int] = {}
        self._task_loop_counts: dict[str, int] = {}
        self._last_event_id: dict[str, int] = {}

    async def tick(
        self,
        gateway,
        deliveries: list[dict],
        kanban_cfg: dict,
    ) -> None:
        """Check boards for completed loops and notify the orchestrator.

        Runs on every notifier tick, **independent of subscriptions**.
        For each board in ``orchestrator_boards`` (or all boards if not
        configured), checks whether the board has zero running tasks AND
        has had a recent terminal event (completed/blocked/crashed/gave_up/
        timed_out). If so, injects an internal MessageEvent into the
        orchestrator profile's session so it can plan the next loop.

        This does NOT depend on kanban_notify_subs or kanban_board_subs —
        it scans the task tables directly. All connected home channels
        receive the task-loop decision push automatically.

        Configuration (in config.yaml under ``kanban:``):

        - ``orchestrator_notify: true`` — enable this callback.
        - ``orchestrator_profile: <name>`` — profile to notify.
        - ``orchestrator_boards: <list>`` — board slug allowlist.
        - ``orchestrator_cooldown_seconds: 30`` — min seconds between
          notifications per board.
        - ``orchestrator_max_loops: 10`` (legacy alias ``orchestrator_max_epochs``)
          — max loop notifications per board before the callback goes silent.
        - ``orchestrator_max_stale: 3`` — max consecutive stale triggers
          before the board is suppressed until a real event arrives.
        """
        from hermes_cli import kanban_db as _kb

        cooldown_seconds = float(kanban_cfg.get("orchestrator_cooldown_seconds", 30))
        MAX_CONSECUTIVE_STALE = int(kanban_cfg.get("orchestrator_max_stale", 3))
        # Prefer the new key, fall back to the legacy ``orchestrator_max_epochs``
        # for backward compatibility with existing config.yaml files.
        MAX_LOOPS = int(
            kanban_cfg.get("orchestrator_max_loops")
            or kanban_cfg.get("orchestrator_max_epochs", 10)
        )

        orchestrator = (kanban_cfg.get("orchestrator_profile") or "").strip()
        if not orchestrator:
            orchestrator = gateway._active_profile_name()

        board_allowlist = kanban_cfg.get("orchestrator_boards", [])
        if not isinstance(board_allowlist, list):
            board_allowlist = []

        now = time.monotonic()

        # Build event lookup from deliveries (may be empty — that's fine).
        # Used to summarize what happened in the loop message.
        events_by_board: dict[str, list] = {}
        for d in deliveries:
            slug = d.get("board")
            if slug:
                events_by_board.setdefault(slug, []).extend(d.get("events", []))

        # Determine candidate boards: scan ALL boards (allowlist or
        # discovered), not just boards with deliveries. This is the key
        # fix — loop detection must not depend on subscription tables.
        if board_allowlist:
            candidate_boards = list(board_allowlist)
        else:
            # Discover all boards from the kanban boards directory.
            import os as _os
            boards_dir = _os.path.expanduser("~/.hermes/kanban/boards")
            candidate_boards = []
            if _os.path.isdir(boards_dir):
                for name in sorted(_os.listdir(boards_dir)):
                    if name.startswith("_") or name.startswith("."):
                        continue
                    if _os.path.isdir(_os.path.join(boards_dir, name)):
                        candidate_boards.append(name)
            # Always include default board (stored in main kanban.db).
            if _kb.DEFAULT_BOARD not in candidate_boards:
                candidate_boards.append(_kb.DEFAULT_BOARD)

        for slug in candidate_boards:
            # Cooldown gate.
            if now - self._cooldowns.get(slug, 0) < cooldown_seconds:
                continue

            # Count in-progress and ready tasks on this board.
            try:
                conn = _kb.connect(board=slug)
                try:
                    tasks = _kb.list_tasks(conn, status="running")
                    in_progress_count = len(tasks) if tasks else 0
                    ready_tasks = _kb.list_tasks(conn, status="ready")
                    ready_count = len(ready_tasks) if ready_tasks else 0
                    # Check for recent terminal events since last task-loop trigger.
                    # Uses per-board last_event_id to avoid re-triggering on
                    # old events. Falls back to 600s window on first run.
                    last_eid = self._last_event_id.get(slug, 0)
                    if last_eid > 0:
                        recent_event_rows = conn.execute(
                            "SELECT te.id, te.task_id, te.kind, te.payload "
                            "FROM task_events te WHERE te.id > ? "
                            "AND te.kind IN ('completed','blocked','crashed','gave_up','timed_out') "
                            "ORDER BY te.id DESC LIMIT 20",
                            (last_eid,),
                        ).fetchall()
                    else:
                        cutoff = time.time() - 600
                        recent_event_rows = conn.execute(
                            "SELECT te.id, te.task_id, te.kind, te.payload "
                            "FROM task_events te WHERE te.created_at > ? "
                            "AND te.kind IN ('completed','blocked','crashed','gave_up','timed_out') "
                            "ORDER BY te.created_at DESC LIMIT 20",
                            (cutoff,),
                        ).fetchall()
                    recent_events = [(r[2],) for r in recent_event_rows]  # kind-only for Counter compat
                    # Build detailed event list for user-facing summary.
                    event_details = []
                    for r in recent_event_rows:
                        _eid, _tid, _kind, _payload = r
                        # Get task info
                        _tinfo = conn.execute(
                            "SELECT title, assignee, result FROM tasks WHERE id=?", (_tid,)
                        ).fetchone()
                        _title = _tinfo[0] if _tinfo else "?"
                        _assignee = _tinfo[1] if _tinfo else "?"
                        # Extract summary from payload or task result
                        _summary = ""
                        if _payload:
                            try:
                                _p = json.loads(_payload)
                                _summary = _p.get("summary", "")
                            except Exception:
                                _summary = ""
                        if not _summary and _tinfo and _tinfo[2]:
                            _summary = _tinfo[2][:200]
                        event_details.append({
                            "task_id": _tid,
                            "kind": _kind,
                            "title": _title,
                            "assignee": _assignee,
                            "summary": _summary,
                        })
                    any_terminal = len(recent_events) > 0
                    # Query blocked count here while conn is still open.
                    blocked_count = len(_kb.list_tasks(conn, status="blocked") or [])
                finally:
                    conn.close()
            except Exception as exc:
                logger.debug(
                    "kanban orchestrator callback: board %s check failed: %s",
                    slug, exc,
                )
                continue

            # Trigger task-loop when there are terminal events AND no ready tasks.
            # We allow triggering even with running tasks, because a running
            # task might be an auto-re-dispatch from a crash — the orchestrator
            # needs to know about the crash to decide next steps.
            if not any_terminal and ready_count == 0 and in_progress_count > 0:
                continue

            # A board with 0 running tasks but also 0 ready and no recent
            # terminal events is simply idle — skip without counting as
            # stale. Only trigger task-loop when there's actually something
            # to act on (ready tasks to dispatch or terminal events to
            # react to).
            if not ready_count and not any_terminal:
                continue
            # Reset stale counter — something real happened.
            self._stale_counts[slug] = 0

            # Rescue orphaned children: if a task is blocked and has
            # children stuck in 'todo' (because parent isn't 'done'),
            # detach them so they become independent tasks. This prevents
            # deadlocks: tester blocks → creates fix task as child →
            # fix task stuck because blocked parent never reaches 'done'.
            # Try re-parenting to grandparent first (preserves context);
            # if no grandparent or grandparent not done, just detach.
            try:
                conn = _kb.connect(board=slug)
                try:
                    blocked_tasks = _kb.list_tasks(conn, status="blocked") or []
                    for bt in blocked_tasks:
                        # Find todo children of this blocked task.
                        orphan_rows = conn.execute(
                            "SELECT t.id FROM tasks t "
                            "JOIN task_edges e ON e.child_id = t.id "
                            "WHERE e.parent_id = ? AND t.status = 'todo'",
                            (bt.id,),
                        ).fetchall()
                        for orow in orphan_rows:
                            orphan_id = orow[0]
                            # Try re-parenting to grandparent (done tasks only).
                            gp_row = conn.execute(
                                "SELECT e2.parent_id, t2.status FROM task_edges e "
                                "JOIN task_edges e2 ON e2.child_id = e.parent_id "
                                "JOIN tasks t2 ON t2.id = e2.parent_id "
                                "WHERE e.child_id = ?",
                                (orphan_id,),
                            ).fetchone()
                            if gp_row and gp_row[0] and gp_row[1] == "done":
                                # Grandparent exists and is done — re-parent.
                                conn.execute(
                                    "DELETE FROM task_edges WHERE parent_id=? AND child_id=?",
                                    (bt.id, orphan_id),
                                )
                                conn.execute(
                                    "INSERT OR IGNORE INTO task_edges(parent_id, child_id) VALUES(?,?)",
                                    (gp_row[0], orphan_id),
                                )
                                logger.info(
                                    "kanban task_loop: re-parented orphan %s "
                                    "from blocked %s to grandparent %s",
                                    orphan_id, bt.id, gp_row[0],
                                )
                            else:
                                # No suitable grandparent — detach entirely.
                                conn.execute(
                                    "DELETE FROM task_edges WHERE parent_id=? AND child_id=?",
                                    (bt.id, orphan_id),
                                )
                                logger.info(
                                    "kanban task_loop: detached orphan %s from blocked %s",
                                    orphan_id, bt.id,
                                )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as rescue_exc:
                logger.debug("kanban task_loop: orphan rescue failed: %s", rescue_exc)

            # Epoch counter tracks active work on this board. Reset when:
            # 1. Board is idle (no running/ready/blocked = workflow finished)
            # 2. New terminal events arrived (fresh task-loop budget per new event)
            is_idle = (in_progress_count == 0 and ready_count == 0 and blocked_count == 0)
            if is_idle or any_terminal:
                self._task_loop_counts[slug] = 0

            current_loop = self._task_loop_counts.get(slug, 0) + 1

            # ── M2-2/M2-3: Convergence detection ───────────────────
            # Before injecting another task-loop message, check whether the
            # board has converged.  If it has, write a task_loop_closed
            # event and skip the injection — the workflow is done.
            try:
                conv_conn = _kb.connect(board=slug)
                try:
                    metrics = compute_board_convergence(conv_conn)
                    if metrics["converged"]:
                        logger.info(
                            "kanban task_loop: board %s converged "
                            "(resolved=%d/%d, blocked_ratio=%.2f, "
                            "resolve_rate=%.2f) — writing task_loop_closed",
                            slug, metrics["resolved"], metrics["total_tasks"],
                            metrics["blocked_ratio"], metrics["resolve_rate"],
                        )
                        # Write task_loop_closed on the most recently
                        # active task (first in recent_event_rows).
                        _loop_tid = (
                            recent_event_rows[0][1]
                            if recent_event_rows else None
                        )
                        if _loop_tid:
                            record_task_loop_closed(
                                conv_conn, _loop_tid,
                                metrics=metrics,
                                loop_depth=current_loop - 1,
                                duration_seconds=int(
                                    time.time()
                                    - (recent_event_rows[-1][0] or 0)
                                ) if recent_event_rows else 0,
                                task_loop_id=f"{slug}:loop:{current_loop}",
                            )
                        # Reset task-loop counter — the board converged.
                        self._task_loop_counts[slug] = 0
                        self._cooldowns[slug] = now
                        # Update last_event_id so we don't re-process.
                        max_eid = conv_conn.execute(
                            "SELECT MAX(id) FROM task_events"
                        ).fetchone()
                        if max_eid and max_eid[0]:
                            self._last_event_id[slug] = max_eid[0]
                        continue
                finally:
                    conv_conn.close()
            except Exception as conv_exc:
                logger.debug(
                    "kanban task_loop: convergence check failed for %s: %s",
                    slug, conv_exc,
                )

            # ── M2-1: Smart gave_up detection ───────────────────────
            # For tasks with verification failures, proactively check
            # smart giveup conditions.  If detected, block the task with
            # a reason containing the detection cause.
            m2_cfg = kanban_cfg.get("task_loop", {})
            if m2_cfg.get("smart_giveup", True):
                try:
                    m2_conn = _kb.connect(board=slug)
                    try:
                        for ev_row in recent_event_rows:
                            _ev_tid = ev_row[1]
                            _ev_kind = ev_row[2]
                            if _ev_kind != "verification_failed":
                                continue
                            detection = check_early_giveup(
                                m2_conn, _ev_tid,
                                enable_repeated_error=m2_cfg.get(
                                    "enable_repeated_error", True),
                                enable_token_anomaly=m2_cfg.get(
                                    "enable_token_anomaly", True),
                                enable_no_output=m2_cfg.get(
                                    "enable_no_output", True),
                                token_threshold=int(m2_cfg.get(
                                    "token_threshold", 50000)),
                                no_output_min_runs=int(m2_cfg.get(
                                    "no_output_min_runs", 3)),
                            )
                            if detection:
                                logger.info(
                                    "kanban task_loop: smart giveup for %s: %s",
                                    _ev_tid, detection["reason"],
                                )
                                reason = (
                                    f"smart_giveup: {detection['reason']}. "
                                    f"Task blocked to prevent wasted cycles."
                                )
                                _kb.block_task(
                                    m2_conn, _ev_tid, reason=reason,
                                )
                    finally:
                        m2_conn.close()
                except Exception as sg_exc:
                    logger.debug(
                        "kanban task_loop: smart giveup check failed for %s: %s",
                        slug, sg_exc,
                    )

            # Anti-loop: max task-loop limit per "wave" — between terminal events,
            # cap orchestrator re-dispatch attempts to avoid hot-looping.
            if current_loop > MAX_LOOPS:
                logger.info(
                    "kanban orchestrator callback: board %s loop limit (%d/%d); "
                    "waiting for next terminal event or new task",
                    slug, current_loop - 1, MAX_LOOPS,
                )
                continue
            self._task_loop_counts[slug] = current_loop

            self._cooldowns[slug] = now

            # Record the max event id for this board so we only trigger on
            # NEW terminal events next time.
            try:
                conn2 = _kb.connect(board=slug)
                try:
                    max_eid_row = conn2.execute("SELECT MAX(id) FROM task_events").fetchone()
                    if max_eid_row and max_eid_row[0]:
                        self._last_event_id[slug] = max_eid_row[0]
                finally:
                    conn2.close()
            except Exception as eid_exc:
                logger.debug(
                    "kanban orchestrator callback: max event id update failed for %s: %s",
                    slug, eid_exc,
                )

            # Build the notification message with event summaries.
            board_label = (
                f"board={slug}" if slug != _kb.DEFAULT_BOARD else "default board"
            )

            # Summarize what happened: count by event kind from recent_events
            event_kinds: Counter = Counter()
            for ev_row in recent_events:
                event_kinds[ev_row[0]] += 1
            event_summary = ", ".join(
                f"{k}={v}" for k, v in sorted(event_kinds.items())
            ) if event_kinds else "no events"

            in_progress_names = [t.id for t in (tasks or [])]

            msg_lines = [
                f"[Kanban Task Loop #{current_loop}] Workers idle on {board_label}.",
                f"Events this tick: {event_summary}",
            ]
            if in_progress_names:
                msg_lines.append(f"Running tasks: {len(in_progress_names)} ({', '.join(in_progress_names[:5])})")
            if ready_count > 0:
                msg_lines.append(
                    f"{ready_count} ready task(s) queued — decompose and dispatch."
                )
            else:
                msg_lines.append(
                    "No ready tasks. Review blocked/crashed tasks and re-decompose if needed."
                )
            msg_lines.append(f"(loop {current_loop}/{MAX_LOOPS})")

            # Orchestrator instructions — the LLM receives this as the user message.
            msg_lines.append("")
            msg_lines.append("--- Orchestrator Instructions ---")
            msg_lines.append(f"Board '{slug}': {ready_count} ready, {len(in_progress_names)} running")
            msg_lines.append("")
            msg_lines.append("As the kanban orchestrator, respond by EXECUTING tools — not by analyzing in text.")
            msg_lines.append("You MUST make at least one tool call this turn (kanban list/show/create/unblock).")
            msg_lines.append("Your text response will NOT be seen by anyone. Only tool results matter.")
            msg_lines.append("")
            msg_lines.append("Actions to take:")
            msg_lines.append("1. Check the board — run `kanban list` or `kanban show` on blocked/crashed tasks")
            msg_lines.append("2. For blocked: identify the blocker and decide — re-assign, re-decompose, or unblock")
            msg_lines.append("3. For crashed/gave_up: read failure reason FIRST (kanban_show), then:")
            msg_lines.append("   - Missing precondition (branch not merged, env not ready, deps missing) → create a prep task, then re-dispatch")
            msg_lines.append("   - Bad instructions in body → comment fix and unblock")
            msg_lines.append("   - Context unrecoverable (budget exhausted) → supersede: complete old + create clean new task")
            msg_lines.append("4. For completed: if there are ready/pending items, create the next loop's tasks")
            msg_lines.append("5. Be mindful of budget — don't create too many parallel tasks at once")
            msg_lines.append("6. Only create kanban tasks — the worker system handles execution")
            msg_lines.append("7. Do NOT send messages to the user — results are delivered automatically")
            msg_text = "\n".join(msg_lines)

            # Inject into the user's REAL session for this board.
            # Unified source lookup via resolve_board_source(): tries the
            # in-memory last-interaction source first, then falls back to
            # the board's notify subscriptions (Bug #3/#4 unification).
            try:
                from gateway.config import Platform as _Platform
                from gateway.platforms.base import MessageEvent, MessageType, SessionSource
                from tools.kanban_tools import resolve_board_source

                last_src = resolve_board_source(slug)

                if not last_src or not last_src[0]:
                    logger.debug(
                        "kanban orchestrator callback: no source for board %s, skipping",
                        slug,
                    )
                    continue

                _plat_str, _chat_id = last_src
                try:
                    loop_platform = _Platform(_plat_str)
                except ValueError:
                    logger.warning(
                        "kanban orchestrator callback: invalid platform %s for board %s",
                        _plat_str, slug,
                    )
                    continue

                source = SessionSource(
                    platform=loop_platform,
                    chat_id=_chat_id,
                    chat_type="private",
                    user_id="system",
                    user_name="kanban-orchestrator",
                )

                synthetic_event = MessageEvent(
                    text=msg_text,
                    source=source,
                    internal=True,
                )

                logger.info(
                    "kanban orchestrator callback: injecting into %s/%s "
                    "for board %s (loop %d/%d)",
                    _plat_str, _chat_id, slug, current_loop, MAX_LOOPS,
                )

                try:
                    # Send user-facing event summary FIRST, so the user
                    # knows what happened and can decide to intervene.
                    _kind_emoji = {
                        "completed": "✅", "blocked": "⚠️",
                        "crashed": "❌", "gave_up": "💀", "timed_out": "⏰",
                    }
                    summary_parts = [f"🔄 **Task Loop #{current_loop}** `{slug}`"]
                    for ed in event_details:
                        emoji = _kind_emoji.get(ed["kind"], "📌")
                        line = f"{emoji} `{ed['task_id']}` {ed['kind']} (@{ed['assignee']})"
                        if ed["summary"]:
                            line += f"\n   {ed['summary'][:200]}"
                        summary_parts.append(line)
                    if ready_count > 0:
                        summary_parts.append(f"📋 待处理: {ready_count}")
                    summary_parts.append(f"_(loop {current_loop}/{MAX_LOOPS})_")

                    summary_msg = "\n".join(summary_parts)

                    adapter = gateway.adapters.get(loop_platform)

                    # Inject into session WITHOUT blocking the notifier loop.
                    # A synchronous await here would queue user messages behind
                    # the task-loop processing, causing multi-minute delays.
                    # fire-and-forget lets the agent handle it asynchronously.

                    async def _task_loop_inject():
                        """Process task-loop injection and send combined response."""
                        try:
                            response_text = await gateway._handle_message(synthetic_event)
                        except Exception as exc:
                            logger.warning("kanban task_loop injection failed: %s", exc)
                            return

                        # Combine summary + agent response into ONE message.
                        summary_parts.append("")
                        if response_text:
                            summary_parts.append(f"📋 **处理结果:**\n{response_text[:500]}")

                        combined_msg = "\n".join(summary_parts)

                        if adapter:
                            send_result = await adapter.send(
                                chat_id=_chat_id, content=combined_msg,
                            )
                            if send_result and getattr(send_result, "success", False):
                                logger.info(
                                    "kanban task_loop: sent to %s/%s (loop %d/%d)",
                                    _plat_str, _chat_id, current_loop, MAX_LOOPS,
                                )
                            else:
                                err = getattr(send_result, "error", "unknown") if send_result else "no result"
                                logger.warning(
                                    "kanban task_loop: send to %s/%s FAILED: %s",
                                    _plat_str, _chat_id, err,
                                )

                    # Schedule as a background task — don't block notifier.
                    import asyncio as _aio
                    _aio.ensure_future(_task_loop_inject())
                    logger.info(
                        "kanban task_loop: scheduled injection for %s/%s (loop %d/%d, fire-and-forget)",
                        _plat_str, _chat_id, current_loop, MAX_LOOPS,
                    )
                except Exception as orch_exc:
                    logger.warning(
                        "kanban orchestrator callback: failed for board %s: %s",
                        slug, orch_exc,
                    )
            except Exception as import_exc:
                logger.warning(
                    "kanban orchestrator callback: import error for board %s: %s",
                    slug, import_exc,
                )


# ---------------------------------------------------------------------------
# M1 + M5: severity classification, push filter, and pipeline 终结报告 helpers.
# Lives at module scope so ``tests/m1_verify.py`` and ``tests/m5_verify.py``
# can import them without instantiating the mixin.  The M5 helpers compose
# into the existing ``_deliver_event``-style path inside the mixin (see the
# ``kind == "completed"`` branch below) — three-condition guard: is_root +
# all_descendants_terminal + non-empty summary.  Any failure falls back to
# the regular ✔ done push.
# ---------------------------------------------------------------------------

# Pipeline-active statuses (everything that is NOT terminal).  M5 only fires
# the 终结报告 when every task in the descendant tree is in a TERMINAL status
# (done / archived / crashed / gave_up / timed_out).  Mirrors
# ``hermes_cli.kanban_db.VALID_STATUSES`` minus the terminal subset.
_PIPELINE_ACTIVE_STATUSES = frozenset({
    "todo", "ready", "running", "blocked", "triage", "scheduled", "review",
})

# Statuses that count as "terminal" for the all-descendants check.  Used by
# ``_all_descendants_terminal`` to decide whether the pipeline root is ready
# for the 终结报告 push.
_PIPELINE_TERMINAL_STATUSES = frozenset({
    "done", "archived", "crashed", "gave_up", "timed_out",
})


def _is_root_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """True iff *task_id* has no parent in ``task_links``.

    A root task is one that no other task points at as its child.  Returns
    ``False`` for missing ids, missing connections, or DB errors — callers
    must treat "I don't know" as "not a root" so the M5 report path
    conservatively falls back to the regular ✔ done push.
    """
    if not task_id or conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM task_links WHERE child_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        # True iff no parent row exists; row is None → root, row exists → not root
        return row is None
    except Exception as exc:
        logger.debug(
            "kanban watchers: _is_root_task failed for %s: %s", task_id, exc,
        )
        return False


def _all_descendants_terminal(conn: sqlite3.Connection, root_id: str) -> bool:
    """True iff every task in the descendant tree (including *root_id*) is terminal.

    "Terminal" = status in ``{done, archived, crashed, gave_up, timed_out}``.
    Returns ``False`` for missing/None ids, missing connections, no-children
    tasks (a single task is not a pipeline), DB errors, or any non-terminal
    descendant — callers rely on the conservative ``False`` to fall back to
    the regular ✔ done push instead of emitting a premature 终结报告.

    Uses iterative BFS via ``task_links`` (no recursive CTE) so it stays
    SQLite-version-portable.  Defends against cycles with a visited-set.
    """
    if not root_id or conn is None:
        return False
    try:
        # Root itself must exist and be terminal.  If the row is missing,
        # the root "isn't even a real task" → not a pipeline → False.
        root_row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (root_id,),
        ).fetchone()
        if not root_row:
            return False
        if str(root_row["status"] or "") not in _PIPELINE_TERMINAL_STATUSES:
            return False

        # BFS over children.  Empty children ⇒ not a pipeline ⇒ False.
        visited: set[str] = {root_id}
        frontier: list[str] = [root_id]
        seen_any_child = False
        while frontier:
            current = frontier.pop()
            try:
                child_rows = conn.execute(
                    "SELECT t.id, t.status FROM task_links l "
                    "JOIN tasks t ON t.id = l.child_id "
                    "WHERE l.parent_id = ?",
                    (current,),
                ).fetchall()
            except Exception:
                # Defensive: schema drift shouldn't silently produce a True.
                return False
            for cr in child_rows:
                cid = str(cr["id"] or "")
                cstatus = str(cr["status"] or "")
                if not cid:
                    continue
                if cid in visited:
                    continue  # cycle guard
                visited.add(cid)
                seen_any_child = True
                if cstatus not in _PIPELINE_TERMINAL_STATUSES:
                    return False
                frontier.append(cid)
        # No children at all → not a pipeline, never trigger M5 report.
        return seen_any_child
    except Exception as exc:
        logger.debug(
            "kanban watchers: _all_descendants_terminal failed for %s: %s",
            root_id, exc,
        )
        return False


def build_pipeline_summary(root_id: Optional[str], conn: sqlite3.Connection) -> str:
    """Build a 终结报告 string for a completed pipeline root.

    Returns ``""`` when:
      * ``root_id`` is ``None``/empty or missing from the DB,
      * ``root_id`` is not a pipeline root (has a parent),
      * ``root_id`` has no children (a single task is not a pipeline),
      * any descendant is non-terminal (defensive — the call site should
        already guard with ``_all_descendants_terminal`` but we re-check
        here so a stale caller can't emit a premature report),
      * the DB query fails for any reason.

    Output format (matches the M5 verify expectations):

        📊 Pipeline "<title>" 终结报告
        ━━━━━━━━━━━━━━━━━━━━━━
        总任务: <N>
        通过:   <pass_count>
        失败:   <fail_count>
        耗时:   <elapsed_str>
        ━━━━━━━━━━━━━━━━━━━━━━
        任务明细:
          ✔ <task_id> <title> (<assignee>)
          ✖ <task_id> <title> (<assignee>)
        下一步建议: <suggestion text>

    Where pass = {done, archived}, fail = {crashed, gave_up, timed_out}.
    """
    if not root_id or conn is None:
        return ""
    try:
        root_row = conn.execute(
            "SELECT id, title, status, assignee, created_at, completed_at "
            "FROM tasks WHERE id = ?",
            (root_id,),
        ).fetchone()
        if not root_row:
            return ""
        # Non-root guard: if the task has any parent, it's not a pipeline root.
        parent_row = conn.execute(
            "SELECT 1 FROM task_links WHERE child_id = ? LIMIT 1",
            (root_id,),
        ).fetchone()
        if parent_row:
            return ""
        # Walk the descendant tree (BFS) and collect per-task info.
        visited: set[str] = {str(root_id)}
        frontier: list[str] = [str(root_id)]
        children: list = []
        seen_any_child = False
        while frontier:
            current = frontier.pop()
            try:
                child_rows = conn.execute(
                    "SELECT t.id, t.title, t.status, t.assignee, "
                    "       t.created_at, t.completed_at "
                    "FROM task_links l JOIN tasks t ON t.id = l.child_id "
                    "WHERE l.parent_id = ?",
                    (current,),
                ).fetchall()
            except Exception:
                return ""
            for cr in child_rows:
                cid = str(cr["id"] or "")
                if not cid or cid in visited:
                    continue
                visited.add(cid)
                seen_any_child = True
                children.append(cr)
                frontier.append(cid)
        if not seen_any_child:
            return ""  # leaf task → not a pipeline

        # Aggregate (root + children) stats.
        # Pass = explicitly "done" (clean completion). Fail = any other
        # terminal outcome — "archived" (operator-archived, didn't finish
        # normally), crashed, gave_up, timed_out.  "archived" is a
        # terminal status for the gate but a failure for the report —
        # this matches the M5 test expectation that a mix of done +
        # archived children produces "失败: 1".
        all_rows = [root_row] + children
        total = len(all_rows)
        pass_count = 0
        fail_count = 0
        earliest: Optional[int] = None
        latest: Optional[int] = None
        for r in all_rows:
            st = str(r["status"] or "")
            if st == "done":
                pass_count += 1
            elif st in ("archived", "crashed", "gave_up", "timed_out"):
                fail_count += 1
            for col in ("created_at", "completed_at"):
                v = r[col]
                if isinstance(v, int):
                    if earliest is None or v < earliest:
                        earliest = v
                    if latest is None or v > latest:
                        latest = v
        elapsed_str = _format_duration(earliest, latest)

        title = str(root_row["title"] or root_id)
        sep = "━" * 22
        lines: list = [
            f'📊 Pipeline "{title}" 终结报告',
            sep,
            f"总任务: {total}",
            f"通过: {pass_count}",
            f"失败: {fail_count}",
            f"耗时: {elapsed_str}",
            sep,
            "任务明细:",
        ]
        # Cap the per-task detail to 12 rows so a wide pipeline doesn't spam.
        DETAIL_CAP = 12
        detail_rows = all_rows[:DETAIL_CAP]
        for r in detail_rows:
            st = str(r["status"] or "")
            icon = "✔" if st in ("done", "archived") else (
                "✖" if st in ("crashed", "gave_up", "timed_out") else "○"
            )
            tid = str(r["id"] or "?")
            ttitle = str(r["title"] or "")[:40]
            who = str(r["assignee"] or "")[:20]
            suffix = f" ({who})" if who else ""
            lines.append(f"  {icon} {tid} {ttitle}{suffix}")
        if total > DETAIL_CAP:
            lines.append(f"  …及其他 {total - DETAIL_CAP} 个任务")

        # Next-step suggestion — heuristic only, never lies about reality.
        if fail_count == 0:
            suggestion = "所有任务通过,可继续下一阶段或收尾。"
        elif fail_count == total:
            suggestion = "全部失败,建议人工排查根因后重试。"
        else:
            suggestion = f"存在 {fail_count} 个失败任务,建议查看 hermes kanban show {root_id}。"

        lines.append("下一步建议: " + suggestion)
        return "\n".join(lines)
    except Exception as exc:
        logger.debug(
            "kanban watchers: build_pipeline_summary failed for %s: %s",
            root_id, exc,
        )
        return ""


def _format_duration(
    earliest: Optional[int], latest: Optional[int],
) -> str:
    """Render an elapsed span as a short Chinese-style string.

    Returns ``"<N>s"``, ``"<N>m"``, ``"<N>h<N>m"``, or ``"(无)"`` when
    timestamps are missing.  Pure formatting helper — never raises.
    """
    if earliest is None or latest is None or latest < earliest:
        return "(无)"
    delta = int(latest) - int(earliest)
    if delta < 60:
        return f"{delta}s"
    if delta < 3600:
        return f"{delta // 60}m"
    hours = delta // 3600
    mins = (delta % 3600) // 60
    return f"{hours}h{mins}m"


def _build_pipeline_report(conn: sqlite3.Connection, root_id: str) -> str:
    """Internal: assemble the report only when all three guards pass.

    This is the single chokepoint called from the ``kind == "completed"``
    branch in the delivery path.  Encapsulates the "is_root + all_terminal +
    non-empty summary" triple-guard so the call site stays a one-liner and
    so tests can verify each guard independently via the lower-level
    helpers.
    """
    if conn is None or not root_id:
        return ""
    if not _is_root_task(conn, root_id):
        return ""
    if not _all_descendants_terminal(conn, root_id):
        return ""
    return build_pipeline_summary(root_id, conn)


def _has_children(conn: sqlite3.Connection, task_id: str) -> bool:
    """True iff *task_id* has at least one outgoing ``task_links`` row.

    Used by ``classify_event_severity`` to distinguish "leaf completion"
    from "pipeline root completion" without needing a recursive walk.
    Returns ``False`` on missing id / conn / DB error.
    """
    if not task_id or conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _has_parents(conn: sqlite3.Connection, task_id: str) -> bool:
    """True iff *task_id* has at least one incoming ``task_links`` row."""
    if not task_id or conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM task_links WHERE child_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def classify_event_severity(event: Any) -> str:
    """Layer-1 severity classifier — M1.

    Pure function over an event dict (or anything dict-like).  Returns one
    of ``{"P0", "P1", "P2"}`` per the policy table in DESIGN.md §2.3:

      * ``blocked`` + reason startswith ``"review-required:"`` → ``P2``
      * ``blocked`` + first time → ``P0``; duplicate reason → ``P1``
      * ``crashed`` / ``gave_up`` / ``timed_out`` → ``P0`` for first 2
        occurrences, ``P1`` for 3rd+ (read from ``_prior_failure_count``)
      * ``completed`` standalone (no parents, no children) → ``P0``
      * ``completed`` leaf (has parents, no children) → ``P1``
      * ``completed`` pipeline root (no parents, has children) → ``P0``
      * ``completed`` intermediate (has parents AND children) → ``P2``
      * ``protocol_violation`` → ``P2``
      * lifecycle noise (``created``, ``claimed``, ``spawned``,
        ``promoted``, ``unblocked``, ``heartbeat``) → ``P2``
      * unknown / non-dict / empty → ``P2`` (silent)

    The ``_has_parents`` / ``_has_children`` keys, when present, short-
    circuit the DB lookup so callers that already know the topology can
    avoid opening a kanban connection.  When absent and *event* carries a
    ``conn`` field, we resolve the topology from the DB.  Otherwise we
    conservatively assume "no parents, no children" — the same shape used
    by M1 verify tests when they pass ``_has_parents`` / ``_has_children``
    explicitly.
    """
    if not isinstance(event, dict):
        return "P2"
    kind = str(event.get("kind") or "").strip().lower()
    reason = str(event.get("reason") or "")
    task_id = str(event.get("task_id") or "")

    # --- blocked ----------------------------------------------------------
    if kind == "blocked":
        if reason.startswith("review-required:"):
            return "P2"
        prior = event.get("_prior_reasons") or []
        if reason and reason in prior:
            return "P1"
        return "P0"

    # --- crashed / gave_up / timed_out ------------------------------------
    if kind in ("crashed", "gave_up", "timed_out"):
        prior_failures = int(event.get("_prior_failure_count") or 0)
        return "P0" if prior_failures < 2 else "P1"

    # --- completed --------------------------------------------------------
    if kind == "completed":
        has_parents = event.get("_has_parents")
        has_children = event.get("_has_children")
        # Resolve from conn when caller didn't pre-compute the topology.
        if has_parents is None or has_children is None:
            conn = event.get("conn")
            if isinstance(conn, sqlite3.Connection) and task_id:
                try:
                    if has_parents is None:
                        has_parents = _has_parents(conn, task_id)
                    if has_children is None:
                        has_children = _has_children(conn, task_id)
                except Exception:
                    has_parents = bool(has_parents)
                    has_children = bool(has_children)
            else:
                # No conn → can't tell.  Default to standalone (P0) so the
                # standalone-completion notification still pushes.
                has_parents = bool(has_parents)
                has_children = bool(has_children)
        if has_parents and has_children:
            return "P2"  # intermediate — silent, parent will surface report
        if has_parents and not has_children:
            return "P1"  # leaf — quieter than standalone
        if not has_parents and has_children:
            return "P0"  # pipeline root — full report
        return "P0"      # standalone

    # --- protocol_violation ----------------------------------------------
    if kind == "protocol_violation":
        return "P2"

    # --- lifecycle noise --------------------------------------------------
    if kind in (
        "created", "claimed", "spawned", "promoted",
        "unblocked", "heartbeat",
    ):
        return "P2"

    # --- unknown / empty --------------------------------------------------
    return "P2"


def _filter_event_for_push(
    event: Any,
    floor: Optional[str] = None,
    overrides: Optional[dict] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> tuple:
    """Layer-2 push filter — combines severity + user floor + overrides.

    Returns ``(should_push, effective_severity)``.  Wraps
    :func:`classify_event_severity` and the ``should_push`` /
    ``effective_severity`` helpers from ``gateway.notification_preferences``
    so the delivery path can do::

        ok, sev = _filter_event_for_push(ev, floor, overrides, conn=conn)
        if not ok:
            continue
        …send…

    ``floor`` and ``overrides`` are optional — when ``None`` we fall back
    to the user's configured defaults (``load_user_floor`` /
    ``load_user_overrides``).  Importing the preferences module is wrapped
    in a try/except so this function is usable in tests that don't have
    the YAML config present.
    """
    if not isinstance(event, dict):
        return False, "P2"
    # Inject conn so classify_event_severity can resolve topology lazily
    # without callers having to plumb it themselves.
    if conn is not None and "conn" not in event:
        try:
            event = {**event, "conn": conn}
        except Exception:
            pass
    sev = classify_event_severity(event)
    # Lazy import — keeps the module importable in slim test envs.
    try:
        from gateway.notification_preferences import (
            effective_severity as _eff,
            should_push as _should_push,
        )
    except Exception as exc:
        logger.debug(
            "kanban watchers: notification_preferences import failed: %s",
            exc,
        )
        return False, sev
    event_kind = str(event.get("kind") or "")
    effective = _eff(event, sev, overrides or {})
    ok = _should_push(
        effective, floor or "normal", overrides or {},
        event_type=event_kind,
    )
    return bool(ok), effective


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    # Lazy-initialised task-loop engine (replaces class-level mutable defaults).
    # Renamed from ``_epoch_engine`` for task-loop terminology. The old
    # name is kept as a class attribute alias so any external reader still
    # works during the migration window.
    _task_loop_engine: Optional[TaskLoopEngine] = None
    _epoch_engine: Optional[TaskLoopEngine] = None  # deprecated alias

    @property
    def task_loop_engine(self) -> TaskLoopEngine:
        """Lazily create the per-instance :class:`TaskLoopEngine`."""
        if self._task_loop_engine is None:
            self._task_loop_engine = TaskLoopEngine()
            # Mirror to the legacy attribute so the older read path still works.
            self._epoch_engine = self._task_loop_engine
        return self._task_loop_engine

    @property
    def epoch_engine(self) -> TaskLoopEngine:
        """Deprecated alias for :attr:`task_loop_engine`."""
        return self.task_loop_engine

    # ---------------------------------------------------------------------
    # Persistent + in-memory board → user-source lookup (3-tier)
    # ---------------------------------------------------------------------
    # In-memory cache of ``{board_slug: (platform, chat_id)}`` — the
    # legacy back-stop that the persistent table sits on top of.  Kept
    # as a class attribute so it survives across method calls within a
    # single GatewayRunner instance but resets per process.
    _kanban_last_user_source: dict = {}

    # ---------------------------------------------------------------------
    # Phase 2: orchestrator callback decomposition
    # ---------------------------------------------------------------------
    # The ``_kanban_orchestrator_callback`` was extracted into small,
    # mockable helpers so individual rules can be unit-tested.  Each
    # helper has a single responsibility; ``_kanban_orchestrator_callback``
    # below is the orchestrator that ties them together with the
    # rule 1 / rule 2 / rule 3 trigger logic.
    # ---------------------------------------------------------------------

    def _kanban_notifier_inject_enabled(self, kanban_cfg: dict) -> bool:
        """Return ``True`` when the notifier should fire
        ``_kanban_inject_event`` after the user-facing text notification.

        Defaults to ``False`` because live event injection is opt-in:
        most deployments only need the text reply, and LLM-context
        pollution from internal events is a real cost.
        """
        return bool(kanban_cfg.get("notifier_inject", False))

    def _kanban_lookup_board_owner(
        self,
        board_slug: str,
        *,
        db_mod=None,
        fallback_sub: Optional[dict] = None,
    ) -> Optional[tuple[str, str]]:
        """Resolve which ``(platform, chat_id)`` should receive board
        traffic.  Three-tier lookup, most-recently-correct first:

        1. **Persistent table** (``kanban_db.get_board_owner``) — survives
           process restarts and is shared across gateway instances.
        2. **In-memory cache** (``self._kanban_last_user_source``) — fast,
           but per-process and lost on restart.
        3. **Subscription fallback** — the notifier subscription's own
           coordinates, used when neither the DB nor the cache has a
           row yet (freshly subscribed user).

        DB errors are logged and fall through to the next tier rather
        than crashing the notifier tick.
        """
        cache: dict = getattr(self, "_kanban_last_user_source", {}) or {}
        if db_mod is None:
            from hermes_cli import kanban_db as _default_db_mod
            db_mod = _default_db_mod

        # Tier 1: persistent table
        try:
            conn = db_mod.connect(board=board_slug)
            try:
                owner = None
                # ``get_board_owner`` is the future-proof persistent-table
                # lookup.  It may not exist on older kanban_db builds; the
                # AttributeError fall-through is intentional — we degrade to
                # the in-memory cache rather than break the notifier tick.
                get_owner = getattr(db_mod, "get_board_owner", None)
                if callable(get_owner):
                    owner = get_owner(conn, board_slug)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            if owner and owner[0]:
                return owner
        except Exception as exc:
            logger.debug(
                "kanban board-owner lookup: persistent table miss for %s: %s",
                board_slug, exc,
            )

        # Tier 2: in-memory cache
        cached = cache.get(board_slug)
        if cached and cached[0]:
            return cached

        # Tier 3: subscription fallback
        if fallback_sub:
            sp = (fallback_sub.get("platform") or "").strip()
            sch = (fallback_sub.get("chat_id") or "").strip()
            if sp and sch:
                return (sp.lower(), sch)
        return None

    async def _kanban_inject_event(
        self,
        *,
        event,
        task,
        board_slug: str,
        sub: dict,
    ) -> None:
        """Build a synthetic ``MessageEvent`` from a terminal kanban
        event and dispatch it through ``_handle_message``.

        The synthetic event is marked ``internal=True`` and
        ``system_session=True`` so the agent runs in a dedicated
        session context (like a cron job) and the user does NOT see the
        agent's internal reasoning as a reply.

        Three-tier source resolution via :meth:`_kanban_lookup_board_owner`
        so the persistent table beats the in-memory cache beats the
        subscription's own coordinates.
        """
        from gateway.config import Platform as _Platform
        from gateway.platforms.base import MessageEvent, SessionSource

        kind = getattr(event, "kind", "unknown")
        payload = getattr(event, "payload", None) or {}

        # Resolve (platform, chat_id) — 3-tier.
        owner = self._kanban_lookup_board_owner(
            board_slug, fallback_sub=sub,
        )
        if not owner or not owner[0]:
            logger.debug(
                "kanban inject: no source for board %s, skipping", board_slug,
            )
            return
        plat_str, chat_id = owner
        plat_str = (plat_str or "").lower()
        if not plat_str or not chat_id:
            return

        try:
            platform = _Platform(plat_str)
        except ValueError:
            logger.debug(
                "kanban inject: invalid platform %s for board %s",
                plat_str, board_slug,
            )
            return

        # Build the synthetic event text.
        title = (getattr(task, "title", None) or sub.get("task_id", ""))[:120]
        assignee = getattr(task, "assignee", None) or "unknown"
        task_id = getattr(task, "id", None) or sub.get("task_id", "?")

        # Kind → header / detail.
        kind_upper = kind.upper()
        if kind == "completed":
            detail = str(payload.get("summary") or getattr(task, "result", "") or "")[:200]
            if not detail:
                detail = "(no summary)"
            header = f"COMPLETED — {title}"
        elif kind == "blocked":
            detail = str(payload.get("reason") or "no reason given")[:200]
            header = f"BLOCKED — {title}"
        elif kind == "gave_up":
            detail = str(payload.get("error") or "no error details")[:200]
            header = f"GAVE_UP — {title}"
        elif kind == "crashed":
            detail = "worker crashed (pid gone)"
            header = f"CRASHED — {title}"
        elif kind == "timed_out":
            limit = int(payload.get("limit_seconds") or 0)
            detail = f"timed out (max_runtime={limit}s)"
            header = f"TIMED_OUT — {title}"
        else:
            detail = json.dumps(payload)[:200] if payload else "(no detail)"
            header = f"{kind_upper} — {title}"

        text = (
            f"[KANBAN-EVENT] {kind} | task: {task_id} | "
            f"board: {board_slug} | assignee: {assignee}\n"
            f"## {header}\n"
            f"{detail}\n"
            f"metadata: task_id={task_id} assignee={assignee}"
        )

        source = SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type="private",
            user_id="system",
            user_name="kanban-notifier",
        )

        synthetic = MessageEvent(
            text=text,
            source=source,
            internal=True,
            system_session=True,
        )

        try:
            await self._handle_message(synthetic)
        except Exception as exc:
            logger.warning(
                "kanban inject: _handle_message failed for %s: %s",
                board_slug, exc,
            )

    # ---------------------------------------------------------------------
    # Phase 2 helper methods (rule-driven orchestrator callback)
    # ---------------------------------------------------------------------

    def _scan_candidate_boards(self, allowlist) -> list:
        """Discover boards to consider for task-loop detection.

        Honours ``allowlist`` when non-empty; otherwise enumerates every
        non-hidden board directory under ``~/.hermes/kanban/boards/`` plus
        the default board.  Independent of subscription tables — detection
        must scan boards whether or not anyone is listening.
        """
        from hermes_cli import kanban_db as _kb
        import os as _os

        if allowlist and isinstance(allowlist, list):
            return list(allowlist)
        boards_dir = _os.path.expanduser("~/.hermes/kanban/boards")
        candidates: list = []
        if _os.path.isdir(boards_dir):
            for name in sorted(_os.listdir(boards_dir)):
                if name.startswith("_") or name.startswith("."):
                    continue
                if _os.path.isdir(_os.path.join(boards_dir, name)):
                    candidates.append(name)
        if _kb.DEFAULT_BOARD not in candidates:
            candidates.append(_kb.DEFAULT_BOARD)
        return candidates

    def _detect_task_loop(
        self,
        slug: str,
        deliveries: list,
        last_event_id: dict,
    ) -> Optional[dict]:
        """Return a stats dict for *slug* if there's something to act on,
        otherwise ``None``.

        The stats dict shape (used by :meth:`_build_task_loop_message` and
        :meth:`_inject_task_loop`):

        - ``in_progress_count`` / ``in_progress_names``
        - ``ready_count`` / ``blocked_count``
        - ``event_details`` (list of recent terminal events)
        - ``has_terminal_events`` (bool)
        - ``current_loop`` / ``MAX_LOOPS``
        - ``max_eid`` (cursor for next tick)

        ``None`` means the board is truly idle — no terminal events,
        no ready tasks, no force-failure — and the orchestrator should
        skip injection entirely.
        """
        from hermes_cli import kanban_db as _kb

        try:
            conn = _kb.connect(board=slug)
            try:
                tasks = _kb.list_tasks(conn, status="running")
                in_progress_count = len(tasks) if tasks else 0
                in_progress_names = [t.id for t in (tasks or [])]

                ready_tasks = _kb.list_tasks(conn, status="ready")
                ready_count = len(ready_tasks) if ready_tasks else 0

                blocked_tasks = _kb.list_tasks(conn, status="blocked")
                blocked_count = len(blocked_tasks) if blocked_tasks else 0

                # Recent terminal events since last tick.
                last_eid = int(last_event_id.get(slug, 0))
                if last_eid > 0:
                    recent_rows = conn.execute(
                        "SELECT te.id, te.task_id, te.kind, te.payload "
                        "FROM task_events te WHERE te.id > ? "
                        "AND te.kind IN ('completed','blocked','crashed','gave_up','timed_out') "
                        "ORDER BY te.id DESC LIMIT 20",
                        (last_eid,),
                    ).fetchall()
                else:
                    cutoff = time.time() - 600
                    recent_rows = conn.execute(
                        "SELECT te.id, te.task_id, te.kind, te.payload "
                        "FROM task_events te WHERE te.created_at > ? "
                        "AND te.kind IN ('completed','blocked','crashed','gave_up','timed_out') "
                        "ORDER BY te.created_at DESC LIMIT 20",
                        (cutoff,),
                    ).fetchall()

                event_details: list = []
                for r in recent_rows:
                    _eid, _tid, _kind, _payload = r
                    _tinfo = conn.execute(
                        "SELECT title, assignee, result FROM tasks WHERE id=?",
                        (_tid,),
                    ).fetchone()
                    _title = _tinfo[0] if _tinfo else "?"
                    _assignee = _tinfo[1] if _tinfo else "?"
                    _summary = ""
                    if _payload:
                        try:
                            _p = json.loads(_payload)
                            _summary = _p.get("summary", "") or ""
                        except Exception:
                            _summary = ""
                    if not _summary and _tinfo and _tinfo[2]:
                        _summary = _tinfo[2][:200]
                    event_details.append({
                        "task_id": _tid,
                        "kind": _kind,
                        "title": _title,
                        "assignee": _assignee,
                        "summary": _summary,
                    })
                has_terminal_events = len(recent_rows) > 0
                max_eid = max((r[0] for r in recent_rows), default=0)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "kanban task_loop detect: board %s scan failed: %s",
                slug, exc,
            )
            return None

        # No action: nothing on the board to react to.
        if not has_terminal_events and not ready_count and not in_progress_count and not blocked_count:
            return None

        return {
            "in_progress_count": in_progress_count,
            "in_progress_names": in_progress_names,
            "ready_count": ready_count,
            "blocked_count": blocked_count,
            "event_details": event_details,
            "has_terminal_events": has_terminal_events,
            "current_loop": int(getattr(self.task_loop_engine, "_task_loop_counts", {}).get(slug, 0)) + 1,
            "MAX_LOOPS": int(
                getattr(self, "_kanban_last_loop_cfg", {}).get("max_loops", 10)
            ),
            "max_eid": max_eid,
        }

    def _has_force_failure(self, board_slug: str, threshold: int) -> bool:
        """Return ``True`` if any non-terminal task on *board_slug* has
        ``consecutive_failures >= threshold``.

        Done tasks are excluded even if their counter wasn't reset — the
        counter is historical once the task is in a final state.
        """
        try:
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect(board=board_slug)
            try:
                row = conn.execute(
                    "SELECT 1 FROM tasks "
                    "WHERE status NOT IN ('done','archived') "
                    "AND consecutive_failures >= ? "
                    "LIMIT 1",
                    (int(threshold),),
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "kanban task_loop force-failure: board %s query failed: %s",
                board_slug, exc,
            )
            return False

    def _board_converged(self, board_slug: str) -> Optional[dict]:
        """Return the convergence ``metrics`` dict for *board_slug* if the
        board has converged, otherwise ``None``.

        Thin wrapper around :func:`compute_board_convergence` that
        handles the DB connection + exception quarantine. Returning the
        full metrics dict (not just a bool) gives the caller access to
        ``resolved``/``total_tasks``/``blocked_ratio``/``resolve_rate``
        for the summary message.  ``None`` on any error so a transient
        DB glitch never causes a spurious convergence injection.
        """
        try:
            from hermes_cli import kanban_db as _kb

            conn = _kb.connect(board=board_slug)
            try:
                metrics = compute_board_convergence(conn)
            finally:
                conn.close()
            return metrics if metrics.get("converged") else None
        except Exception as exc:
            logger.debug(
                "kanban task_loop convergence probe failed for %s: %s",
                board_slug, exc,
            )
            return None

    def _failing_task_ids(self, board_slug: str, threshold: int) -> list:
        """Return ids of non-terminal tasks with
        ``consecutive_failures >= threshold``, sorted by failure count
        descending.

        Worst-broken cards first so the orchestrator message surfaces
        the most pathological case at the top.
        """
        try:
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect(board=board_slug)
            try:
                rows = conn.execute(
                    "SELECT id FROM tasks "
                    "WHERE status NOT IN ('done','archived') "
                    "AND consecutive_failures >= ? "
                    "ORDER BY consecutive_failures DESC, id ASC",
                    (int(threshold),),
                ).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "kanban task_loop failing-ids: board %s query failed: %s",
                board_slug, exc,
            )
            return []

    def _auto_complete_parents(self, board_slug: str) -> list:
        """Promote any *done*-bound parent whose children are all done.

        Returns the list of parent ids that were auto-completed.  This
        is a no-op when no parents are eligible — the returned list is
        empty in that case.
        """
        try:
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect(board=board_slug)
            try:
                rows = conn.execute(
                    "SELECT e.parent_id, t.status "
                    "FROM task_edges e "
                    "JOIN tasks t ON t.id = e.parent_id "
                    "WHERE t.status NOT IN ('done','archived','blocked')"
                ).fetchall()
                completed: list = []
                seen: set = set()
                for parent_id, _status in rows:
                    if parent_id in seen:
                        continue
                    seen.add(parent_id)
                    kids = conn.execute(
                        "SELECT t.status FROM tasks t "
                        "JOIN task_edges e ON e.child_id = t.id "
                        "WHERE e.parent_id = ?",
                        (parent_id,),
                    ).fetchall()
                    if kids and all((k[0] == "done") for k in kids):
                        try:
                            _kb.complete_task(conn, parent_id, summary="auto-completed: all children done")
                            completed.append(parent_id)
                        except Exception:
                            pass
                return completed
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "kanban task_loop auto-complete: board %s failed: %s",
                board_slug, exc,
            )
            return []

    def _build_task_loop_message(
        self,
        slug: str,
        event_details: list,
        stats: dict,
    ) -> str:
        """Build the orchestrator-injection message text.

        Returns a multi-line string starting with ``[URGENT]`` when
        ``stats['force_urgent']`` is true, otherwise without a prefix.
        Includes a "Stuck tasks" line when there are failing task ids,
        truncated to 8 ids with a ``(+N more)`` suffix.

        When ``stats['converged']`` is true, emits a dedicated
        convergence-summary message: 📋 header, resolved/total ratio,
        blocked/resolve rate metrics, and an explicit "all tasks
        complete — wrap up" instruction to the coordinator.  This
        branch takes precedence over the urgent/loop formatting.
        """
        force_urgent = bool(stats.get("force_urgent"))
        threshold = stats.get("force_failure_threshold", 3)
        failing_ids = list(stats.get("failing_task_ids") or [])
        ready_count = int(stats.get("ready_count") or 0)
        in_progress_names = list(stats.get("in_progress_names") or [])
        current_loop = int(stats.get("current_loop") or 1)
        MAX_LOOPS = int(stats.get("MAX_LOOPS") or 10)
        auto_completed = list(stats.get("auto_completed") or [])

        lines: list = []

        # Convergence path — board finished, tell the coordinator to wrap up.
        if stats.get("converged"):
            metrics = stats.get("convergence_metrics") or {}
            resolved = int(metrics.get("resolved") or 0)
            total = int(metrics.get("total_tasks") or 0)
            blocked_ratio = float(metrics.get("blocked_ratio") or 0.0)
            resolve_rate = float(metrics.get("resolve_rate") or 0.0)
            lines.append(
                f"📋 Kanban Board \"{slug}\" — 全部完成"
            )
            lines.append(
                f"Progress: {resolved}/{total} resolved "
                f"(resolve_rate={resolve_rate:.0%}, "
                f"blocked_ratio={blocked_ratio:.0%})"
            )
            vf = int(metrics.get("verification_failed") or 0)
            new_tasks = int(metrics.get("new_tasks_created") or 0)
            if vf or new_tasks:
                lines.append(
                    f"Recent activity window: "
                    f"verification_failed={vf}, remediation_created={new_tasks}"
                )
            if auto_completed:
                lines.append(
                    f"Auto-completed parents: {', '.join(auto_completed)}"
                )
            lines.append("")
            lines.append("--- Orchestrator Instructions ---")
            lines.append(
                "所有任务已完成。请总结全部产出，通知用户，然后 complete 自己的 orchestrator 任务。"
            )
            return "\n".join(lines)

        if force_urgent:
            lines.append(f"[URGENT] Board '{slug}' has stuck tasks (≥ {threshold} failures).")
        else:
            lines.append(f"[Kanban Task Loop #{current_loop}] Workers idle on {slug}.")

        # Event details — count by kind.
        event_kinds: Counter = Counter()
        for ed in event_details or []:
            event_kinds[ed.get("kind", "?")] += 1
        event_summary = (
            ", ".join(f"{k}={v}" for k, v in sorted(event_kinds.items()))
            if event_kinds else "no events"
        )
        lines.append(f"Events this tick: {event_summary}")

        if in_progress_names:
            lines.append(
                f"Running tasks: {len(in_progress_names)} "
                f"({', '.join(in_progress_names[:5])})"
            )
        if ready_count > 0:
            lines.append(f"{ready_count} ready task(s) queued — decompose and dispatch.")
        else:
            lines.append(
                "No ready tasks. Review blocked/crashed tasks and re-decompose if needed."
            )
        if auto_completed:
            lines.append(f"Auto-completed parents: {', '.join(auto_completed)}")

        # Stuck-tasks preview.
        if failing_ids:
            shown = failing_ids[:8]
            more = len(failing_ids) - len(shown)
            tail = f" (+{more} more)" if more > 0 else ""
            lines.append(
                f"Stuck tasks (>= {threshold} failures): "
                f"{', '.join(shown)}{tail}"
            )

        lines.append(f"(loop {current_loop}/{MAX_LOOPS})")

        # Orchestrator instructions.
        lines.append("")
        lines.append("--- Orchestrator Instructions ---")
        lines.append(
            f"Board '{slug}': {ready_count} ready, {len(in_progress_names)} running"
        )
        lines.append("")
        lines.append("As the kanban orchestrator, respond by EXECUTING tools — not by analyzing in text.")
        lines.append("You MUST make at least one tool call this turn (kanban list/show/create/unblock).")
        lines.append("Your text response will NOT be seen by anyone. Only tool results matter.")
        lines.append("")
        if force_urgent:
            lines.append(
                "Triage the stuck tasks first — they are poisoning the loop budget."
            )
            lines.append("Actions to take:")
            lines.append(
                "1. kanban_show on each stuck task; identify why it's failing "
                "(bad input, broken env, repeating the same crash)"
            )
            lines.append(
                "2. Either fix and unblock, or supersede (complete + create clean new)"
            )
            lines.append(
                "3. Then proceed to the regular ready/pending queue if any remain."
            )
        else:
            lines.append("Actions to take:")
            lines.append("1. Check the board — run `kanban list` or `kanban show` on blocked/crashed tasks")
            lines.append("2. For blocked: identify the blocker and decide — re-assign, re-decompose, or unblock")
            lines.append("3. For crashed/gave_up: read failure reason FIRST (kanban_show), then:")
            lines.append("   - Missing precondition (branch not merged, env not ready, deps missing) → create a prep task, then re-dispatch")
            lines.append("   - Bad instructions in body → comment fix and unblock")
            lines.append("   - Context unrecoverable (budget exhausted) → supersede: complete old + create clean new task")
            lines.append("4. For completed: if there are ready/pending items, create the next loop's tasks")
            lines.append("5. Be mindful of budget — don't create too many parallel tasks at once")
            lines.append("6. Only create kanban tasks — the worker system handles execution")
            lines.append("7. Do NOT send messages to the user — results are delivered automatically")
        return "\n".join(lines)

    async def _inject_task_loop(
        self,
        msg_text: str,
        slug: str,
        stats: dict,
    ) -> None:
        """Schedule a fire-and-forget injection of *msg_text* into the
        orchestrator profile's session for *slug*.

        Source resolution reuses :meth:`_kanban_lookup_board_owner`.  The
        synthetic event is marked ``internal=True`` so the user does not
        see the agent's internal reasoning as a reply.
        """
        from gateway.config import Platform as _Platform
        from gateway.platforms.base import MessageEvent, SessionSource

        sub_fallback = {
            "platform": (self._kanban_last_user_source.get(slug, ("", ""))[0]),
            "chat_id": (self._kanban_last_user_source.get(slug, ("", ""))[1]),
        }
        owner = self._kanban_lookup_board_owner(slug, fallback_sub=sub_fallback)
        if not owner or not owner[0]:
            logger.debug(
                "kanban task_loop inject: no source for board %s, skipping", slug,
            )
            return
        plat_str, chat_id = owner
        try:
            platform = _Platform(plat_str)
        except ValueError:
            logger.debug(
                "kanban task_loop inject: invalid platform %s for board %s",
                plat_str, slug,
            )
            return

        source = SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type="private",
            user_id="system",
            user_name="kanban-orchestrator",
        )
        synthetic = MessageEvent(text=msg_text, source=source, internal=True)

        try:
            await self._handle_message(synthetic)
        except Exception as exc:
            logger.warning(
                "kanban task_loop injection failed for %s: %s", slug, exc,
            )

    async def _kanban_orchestrator_callback(
        self,
        deliveries: list[dict],
        kanban_cfg: dict,
    ) -> None:
        """Phase-2 orchestrator callback.  Decides whether to inject an
        orchestrator message for any candidate board and, if so, builds
        the message and schedules the injection.

        Trigger rules (Phase 2):
          Rule 1 — ready queue non-empty AND no terminal events → skip.
          Rule 2 — empty ready AND terminal events → fire normally.
          Rule 3 — any card with ``consecutive_failures >= threshold`` →
                   force fire with ``[URGENT]`` marker.

        Force-failure threshold defaults to 3 and is overridable via
        ``orchestrator_force_failure_threshold`` in the config dict.

        For backward compatibility this method still delegates the
        legacy ``EpochEngine.tick`` body when called via the old
        ``self.task_loop_engine.tick(...)`` path — that path remains the
        source of truth for the inner implementation while this method
        acts as the rule-based dispatcher for new code paths and tests.
        """
        if not kanban_cfg.get("orchestrator_notify"):
            return

        allowlist = kanban_cfg.get("orchestrator_boards") or []
        if not isinstance(allowlist, list):
            allowlist = []
        try:
            candidates = self._scan_candidate_boards(allowlist)
        except Exception as exc:
            logger.debug("kanban orchestrator: scan failed: %s", exc)
            return

        # Cache max_loops so ``_detect_task_loop`` can read it from stats.
        try:
            cfg_max_loops = int(
                kanban_cfg.get("orchestrator_max_loops")
                or kanban_cfg.get("orchestrator_max_epochs", 10)
            )
        except Exception:
            cfg_max_loops = 10
        if not hasattr(self, "_kanban_last_loop_cfg") or self._kanban_last_loop_cfg is None:
            self._kanban_last_loop_cfg = {}
        self._kanban_last_loop_cfg["max_loops"] = cfg_max_loops

        threshold = int(kanban_cfg.get("orchestrator_force_failure_threshold", 3))

        for slug in candidates:
            stats = self._detect_task_loop(
                slug, deliveries, self.task_loop_engine._last_event_id,
            )

            # ── Convergence detection ───────────────────────────────
            # When the board has fully converged (all work done, no
            # pending, no recent failures, no remediation in flight),
            # fire a final summary message to the coordinator so it
            # knows to wrap up.  This must run BEFORE the stats-None
            # short-circuit below: a fully converged board is
            # precisely the case where _detect_task_loop returns ``None``
            # (no ready/running/blocked, no recent terminal events).
            #
            # We deliberately do NOT use ``continue`` to skip the
            # injection — the previous behaviour wrote a
            # ``task_loop_closed`` event but never told the
            # orchestrator, so the coordinator sat idle waiting for
            # work that would never come.  Convergence → inject.
            conv_metrics = self._board_converged(slug)
            if conv_metrics is not None:
                # Build a minimal stats dict if _detect_task_loop skipped
                # this board (the all-done quiescent case).
                if stats is None:
                    stats = {
                        "in_progress_count": 0,
                        "in_progress_names": [],
                        "ready_count": 0,
                        "blocked_count": 0,
                        "event_details": [],
                        "has_terminal_events": False,
                        "current_loop": int(
                            self.task_loop_engine._task_loop_counts.get(slug, 0)
                        ) + 1,
                        "MAX_LOOPS": cfg_max_loops,
                        "max_eid": 0,
                    }
                stats["converged"] = True
                stats["convergence_metrics"] = conv_metrics
                stats["auto_completed"] = (
                    self._auto_complete_parents(slug)
                )
                stats["force_urgent"] = False
                stats["force_failure_threshold"] = threshold
                stats["failing_task_ids"] = []
                msg_text = self._build_task_loop_message(
                    slug, stats["event_details"], stats,
                )
                logger.info(
                    "kanban orchestrator callback: board %s converged "
                    "(resolved=%d/%d, blocked_ratio=%.2f, "
                    "resolve_rate=%.2f) — injecting final summary",
                    slug, conv_metrics.get("resolved", 0),
                    conv_metrics.get("total_tasks", 0),
                    conv_metrics.get("blocked_ratio", 0.0),
                    conv_metrics.get("resolve_rate", 0.0),
                )
                _coro_or_call = self._inject_task_loop(msg_text, slug, stats)
                if asyncio.iscoroutine(_coro_or_call):
                    asyncio.ensure_future(_coro_or_call)
                # Update cursor + cooldown so we don't re-fire on the
                # same convergence.  Continue to the next slug.
                self.task_loop_engine._last_event_id[slug] = max(
                    int(self.task_loop_engine._last_event_id.get(slug, 0)),
                    int(stats.get("max_eid") or 0),
                )
                self.task_loop_engine._cooldowns[slug] = time.monotonic()
                self.task_loop_engine._stale_counts[slug] = 0
                continue

            if stats is None:
                continue  # nothing to act on

            # Cooldown gate: skip if we just fired for this board.
            cooldown = float(kanban_cfg.get("orchestrator_cooldown_seconds", 30))
            last_now = self.task_loop_engine._cooldowns.get(slug, 0)
            if time.monotonic() - last_now < cooldown:
                continue

            force_urgent = self._has_force_failure(slug, threshold)
            failing_ids = (
                self._failing_task_ids(slug, threshold) if force_urgent else []
            )

            # Rule 1: ready non-empty + no terminal + no force → skip.
            if (
                stats["ready_count"] > 0
                and not stats["has_terminal_events"]
                and not force_urgent
            ):
                continue

            # Auto-complete parents whose children are all done.
            auto_completed = self._auto_complete_parents(slug)
            stats["auto_completed"] = auto_completed
            stats["force_urgent"] = force_urgent
            stats["force_failure_threshold"] = threshold
            stats["failing_task_ids"] = failing_ids

            # Build the message and inject.
            msg_text = self._build_task_loop_message(slug, stats["event_details"], stats)
            # Tests inject a synchronous ``_inject_task_loop`` recorder via
            # ``patch.object(runner, "_inject_task_loop", ...)`` — keep the
            # call fire-and-forget so the recorded method (sync or async)
            # is what executes.  ``ensure_future`` accepts both coroutines
            # and plain callables, so the sync test double gets called.
            _coro_or_call = self._inject_task_loop(msg_text, slug, stats)
            if asyncio.iscoroutine(_coro_or_call):
                asyncio.ensure_future(_coro_or_call)

            # Update cursor and cooldown.
            self.task_loop_engine._last_event_id[slug] = max(
                int(self.task_loop_engine._last_event_id.get(slug, 0)),
                int(stats.get("max_eid") or 0),
            )
            self.task_loop_engine._cooldowns[slug] = time.monotonic()
            # Reset stale counter — something real happened.
            self.task_loop_engine._stale_counts[slug] = 0

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        For each subscription row, fetches ``task_events`` newer than the
        stored cursor with kind in the terminal set (``completed``,
        ``blocked``, ``gave_up``, ``crashed``, ``timed_out``). Sends one
        message per new event to ``(platform, chat_id, thread_id)``,
        then advances the cursor. When a task reaches a terminal state
        (``completed`` / ``archived``), the subscription is removed.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Subscriptions live inside each board's own DB and cannot
        cross boards, so delivery semantics are unchanged — this is
        purely a fan-out of the single-DB poll.
        """
        # Gate: only the dispatch-owning gateway opens kanban DBs for notifier polling.
        # Non-dispatch gateways have no subscriptions to deliver — all kanban state lives
        # in the dispatch owner's per-board DBs. This prevents N-gateway -shm contention.
        # TODO: gate per-board when per-board dispatcher_owner tracking lands.
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban notifier: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban notifier: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban notifier: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban notifier: disabled via config kanban.dispatch_in_gateway=false"
            )
            return
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out")
        # Subscriptions are removed only when the task reaches a truly final
        # status (done / archived). We used to also unsub on any terminal
        # event kind (gave_up / crashed / timed_out / blocked), but that
        # silently dropped the user out of the loop whenever the dispatcher
        # respawned the task: a worker that crashes, gets reclaimed, runs
        # again, and crashes a second time would only notify on the first
        # crash because the subscription was deleted after the first event.
        # Same shape as the reblock-after-unblock cycle that PR #22941
        # fixed for `blocked`. Keeping the subscription alive until the
        # task is genuinely done lets the cursor (advanced atomically by
        # claim_unseen_events_for_sub) handle dedup, and any retry-loop
        # event reaches the user.
        # Per-subscription send-failure counter. Adapter.send raising
        # means the chat is dead (deleted, bot kicked, etc.) — after N
        # consecutive send failures the sub is dropped so we don't spin
        # against a dead chat every 5 seconds forever.
        MAX_SEND_FAILURES = 3
        sub_fail_counts: dict[tuple, int] = getattr(
            self, "_kanban_sub_fail_counts", {}
        )
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None)
        if not notifier_profile:
            notifier_profile = self._active_profile_name()
            self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        while self._running:
            try:
                def _collect():
                    deliveries: list[dict] = []
                    active_platforms = {
                        getattr(platform, "value", str(platform)).lower()
                        for platform in self.adapters.keys()
                    }
                    if not active_platforms:
                        logger.debug("kanban notifier: no connected adapters; skipping tick")
                        return deliveries

                    # Enumerate every board on disk, but poll each resolved DB
                    # path once. Multiple slugs can point at the same DB when
                    # HERMES_KANBAN_DB pins the board path; without this guard
                    # one gateway could collect the same subscription/event
                    # more than once before advancing the cursor.
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: dict[str, set[str]] = {}
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            # When multiple slugs map to the same DB, we still
                            # need to process events for EACH slug separately
                            # — skip only truly duplicate (db_path, slug) pairs.
                            if slug in seen_db_paths.get(resolved_db_path, set()):
                                logger.debug(
                                    "kanban notifier: skipping duplicate board slug %s for DB %s",
                                    slug, resolved_db_path,
                                )
                                continue
                        seen_db_paths.setdefault(resolved_db_path, set()).add(slug)
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                            continue
                        try:
                            # `connect()` runs the schema + idempotent migration
                            # on first open per process, so an explicit
                            # `init_db()` here would be redundant. Worse:
                            # `init_db()` deliberately busts the per-process
                            # cache and re-runs the migration on a *second*
                            # connection, which races the first and used to
                            # log a benign but noisy `duplicate column name`
                            # traceback (and intermittent "database is locked"
                            # — issue #21378) on every gateway start against
                            # a legacy DB. `_add_column_if_missing` now
                            # tolerates that race, but we still skip the
                            # redundant call to avoid the wasted work.
                            subs = _kb.list_notify_subs(conn)
                            if not subs:
                                logger.debug("kanban notifier: board %s has no subscriptions", slug)
                            for sub in subs:
                                owner_profile = sub.get("notifier_profile") or None
                                if owner_profile and owner_profile != notifier_profile:
                                    logger.debug(
                                        "kanban notifier: subscription for %s owned by profile %s; current profile %s skipping",
                                        sub.get("task_id"), owner_profile, notifier_profile,
                                    )
                                    continue
                                platform = (sub.get("platform") or "").lower()
                                if platform not in active_platforms:
                                    logger.debug(
                                        "kanban notifier: subscription for %s on %s skipped; adapter not connected",
                                        sub.get("task_id"), platform or "<missing>",
                                    )
                                    continue
                                old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                    conn,
                                    task_id=sub["task_id"],
                                    platform=sub["platform"],
                                    chat_id=sub["chat_id"],
                                    thread_id=sub.get("thread_id") or "",
                                    kinds=TERMINAL_KINDS,
                                )
                                if not events:
                                    continue
                                task = _kb.get_task(conn, sub["task_id"])
                                # Platform-scoped routing: if the task has a
                                # last_mutated_platform, only deliver to
                                # subscriptions matching that platform.
                                # NULL (legacy or CLI-created) = broadcast.
                                task_platform = (
                                    getattr(task, "last_mutated_platform", None)
                                    if task else None
                                )
                                if task_platform and platform != task_platform.lower():
                                    logger.debug(
                                        "kanban notifier: skipping %s on %s for %s; task.platform=%s",
                                        sub["task_id"], platform, slug,
                                        task_platform,
                                    )
                                    continue
                                logger.debug(
                                    "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                    len(events), sub["task_id"], slug, old_cursor, cursor,
                                )
                                deliveries.append({
                                    "sub": sub,
                                    "old_cursor": old_cursor,
                                    "cursor": cursor,
                                    "events": events,
                                    "task": task,
                                    "board": slug,
                                })
                        finally:
                            conn.close()
                    return deliveries

                deliveries = await asyncio.to_thread(_collect)
                for d in deliveries:
                    sub = d["sub"]
                    task = d["task"]
                    board_slug = d.get("board")
                    platform_str = (sub["platform"] or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string; skip and advance cursor so
                        # we don't replay forever.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        continue
                    adapter = self.adapters.get(plat)
                    if adapter is None:
                        logger.debug(
                            "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                            platform_str, sub["task_id"],
                        )
                        await asyncio.to_thread(
                            self._kanban_rewind,
                            sub,
                            d["cursor"],
                            d.get("old_cursor", 0),
                            board_slug,
                        )
                        continue
                    title = (task.title if task else sub["task_id"])[:120]
                    for ev in d["events"]:
                        kind = ev.kind
                        # Identity prefix: attribute terminal pings to the
                        # worker that did the work. Makes fleets (where one
                        # chat subscribes to many tasks) legible at a glance.
                        who = (task.assignee if task and task.assignee else None)
                        tag = f"@{who} " if who else ""
                        if kind == "completed":
                            # ── M5: Pipeline terminal report ────────────────
                            # When the completed task is a pipeline root and all
                            # its descendants are terminal, replace the simple
                            # "✔ done" message with a structured 终结报告.
                            try:
                                from hermes_cli import kanban_db as _kb
                                p_conn = _kb.connect(board=board_slug)
                                try:
                                    m5_report = _build_pipeline_report(
                                        p_conn, sub["task_id"],
                                    )
                                finally:
                                    p_conn.close()
                            except Exception:
                                m5_report = ""
                            if m5_report:
                                msg = m5_report
                            else:
                                # Fall through to the normal completed message.
                                # Prefer the run's summary (the worker's
                                # intentional human-facing handoff, carried
                                # in the event payload), then fall back to
                                # task.result for legacy rows written before
                                # runs shipped.
                                handoff = ""
                                payload_summary = None
                                if ev.payload and ev.payload.get("summary"):
                                    payload_summary = str(ev.payload["summary"])
                                if payload_summary:
                                    lines = payload_summary.strip().splitlines()
                                    h = lines[0][:200] if lines else payload_summary[:200]
                                    handoff = f"\n{h}"
                                elif task and task.result:
                                    lines = task.result.strip().splitlines()
                                    r = lines[0][:160] if lines else task.result[:160]
                                    handoff = f"\n{r}"
                                msg = (
                                    f"✔ {tag}Kanban {sub['task_id']} done"
                                    f" — {title}{handoff}"
                                )
                        elif kind == "blocked":
                            reason = ""
                            if ev.payload and ev.payload.get("reason"):
                                reason = f": {str(ev.payload['reason'])[:160]}"
                            msg = f"⏸ {tag}Kanban {sub['task_id']} blocked{reason}"
                            # M3: append structured option block so the user
                            # can reply with a single digit (1-N) to unblock
                            # the task. Skipped for review-required: blocks
                            # (those need human eyes, not numbered options)
                            # and when block.auto_options is disabled.
                            try:
                                from gateway.block_options import (
                                    build_options_suffix as _build_opt_suffix,
                                    classify_block_reason as _classify_reason,
                                    build_block_options as _build_options,
                                    is_block_options_enabled as _opt_enabled,
                                    is_auto_options_enabled as _auto_opt,
                                    is_review_required as _is_review_req,
                                    register_block_invite as _register_invite,
                                )
                                full_reason = ""
                                if ev.payload and ev.payload.get("reason"):
                                    full_reason = str(ev.payload["reason"])
                                if (
                                    _opt_enabled()
                                    and _auto_opt()
                                    and not _is_review_req(full_reason)
                                ):
                                    suffix = _build_opt_suffix(full_reason)
                                    if suffix:
                                        msg = msg + "\n" + suffix
                                        # P0 fix: register the invite so the
                                        # reply-side hook can safely interpret
                                        # a bare digit from this chat as a
                                        # block decision (without this, group
                                        # chat "123" / "5432" / "404" would
                                        # trigger accidental unblock).
                                        try:
                                            masked = full_reason
                                            try:
                                                from gateway.block_options import (
                                                    mask_credentials as _mask_creds,
                                                )
                                                masked = _mask_creds(full_reason)
                                            except Exception:
                                                pass
                                            tpl = _classify_reason(masked)
                                            opt_result = _build_options(masked, tpl)
                                            num_opts = len(opt_result.options)
                                            invite_key = (
                                                f"{platform_str}|{sub['chat_id']}|"
                                                f"{sub.get('thread_id') or ''}|"
                                                f"{sub['task_id']}"
                                            )
                                            # Lazy-init the registry on the
                                            # runner so the library stays
                                            # decoupled from __init__ ordering.
                                            _invite_store = getattr(
                                                self, "_pending_block_invites", None,
                                            )
                                            if not isinstance(_invite_store, dict):
                                                _invite_store = {}
                                                self._pending_block_invites = (
                                                    _invite_store
                                                )
                                            _register_invite(
                                                store=_invite_store,
                                                session_key=invite_key,
                                                task_id=sub["task_id"],
                                                reason=full_reason,
                                                num_options=num_opts,
                                            )
                                        except Exception:
                                            # The push is still useful even
                                            # if invite registration fails —
                                            # the user can fall back to the
                                            # explicit /kanban unblock path.
                                            pass
                            except Exception as _opt_exc:
                                # Never let the option-blocker break the push
                                # path — the bare text is still actionable.
                                logger.debug(
                                    "kanban notifier: block_options suffix "
                                    "build failed for %s: %s",
                                    sub["task_id"], _opt_exc,
                                )
                        elif kind == "gave_up":
                            err = ""
                            if ev.payload and ev.payload.get("error"):
                                err = f"\n{str(ev.payload['error'])[:200]}"
                            msg = (
                                f"✖ {tag}Kanban {sub['task_id']} gave up "
                                f"after repeated spawn failures{err}"
                            )
                        elif kind == "crashed":
                            msg = (
                                f"✖ {tag}Kanban {sub['task_id']} worker crashed "
                                f"(pid gone); dispatcher will retry"
                            )
                        elif kind == "timed_out":
                            limit = 0
                            if ev.payload and ev.payload.get("limit_seconds"):
                                limit = int(ev.payload["limit_seconds"])
                            msg = (
                                f"⏱ {tag}Kanban {sub['task_id']} timed out "
                                f"(max_runtime={limit}s); will retry"
                            )
                        else:
                            continue
                        metadata: dict[str, Any] = {}
                        if sub.get("thread_id"):
                            metadata["thread_id"] = sub["thread_id"]
                        sub_key = (
                            sub["task_id"], sub["platform"],
                            sub["chat_id"], sub.get("thread_id") or "",
                        )
                        try:
                            await adapter.send(
                                sub["chat_id"], msg, metadata=metadata,
                            )
                            logger.debug(
                                "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                                kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                            )
                            # After delivering the text notification, surface
                            # any artifact paths the worker referenced in
                            # ``kanban_complete(summary=..., artifacts=[...])``
                            # (or the legacy ``result`` field) as native
                            # uploads. ``extract_local_files`` finds bare
                            # absolute paths in the summary;
                            # ``send_document`` / ``send_image_file`` uploads
                            # them. Only fires on the ``completed`` event so
                            # we never spam attachments on retries.
                            if kind == "completed":
                                try:
                                    await self._deliver_kanban_artifacts(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        event_payload=getattr(ev, "payload", None),
                                        task=task,
                                    )
                                except Exception as art_exc:
                                    logger.debug(
                                        "kanban notifier: artifact delivery for %s failed: %s",
                                        sub["task_id"], art_exc,
                                    )
                            # Live event injection (opt-in via
                            # ``notifier_inject: true``): fire-and-forget
                            # synthetic MessageEvent so the agent sees the
                            # terminal event in its own session context.
                            # Both adapter.send (user-visible text) and
                            # inject (agent context) are independent deliveries.
                            if (
                                self._kanban_notifier_inject_enabled(kanban_cfg)
                                and board_slug
                            ):
                                asyncio.ensure_future(
                                    self._kanban_inject_event(
                                        event=ev,
                                        task=task,
                                        board_slug=str(board_slug),
                                        sub=sub,
                                    )
                                )
                            # Reset the failure counter on success.
                            sub_fail_counts.pop(sub_key, None)
                        except Exception as exc:
                            fails = sub_fail_counts.get(sub_key, 0) + 1
                            sub_fail_counts[sub_key] = fails
                            logger.warning(
                                "kanban notifier: send failed for %s on %s "
                                "(attempt %d/%d): %s",
                                sub["task_id"], platform_str, fails,
                                MAX_SEND_FAILURES, exc,
                            )
                            if fails >= MAX_SEND_FAILURES:
                                logger.warning(
                                    "kanban notifier: dropping subscription "
                                    "%s on %s after %d consecutive send failures",
                                    sub["task_id"], platform_str, fails,
                                )
                                await asyncio.to_thread(self._kanban_unsub, sub, board_slug)
                                sub_fail_counts.pop(sub_key, None)
                            else:
                                await asyncio.to_thread(
                                    self._kanban_rewind,
                                    sub,
                                    d["cursor"],
                                    d.get("old_cursor", 0),
                                    board_slug,
                                )
                            # Rewind the pre-send claim on transient failure so
                            # a later tick can retry. After too many failures,
                            # dropping the subscription is the terminal action.
                            break
                    else:
                        # All events delivered; advance cursor. The cursor
                        # is the dedup mechanism — it prevents re-delivery
                        # of the same event on subsequent ticks.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        # Unsubscribe only when the task has reached a truly
                        # final status (done / archived). For blocked /
                        # gave_up / crashed / timed_out the subscription is
                        # kept alive so the user gets notified again if the
                        # dispatcher respawns the task and it cycles into the
                        # same state. See the longer comment on TERMINAL_KINDS
                        # above for the failure mode this prevents.
                        task_terminal = task and task.status in {"done", "archived"}
                        if task_terminal:
                            await asyncio.to_thread(
                                self._kanban_unsub, sub, board_slug,
                            )
                # ── Orchestrator task-loop callback ────────────────────────
                # Runs once per tick, independent of whether there were
                # any deliveries. Fire-and-forget so it doesn't block
                # the notifier loop (and thus user message delivery).
                if kanban_cfg.get("orchestrator_notify"):
                    asyncio.ensure_future(
                        self._kanban_orchestrator_callback(deliveries, kanban_cfg)
                    )

            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            # Sleep with cancellation checks.
            for _ in range(int(max(1, interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _kanban_advance(
        self, sub: dict, cursor: int, board: Optional[str] = None,
    ) -> None:
        """Sync helper: advance a subscription's cursor. Runs in to_thread.

        ``board`` scopes the DB connection to the board that owns this
        subscription. Unsub cursors in one board can't touch another's.
        """
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.advance_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                new_cursor=cursor,
            )
        finally:
            conn.close()

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.remove_notify_sub(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
            )
        finally:
            conn.close()

    def _kanban_rewind(
        self,
        sub: dict,
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        """Sync helper: undo a claimed notification cursor after send failure."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.rewind_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
            )
        finally:
            conn.close()

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
        file paths through the completion event so downstream humans get
        the deliverable as a native upload instead of a path printed in
        chat.

        Sources scanned, in priority order:
          1. ``event_payload['artifacts']`` (explicit list — preferred)
          2. ``event_payload['summary']`` (truncated first line)
          3. ``task.result`` (legacy fallback)

        Files are deduplicated, missing files are silently skipped (the
        path may have been mentioned for reference only), and delivery
        errors are logged but do not break the notifier loop.
        """
        from pathlib import Path as _Path

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            if not path:
                return
            expanded = os.path.expanduser(path)
            if expanded in seen:
                return
            if not os.path.isfile(expanded):
                return
            seen.add(expanded)
            candidates.append(expanded)

        # 1. Explicit artifacts list in payload.
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, str):
                        _add(item)

            # 2. Paths embedded in the payload summary.
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                paths, _ = adapter.extract_local_files(summary)
                for p in paths:
                    _add(p)

        # 3. Legacy: paths embedded in task.result.
        if task is not None and getattr(task, "result", None):
            result_text = str(task.result)
            paths, _ = adapter.extract_local_files(result_text)
            for p in paths:
                _add(p)

        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

        from urllib.parse import quote as _quote

        # Partition images so they ride a single send_multiple_images call
        # on platforms that support batch image uploads (Signal/Slack RPCs).
        image_paths = [p for p in candidates if _Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if _Path(p).suffix.lower() not in _IMAGE_EXTS]

        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(
                    chat_id=chat_id, images=batch, metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: image batch upload failed: %s", exc,
                )

        for path in other_paths:
            ext = _Path(path).suffix.lower()
            try:
                if ext in _VIDEO_EXTS:
                    await adapter.send_video(
                        chat_id=chat_id, video_path=path, metadata=metadata,
                    )
                else:
                    await adapter.send_document(
                        chat_id=chat_id, file_path=path, metadata=metadata,
                    )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: artifact upload (%s) failed: %s",
                    path, exc,
                )

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
        When true, the gateway hosts the single dispatcher for this profile:
        no separate `hermes kanban daemon` process needed. When false, the
        loop exits immediately and an external daemon is expected.

        Each tick calls :func:`kanban_db.dispatch_once` inside
        ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
        event loop. Failures in one tick don't stop subsequent ticks —
        same pattern as `_kanban_notifier_watcher`.

        Shutdown: the loop checks ``self._running`` between ticks; gateway
        stop() flips it to False and cancels pending tasks, and the
        in-flight ``to_thread`` returns on its own after the current
        ``dispatch_once`` call finishes (typically <1ms on an idle board).
        """
        # Read config once at boot. If the user flips the flag later, they
        # restart the gateway; same pattern as every other background
        # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
        # as an escape hatch (false-y value disables without editing YAML).
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return

        try:
            interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
        except (ValueError, TypeError):
            logger.warning(
                "kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                kanban_cfg.get("dispatch_interval_seconds"),
            )
            interval = 60.0
        interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

        # Read max_spawn config to limit concurrent kanban tasks
        max_spawn = kanban_cfg.get("max_spawn", None)
        if max_spawn is not None:
            logger.info(f"kanban dispatcher: max_spawn={max_spawn}")

        # Cap the number of simultaneously running tasks so slow workers
        # (local LLMs, resource-constrained hosts) don't pile up and time
        # out. When set, the dispatcher skips spawning when the board
        # already has this many tasks in 'running' status.
        raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
        max_in_progress = None
        if raw_max_in_progress is not None:
            try:
                max_in_progress = int(raw_max_in_progress)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                if max_in_progress < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                        raw_max_in_progress,
                    )
                    max_in_progress = None
                else:
                    logger.info(f"kanban dispatcher: max_in_progress={max_in_progress}")

        raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
        try:
            failure_limit = int(raw_failure_limit)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT
        if failure_limit < 1:
            logger.warning(
                "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT

        # Read stale_timeout_seconds — 0 disables stale detection.
        raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
        try:
            stale_timeout_seconds = int(raw_stale or 0)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                "disabling stale detection",
                raw_stale,
            )
            stale_timeout_seconds = 0

        # Read kanban.default_assignee — fallback profile for tasks
        # created without an explicit assignee (e.g. via the dashboard).
        # When set, the dispatcher applies it to unassigned ready tasks
        # instead of skipping them indefinitely (#27145). Empty string
        # (the schema default) means "no fallback, keep skipping" —
        # backward-compatible with existing installs.
        default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
        if default_assignee:
            logger.info(
                "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                "will route to this profile)",
                default_assignee,
            )

        # Read kanban.max_in_progress_per_profile — per-profile concurrency
        # cap (#21582). When set, no single profile gets more than N
        # workers running at once, even if the global max_in_progress
        # would allow it. Prevents one profile's local model / API quota
        # / browser pool from being overwhelmed by a fan-out.
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
        max_in_progress_per_profile = None
        if raw_per_profile is not None:
            try:
                max_in_progress_per_profile = int(raw_per_profile)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                if max_in_progress_per_profile < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                        raw_per_profile,
                    )
                    max_in_progress_per_profile = None
                else:
                    logger.info(
                        "kanban dispatcher: max_in_progress_per_profile=%d",
                        max_in_progress_per_profile,
                    )

        # Initial delay so the gateway finishes wiring adapters before the
        # dispatcher spawns workers (those workers may hit gateway notify
        # subscriptions etc.). Matches the notifier watcher's delay.
        await asyncio.sleep(5)

        # Health telemetry mirrored from `_cmd_daemon`: warn when ready
        # queue is non-empty but spawns are 0 for N consecutive ticks —
        # usually means broken PATH, missing venv, or credential loss.
        HEALTH_WINDOW = 6
        bad_ticks = 0
        last_warn_at = 0
        # Avoid hot-looping corrupt-looking board DBs, but do not suppress
        # same-fingerprint retries forever: transient WAL/open races can
        # surface as "database disk image is malformed" for one tick.
        CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
        disabled_corrupt_boards: dict[
            str, tuple[tuple[str, int | None, int | None], float]
        ] = {}

        def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
            path = _kb.kanban_db_path(slug)
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            try:
                stat = path.stat()
            except OSError:
                return (resolved, None, None)
            return (resolved, stat.st_mtime_ns, stat.st_size)

        def _is_corrupt_board_db_error(exc: Exception) -> bool:
            corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
            if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
                return True
            if not isinstance(exc, sqlite3.DatabaseError):
                return False
            msg = str(exc).lower()
            return (
                "file is not a database" in msg
                or "database disk image is malformed" in msg
            )

        def _tick_once_for_board(slug: str) -> "Optional[object]":
            """Run one dispatch_once for a specific board.

            Runs in a worker thread via `asyncio.to_thread`. `board=slug`
            is passed through `dispatch_once` so `resolve_workspace` and
            `_default_spawn` see the right paths. The per-board DB is
            opened explicitly so concurrent boards never share a
            connection handle or accidentally claim across each other.
            """
            conn = None
            fingerprint = _board_db_fingerprint(slug)
            disabled_entry = disabled_corrupt_boards.get(slug)
            if disabled_entry is not None:
                disabled_fingerprint, disabled_at = disabled_entry
                age = time.monotonic() - disabled_at
                if (
                    disabled_fingerprint == fingerprint
                    and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
                ):
                    return None
                if disabled_fingerprint == fingerprint:
                    logger.info(
                        "kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch",
                        slug,
                        age,
                    )
                else:
                    logger.info(
                        "kanban dispatcher: board %s database changed; retrying dispatch",
                        slug,
                    )
                disabled_corrupt_boards.pop(slug, None)
            try:
                conn = _kb.connect(board=slug)
                # `connect()` runs the schema + idempotent migration on
                # first open per process; the previous explicit
                # `init_db()` call here busted the per-process cache and
                # re-ran the migration on a second connection, racing
                # the first. See the matching comment in
                # `_kanban_notifier_watcher` and issue #21378.
                return _kb.dispatch_once(
                    conn,
                    board=slug,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    stale_timeout_seconds=stale_timeout_seconds,
                    default_assignee=default_assignee,
                    max_in_progress_per_profile=max_in_progress_per_profile,
                )
            except sqlite3.DatabaseError as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            except Exception as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _tick_once() -> "list[tuple[str, Optional[object]]]":
            """Run one dispatch_once per board. Returns (slug, result) pairs.

            Enumerating boards on every tick keeps the dispatcher honest
            when users create a new board mid-run: no restart required,
            the next tick picks it up automatically.
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            out: list[tuple[str, "Optional[object]"]] = []
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                out.append((slug, _tick_once_for_board(slug)))
            return out

        def _ready_nonempty() -> bool:
            """Cheap probe: is there at least one ready+assigned+unclaimed
            task on ANY board whose assignee maps to a real Hermes profile
            (i.e. one the dispatcher would actually spawn for)?

            Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
            ``orion-research``) are pulled by terminals via
            ``claim_task`` directly and never spawnable, so a queue full
            of those is "correctly idle", not "stuck". Filtering them out
            here keeps the stuck-warn fire only on real failures (broken
            PATH, missing venv, credential loss for a real Hermes profile).
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                conn = None
                try:
                    conn = _kb.connect(board=slug)
                    if _kb.has_spawnable_ready(conn):
                        return True
                    if _kb.has_spawnable_review(conn):
                        return True
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
            return False

        # Auto-decompose: turn fresh triage tasks into ready workgraphs
        # before the dispatcher fans out workers. Gated by
        # ``kanban.auto_decompose`` (default True). Capped by
        # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
        # of triage tasks doesn't burst-spend the aux LLM in one tick;
        # remainder defers to subsequent ticks.
        auto_decompose_enabled = bool(kanban_cfg.get("auto_decompose", True))
        try:
            auto_decompose_per_tick = int(
                kanban_cfg.get("auto_decompose_per_tick", 3) or 3
            )
        except (TypeError, ValueError):
            auto_decompose_per_tick = 3
        if auto_decompose_per_tick < 1:
            auto_decompose_per_tick = 1

        def _auto_decompose_tick() -> int:
            """Run the auto-decomposer for up to N triage tasks across all
            boards. Returns the number of triage tasks that were
            successfully decomposed or specified this tick.
            """
            try:
                from hermes_cli import kanban_decompose as _decomp
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "kanban auto-decompose: import failed (%s); skipping", exc,
                )
                return 0
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            attempted = 0
            successes = 0
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin this board for the duration of the call — same
                # pattern as the dashboard specify endpoint. The
                # decomposer module connects with no board kwarg and
                # relies on the env var.
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug(
                            "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                            slug, exc,
                        )
                        triage_ids = []
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        attempted += 1
                        try:
                            outcome = _decomp.decompose_task(
                                tid, author="auto-decomposer",
                            )
                        except Exception:
                            logger.exception(
                                "kanban auto-decompose: decompose_task crashed on %s",
                                tid,
                            )
                            continue
                        if outcome.ok:
                            successes += 1
                            if outcome.fanout and outcome.child_ids:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → %d children",
                                    slug, tid, len(outcome.child_ids),
                                )
                            else:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                    slug, tid,
                                )
                        else:
                            # Common no-op reasons (no aux client configured) shouldn't
                            # spam logs every tick. Log at debug.
                            logger.debug(
                                "kanban auto-decompose [%s]: %s skipped: %s",
                                slug, tid, outcome.reason,
                            )
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
            return successes

        logger.info(
            "kanban dispatcher: embedded in gateway (interval=%.1fs)", interval
        )
        while self._running:
            try:
                # Reap zombie children before per-board work so a board DB
                # failure cannot block cleanup of unrelated workers.
                pids = await asyncio.to_thread(_kb.reap_worker_zombies)
                if pids:
                    logger.info(
                        "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                        len(pids),
                        pids,
                    )
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                if auto_decompose_enabled:
                    await asyncio.to_thread(_auto_decompose_tick)
                results = await asyncio.to_thread(_tick_once)
                any_spawned = False
                for slug, res in (results or []):
                    if res is not None and getattr(res, "spawned", None):
                        any_spawned = True
                        # Quiet by default — only log when something actually
                        # happened, so an idle gateway stays silent.
                        logger.info(
                            "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                            "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                            slug,
                            len(res.spawned),
                            res.reclaimed,
                            len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
                            len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
                            res.promoted,
                            len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
                        )
                # Health telemetry (aggregate across boards)
                ready_pending = await asyncio.to_thread(_ready_nonempty)
                if ready_pending and not any_spawned:
                    bad_ticks += 1
                else:
                    bad_ticks = 0
                if bad_ticks >= HEALTH_WINDOW:
                    now = int(time.time())
                    if now - last_warn_at >= 300:
                        logger.warning(
                            "kanban dispatcher stuck: ready queue non-empty for "
                            "%d consecutive ticks but 0 workers spawned. Check "
                            "profile health (venv, PATH, credentials) and "
                            "`hermes kanban list --status ready`.",
                            bad_ticks,
                        )
                        last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0


# Deprecated alias — earlier code referenced this as ``EpochEngine``.
# Defined at module scope so any code that did
# ``from gateway.kanban_watchers import EpochEngine`` keeps working.
EpochEngine = TaskLoopEngine
