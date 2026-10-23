"""Board admission (sibling of ``kanban_db``): the ACTIVE-card ceiling, the
promotion pass that honours it, and board-level quota state.

``kanban.max_active_per_board`` caps the cards in an ACTIVE status on one board;
``board_state`` rows record provider-quota exhaustion per profile so the
dispatcher can hold a whole profile and a monitor can read starvation without
grepping ``task_events``. Late-binds the ``kanban_db`` facade inside functions.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional


def _kb():
    from hermes_cli import kanban_db
    return kanban_db


# ``max_active`` default for the admission points (``create_task``,
# ``recompute_ready``): read ``kanban.max_active_per_board`` live. Callers
# pass ``None`` for an explicit "uncapped" or an int to pin the ceiling.
CEILING_FROM_CONFIG: Any = object()


def _resolve_max_active(value: Any) -> Optional[int]:
    if value is CEILING_FROM_CONFIG:
        return configured_max_active_per_board()
    return int(value) if value is not None and int(value) >= 1 else None


def admit_or_defer(
    conn: sqlite3.Connection, task_status: str, max_active: Any,
) -> tuple[str, Optional[dict]]:
    """Admission decision for a card about to be created in ``task_status``.

    A creator (a 5-minute watchdog, a decomposer) must not be able to flood the
    board past the ceiling. Returns ``(status, ceiling_wait)``: unchanged when
    the status is not ACTIVE or the board has room; otherwise ``("todo",
    payload)`` where the payload is the ``ceiling_wait`` event every create
    surface reports (tool result, CLI, dashboard). ``recompute_ready`` admits
    the card when a slot frees.
    """
    if task_status not in ACTIVE_STATUSES:
        return task_status, None
    ceiling = _resolve_max_active(max_active)
    if ceiling is None:
        return task_status, None
    active = count_active_tasks(conn)
    if active < ceiling:
        return task_status, None
    return "todo", {
        "reason": "board_active_ceiling",
        "ceiling": ceiling,
        "active": active,
        "intended_status": task_status,
    }


def _configured_positive_int(key: str) -> Optional[int]:
    """``kanban.<key>`` from config as a positive int, else None (unset/invalid)."""
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(key)
    except Exception:  # health: allow BLE001 -- an unreadable config must never break admission; uncapped, like configured_max_in_progress
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def configured_max_active_per_board() -> Optional[int]:
    """``kanban.max_active_per_board``: the ACTIVE-card ceiling (ready + running +
    review + scheduled + triage; ``blocked`` is cap-invisible). None = uncapped.

    Read live (config load is file-signature cached) by every admission point —
    ``create_task``, ``recompute_ready`` and the dispatcher tick — so the three
    never disagree on whether the board is full.
    """
    return _configured_positive_int("max_active_per_board")


# Statuses that count toward ``kanban.max_active_per_board``. ``blocked`` is
# deliberately absent: a blocked card consumes no worker and the standing board
# rule is that it does not occupy a slot.
ACTIVE_STATUSES: tuple[str, ...] = ("ready", "running", "review", "scheduled", "triage")
_ACTIVE_STATUS_SQL = ",".join(f"'{s}'" for s in ACTIVE_STATUSES)


def count_active_tasks(conn: sqlite3.Connection) -> int:
    """Number of cards in an ACTIVE status (see :data:`ACTIVE_STATUSES`)."""
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM tasks WHERE status IN ({_ACTIVE_STATUS_SQL})"
        ).fetchone()[0]
    )


def ceiling_deferral(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """The ``ceiling_wait`` payload when ``task_id`` is still parked in ``todo`` by
    the ACTIVE ceiling, else None. One query shared by every create surface."""
    row = conn.execute(
        "SELECT t.status, e.payload FROM tasks t "
        "JOIN task_events e ON e.task_id = t.id AND e.kind = 'ceiling_wait' "
        "WHERE t.id = ? ORDER BY e.id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row is None or row["status"] != "todo":
        return None
    return _kb()._json_dict(row["payload"]) or None


def active_headroom(conn: sqlite3.Connection, max_active: Optional[int]) -> Optional[int]:
    """Cards the board may still admit under the ceiling; None = uncapped."""
    if max_active is None:
        return None
    return max(int(max_active) - count_active_tasks(conn), 0)


# --- Board-level quota exhaustion state -------------------------------------
#
# A rate-limited worker exit (``KANBAN_RATE_LIMIT_EXIT_CODE``) used to leave
# evidence only on the card that died (``task_runs.outcome='rate_limited'`` +
# a per-task respawn cooldown). Every OTHER card for the same profile kept
# spawning into the same exhausted quota: spawn -> burn -> 429 -> cooldown ->
# respawn. The board now carries one durable row per profile so (a) the
# dispatcher holds the whole profile for the cooldown and (b) a monitor can
# answer "has this board been quota-starved since T?" with one SELECT.

def _quota_key(profile: str) -> str:
    return f"quota_exhausted:{profile}"


def note_quota_exhausted(
    conn: sqlite3.Connection, profile: Optional[str], task_id: str, *, now: Optional[int] = None,
) -> None:
    """Record a rate-limited worker exit against ``profile`` on this board.

    ``since`` is preserved across repeats so it marks the START of the
    starvation window; ``last_seen`` and ``count`` advance every time. Caller
    holds the write transaction.
    """
    if not profile:
        return
    now = int(time.time()) if now is None else int(now)
    key = _quota_key(profile)
    row = conn.execute("SELECT value FROM board_state WHERE key = ?", (key,)).fetchone()
    prev = _kb()._json_dict(row["value"]) if row is not None else {}
    state = {
        "profile": profile,
        "since": int(prev.get("since") or now),
        "last_seen": now,
        "last_task": task_id,
        "count": int(prev.get("count") or 0) + 1,
    }
    conn.execute(
        "INSERT INTO board_state (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(state), now),
    )


def clear_quota_exhausted(conn: sqlite3.Connection, profile: Optional[str]) -> bool:
    """A profile's run reached a terminal board call: its quota is back. Returns
    whether a starvation row was open. Caller holds the write transaction."""
    if not profile:
        return False
    cur = conn.execute("DELETE FROM board_state WHERE key = ?", (_quota_key(profile),))
    return bool(cur.rowcount)


def quota_state(conn: sqlite3.Connection, *, now: Optional[int] = None) -> dict[str, dict]:
    """``{profile: {since, last_seen, last_task, count, cooldown_until, in_cooldown}}``
    for every profile with an open quota-exhaustion row on this board."""
    now = int(time.time()) if now is None else int(now)
    cooldown = _kb()._resolve_rate_limit_cooldown_seconds()
    out: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT key, value FROM board_state WHERE key LIKE 'quota_exhausted:%'"
    ):
        state = _kb()._json_dict(row["value"])
        profile = state.get("profile") or row["key"].split(":", 1)[1]
        until = int(state.get("last_seen") or 0) + cooldown
        state["cooldown_until"] = until
        state["in_cooldown"] = now < until
        out[profile] = state
    return out


def profiles_in_quota_cooldown(conn: sqlite3.Connection, *, now: Optional[int] = None) -> set[str]:
    """Profiles the dispatcher must not spawn for this tick."""
    return {p for p, s in quota_state(conn, now=now).items() if s["in_cooldown"]}


# --- Promotion (todo/blocked -> active), ceiling-aware ------------------------

def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """``review`` when the newest lifecycle event carries a review
    ``resume_status``/``retry_status``/``source_status``, else ``ready`` (legacy)."""
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'ceiling_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    payload = _kb()._json_dict(_kb()._row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    # A ``triage=True`` create parked by the ACTIVE ceiling still owes a triage pass.
    if payload.get("reason") == "board_active_ceiling" and payload.get("intended_status") == "triage":
        return "triage"
    return "ready"


def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the newest ``blocked``/``unblocked``/``gave_up`` event says the
    block must wait for an operator: an explicit ``kanban_block`` (#28712), or a
    breaker trip ``_record_task_failure`` stamped ``sticky`` — the clean-exit
    protocol-violation budget or a systemic same-error wave. Those trip on a
    policy independent of ``consecutive_failures``, so ``recompute_ready``'s
    counter check cannot see them — without this the trip is promoted back to
    ``ready`` in the same tick and the card respawns forever. A plain
    (unified-budget) ``gave_up`` carries no marker and is judged by the counter,
    so raising ``failure_limit`` or ``assign_task`` to a fresh profile still
    releases it; a task with no such event at all (direct DB edit) auto-recovers.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row and row["kind"] == "blocked":
        return True
    trip = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'gave_up' AND id > COALESCE("
        "  (SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'unblocked'), 0) "
        "ORDER BY id DESC LIMIT 1", (task_id, task_id),
    ).fetchone()
    return bool(trip) and bool(_kb()._json_dict(trip["payload"]).get("sticky"))


def recompute_ready(
    conn: sqlite3.Connection, failure_limit: int | None = None, max_active: Any = CEILING_FROM_CONFIG,
) -> int:
    """Promote ``todo``/``blocked`` tasks whose parents are all done/archived;
    returns the count. Opens its own IMMEDIATE txn — call OUTSIDE any write txn.

    ``blocked`` is skipped when sticky (explicit ``kanban_block``) or when
    ``consecutive_failures`` reached the limit (else the breaker could never
    trip). Limit order matches ``_record_task_failure``: ``max_retries`` >
    ``failure_limit`` > ``_kb().DEFAULT_FAILURE_LIMIT``.

    ``max_active`` (default ``kanban.max_active_per_board``): every promotion
    lands in an ACTIVE status, so promotions stop once the board is at the
    ceiling — highest priority, then oldest, first. The rest stay ``todo`` /
    ``blocked`` for a later pass; nothing is dropped.

    1. The most recent block event was a worker-initiated ``kanban_block`` — those stay blocked until an
    explicit ``kanban_unblock`` (#28712).
    """
    if failure_limit is None:
        failure_limit = _kb().DEFAULT_FAILURE_LIMIT
    promoted = 0
    with _kb().write_txn(conn):
        headroom = active_headroom(conn, _resolve_max_active(max_active))
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked') "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()
        for row in todo_rows:
            if headroom is not None and promoted >= headroom:
                break
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Explicit human-intervention block; only ``unblock_task`` may exit it.
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?", (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                resume_status = _resume_status_from_events(conn, task_id)
                if cur_status == "blocked":
                    # At the breaker limit, no auto-recovery (else block ->
                    # recover -> respawn -> exhaust -> block forever). The
                    # counter is preserved so it accumulates across cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked'", (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                        (resume_status, task_id),
                    )
                _kb()._append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted
