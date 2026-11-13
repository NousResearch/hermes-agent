"""Respawn containment: bounded rate-limit retries, parking cards no tick can
clear, resume after a block, and lifecycle attribution."""
from __future__ import annotations

import sqlite3
from typing import Optional

# Consecutive rate-limited runs double the rate-limit cooldown up to this cap,
# and after this many the card parks ``blocked`` for an operator instead of
# probing the quota wall forever (~8h of backoff at the defaults). Overridable
# via ``HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS`` /
# ``HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE``.
DEFAULT_RATE_LIMIT_MAX_COOLDOWN_SECONDS = 3600
DEFAULT_RATE_LIMIT_MAX_CONSECUTIVE = 12

# Guard reasons that no tick can clear on its own, per lane: the dispatcher
# parks the card ``blocked`` (sticky) once rather than skipping it on every
# tick. In review, ``blocker_auth`` usually reads the implementer's stale quota
# stamp (#119070), a false positive that must not escalate to an operator.
_PARKING_GUARD_REASONS = {
    "ready": frozenset({"blocker_auth", "rate_limit_exhausted"}),
    "review": frozenset({"rate_limit_exhausted"}),
}


def _attribution(actor: Optional[str], source: str) -> dict:
    """Who initiated a lifecycle transition and through which surface. An
    unidentified caller is recorded as ``"unknown"``, never a guessed name."""
    return {"actor": actor or "unknown", "source": source}


def retry_of(conn: sqlite3.Connection, task_id: str, run_id: int) -> dict:
    """``{"retry_of": {run_id, outcome}}`` naming the latest ended run before
    ``run_id``, so a respawn's reason is auditable; ``{}`` on a first run."""
    prior = conn.execute(
        "SELECT id, outcome FROM task_runs WHERE task_id = ? AND id != ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT 1", (task_id, run_id),
    ).fetchone()
    return {"retry_of": {"run_id": prior["id"], "outcome": prior["outcome"]}} if prior else {}


def rate_limit_guard(
    conn: sqlite3.Connection, task_id: str, ended_at: object, cooldown: int, now: int,
) -> Optional[str]:
    """Guard for a card whose latest run hit the quota wall. The trailing streak
    of rate-limited runs doubles ``cooldown`` up to the cap; at the streak limit
    the card is exhausted. The limit is checked first so a zero cooldown (no
    wait) cannot switch it off. Counted from ``task_runs``: it survives
    dispatcher restarts and one worker run counts once."""
    from hermes_cli.kanban_db import _env_int, _to_epoch

    max_streak = _env_int("HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE", DEFAULT_RATE_LIMIT_MAX_CONSECUTIVE, minimum=1)
    recent = conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT ?", (task_id, max_streak),
    ).fetchall()
    streak = next((i for i, r in enumerate(recent) if r["outcome"] != "rate_limited"), len(recent))
    if streak >= max_streak:
        return "rate_limit_exhausted"
    if cooldown <= 0:
        return None
    cap = _env_int("HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS", DEFAULT_RATE_LIMIT_MAX_COOLDOWN_SECONDS)
    backoff = min(cooldown << (streak - 1), max(cooldown, cap))
    ended = _to_epoch(ended_at)
    if ended is not None and (now - ended) < backoff:
        return "rate_limit_cooldown"
    return None


def record_guard(conn: sqlite3.Connection, task_id: str, reason: str, *, lane: str) -> None:
    """Record a dispatcher skip once per reason (same rule as
    ``skipped_nonspawnable``), and park a card no tick can clear instead of
    showing a ``ready``/``review`` card that is skipped forever. Counters and
    ``last_failure_error`` are kept; ``unblock_task`` resets them and resumes
    the lane the card left (``source_status``)."""
    from hermes_cli.kanban_db import _append_event, _json_or_null, block_task, write_txn

    payload = {"reason": reason, "actor": "dispatcher"}
    with write_txn(conn):
        # By id, not created_at: insertion order survives a clock step.
        last = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if last is None or last["kind"] != "respawn_guarded" or last["payload"] != _json_or_null(payload):
            _append_event(conn, task_id, "respawn_guarded", payload)
    if reason in _PARKING_GUARD_REASONS.get(lane, ()):
        block_task(conn, task_id, kind="needs_input", from_review=lane == "review",
                   reason=f"dispatcher parked: {reason}", actor="dispatcher", source="dispatcher")


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
    from hermes_cli.kanban_db import _json_dict

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
    return bool(trip) and bool(_json_dict(trip["payload"]).get("sticky"))


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """``review`` when the newest lifecycle event carries a review
    ``resume_status``/``retry_status``/``source_status``, else ``ready`` (legacy)."""
    from hermes_cli.kanban_db import _json_dict, _row_get

    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    payload = _json_dict(_row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"
