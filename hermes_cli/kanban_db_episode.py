"""Wait episodes for the Kanban DB: the locked park (``scheduled``) and unblock bodies
shared by the legacy verbs, and the governed WAIT_AND_RESUME exact-episode CAS on top.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import time
from typing import Any, Optional


def _unblock_locked(conn: sqlite3.Connection, task_id: str, now: int) -> Optional[str]:
    """Body of :func:`kanban_db.unblock_task` under the caller's write txn; the landing
    status, or ``None`` when the task is not ``blocked``/``scheduled``."""
    resume_status = (
        _kb._resume_status_from_events(conn, task_id)
        if _kb._task_status(conn, task_id) == "blocked"
        else "ready"
    )
    _kb._reclaim_dangling_run(
        conn, task_id, statuses=("blocked", "scheduled"), now=now,
        note="invariant recovery on unblock",
    )
    # Re-gate on parent completion before restoring the source phase.
    landing_status = _kb._landing_status_after_parents(conn, task_id)
    new_status = (
        "review"
        if landing_status == "ready" and resume_status == "review"
        else landing_status
    )
    # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
    # resetting them is the amnesia that let cron-unblock <-> re-block loop
    # unbounded; only complete_task clears them. ``consecutive_failures``
    # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
    # is a fresh start for the retry budget.
    cur = conn.execute(
        "UPDATE tasks SET status = ?, current_run_id = NULL, "
        "consecutive_failures = 0, last_failure_error = NULL "
        "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
    )
    if cur.rowcount != 1:
        return None
    _kb._append_event(
        conn, task_id, "unblocked",
        (
            {"status": new_status, "resume_status": resume_status}
            if new_status != "ready" or resume_status != "ready"
            else None
        ),
    )
    return new_status


def _schedule_locked(
    conn: sqlite3.Connection, task_id: str, guard: str, params: list[Any],
    payload: dict, reason: Optional[str],
) -> bool:
    """Body of :func:`kanban_db.schedule_task` under the caller's write txn: the
    ``guard``-ed flip to ``scheduled``, run close and ``scheduled`` event. False when
    the guard matched nothing (no write happened)."""
    sql = """
        UPDATE tasks
           SET status       = 'scheduled',
               claim_lock   = NULL,
               claim_expires= NULL,
               worker_pid   = NULL
         WHERE id = ?
    """ + guard
    if conn.execute(sql, [task_id, *params]).rowcount != 1:
        return False
    run_id = _kb._end_or_synthesize_run(
        conn, task_id, outcome="scheduled", status="scheduled", summary=reason, synthesize=bool(reason),
    )
    _kb._append_event(conn, task_id, "scheduled", payload, run_id=run_id)
    return True


# --- Governed WAIT_AND_RESUME episodes (exact-episode park/unblock CAS) ---
#
# A governed park mints an opaque, versioned, self-namespaced episode token
# (128-bit random nonce) and stores it in the ``scheduled`` event payload in the
# SAME write txn as the park. A governed unblock proves, in its own write txn and
# before any side effect, that this park is still the task's current wait
# episode. The token is a precondition, never an authority: legacy/human
# block/schedule/unblock paths are unchanged and mint nothing, and any later
# non-neutral event (theirs included) makes an earlier token stale.

EPISODE_TOKEN_SCHEME = "hkep1"
_EPISODE_TOKEN_RE = re.compile(r"^hkep1\.(?P<task_id>\S+)\.(?P<nonce>[0-9a-f]{32})$")

# CLOSED allow-list of event kinds that never end a wait episode. Every other
# kind — including any kind added in the future — is an episode boundary, so a
# new lifecycle writer fails closed instead of silently passing the fence.
# ``terminal_worker_reaped`` is the dispatcher killing the leftover worker of the
# run a MID_RUN park already ended; it changes no task or run state.
EPISODE_NEUTRAL_EVENT_KINDS = frozenset({
    "commented", "edited", "reprioritized", "attached", "attachment_removed",
    "terminal_worker_reaped",
})

GOVERNED_PARK_MODES = ("PRE_LAUNCH", "MID_RUN")

_WAIT_STATUSES = ("blocked", "scheduled", "triage")


class EpisodeConflict(ValueError):
    """A governed park/unblock refused with zero lifecycle mutation. ``code`` is one of
    TASK_NOT_FOUND, RUN_MISMATCH, NOT_IDLE, SOURCE_IS_WAIT, STATUS_NOT_ALLOWED,
    TOKEN_MALFORMED, TOKEN_FOREIGN, EPISODE_STALE."""

    def __init__(self, code: str, detail: str, *, current_status: Optional[str] = None):
        super().__init__(detail)
        self.code = code
        self.current_status = current_status


def _mint_episode_token(task_id: str) -> str:
    return f"{EPISODE_TOKEN_SCHEME}.{task_id}.{secrets.token_hex(16)}"


def schedule_task_governed(
    conn: sqlite3.Connection, task_id: str, *, mode: str,
    expected_run_id: Optional[int] = None, reason: Optional[str] = None,
) -> str:
    """Governed WAIT_AND_RESUME park into ``scheduled``; returns the episode token.

    ``MID_RUN`` requires ``status='running'`` AND ``current_run_id = expected_run_id``;
    ``PRE_LAUNCH`` requires an idle ``todo``/``ready`` task (no run, no claim). A
    task already waiting (``blocked``/``scheduled``/``triage``) is never re-parked.
    Refusals raise :class:`EpisodeConflict` with nothing written. The token is
    stored in the park event inside the park's write txn and returned only after
    that txn has committed (a failed COMMIT raises instead)."""
    if mode == "MID_RUN":
        if expected_run_id is None:
            raise ValueError("MID_RUN park requires expected_run_id")
        guard, params = " AND status = 'running' AND current_run_id = ?", [int(expected_run_id)]
    elif mode == "PRE_LAUNCH":
        if expected_run_id is not None:
            raise ValueError("PRE_LAUNCH park takes no expected_run_id")
        guard = (" AND status IN ('todo', 'ready') AND current_run_id IS NULL"
                 " AND claim_lock IS NULL")
        params = []
    else:
        raise ValueError(f"mode must be one of {GOVERNED_PARK_MODES}")
    with _kb.write_txn(conn):
        token = _mint_episode_token(task_id)
        payload = {"reason": reason, "episode_token": token, "mode": mode}
        if not _schedule_locked(conn, task_id, guard, params, payload, reason):
            raise _governed_park_refusal(conn, task_id, mode)
    return token


def _governed_park_refusal(conn: sqlite3.Connection, task_id: str, mode: str) -> EpisodeConflict:
    """Classify a governed park whose guard matched nothing (read in the same txn)."""
    status = _kb._task_status(conn, task_id)
    if status is None:
        return EpisodeConflict("TASK_NOT_FOUND", f"task {task_id} not found")
    if status in _WAIT_STATUSES:
        code = "SOURCE_IS_WAIT"
    elif mode == "MID_RUN":
        code = "RUN_MISMATCH" if status == "running" else "STATUS_NOT_ALLOWED"
    else:
        code = "NOT_IDLE" if status in ("todo", "ready", "running") else "STATUS_NOT_ALLOWED"
    return EpisodeConflict(code, f"cannot park {task_id} ({mode}): {code} (status {status!r})",
                           current_status=status)


def unblock_task_governed(conn: sqlite3.Connection, task_id: str, *, expected_episode_token: str) -> str:
    """Exact-episode unblock: resume ``task_id`` only while ``expected_episode_token``
    is still its current wait episode; returns the landing status.

    The episode check and the unblock (same semantics as :func:`kanban_db.unblock_task`)
    run in ONE write txn, the check before any reclaim/re-gate/mutation. Refusals raise
    :class:`EpisodeConflict` with nothing written."""
    match = _EPISODE_TOKEN_RE.match(expected_episode_token or "")
    if match is None:
        raise EpisodeConflict("TOKEN_MALFORMED", "episode token is malformed or of an unknown version")
    if match["task_id"] != task_id:
        raise EpisodeConflict("TOKEN_FOREIGN", f"episode token was not issued for task {task_id}")
    now = int(time.time())
    with _kb.write_txn(conn):
        _require_current_episode(conn, task_id, expected_episode_token)
        new_status = _unblock_locked(conn, task_id, now)
        if new_status is None:  # unreachable: 'scheduled' was proven in this txn
            raise EpisodeConflict("EPISODE_STALE", f"task {task_id} left its wait episode")
    return new_status


def _require_current_episode(conn: sqlite3.Connection, task_id: str, token: str) -> None:
    """Raise unless ``task_id`` is ``scheduled`` and its newest non-neutral event is
    the governed park that minted ``token``. Caller holds the write txn."""
    status = _kb._task_status(conn, task_id)
    if status is None:
        raise EpisodeConflict("TASK_NOT_FOUND", f"task {task_id} not found")
    neutral = sorted(EPISODE_NEUTRAL_EVENT_KINDS)
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        f"AND kind NOT IN ({', '.join('?' for _ in neutral)}) ORDER BY id DESC LIMIT 1",
        (task_id, *neutral),
    ).fetchone()
    current = _kb._json_dict(row["payload"]).get("episode_token") if row and row["kind"] == "scheduled" else None
    if status != "scheduled" or current != token:
        raise EpisodeConflict(
            "EPISODE_STALE", f"task {task_id} is no longer in the expected wait episode",
            current_status=status,
        )


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
