"""Respawn guard: when one ready/review card must NOT be re-spawned yet.

Split out of ``hermes_cli.kanban_db_dispatch`` so that module stays inside its code-health line
cap while the guard keeps its own cohesive home. The dispatcher calls :func:`check_respawn_guard`
per ready/review row before any claim attempt; :func:`_rate_limit_reset_hint` and
:func:`_rate_limit_streak` are the quota-wall half of that decision.

A provider quota wall is not the card's failure: the run is requeued without counting a failure
and the guard spaces the next attempt. The spacing is not a fixed cooldown any more — a weekly
plan window would otherwise be probed every few minutes for hours (27+ ``rate_limited`` runs on
one card). Two rules replace it:

* the provider's own reset instant, when the failed run recorded one, is honoured;
* otherwise the hold doubles per consecutive quota-wall run up to a ceiling.

The task stays ``ready`` throughout, so the ordinary dispatcher re-spawns it when the hold
lapses: no separate queue, no parked state nothing wakes, no extra approval.

Origin-resident helpers are reached late-bound via ``_kb`` (import-cycle breaking), exactly as
``kanban_db_dispatch`` does, so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any
from typing import Iterator
from typing import Optional

# The timestamp column each guarded table orders by (see :func:`_rows_since`).
_GUARD_TIME_COLS = {"task_runs": "ended_at", "task_events": "created_at", "task_comments": "created_at"}

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ation)|authz|"
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

# A quota wall that keeps returning must not be probed at the base cooldown for days: the hold
# doubles per consecutive rate-limited run and stops widening here. A weekly plan window then
# costs a handful of spaced attempts instead of one every few minutes.
DEFAULT_RATE_LIMIT_MAX_COOLDOWN_SECONDS = 21600  # 6 hours

# A provider that names when its window reopens ("resets Nov 20, 3pm (Asia/Tokyo)") is held to
# that instant instead of a guessed backoff. A hint further out than this is not trusted.
DEFAULT_RATE_LIMIT_RESET_MAX_SECONDS = 1209600  # 14 days


def _resolve_rate_limit_max_cooldown_seconds() -> int:
    """``HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS``, never below the base cooldown.

    A ceiling under the base would make the first hold the longest one.
    """
    ceiling = _kb._env_int("HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS",
                           DEFAULT_RATE_LIMIT_MAX_COOLDOWN_SECONDS)
    return max(ceiling, _kb._resolve_rate_limit_cooldown_seconds())


def _resolve_rate_limit_reset_max_seconds() -> int:
    """``HERMES_KANBAN_RATE_LIMIT_RESET_MAX_SECONDS`` else the class constant."""
    return _kb._env_int("HERMES_KANBAN_RATE_LIMIT_RESET_MAX_SECONDS", DEFAULT_RATE_LIMIT_RESET_MAX_SECONDS)

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


def _iso_utc(epoch: float) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` for board copy (no locale, no local-time guess)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))


def _rate_limit_reset_hint(
    conn: Optional[sqlite3.Connection], task_id: str, *, board: Optional[str] = None,
) -> Optional[float]:
    """Epoch the failed run's provider said its usage window reopens, or None.

    Only one run's evidence is read: the run in flight when the authority books the death, else
    the newest closed run (the one ``rate_limit_hold`` already established ended on a provider
    wall). Its structured ``resets_at``, its ``worker_result`` output (managed run), or its worker
    log (CLI run) supplies the instant. Output from any other run is never consulted, so an
    unrelated mention of a reset cannot park a card.
    """
    from agent.retry_utils import reset_instant_from_message
    # Late-bound: the worker-log reader lives in the dispatcher, which imports this module.
    from hermes_cli.kanban_db_dispatch import _worker_final_output

    now = time.time()
    horizon = _resolve_rate_limit_reset_max_seconds()
    texts: list[str] = []
    if conn is not None:
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is None:
            # Guard time: the quota-walled run is already closed, so its result is the newest row.
            closed = conn.execute(
                "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            run_id = int(closed["id"]) if closed is not None else None
        if run_id is not None:
            row = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
                "AND kind = 'worker_result' ORDER BY id DESC LIMIT 1",
                (task_id, run_id),
            ).fetchone()
            payload = _kb._json_dict(row["payload"]) if row is not None else {}
            for key in ("resets_at", "quota_resets_at"):
                value = payload.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if 0 < float(value) - now <= horizon:
                        return float(value)
            if isinstance(payload.get("last_output"), str):
                texts.append(payload["last_output"])
    texts.append(_worker_final_output(task_id, board=board))
    for text in texts:
        instant = reset_instant_from_message(text, now=now)
        if instant is not None and instant - now <= horizon:
            return instant
    return None


def _rate_limit_streak(conn: sqlite3.Connection, task_id: str, *, limit: int = 12) -> int:
    """Consecutive ``rate_limited`` runs at the newest closed run (0 when it ended otherwise).

    Run history is the only state the hold needs: any completed / crashed / blocked / scheduled
    run ends the streak, so one throttle can never inherit a long hold and a recovered provider
    puts the task back on the base cooldown.
    """
    rows = conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT ?",
        (task_id, max(1, int(limit))),
    ).fetchall()
    streak = 0
    for row in rows:
        if row["outcome"] != "rate_limited":
            break
        streak += 1
    return streak


def rate_limit_hold(
    conn: sqlite3.Connection, task_id: str, latest_run, now: int, base_cooldown: int,
) -> Optional[str]:
    """``"rate_limit_cooldown"`` while a quota-wall run's next attempt is still held, else None.

    ``latest_run`` is the task's newest closed run and must be its ``rate_limited`` one. The
    provider's recorded reset instant wins: a weekly plan wall names when it reopens, and
    probing it every cooldown only burns a worker slot per probe. Without a usable instant the
    hold doubles per consecutive quota-wall run (base, 2x, 4x …) up to the ceiling, so a long
    window costs a handful of spaced attempts instead of one every few minutes. Either way the
    task stays ``ready``: the ordinary tick re-spawns it when the hold lapses — no separate
    queue, no parked state nothing wakes. Any other run outcome ends the streak, so one
    throttle never inherits a multi-hour hold.
    """
    ended_at = _kb._to_epoch(latest_run["ended_at"])
    if ended_at is None:
        return None
    resets_at = _kb._json_dict(latest_run["metadata"]).get("quota_resets_at")
    if not isinstance(resets_at, (int, float)) or isinstance(resets_at, bool):
        # A run booked before this guard recorded the instant (or whose worker_result landed after
        # the reap) still names it in its own output: read that run, never a guessed cooldown.
        resets_at = _rate_limit_reset_hint(conn, task_id)
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool):
        horizon = _resolve_rate_limit_reset_max_seconds()
        if now < float(resets_at) and float(resets_at) - now <= horizon:
            return "rate_limit_cooldown"
    hold = base_cooldown
    ceiling = _resolve_rate_limit_max_cooldown_seconds()
    for _ in range(max(0, _rate_limit_streak(conn, task_id) - 1)):
        if hold >= ceiling:
            break
        hold = min(hold * 2, ceiling)
    return "rate_limit_cooldown" if (now - ended_at) < hold else None


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review row before any claim attempt. Priority order:
    ``"infrastructure_cooldown"`` (latest run is a ``spawn_failed`` the host
    refused — no restart-safe scope — within the cooldown; never counted),
    ``"rate_limit_cooldown"`` (latest run ``rate_limited``: held to the provider's
    reset instant when the run recorded one, otherwise to a hold that doubles per
    consecutive quota-wall run up to a ceiling — so a weekly wall costs spaced
    attempts instead of a respawn every cooldown; checked BEFORE ``blocker_auth``
    because the requeue stamps a quota-flavored ``last_failure_error`` that would
    otherwise park the task forever — that path never increments
    ``consecutive_failures``), ``"blocker_auth"``
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
            ended_at = _kb._to_epoch(latest_run["ended_at"])
            if ended_at is not None and (now - ended_at) < rl_cooldown:
                return "infrastructure_cooldown"
    if latest_run is not None and latest_run["outcome"] == "rate_limited":
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, skipping blocker_auth so
            # the stamped rate-limit text doesn't re-trap the task.
            return None
        # Hold elapsed returns None here (not a fall-through): the stamped quota text would
        # otherwise re-trap the task in ``blocker_auth``.
        return rate_limit_hold(conn, task_id, latest_run, now, rl_cooldown)

    # 2. Quota / auth blocker: retrying immediately will not help.  A plain
    # crash is different: its persisted error includes the worker's last
    # captured output, which is context rather than a diagnosis and may contain
    # benign commands such as ``claude auth status`` (#117097).
    err = _kb._lossy_text(row["last_failure_error"])
    latest_outcome = latest_run["outcome"] if latest_run is not None else None
    if err and latest_outcome != "crashed" and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR URL
    # are the canonical *inputs* to a review handoff, not duplicate-work signals.
    if lane == "review":
        return None

    # 3. Completed run within guard window. Exception: an explicit re-queue
    #    AFTER that success (done→ready drag, re-promotion, unblock, reclaim) is
    #    a deliberate "run it again" — otherwise a manual done→ready would sit
    #    silently held until the window elapses.
    success = max((t for t, _ in _rows_since(
        conn, "task_runs", task_id, now - _RESPAWN_GUARD_SUCCESS_WINDOW, "AND outcome = 'completed'")), default=None)
    if success is not None and not any(_rows_since(
            conn, "task_events", task_id, success, "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed')")):
        return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    #    Exception: a handoff AFTER the newest PR comment (operator reassign,
    #    reviewer changes_requested, review reopen) names the profile that must
    #    now work on THAT PR — a closer or the implementer finishing it, not a
    #    duplicate implementation (#111910). A crash/reclaim is not a handoff,
    #    so the worker that opened the PR is still not re-spawned against it.
    pr_times = [t for t, c in _rows_since(conn, "task_comments", task_id, now - _RESPAWN_GUARD_PR_WINDOW,
                                          cols=", body", unreadable=now)  # duplicate-PR risk: fail closed
                if _RESPAWN_GUARD_PR_URL_RE.search(_kb._lossy_text(c["body"]) or "")]
    if pr_times:
        # Strictly after: a same-second tie stays guarded (fail closed).
        if any(_is_handoff_event(e["kind"], e["payload"]) for _, e in _rows_since(
                conn, "task_events", task_id, max(pr_times),
                "AND kind IN ('assigned', 'changes_requested', 'review_reopened')", cols=", kind, payload", strict=True)):
            return None
        return "active_pr"

    return None


def _rows_since(conn: sqlite3.Connection, table: str, task_id: str, since: int, where: str = "", *,
                cols: str = "", strict: bool = False, unreadable: Optional[int] = None) -> Iterator[tuple[int, Any]]:
    """``(epoch, row)`` for ``task_id``'s ``table`` rows timed at/after ``since`` (after, if ``strict``).
    SQLite sorts TEXT above every INTEGER, so the indexed range also returns every text stamp (written
    outside the mutators, #97455); those are decided on their parsed epoch. An unreadable time is
    dropped (never guards or lifts) unless the caller substitutes ``unreadable``."""
    col = _GUARD_TIME_COLS[table]
    sql = f"SELECT {col}{cols} FROM {table} WHERE task_id = ? {where} AND {col} {'>' if strict else '>='} ?"
    for row in conn.execute(sql, (task_id, since)):
        t = _kb._to_epoch(row[0])
        t = unreadable if t is None else t
        if t is not None and (t > since if strict else t >= since):
            yield t, row


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


from hermes_cli import kanban_db as _kb
