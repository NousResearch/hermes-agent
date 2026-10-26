"""The respawn guard: whether a ready/review card may be spawned again yet.

Split out of ``hermes_cli.kanban_db_dispatch``; that module stays the facade and
re-exports these names, so ``kanban_db_dispatch.check_respawn_guard`` and the
``_RESPAWN_*`` constants keep resolving for callers and tests.

``tasks.last_failure_error`` is read here as a CACHE of the last run that failed,
never as the latest run's verdict: the requeue path stamps it and a later handoff
never clears it. Every attribution question (which run, which lane, how long ago)
is settled in this module, so the rest of the dispatcher never has to reason about
a stale column.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any
from typing import Iterator
from typing import Optional

# Late-bound origin namespace: the facade imports this module at its own bottom,
# so the import order is settled by then (same seam the facade's own ``_kb`` uses).
from hermes_cli import kanban_db as _kb

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

# Outcomes that prove ``tasks.last_failure_error`` describes a SUPERSEDED run: the
# latest run ended by handing the card to its next owner (or finishing it) instead of
# failing it, and nothing on that path clears the column. Deliberately a denylist, not
# a list of failure outcomes: an outcome this dispatcher does not know yet must keep
# today's ``blocker_auth`` behaviour, so an unfamiliar run fails closed rather than
# letting a genuine quota/auth wall spin against the provider every tick.
_RESPAWN_SUPERSEDING_OUTCOMES = frozenset({"completed", "review_requested", "changes_requested"})

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
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
    (quota/auth pattern in a failure text still attributable to the latest run —
    a text a later handoff run superseded is stale evidence, see
    ``_RESPAWN_SUPERSEDING_OUTCOMES``; the breaker still trips eventually), then
    for the ready lane only ``"recent_success"`` (completed run within the window, unless
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
        ended_at = _kb._to_epoch(latest_run["ended_at"])
        if ended_at is not None and (now - ended_at) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — return early so blocker_auth doesn't catch the
        # stamped rate-limit text; this path intentionally retries forever
        # (spaced by the cooldown) until quota returns or a real run supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.  A plain
    # crash is different: its persisted error includes the worker's last
    # captured output, which is context rather than a diagnosis and may contain
    # benign commands such as ``claude auth status`` (#117097).
    #
    # The column is a CACHE of the last run that failed, not of the latest run: the
    # rate-limited requeue stamps it (``_reclaim_dead_workers``) and a later handoff
    # run never clears it. Reading the text alone therefore parks a card forever on
    # evidence about a run that has since been superseded — the review lane first,
    # because this check runs before its exemption. Attribute the text to the latest
    # run before trusting it; a superseded one leaves nothing for ``blocker_auth``.
    err = _kb._lossy_text(row["last_failure_error"])
    latest_outcome = latest_run["outcome"] if latest_run is not None else None
    if (err and latest_outcome not in _RESPAWN_SUPERSEDING_OUTCOMES
            and latest_outcome != "crashed" and _RESPAWN_BLOCKER_RE.search(err)):
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


_GUARD_TIME_COLS = {"task_runs": "ended_at", "task_events": "created_at", "task_comments": "created_at"}


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
