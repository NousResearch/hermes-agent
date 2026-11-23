"""The respawn circuit breaker: quota/auth blocker patterns, the profile allowlist, and spawnable lane probes."""

from __future__ import annotations

from hermes_cli.kanban_db_dispatch import (
    _kb,
)

from typing import Any
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Optional
import re
import sqlite3
import time

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
    #    condition is not the card's (its text may still say 403), so it retries
    #    forever, spaced, and reaches neither the breaker nor blocker_auth.
    rl_cooldown = _kb._resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT 1",  # runs end on whole seconds: the later insert wins a tie
        (task_id,),
    ).fetchone()
    infrastructure = (latest_run is not None and latest_run["outcome"] == "spawn_failed"
                      and bool(_kb._json_dict(latest_run["metadata"]).get("infrastructure")))
    ended_at = _kb._to_epoch(latest_run["ended_at"]) if infrastructure else None
    if rl_cooldown > 0 and ended_at is not None and (now - ended_at) < rl_cooldown:
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
    err = _kb._lossy_text(row["last_failure_error"])
    latest_outcome = latest_run["outcome"] if latest_run is not None else None
    if err and latest_outcome != "crashed" and not infrastructure and _RESPAWN_BLOCKER_RE.search(err):
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

def _profile_exists_fn() -> Optional[Callable[[str], bool]]:
    """``profile_exists`` from ``hermes_cli.profiles``; ``None`` when it cannot be
    imported and no allowlist is set (callers then trust the assignee).

    With ``kanban.dispatch_profiles`` set (#110995) the predicate also requires the
    assignee to be listed, fail-closed: a ``default`` card is only claimable by homes that
    opted in, and a set allowlist whose gate cannot be built claims nothing (#147531)."""
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        if _dispatch_profile_allowlist(str) is None:
            return None
        _kb._log.warning("kanban: hermes_cli.profiles unimportable — claiming no cards", exc_info=True)
        return lambda _name: False
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
    behavior: any existing profile is claimable). A present value is fail-closed:
    an empty list, ``null`` or a bare ``dispatch_profiles:`` claims nothing, and
    so does a non-mapping ``kanban`` section (a bare ``kanban:`` is absent) or an
    unreadable config (#113620, #147532) — the user layer is read without the
    ``DEFAULT_CONFIG`` merge, whose ``None`` placeholder fakes the key's presence.
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban")
        if kanban is not None and not isinstance(kanban, Mapping):
            raise TypeError(f"the `kanban` section is {type(kanban).__name__}, not a mapping")
    except Exception as exc:
        _kb._log.warning(
            "kanban: kanban.dispatch_profiles cannot be resolved (%s: %s) — this home claims no cards",
            type(exc).__name__, exc,
        )
        return frozenset()
    if "dispatch_profiles" not in (kanban or {}):
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
    """What this home may claim, for ``hermes kanban diagnostics`` (#113620): ``any``
    (key absent; also when profiles cannot import, as the gate then stays open too),
    the sorted allowed names, or ``none (fail-closed: ...)``, as `_profile_exists_fn`."""
    try:
        from hermes_cli.profiles import normalize_profile_name
    except Exception as exc:
        if _dispatch_profile_allowlist(str) is None:
            return "any"
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
    """True iff a ready+assigned+unclaimed task maps to a real Hermes profile.

    Lets health telemetry tell "stuck" (``0 spawned`` with spawnable work) from
    "correctly idle" (only control-plane lanes waiting on ``claim_task``). Falls
    back to "any assigned" when profiles cannot import and no allowlist is set.
    """
    return _has_spawnable(conn, "ready")

def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """:func:`has_spawnable_ready` for the review column."""
    return _has_spawnable(conn, "review")

def review_dispatch_enabled() -> bool:
    """Whether review tasks dispatch automatically. Default true (Hermes ships
    ``sdlc-review``); operators disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("kanban", {}).get("review_dispatch", True))
    except Exception:
        return True
