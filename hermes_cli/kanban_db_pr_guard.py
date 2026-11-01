"""PR-state and card-lane awareness for the dispatcher's ``active_pr`` guard.

Split out of ``hermes_cli.kanban_db_dispatch`` — the code-health ratchet does
not let that file grow, and this is behaviour rather than wiring. The guard
itself stays in the dispatcher; the bounded ``gh`` lookup, its cache, the
"can this card publish a PR" heuristic, the handoff rule and the operator
``unguard`` lever live here.

``kanban_db`` and ``kanban_db_dispatch`` are imported inside functions only: the
facade imports the dispatcher at its tail and the dispatcher imports this module
at its top, so a module-level import either way would close a cycle.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import time
from typing import Optional

# Window (seconds) in which a GitHub PR URL in a comment blocks re-spawn.
# Overridable via ``HERMES_KANBAN_RESPAWN_GUARD_PR_WINDOW_SECONDS`` or
# ``kanban.respawn_guard_pr_window_seconds``; 0 disables the guard.
DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS = 86400  # 24 hours

RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)

_PR_URL_PARTS_RE = re.compile(
    r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)",
    re.IGNORECASE,
)

# ``gh api`` lookups are cached — successes AND failures — so a host without
# ``gh`` (or without auth) does not fork a subprocess per card per tick.
_RESPAWN_GUARD_PR_CACHE_SECONDS = 300
_PR_STATE_CACHE: dict[str, tuple[int, Optional[str]]] = {}


def _resolve_respawn_guard_pr_window_seconds() -> int:
    """``active_pr`` window: env override, then ``kanban.respawn_guard_pr_window_seconds``.

    ``0`` disables the PR guard; a negative or non-integer value falls back to
    the default. Read here (not threaded through ``DispatchContext``) so the
    dispatcher picks up a config edit without a restart, like the rate-limit
    cooldown.
    """
    raw = os.environ.get("HERMES_KANBAN_RESPAWN_GUARD_PR_WINDOW_SECONDS", "").strip()  # health: allow HX002 -- per-process pin for tests and one-off dispatcher runs; config.yaml is the user-facing knob
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS
        return value if value >= 0 else DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS
    try:
        # Local import: ``hermes_cli.config`` imports the kanban facade.
        from hermes_cli.config import load_config

        configured = (load_config().get("kanban") or {}).get("respawn_guard_pr_window_seconds")
        if isinstance(configured, bool) or not isinstance(configured, (int, str)):
            return DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS
        value = int(configured)
    except Exception:  # health: allow BLE001 -- any config read/parse failure means the default window (fail safe)
        return DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS
    return value if value >= 0 else DEFAULT_RESPAWN_GUARD_PR_WINDOW_SECONDS


def _pr_url_parts(url: str) -> Optional[tuple[str, str, int]]:
    """``(owner, repo, number)`` for a GitHub PR URL, or ``None`` when it is not one."""
    match = _PR_URL_PARTS_RE.fullmatch(url.strip()) if isinstance(url, str) else None
    if match is None:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def _lookup_pr_state(url: str) -> Optional[str]:
    """One bounded ``gh api`` call: ``"OPEN"`` / ``"MERGED"`` / ``"CLOSED"`` or ``None``."""
    parts = _pr_url_parts(url)
    if parts is None:
        return None
    owner, repo, number = parts
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/pulls/{number}",
             "--jq", '.state + " " + (.merged|tostring)'],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        # ``gh`` absent (FileNotFoundError) or the call timed out.
        return None
    if result.returncode != 0:
        return None
    tokens = (result.stdout or "").strip().lower().split()
    if tokens and tokens[0] == "open":
        return "OPEN"
    if tokens == ["closed", "true"]:
        return "MERGED"
    if tokens == ["closed", "false"]:
        return "CLOSED"
    return None


def _reset_pr_state_cache() -> None:
    """Clear the ``gh`` PR-state cache (tests isolate on it)."""
    _PR_STATE_CACHE.clear()


def _resolve_pr_state(url: str) -> Optional[str]:
    """Cached :func:`_lookup_pr_state` — failures are cached too (fail closed, no fork storm)."""
    now = int(time.time())
    cached = _PR_STATE_CACHE.get(url)
    if cached is not None and now - cached[0] < _RESPAWN_GUARD_PR_CACHE_SECONDS:
        return cached[1]
    state = _lookup_pr_state(url)
    _PR_STATE_CACHE[url] = (now, state)
    return state


def _task_publishes_pr(conn: sqlite3.Connection, task_id: str, row: sqlite3.Row) -> bool:
    """Whether ``task_id`` is implementer work that can open a PR (the ``active_pr`` lane).

    ``completion_contract`` is the card's own declaration: an unset/empty value
    stays PR-bearing for compatibility with legacy cards (fail closed); anything
    other than exactly ``local-only`` (a repo slug or a PR URL) is PR-bearing.
    ``local-only`` is the report/notify exemption — unless one of the card's runs
    actually recorded ``published_pr``, which makes it implementer work whatever
    it declared.

    A stored ``local-only`` alone is not proof of the declaration:
    ``validate_contract`` folds an *omitted* contract into the same value, so
    the exemption also requires the ``created`` event to record
    ``local-only`` as the raw creation argument. Every other ``local-only``
    row — legacy cards created before the argument was recorded, decompose
    sub-cards, a missing or unparseable payload — fails closed to guarded.
    """
    from hermes_cli import kanban_db as _kb

    contract = (row["completion_contract"] or "").strip()
    if not contract or contract != "local-only":
        return True
    for run in conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ?", (task_id,),
    ):
        if _kb._json_dict(run["metadata"]).get("published_pr"):
            return True
    # Only a card that explicitly declared ``local-only`` at creation time is
    # exempt; one indexed SELECT, reached solely on this no-published_pr
    # local-only path. Anything else stays guarded (fail closed).
    created = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created' "
        "ORDER BY id LIMIT 1",
        (task_id,),
    ).fetchone()
    if created is not None and _kb._json_dict(created["payload"]).get(
        "completion_contract"
    ) == "local-only":
        return False
    return True


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed.

    Every other kind in the guard's filter is a handoff by construction:
    ``changes_requested`` / ``review_reopened`` come from the review lane, and
    ``unguarded`` is an explicit operator action (``hermes kanban unguard``) —
    the sanctioned lever for a single-owner or report card that cannot be
    reassigned.
    """
    if kind == "unguarded":
        return True
    if kind != "assigned":
        return True
    from hermes_cli import kanban_db as _kb

    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def _pr_guard_reason(conn: sqlite3.Connection, task_id: str, row: sqlite3.Row, now: int,
                     window: int, resolve_state) -> Optional[str]:
    """The ``active_pr`` step: ``"active_pr"`` when duplicate work is still possible, else ``None``.

    ``resolve_state`` is the dispatcher's :func:`_resolve_pr_state`, passed in so
    tests patch one seam and the lookup stays cached per process. A handoff
    strictly after the newest PR comment lifts the guard; a card that cannot
    publish a PR (the ``local-only`` report lane) is not guarded at all and costs
    no lookup; and once every distinct referenced PR is merged or closed there is
    no duplicate left to open. An open PR, or an unresolved lookup, fails closed.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    pr_times: list[int] = []
    pr_urls: dict[tuple[str, str, int], str] = {}
    for t, c in _kbd._rows_since(conn, "task_comments", task_id, now - window,
                                 cols=", body", unreadable=now):  # duplicate-PR risk: fail closed
        body = _kb._lossy_text(c["body"]) or ""
        matches = list(RESPAWN_GUARD_PR_URL_RE.finditer(body))
        if not matches:
            continue
        pr_times.append(t)
        for match in matches:
            parts = _pr_url_parts(match.group(0))
            if parts is not None:
                pr_urls.setdefault(parts, match.group(0))
    if not pr_times:
        return None
    # Strictly after: a same-second tie stays guarded (fail closed).
    if any(_is_handoff_event(e["kind"], e["payload"]) for _, e in _kbd._rows_since(
            conn, "task_events", task_id, max(pr_times),
            "AND kind IN ('assigned', 'changes_requested', 'review_reopened', 'unguarded')",
            cols=", kind, payload", strict=True)):
        return None
    if not _task_publishes_pr(conn, task_id, row):
        return None
    if pr_urls and all(resolve_state(url) in ("MERGED", "CLOSED") for url in pr_urls.values()):
        return None
    return "active_pr"


def unguard_task(conn: sqlite3.Connection, task_id: str, *, actor: str) -> bool:
    """Lift the ``active_pr`` respawn guard explicitly (``hermes kanban unguard <id>``).

    Records an ``unguarded`` event, which ``check_respawn_guard`` treats as a
    handoff: the sanctioned lever when a card cannot be reassigned to a different
    profile (a single-owner card, or a report/notify card) and the referenced PR
    has merged or closed without the dispatcher's lookup seeing it. Returns
    ``False`` for an unknown id.
    """
    from hermes_cli import kanban_db as _kb

    with _kb.write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
            return False
        _kb._append_event(conn, task_id, "unguarded", {"by": actor})
    return True
