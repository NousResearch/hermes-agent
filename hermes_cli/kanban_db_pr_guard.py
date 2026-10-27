"""PR-state and card-lane awareness for the dispatcher's ``active_pr`` guard.

Split out of ``hermes_cli.kanban_db_dispatch`` — the code-health ratchet does
not let that file grow, and this is behaviour rather than wiring. The guard
itself stays in the dispatcher; the bounded ``gh`` lookup, its cache and the
"can this card publish a PR" heuristic live here.

``kanban_db`` is imported inside :func:`_task_publishes_pr` only: the facade
imports the dispatcher at its tail, so a module-level import here would close
an import cycle.
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
    other than exactly ``local-only`` (a repo slug or a PR URL) is PR-bearing;
    ``local-only`` is the report/notify exemption — unless one of the card's runs
    actually recorded ``published_pr``, which makes it implementer work whatever
    it declared.
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
    return False
