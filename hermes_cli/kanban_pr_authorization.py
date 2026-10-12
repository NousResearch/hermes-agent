"""Operator-scoped, single-use authorization for same-PR remediation dispatch.

The ``active_pr`` respawn guard holds a ready card while a recent comment links
a GitHub PR: a fresh worker would open a duplicate. When an operator wants the
SAME card to amend that PR (review remediation), ``hermes kanban
authorize-existing-pr`` records an authorization bound to the exact task, PR
URL, repository and head branch, after a read-only GitHub lookup proves the PR
is open and its head branch / repository match.

The guard lifts only while that authorization is unexpired AND every PR URL in
the guard window is the authorized one. It expires on the first run claimed
after it was granted: the grant stores the task's highest run id as a floor,
and any run above it ends the authorization. ``claim_task``'s status CAS lets
exactly one concurrent dispatcher claim the card, so the authorization yields
one dispatch; a crash, reclaim or later run cannot reuse it. It lives in
``task_events`` keyed by ``task_id``, so it can never lift another card's guard.

Nothing here creates, closes, merges, rebases or retargets a PR: the only
GitHub call is a read-only ``GET repos/{repo}/pulls/{number}``.
"""
from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from typing import Callable, Optional

AUTH_EVENT = "existing_pr_authorized"
CONSUMED_EVENT = "existing_pr_authorization_consumed"

_PR_URL = re.compile(r"https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_REPO = re.compile(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+")
# git check-ref-format subset: no whitespace, no leading '-', no '..', no trailing '/' or '.lock'.
_BRANCH = re.compile(r"(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+(?<![/.])(?<!\.lock)")
_GUARD_PR_URL = re.compile(r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+", re.IGNORECASE)

PrLookup = Callable[[str, int], dict]


class AuthorizationError(ValueError):
    """The authorization request was refused; no state was written."""


def parse_pr_url(url: str) -> tuple[str, int]:
    """``("OWNER/REPO", number)`` for an exact canonical PR URL, else raise."""
    match = _PR_URL.fullmatch(url or "")
    if match is None:
        raise AuthorizationError(
            f"malformed PR URL {url!r}: expected https://github.com/OWNER/REPO/pull/NUMBER")
    return f"{match[1]}/{match[2]}", int(match[3])


def github_pr_lookup(repo: str, number: int) -> dict:
    """Read-only PR fetch through the operator's ``gh`` login."""
    from hermes_cli.kanban_pr_acceptance import _api
    return _api(f"repos/{repo}/pulls/{number}")


def _check_inputs(task_id: str, pr_url: str, repo: str, branch: str, reason: str) -> int:
    if not (task_id or "").strip() or task_id != task_id.strip():
        raise AuthorizationError("an exact task id is required")
    url_repo, number = parse_pr_url(pr_url)
    if not _REPO.fullmatch(repo or ""):
        raise AuthorizationError(f"malformed repository {repo!r}: expected OWNER/REPO")
    if repo != url_repo:
        raise AuthorizationError(f"repository {repo!r} does not match PR URL repository {url_repo!r}")
    if not _BRANCH.fullmatch(branch or ""):
        raise AuthorizationError(f"malformed branch {branch!r}")
    if not (reason or "").strip():
        raise AuthorizationError("a reason is required")
    return number


def _check_pr(pr: dict, repo: str, branch: str) -> None:
    if not isinstance(pr, dict):
        raise AuthorizationError("GitHub returned no PR data")
    if pr.get("state") != "open" or pr.get("merged"):
        raise AuthorizationError(f"PR must be open; GitHub reports state={pr.get('state')!r}, merged={bool(pr.get('merged'))}")
    head = pr.get("head") or {}
    base_repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name") or ""
    head_repo = (head.get("repo") or {}).get("full_name") or ""
    if base_repo.lower() != repo.lower():
        raise AuthorizationError(f"PR base repository {base_repo!r} is not {repo!r}")
    if head_repo.lower() != repo.lower():
        raise AuthorizationError(f"PR head comes from {head_repo!r}, not {repo!r}")
    if head.get("ref") != branch:
        raise AuthorizationError(f"PR head branch {head.get('ref')!r} is not {branch!r}")


def _check_task(conn: sqlite3.Connection, task_id: str, pr_url: str, branch: str) -> sqlite3.Row:
    task = conn.execute(
        "SELECT id, status, body, branch_name FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task is None:
        raise AuthorizationError(f"unknown task {task_id!r}")
    if task["status"] in ("done", "archived"):
        raise AuthorizationError(f"task {task_id} is {task['status']}")
    if task["branch_name"] and task["branch_name"] != branch:
        raise AuthorizationError(
            f"task branch {task['branch_name']!r} does not match PR branch {branch!r}")
    texts = [task["body"] or ""] + [
        r["body"] or "" for r in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ?", (task_id,))]
    if not any(pr_url in _GUARD_PR_URL.findall(t) for t in texts):
        raise AuthorizationError(f"task {task_id} never referenced {pr_url}; not its existing PR")
    return task


def authorize_existing_pr(
    conn: sqlite3.Connection, task_id: str, *, pr_url: str, repo: str, branch: str,
    reason: str, operator: str, pr_lookup: Optional[PrLookup] = None,
) -> dict:
    """Validate every identifier, verify the PR live, then persist one audit event.

    Raises :class:`AuthorizationError` before any write when anything is off.
    """
    from hermes_cli import kanban_db as kb
    number = _check_inputs(task_id, pr_url, repo, branch, reason)
    if not (operator or "").strip():
        raise AuthorizationError("operator identity is required")
    _check_task(conn, task_id, pr_url, branch)
    try:
        pr = (pr_lookup or github_pr_lookup)(repo, number)
    except AuthorizationError:
        raise
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        # gh missing / not logged in / denied / bad JSON: fail closed, never persist gh stderr.
        raise AuthorizationError(f"could not verify PR state on GitHub ({type(exc).__name__})") from None
    _check_pr(pr, repo, branch)
    with kb.write_txn(conn):
        # Re-read under the write lock: the task may have moved since the lookup.
        _check_task(conn, task_id, pr_url, branch)
        floor = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0]
        payload = {
            "operator": operator.strip(), "task_id": task_id, "pr_url": pr_url, "repo": repo,
            "branch": branch, "reason": reason.strip(), "authorized_at": int(time.time()),
            "run_id_floor": int(floor), "pr_state": "open",
        }
        kb._append_event(conn, task_id, AUTH_EVENT, payload)
    return payload


def active_authorization(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """The newest grant for ``task_id`` while no run has been claimed since it."""
    from hermes_cli import kanban_db as kb
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1", (task_id, AUTH_EVENT)).fetchone()
    if row is None:
        return None
    grant = kb._json_or(row["payload"], {})
    if not isinstance(grant, dict) or grant.get("task_id") != task_id:
        return None
    floor = grant.get("run_id_floor")
    if not isinstance(floor, int):
        return None
    later = conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id = ? AND id > ? LIMIT 1", (task_id, floor)).fetchone()
    return None if later else grant


def permits_active_pr_bypass(conn: sqlite3.Connection, task_id: str, since: int) -> bool:
    """True only when an unexpired grant names EVERY PR URL commented since ``since``."""
    grant = active_authorization(conn, task_id)
    if grant is None:
        return False
    urls = set()
    for c in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?", (task_id, since)):
        urls.update(_GUARD_PR_URL.findall(c["body"] or ""))
    return urls == {grant["pr_url"]}


def record_consumed(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> None:
    """Audit the claim that spent the newest grant: the first run above its floor.

    Expiry itself is that run row (see :func:`active_authorization`); this event
    is the operator-facing receipt, written at most once per grant.
    """
    from hermes_cli import kanban_db as kb
    row = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (task_id, AUTH_EVENT)).fetchone()
    if row is None or run_id is None:
        return
    grant = kb._json_or(row["payload"], {})
    floor = grant.get("run_id_floor") if isinstance(grant, dict) else None
    if not isinstance(floor, int):
        return
    with kb.write_txn(conn):
        first = conn.execute(
            "SELECT MIN(id) FROM task_runs WHERE task_id = ? AND id > ?", (task_id, floor)).fetchone()[0]
        spent = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? AND id > ?",
            (task_id, CONSUMED_EVENT, row["id"])).fetchone()
        if first == run_id and not spent:
            kb._append_event(conn, task_id, CONSUMED_EVENT,
                             {"pr_url": grant.get("pr_url"), "operator": grant.get("operator")},
                             run_id=run_id)
