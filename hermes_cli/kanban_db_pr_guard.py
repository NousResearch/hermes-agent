"""Active-PR respawn guard: decides whether a card already has a live PR of its own.

Split out of ``hermes_cli.kanban_db_dispatch``; ``check_respawn_guard`` reaches
:func:`_active_pr_reason` through a local import. Origin-resident helpers are
reached late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working, and the GitHub read goes through
``hermes_cli.kanban_pr_acceptance`` looked up at call time.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from typing import Optional


# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


def _active_pr_reason(conn: sqlite3.Connection, task_id: str, now: int) -> Optional[str]:
    """``"active_pr"`` when the card has a live PR of its own, else None.

    Comment evidence: a PR URL in a comment from the last 24h guards unless a
    handoff followed the comment (that comment is then skipped, and a newer own
    publication is still evaluated on its own timestamp). A direct parent's publication is foreign and
    ignored; a URL with no ownership evidence stays guarded; a URL the card
    owns (its contract or a run's ``published_pr``) guards until a fresh read
    proves it closed. With no comment, only an actual ``published_pr`` run
    ended within 24h guards, timed per URL; a contract alone (never run) is an
    input, not a publication, and does not. The card's own PR is read at most
    once per call; a second unresolved own PR fails closed.
    """
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    own_prs = _published_pr_urls(conn, [task_id])
    foreign_prs = _published_pr_urls(
        conn,
        [r[0] for r in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?", (task_id,),
        ).fetchall()],
    ) - own_prs
    reads: dict[str, bool] = {}  # url -> own PR proven closed

    def closed(url: str) -> bool:
        if url not in reads:
            if reads:  # the one lookup is spent: fail closed
                return False
            reads[url] = _own_pr_is_closed(conn, task_id, url)
        return reads[url]

    for c in conn.execute(
        "SELECT body, created_at FROM task_comments "
        "WHERE task_id = ? AND created_at >= ? ORDER BY created_at DESC",
        (task_id, pr_cutoff),
    ).fetchall():
        body = _kb._lossy_text(c["body"])
        urls = sorted(set(_RESPAWN_GUARD_PR_URL_RE.findall(body)) - foreign_prs) if body else []
        if not urls:
            continue
        if _handoff_after(conn, task_id, int(c["created_at"] or 0)):
            continue  # this comment is stale evidence; a newer publication may still guard
        for url in urls:
            if url not in own_prs or not closed(url):
                return "active_pr"

    published = _own_publication_times(conn, task_id)
    for url in sorted(published):
        if url in reads or published[url] < pr_cutoff:
            continue
        if _handoff_after(conn, task_id, published[url]):
            continue
        if not closed(url):
            return "active_pr"
    return None


def _own_publication_times(conn: sqlite3.Connection, task_id: str) -> dict[str, int]:
    """PR URL -> newest ``ended_at`` of a run of this card that published it."""
    times: dict[str, int] = {}
    for r in conn.execute(
        "SELECT metadata, ended_at FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL",
        (task_id,),
    ).fetchall():
        url = _kb._json_dict(r["metadata"]).get("published_pr")
        if isinstance(url, str) and _RESPAWN_GUARD_PR_URL_RE.fullmatch(url):
            times[url] = max(times.get(url, 0), int(r["ended_at"]))
    return times


def _handoff_after(conn: sqlite3.Connection, task_id: str, since: int) -> bool:
    """True when a handoff event landed strictly after ``since`` (a same-second
    tie stays guarded: fail closed)."""
    return any(
        _is_handoff_event(e["kind"], e["payload"])
        for e in conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND created_at > ? "
            "AND kind IN ('assigned', 'changes_requested', 'review_reopened')",
            (task_id, since),
        ).fetchall()
    )


def _published_pr_urls(conn: sqlite3.Connection, task_ids: list) -> set[str]:
    """Exact PR URLs the given tasks published: ``completion_contract`` or a
    run's ``metadata.published_pr``."""
    found: set[str] = set()
    for tid in task_ids:
        values = [r[0] for r in conn.execute(
            "SELECT completion_contract FROM tasks WHERE id = ?", (tid,),
        ).fetchall()]
        for r in conn.execute("SELECT metadata FROM task_runs WHERE task_id = ?", (tid,)).fetchall():
            values.append(_kb._json_dict(r[0]).get("published_pr"))
        found.update(
            v for v in values
            if isinstance(v, str) and _RESPAWN_GUARD_PR_URL_RE.fullmatch(v)
        )
    return found


def _own_pr_is_closed(conn: sqlite3.Connection, task_id: str, url: str) -> bool:
    """Fresh read of the card's own PR as its assignee profile; True only when
    the exact PR reports ``closed`` (merged or not). An expected read failure or
    malformed answer is False."""
    from hermes_cli import kanban_pr_acceptance as kpa
    try:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assignee = row["assignee"] if row else None
        if not assignee:
            return False  # no profile to read as: never fall back to ambient identity
        home = kpa._assignee_profile_home(assignee)
        if not home:
            return False
        repo, number = url.split("/pull/")
        repo = repo.split("github.com/", 1)[1]
        pr = kpa._api(f"repos/{repo}/pulls/{int(number)}", profile_home=home)
        return (
            isinstance(pr, dict)
            and pr.get("html_url") == url
            and pr.get("number") == int(number)
            and pr.get("state") == "closed"
        )
    except (kpa._GateAuthError, OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        return False


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


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` is touched.
from hermes_cli import kanban_db as _kb
