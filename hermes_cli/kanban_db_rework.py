"""The rework-handoff contract: when an ``active_pr`` hold may be lifted.

A ``ready`` card whose recent comment names a GitHub PR is held for 24h by
``check_respawn_guard`` (``hermes_cli.kanban_db_dispatch``): re-spawning the
implementer risks a duplicate PR (#111910). Two operator actions legitimately
end that hold after rework, and both record themselves on the card:

* ``unblocked`` — ``hermes kanban unblock --reason <marker>`` (the reason rides
  the event payload; a comment alone is not evidence);
* ``promoted_manual`` — ``hermes kanban promote --reason <marker>``.

The reason carries the marker ``decisao=<class> pr=<n> head=<sha> causa=<cause>
resume=<action>``. Only a fully-formed marker naming THE PR the newest comment
armed the guard with lifts it: the ``UNBLOCK:`` comment beside the event is
prose any writer can post, so it is never authority. Everything here is
fail-closed — parsing, ordering, and both closed lists.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Optional

# Guard tables carry their timestamp in one column each; ``task_runs`` ends a
# run at ``ended_at``, events and comments are created at ``created_at``.
_GUARD_TIME_COLS = {
    "task_runs": "ended_at",
    "task_events": "created_at",
    "task_comments": "created_at",
}

# A GitHub PR URL; group 1 is the PR number the guard keys on.
_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)",
    re.IGNORECASE,
)

# ``resume`` must match the event kind: an unblock cannot authorise a promotion
# and a promotion cannot authorise an unblock.
_REWORK_RESUME_KINDS = {"unblocked": "unblock", "promoted_manual": "promote"}

# What the emitter decided. Closed on purpose: this is the class of rework the
# operator observed, and free text here would let any writer invent authority.
_REWORK_DECISIONS = frozenset({
    "ci-vermelho",
    "push-novo",
    "mudancas-pedidas",
    "aprovacao-dispensada",
    "fila-saiu",
})

# What the emitter observed: a GitHub ``synchronize`` / ``changes_requested``
# delivery, the auto-approver skipping a head, the queue label landing, or any
# CI conclusion it read (raw GitHub values use underscores; hyphenated aliases
# remain accepted for compatibility).
_REWORK_CAUSES = frozenset({
    "synchronize",
    "changes_requested",
    "auto-approve-skipped",
    "mergify-queue",
    "success",
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
    "timed-out",
    "action-required",
    "neutral",
    "skipped",
    "stale",
    "startup-failure",
    "in-progress",
    "queued",
})

_HEAD_RE = re.compile(r"[0-9a-fA-F]{7,40}")


def unblocked_payload(
    new_status: str, resume_status: str, reason: Optional[str],
) -> Optional[dict[str, Any]]:
    """Payload for the ``unblocked`` event: the historical keys, plus ``reason``.

    Without a reason this is exactly what the event always carried, so old call
    sites keep their payload byte for byte. The status keys are never rewritten:
    the reason is an audit trail, not a state override.
    """
    payload = (
        {"status": new_status, "resume_status": resume_status}
        if new_status != "ready" or resume_status != "ready"
        else None
    )
    return {**(payload or {}), "reason": reason} if reason else payload


def rows_since(
    conn, table: str, task_id: str, since: int, where: str = "", *,
    cols: str = "", strict: bool = False, unreadable: Optional[int] = None,
) -> Iterator[tuple[int, Any]]:
    """``(epoch, row)`` for ``task_id``'s ``table`` rows timed at/after ``since`` (after, if ``strict``).

    SQLite sorts TEXT above every INTEGER, so the indexed range also returns every text stamp (written
    outside the mutators, #97455); those are decided on their parsed epoch. An unreadable time is
    dropped (never guards or lifts) unless the caller substitutes ``unreadable``.
    """
    from hermes_cli import kanban_db as _kb

    col = _GUARD_TIME_COLS[table]
    sql = f"SELECT {col}{cols} FROM {table} WHERE task_id = ? {where} AND {col} {'>' if strict else '>='} ?"
    for row in conn.execute(sql, (task_id, since)):
        t = _kb._to_epoch(row[0])
        t = unreadable if t is None else t
        if t is not None and (t > since if strict else t >= since):
            yield t, row


def pr_numbers(body: str) -> frozenset[int]:
    """PR numbers named by GitHub PR URLs in a comment body."""
    return frozenset(int(number) for number in _PR_URL_RE.findall(body))


def _rework_field(reason: str, key: str) -> Optional[str]:
    """One ``key=value`` marker token from ``reason``; ``None`` unless it appears
    exactly once — a duplicated key is ambiguous, so it authorises nothing."""
    hits = re.findall(rf"(?:^|\s){re.escape(key)}=(\S+)", reason)
    return hits[0] if len(hits) == 1 else None


def handoff_lifts_active_pr(kind: str, payload: Optional[str], prs: frozenset) -> bool:
    """True only for a fully-formed rework marker on ``kind`` naming THE PR.

    ``unblocked``/``promoted_manual`` lift ``active_pr`` only when the payload's
    ``reason`` carries ``decisao=<class> pr=<n> head=<sha> causa=<cause>
    resume=<action>`` and every part checks out against the newest PR comment:
    the marker names that PR, ``head`` is a hex SHA, ``decisao`` and ``causa``
    are in their closed lists, and ``resume`` matches the event kind. Anything
    missing, duplicated or divergent keeps the guard — the alternative is a
    second worker on a live PR, which is what the guard exists to prevent.
    """
    from hermes_cli import kanban_db as _kb

    if len(prs) != 1:
        return False
    data = _kb._json_or(payload, None)
    if not isinstance(data, dict):
        return False
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False
    number = _rework_field(reason, "pr")
    if number is None or not number.isdigit() or int(number) not in prs:
        return False
    head = _rework_field(reason, "head")
    if head is None or _HEAD_RE.fullmatch(head) is None:
        return False
    if _rework_field(reason, "decisao") not in _REWORK_DECISIONS:
        return False
    if _rework_field(reason, "causa") not in _REWORK_CAUSES:
        return False
    return _rework_field(reason, "resume") == _REWORK_RESUME_KINDS.get(kind)


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed.
    """
    from hermes_cli import kanban_db as _kb

    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def active_pr_engaged(conn, task_id: str, *, since: int, unreadable: int) -> bool:
    """True when a PR named in a recent comment still holds the card.

    The guard is armed by any comment younger than the window that names a
    GitHub PR. It lifts only for a handoff STRICTLY after the newest such
    comment: a reassign to another profile, reviewer ``changes_requested``,
    review reopen, or an ``unblocked``/``promoted_manual`` carrying the
    validated rework marker (``handoff_lifts_active_pr``). An event in the
    same second stays guarded, and a crash/reclaim is not a handoff at all —
    the worker that opened the PR is not re-spawned against it.
    """
    pr_comments = [
        (t, pr_numbers(_kb_lossy(c["body"])))
        for t, c in rows_since(conn, "task_comments", task_id, since,
                               cols=", body", unreadable=unreadable)
    ]
    pr_comments = [(t, numbers) for t, numbers in pr_comments if numbers]
    if not pr_comments:
        return False
    newest = max(t for t, _ in pr_comments)
    # Several distinct PRs in the newest comment is ambiguous: a marker then
    # names no single PR to work on, so it lifts nothing (fail closed).
    newest_prs = frozenset().union(*(numbers for t, numbers in pr_comments if t == newest))
    for _, event in rows_since(
            conn, "task_events", task_id, newest,
            "AND kind IN ('assigned', 'changes_requested', 'review_reopened', "
            "'unblocked', 'promoted_manual')", cols=", kind, payload", strict=True):
        if event["kind"] in _REWORK_RESUME_KINDS:
            if handoff_lifts_active_pr(event["kind"], event["payload"], newest_prs):
                return False
        elif _is_handoff_event(event["kind"], event["payload"]):
            return False
    return True


def _kb_lossy(value: Any) -> str:
    """``kanban_db._lossy_text`` for a comment body (late import: no cycle)."""
    from hermes_cli import kanban_db as _kb

    return _kb._lossy_text(value) or ""
