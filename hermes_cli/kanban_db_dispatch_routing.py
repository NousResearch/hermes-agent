"""Routing helpers split from the Kanban dispatcher facade."""
from __future__ import annotations

from typing import Any, Callable, Optional


def costar_admission_rejected(conn, row, *, dry_run: bool, write_txn: Callable, append_event: Callable) -> bool:
    """Record a CoStar refusal before any claim or worker spawn."""
    from hermes_cli.kanban_costar_policy import admission_reason

    reason = admission_reason(conn, row["id"], row["title"], row["body"])
    if reason is None:
        return False
    if not dry_run:
        with write_txn(conn):
            append_event(conn, row["id"], "costar_dispatch_rejected", {"reason": reason})
    return True


def resolve_default_assignee(
    default_assignee: Optional[str], profile_exists_fn: Callable[[], Optional[Callable[[str], bool]]],
) -> Optional[str]:
    """Return a configured default only when its profile is locally spawnable."""
    name = (default_assignee or "").strip() or None
    if name:
        profile_exists = profile_exists_fn()
        if profile_exists is not None and not profile_exists(name):
            return None
    return name
