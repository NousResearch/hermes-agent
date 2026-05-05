"""Idempotent user-list → routing-table sync.

This is the *core* of feishu-sync — the integration with Feishu's HR APIs is
left to the deployment (cron / webhook / manual export). All this module
asserts is:

  Given a desired ``users`` list, mutate ``RoutingTable`` so the active set
  matches. Specifically:
    * Insert / refresh routes present in the list.
    * Soft-delete routes present in the table but missing from the list.
    * Do nothing for matching rows (no version bumps, no last_active_at
      churn — those are router-only fields).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class UserSpec:
    """One user → one profile route. Mirrors the columns feishu-sync writes."""
    user_id: str
    profile_name: str
    open_id: str
    union_id: Optional[str] = None


def apply_users(table, users: Iterable[UserSpec]) -> dict[str, int]:
    """Reconcile ``table`` to match ``users``.

    Returns a counters dict::

        {"upserted": int, "soft_deleted": int, "kept": int}

    The numbers are per-row-action, useful for ops dashboards and tests.
    """
    desired = {u.user_id: u for u in users}

    # Snapshot current active rows so we can detect deletes.
    cur = table._conn.execute(
        "SELECT user_id, profile_name, open_id, union_id"
        " FROM multitenancy_routing WHERE active = 1"
    )
    current = {r["user_id"]: dict(r) for r in cur.fetchall()}

    upserted = 0
    soft_deleted = 0
    kept = 0

    # 1. Upsert / refresh
    for u in desired.values():
        existing = current.get(u.user_id)
        if existing and (
            existing["profile_name"] == u.profile_name
            and existing["open_id"] == u.open_id
            and (existing["union_id"] or None) == u.union_id
        ):
            kept += 1
            continue
        table.upsert(
            user_id=u.user_id,
            profile_name=u.profile_name,
            open_id=u.open_id,
            union_id=u.union_id,
        )
        upserted += 1

    # 2. Soft-delete rows in table but missing from desired
    for user_id in current:
        if user_id not in desired:
            if table.soft_delete(user_id):
                soft_deleted += 1

    return {"upserted": upserted, "soft_deleted": soft_deleted, "kept": kept}
