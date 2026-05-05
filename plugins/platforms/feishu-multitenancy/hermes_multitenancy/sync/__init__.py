"""feishu-sync keeps multitenancy_routing in sync with an external user list.

Reference implementation. Real deployments wire this into a Feishu HR webhook
(``feishu_hr.subscribe_events``); this package exposes the ``apply_users`` core
so a CLI, cron job, or webhook can all share it.

Contract:
  - ``apply_users(table, users)`` is idempotent. Re-running with the same
    list is a no-op (modulo synced_at + version bumps in the row).
  - Users present in the table but absent from ``users`` get **soft-deleted**.
    The plugin treats soft-deleted rows as routing-miss (per US-007 schema).
"""
from __future__ import annotations

from .feishu_hr import UserSpec, apply_users

__all__ = ["UserSpec", "apply_users"]
