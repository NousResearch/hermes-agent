"""Money left per provider for the month view (``/api/analytics/month``), from the account-usage cache.

The month endpoint must not hang on provider APIs. A provider with a fresh cached snapshot refreshes in
the background for the next read; only a cold one (never fetched, or past the stale bound) gets one
bounded wait, so the first open of Command Center → Usage can still show its balance.
"""

from __future__ import annotations

import time
from typing import Optional

# One balance round-trip fits; a dead endpoint never stalls the panel for longer.
BALANCE_WAIT_S = 3.0


def balance_view(provider: str) -> Optional[dict]:
    """``None``: no money balance for this provider (no usage source, or one that reports only
    percentage windows). ``{"state": "unknown"}``: it has a source, but nothing fresh is known (not
    fetched yet, or the fetch failed): never rendered as an amount. ``{"state": "ready", ...}``: the
    amounts and when they were fetched."""
    from agent.account_usage_cache import cached_account_usage, has_account_usage, snapshot_is_stale

    if not has_account_usage(provider):
        return None
    snapshot = cached_account_usage(provider)
    if snapshot is None or snapshot_is_stale(snapshot):
        return {"state": "unknown"}
    if not snapshot.balances:
        return None
    return {
        "state": "ready",
        "fetched_at": snapshot.fetched_at.isoformat(),
        "amounts": [{"label": b.label, "amount": b.amount, "currency": b.currency} for b in snapshot.balances],
    }


def attach_balances(rows: list[dict], *, wait_s: float = BALANCE_WAIT_S) -> None:
    """Set ``row["balance"]`` (see :func:`balance_view`) on each month row. Cold providers are fetched
    and awaited for at most *wait_s* in total; warm ones refresh in the background. Runs inside the
    caller's profile scope, which the refresh threads inherit."""
    from agent.account_usage_cache import (
        cached_account_usage, has_account_usage, refresh_account_usage_async, snapshot_is_stale,
    )

    supported = [row["provider"] for row in rows if has_account_usage(row["provider"])]
    cold = [provider for provider in supported if snapshot_is_stale(cached_account_usage(provider))]
    deadline = time.monotonic() + wait_s
    for thread in refresh_account_usage_async(cold):
        thread.join(max(0.0, deadline - time.monotonic()))
    refresh_account_usage_async(provider for provider in supported if provider not in cold)
    for row in rows:
        row["balance"] = balance_view(row["provider"])
