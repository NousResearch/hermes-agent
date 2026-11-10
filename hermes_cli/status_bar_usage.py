"""The opt-in ``weekly_usage`` status-bar field (``wk 49%``): the share of the provider's weekly
allowance already spent.

The number comes from the provider's usage API (:mod:`agent.account_usage`), so the bar never
fetches it on the render path. It reads the shared cache (:mod:`agent.account_usage_cache`) and
asks for a background refresh, which that cache throttles to one try per provider every
``REFRESH_AFTER_S``. Before the first fetch lands, for a provider without a usage API, and for a
snapshot past the cache's staleness bound, the segment stays hidden instead of guessing.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

logger = logging.getLogger(__name__)

# Labels Hermes's usage fetchers give a provider's account-wide weekly window: Anthropic's
# ``seven_day`` ("Current week"), Codex's 7-day window and plugin providers such as opencode-zen
# ("Weekly"). Model-scoped weeks (Anthropic's Opus/Sonnet) never stand for the account.
WEEKLY_WINDOW_LABELS = frozenset({"Current week", "Weekly"})


def weekly_window(snapshot: Optional[AccountUsageSnapshot]) -> Optional[AccountUsageWindow]:
    """The account-scoped weekly window that carries a reading, or None."""
    for window in getattr(snapshot, "windows", ()) or ():
        if window.scope == "account" and window.label in WEEKLY_WINDOW_LABELS and window.used_percent is not None:
            return window
    return None


def weekly_usage_percent(provider: Optional[str]) -> Optional[int]:
    """Percent of *provider*'s weekly allowance spent, rounded up so the bar never understates how
    close the wall is; None while unknown. Never waits on the network, and never raises: it runs on
    every repaint, where a failure would cost the whole bar."""
    try:
        return _cached_weekly_usage_percent(provider)
    except Exception:
        logger.debug("weekly usage reading failed for provider %r", provider, exc_info=True)
        return None


def _cached_weekly_usage_percent(provider: Optional[str]) -> Optional[int]:
    from agent.account_usage_cache import (
        cached_account_usage, has_account_usage, refresh_account_usage_async, snapshot_is_stale,
    )

    slug = str(provider or "").strip().lower()
    if not slug or not has_account_usage(slug):
        return None
    refresh_account_usage_async([slug])
    snapshot = cached_account_usage(slug)
    if snapshot_is_stale(snapshot):
        return None
    window = weekly_window(snapshot)
    used = window.used_percent if window is not None else None
    if used is None:
        return None
    return max(0, min(100, math.ceil(float(used))))
