"""The opt-in ``weekly_usage`` status-bar field (``wk 49%``): the share of the provider's weekly
allowance already spent.

The number comes from the provider's usage API (:mod:`agent.account_usage`), so the bar never
fetches it on the render path. It reads the shared cache (:mod:`agent.account_usage_cache`) and
asks for a background refresh, which that cache throttles to one try per provider every
``REFRESH_AFTER_S``. The segment stays hidden instead of guessing: before the first fetch lands,
for a provider without a usage API, for a fresh reading with no account-wide weekly window (an
Anthropic API key, OpenRouter; such a provider is asked again only once the reading goes stale),
for a credential pool of several accounts, and for a snapshot past the cache's staleness bound.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

logger = logging.getLogger(__name__)

# Labels Hermes's usage fetchers give a provider's account-wide weekly window: Anthropic's
# ``seven_day`` ("Current week"), Codex's 7-day window and plugin providers such as OpenCode Go
# (``opencode-go``, "Weekly"). Model-scoped weeks (Anthropic's Opus/Sonnet) never stand for the account.
WEEKLY_WINDOW_LABELS = frozenset({"Current week", "Weekly"})
# The cached reading belongs to one login, so a pool of this many accounts gets none: it may not be
# the account serving the session (the /model picker's rule for its usage gauge).
MULTI_ACCOUNT_POOL_SIZE = 2
# Readings arrive as floats: 7.000000000000001 is 7%, not a sliver over it, before rounding up.
_FLOAT_NOISE_DECIMALS = 6

# (provider, error type) pairs already warned about. A repaint runs every second, so a lasting
# failure warns once with its traceback and then logs one quiet line per repaint.
_warned_failures: set[tuple[str, str]] = set()


def weekly_window(snapshot: Optional[AccountUsageSnapshot]) -> Optional[AccountUsageWindow]:
    """The account-scoped weekly window that carries a finite reading, or None."""
    for window in (snapshot.windows if snapshot else ()):
        if (window.scope == "account" and window.label in WEEKLY_WINDOW_LABELS
                and window.used_percent is not None and math.isfinite(window.used_percent)):
            return window
    return None


def weekly_usage_percent(provider: Optional[str], credential_pool: Any = None) -> Optional[int]:
    """Percent of *provider*'s weekly allowance spent, rounded up so the bar never understates how
    close the wall is; None while unknown. *credential_pool* is the session's in-memory pool, if any.
    Never waits on the network, and never raises: it runs on every repaint, where a failure would
    cost the whole bar."""
    try:
        return _cached_weekly_usage_percent(provider, credential_pool)
    except Exception as error:
        if _first_failure(provider, error):
            logger.warning("weekly usage segment hidden for provider %r", provider, exc_info=True)
        else:
            logger.debug("weekly usage segment still hidden for provider %r: %s", provider, error)
        return None


def _first_failure(provider: Optional[str], error: Exception) -> bool:
    key = (str(provider), type(error).__name__)
    first = key not in _warned_failures
    _warned_failures.add(key)
    return first


def _cached_weekly_usage_percent(provider: Optional[str], credential_pool: Any) -> Optional[int]:
    from agent.account_usage_cache import (
        cached_account_usage, has_account_usage, refresh_account_usage_async, snapshot_is_stale,
    )

    if not provider or not has_account_usage(provider) or _pooled_accounts(credential_pool) >= MULTI_ACCOUNT_POOL_SIZE:
        return None
    snapshot = cached_account_usage(provider)
    fresh = not snapshot_is_stale(snapshot)
    window = weekly_window(snapshot) if fresh else None
    if fresh and window is None:
        return None  # this provider reports no weekly window: ask again once the reading goes stale
    refresh_account_usage_async([provider])
    if window is None:
        return None
    return max(0, min(100, math.ceil(round(float(window.used_percent), _FLOAT_NOISE_DECIMALS))))


def _pooled_accounts(credential_pool: Any) -> int:
    return len(credential_pool.entries()) if credential_pool is not None else 0
