"""What ``model.options`` knows about subscription usage without waiting on the network.

Usage windows come from a provider API (:func:`agent.account_usage.fetch_account_usage`), so a
picker reads this cache and asks for a background refresh instead of fetching inline. Every fetch
lands here, including the one ``session.usage`` runs after each turn, so a picker is at most one
turn or one refresh interval behind. Keyed per profile home: one process may serve many profiles.
"""

from __future__ import annotations

import threading
import time
from contextvars import copy_context
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot

# A picker opened again within this long reuses what it has; usage windows move in minutes, not seconds.
REFRESH_AFTER_S = 120.0

_lock = threading.Lock()
_snapshots: dict[tuple[str, str], "AccountUsageSnapshot"] = {}
_last_try: dict[tuple[str, str], float] = {}
_inflight: set[tuple[str, str]] = set()


def _key(provider: str) -> tuple[str, str]:
    from hermes_constants import hermes_home_key

    return hermes_home_key(), str(provider or "").strip().lower()


def has_account_usage(provider: str) -> bool:
    """Whether *provider* can report usage windows at all (a built-in fetcher or a plugin hook)."""
    from agent.account_usage import _USAGE_FETCHERS
    from providers import get_provider_profile
    from providers.base import ProviderProfile

    slug = str(provider or "").strip().lower()
    if slug in _USAGE_FETCHERS:
        return True
    profile = get_provider_profile(slug) if slug else None
    return profile is not None and type(profile).fetch_account_usage is not ProviderProfile.fetch_account_usage


def remember_account_usage(provider: Optional[str], snapshot: Optional["AccountUsageSnapshot"]) -> None:
    if provider and snapshot is not None and snapshot.windows:
        with _lock:
            _snapshots[_key(provider)] = snapshot


def cached_account_usage(provider: str) -> Optional["AccountUsageSnapshot"]:
    with _lock:
        return _snapshots.get(_key(provider))


def refresh_account_usage_async(providers: Iterable[str]) -> list[threading.Thread]:
    """Fetch usage in the background for each provider not tried within ``REFRESH_AFTER_S``. A failed
    or empty fetch still counts as a try, so a provider with no usage API isn't re-asked every open."""
    started: list[threading.Thread] = []
    now = time.monotonic()
    for provider in dict.fromkeys(providers):
        key = _key(provider)
        with _lock:
            if key in _inflight or now - _last_try.get(key, float("-inf")) < REFRESH_AFTER_S:
                continue
            _inflight.add(key)
            _last_try[key] = now
        thread = threading.Thread(target=copy_context().run, args=(_refresh, provider, key),
                                  name="hermes-account-usage-refresh", daemon=True)
        thread.start()
        started.append(thread)
    return started


def _refresh(provider: str, key: tuple[str, str]) -> None:
    from agent.account_usage import fetch_account_usage

    try:
        fetch_account_usage(provider)  # remembers its own result
    finally:
        with _lock:
            _inflight.discard(key)
