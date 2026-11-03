"""Conservative account-quota evidence for the existing routing chain.

A verdict is an observation, not a reservation. Unknown evidence never authorizes a
new destination. Only Codex's native account windows currently have an unambiguous
shared-account contract; plugin/model-specific gauges deliberately remain unknown.
The bounded process-local cache is shared by main agents and threaded children,
partitioned by profile, endpoint and account (never by model).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from threading import RLock
import time
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class QuotaVerdict:
    state: str  # available | exhausted | unknown | fetch_failed
    reason: str
    reset_at: datetime | None = None


MAX_AGE_SECONDS = 60.0
FAILURE_TTL_SECONDS = 15.0
MAX_CACHE_ENTRIES = 256
_lock = RLock()
_cache: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()


def quota_group(provider: str, base_url: str, api_key: Any) -> tuple | None:
    """No implicit credential resolution: a probe must own an exact bound key."""
    if provider != "openai-codex" or not isinstance(api_key, str) or not api_key.strip():
        return None
    from agent.account_usage_cache import _identity_id_for
    from hermes_constants import hermes_home_key

    from agent.account_usage import _codex_backend_urls
    identity = _identity_id_for(provider, SimpleNamespace(runtime_api_key=api_key))
    return hermes_home_key(), provider, _codex_backend_urls(base_url)[0], identity


def evaluate_snapshot(snapshot: Any, *, now: datetime | None = None) -> QuotaVerdict:
    """Do not confuse snapshot.available (has information) with remaining quota."""
    now = now or datetime.now(UTC)
    if snapshot is None:
        return QuotaVerdict("fetch_failed", "usage_fetch_failed")
    if snapshot.provider != "openai-codex" or snapshot.source != "usage_api":
        return QuotaVerdict("unknown", "unsupported_quota_scope")
    fetched = snapshot.fetched_at
    if fetched.tzinfo is None or not 0 <= (now - fetched).total_seconds() <= MAX_AGE_SECONDS:
        return QuotaVerdict("unknown", "stale_snapshot")
    if snapshot.unavailable_reason:
        return QuotaVerdict("fetch_failed", "usage_unavailable")
    windows = [w for w in snapshot.windows if w.scope == "account"]
    valid = []
    for window in windows:
        used, reset = window.used_percent, window.reset_at
        if isinstance(used, bool) or not isinstance(used, (float, int)) or not math.isfinite(used) or used < 0:
            continue
        if reset is not None and (reset.tzinfo is None or reset <= now):
            continue  # a past reset invalidates the observation, not proof of recovery
        valid.append(window)
    exhausted = [w for w in valid if w.used_percent >= 100]
    if exhausted:
        resets = [w.reset_at for w in exhausted if w.reset_at is not None]
        return QuotaVerdict("exhausted", "account_window_exhausted", max(resets) if resets else None)
    if valid and len(valid) == len(windows):
        return QuotaVerdict("available", "observed_remaining_quota")
    return QuotaVerdict("unknown", "incomplete_quota_windows")


def account_quota(provider: str, base_url: str = "", api_key: Any = None) -> QuotaVerdict:
    group = quota_group(provider, base_url, api_key)
    if group is None:
        return QuotaVerdict("unknown", "unbound_or_unsupported_provider")
    from agent.account_usage import fetch_account_usage

    # Serialize refreshes so simultaneous children cannot stampede the usage endpoint.
    # Fetches use the read-only native path (15 s HTTP timeout); no credential rotation.
    with _lock:
        now = time.monotonic()
        cached = _cache.get(group)
        if cached is not None:
            expires, snapshot = cached
            verdict = evaluate_snapshot(snapshot)
            if now < expires:
                _cache.move_to_end(group)
                return verdict
        try:
            snapshot = fetch_account_usage(provider, base_url=base_url, api_key=api_key,
                                           read_only=True, identity_id=group[-1])
        except Exception:  # health: allow BLE001 -- optional preflight fails open; exception text/tracebacks may contain credentials
            # Never log exception text: providers can echo credentials in errors.
            snapshot = None
        verdict = evaluate_snapshot(snapshot)
        ttl = MAX_AGE_SECONDS if verdict.state in {"available", "exhausted"} else FAILURE_TTL_SECONDS
        if snapshot is not None:
            wall_now = datetime.now(UTC)
            resets = [(w.reset_at - wall_now).total_seconds() for w in snapshot.windows
                      if w.reset_at is not None and w.reset_at.tzinfo is not None and w.reset_at > wall_now]
            if resets:
                ttl = min(ttl, min(resets))
        _cache[group] = (time.monotonic() + ttl, snapshot)
        _cache.move_to_end(group)
        while len(_cache) > MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)
        return verdict


def invalidate_quota(provider: str, base_url: str = "", api_key: Any = None) -> None:
    group = quota_group(provider, base_url, api_key)
    if group is None:
        return
    with _lock:
        _cache.pop(group, None)
