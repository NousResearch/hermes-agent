"""Codex pool selection from cached allowance windows, without moving live sessions."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent import account_usage_cache as cache

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)
_PROVIDER = "openai-codex"
_COLD_WAIT_SECONDS = 2.0
# A reset a few seconds away must not attract every new session on an old reading.
_MIN_REMAINING_PERIOD = 0.05


def _path(identity: str) -> Path:
    from hermes_constants import get_hermes_home

    digest = hashlib.sha256(identity.encode()).hexdigest()
    return get_hermes_home() / "cache" / "credential-usage" / f"{digest}.json"


def _number(value: object) -> bool:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _windows(raw: dict) -> list[dict] | None:
    rate_limit = raw.get("rate_limit")
    if not isinstance(rate_limit, dict):
        return None
    windows = []
    for name in ("primary_window", "secondary_window"):
        window = rate_limit.get(name)
        if window is None:
            continue
        if not isinstance(window, dict):
            return None
        fields = {key: window.get(key) for key in ("used_percent", "reset_at", "limit_window_seconds")}
        if not all(_number(value) for value in fields.values()):
            return None
        if not 0 <= fields["used_percent"] <= 100 or fields["limit_window_seconds"] <= 0:
            return None
        windows.append(fields)
    return windows or None


def remember_usage(identity: str, snapshot: AccountUsageSnapshot | None) -> None:
    """Persist only numeric windows for opt-in routing, never raw payloads or credentials."""
    if snapshot is None or snapshot.provider != _PROVIDER or snapshot.unavailable_reason:
        return
    windows = _windows(snapshot.raw or {})
    if windows is None:
        return
    from utils import atomic_json_write
    from hermes_constants import mkdir_under_hermes_home

    path = _path(snapshot.identity or identity)
    try:
        mkdir_under_hermes_home(path.parent)
        atomic_json_write(path, {"fetched_at": snapshot.fetched_at.timestamp(), "windows": windows}, mode=0o600)
    except OSError:
        logger.debug("Could not persist credential usage", exc_info=True)


def _reading(entry: PooledCredential) -> dict | None:
    identity = cache._identity_id_for(_PROVIDER, entry)
    snapshot = cache.cached_account_usage(_PROVIDER, identity_id=identity)
    if snapshot is not None and not cache.snapshot_is_stale(snapshot) and not snapshot.unavailable_reason:
        windows = _windows(snapshot.raw or {})
        if windows is not None:
            return {"fetched_at": snapshot.fetched_at.timestamp(), "windows": windows}
    try:
        value = json.loads(_path(identity).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _score(reading: dict | None, now: float) -> float | None:
    if reading is None or not _number(reading.get("fetched_at")):
        return None
    age = now - reading["fetched_at"]
    if not 0 <= age <= cache.SNAPSHOT_STALE_AFTER_S:
        return None
    windows = reading.get("windows")
    if not isinstance(windows, list) or not windows:
        return None
    scores = [_window_score(window, now) for window in windows]
    if 0 in scores:
        return 0.0  # An active exhausted week still blocks when the short window has reset.
    return None if any(score is None for score in scores) else min(scores)


def _window_score(window: object, now: float) -> float | None:
    if not isinstance(window, dict):
        return None
    used, reset, period = (window.get(key) for key in ("used_percent", "reset_at", "limit_window_seconds"))
    if not all(_number(value) for value in (used, reset, period)):
        return None
    if not 0 <= used <= 100 or period <= 0 or reset <= now:
        return None  # A crossed reset needs a new reading, not an assumed refill.
    fraction_of_period = max((reset - now) / period, _MIN_REMAINING_PERIOD)
    return (1 - used / 100) / fraction_of_period


def prepare_selection(provider: str, entries: list[PooledCredential]) -> None:
    """Refresh outside the pool lock; a cold CLI gets a bounded chance to read its accounts."""
    if provider != _PROVIDER or len(entries) < 2:
        return
    now = time.time()
    requests = []
    cold = False
    for entry in entries:
        if entry.last_status in {"dead", "exhausted"}:
            continue
        reading = _reading(entry)
        cold |= _score(reading, now) is None
        # A recent persisted reading also throttles fresh CLI processes.
        if _score(reading, now) is not None and now - reading["fetched_at"] < cache.REFRESH_AFTER_S:
            continue
        requests.append({"provider": provider, "identity_id": cache._identity_id_for(provider, entry),
                         "base_url": entry.base_url, "api_key": entry.runtime_api_key, "persist_usage": True})
    workers = cache.refresh_account_usage_entries_async(requests)
    if cold:
        deadline = time.monotonic() + _COLD_WAIT_SECONDS
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))


def choose_entry(provider: str, entries: list[PooledCredential]) -> PooledCredential:
    """Weighted pacing across both windows; incomplete observations retain round-robin order."""
    if provider != _PROVIDER:
        return entries[0]
    now = time.time()
    weights = [_score(_reading(entry), now) for entry in entries]
    candidates = [(entry, weight) for entry, weight in zip(entries, weights) if weight != 0]
    if not candidates:
        return entries[0]  # Preserve the existing request/error-driven exhaustion path.
    if any(weight is None for _, weight in candidates):
        return candidates[0][0]
    return random.choices([entry for entry, _ in candidates], weights=[weight for _, weight in candidates], k=1)[0]
