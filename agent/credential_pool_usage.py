"""Codex pool selection from cached allowance windows, without moving live sessions."""
from __future__ import annotations

import hashlib
import json
import logging
import math
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


def _next_reset(reading: dict | None, now: float) -> float | None:
    if reading is None or not _number(reading.get("fetched_at")):
        return None
    age = now - reading["fetched_at"]
    if not 0 <= age <= cache.SNAPSHOT_STALE_AFTER_S:
        return None
    windows = reading.get("windows")
    if not isinstance(windows, list) or not windows:
        return None
    resets = [_window_reset(window, now) for window in windows]
    if 0 in resets:
        return 0.0  # An active exhausted week still blocks when the short window has reset.
    return None if any(reset is None for reset in resets) else min(resets)


def _window_reset(window: object, now: float) -> float | None:
    if not isinstance(window, dict):
        return None
    used, reset, period = (window.get(key) for key in ("used_percent", "reset_at", "limit_window_seconds"))
    if not all(_number(value) for value in (used, reset, period)):
        return None
    if not 0 <= used <= 100 or period <= 0 or reset <= now:
        return None  # A crossed reset needs a new reading, not an assumed refill.
    return 0.0 if used == 100 else reset


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
        cold |= _next_reset(reading, now) is None
        # A recent persisted reading also throttles fresh CLI processes.
        if _next_reset(reading, now) is not None and now - reading["fetched_at"] < cache.REFRESH_AFTER_S:
            continue
        requests.append({"provider": provider, "identity_id": cache._identity_id_for(provider, entry),
                         "base_url": entry.base_url, "api_key": entry.runtime_api_key, "persist_usage": True})
    workers = cache.refresh_account_usage_entries_async(requests)
    if cold:
        deadline = time.monotonic() + _COLD_WAIT_SECONDS
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))


def choose_entry(provider: str, entries: list[PooledCredential]) -> PooledCredential:
    """Use the account resetting soonest; preserve round-robin for ties and unknown usage."""
    if provider != _PROVIDER:
        return entries[0]
    now = time.time()
    resets = [(entry, _next_reset(_reading(entry), now)) for entry in entries]
    known = [(entry, reset) for entry, reset in resets if reset is not None and reset > 0]
    if known:
        return min(known, key=lambda candidate: candidate[1])[0]
    return next((entry for entry, reset in resets if reset is None), entries[0])
