"""Profile-scoped, best-effort usage ranking for Codex subscription credentials."""
from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING

from agent.credential_persistence import fingerprint_secret_value
from hermes_constants import hermes_home_key

if TYPE_CHECKING:
    from agent.credential_pool import CredentialPool, PooledCredential

_cache: dict[tuple, tuple] = {}
_cache_lock = threading.Lock()


def _settings() -> dict:
    from agent.credential_pool import _load_config_safe

    raw = (_load_config_safe() or {}).get("codex_usage") or {}
    defaults = {"cache_ttl": 180.0, "fetch_timeout": 5.0, "fetch_budget": 2, "exhausted_pct": 98.0}
    values = {}
    for key, default in defaults.items():
        try:
            value = float(raw.get(key, default))
            if not math.isfinite(value) or value < 0 or (key == "fetch_timeout" and value == 0):
                value = default
            if key == "exhausted_pct" and not 0 < value <= 100:
                value = default
        except (AttributeError, TypeError, ValueError):
            value = default
        values[key] = int(value) if key == "fetch_budget" else value
    return values


def _cache_key(entry: PooledCredential) -> tuple:
    return (hermes_home_key(), entry.id, fingerprint_secret_value(entry.runtime_api_key),
            entry.runtime_base_url)


def _cached(entry: PooledCredential, ttl: float):
    with _cache_lock:
        item = _cache.get(_cache_key(entry))
    if item is not None and time.monotonic() - item[1] <= ttl:
        return item[0]
    return None


def refresh_codex_usage(
    pool: CredentialPool, *, api_key_hint=None, credential_id=None, after_failure=False,
) -> None:
    """Bound HTTP probes outside the pool lock; a quota error refreshes its own account."""
    if pool.provider != "openai-codex":
        return
    entries = pool.entries()
    if after_failure:
        entry = next((e for e in entries if e.id == credential_id
                      and (not api_key_hint or e.runtime_api_key == api_key_hint)), None)
        if entry is None and api_key_hint:
            entry = next((e for e in entries if e.runtime_api_key == api_key_hint), None)
        if entry is None and not (api_key_hint or credential_id):
            entry = pool.current()
        entries = [entry] if entry is not None else []
    elif len(entries) < 2:
        return
    from agent.account_usage import fetch_codex_usage_for_token

    settings = _settings()
    budget = 1 if after_failure else settings["fetch_budget"]
    for entry in entries:
        # Gateway API keys on this provider are not ChatGPT subscription OAuth grants.
        if entry.auth_type != "oauth" or entry.last_status == "dead" or not entry.access_token:
            continue
        if budget <= 0:
            break
        if not after_failure and _cached(entry, settings["cache_ttl"]) is not None:
            continue
        snapshot = fetch_codex_usage_for_token(
            entry.access_token, entry.runtime_base_url, timeout=settings["fetch_timeout"])
        budget -= 1
        if snapshot is not None and snapshot.available:
            with _cache_lock:
                # Bound process lifetime growth across token rotations and served profiles.
                if len(_cache) >= 256:
                    _cache.pop(next(iter(_cache)))
                _cache[_cache_key(entry)] = (snapshot, time.monotonic())


def usage_prioritized(available: list[PooledCredential]) -> list[PooledCredential]:
    """Keep healthy, unknown, session-spent and weekly-spent tiers in priority order."""
    settings = _settings()
    groups = {label: [] for label in ("healthy", "unknown", "Session", "Weekly", "Monthly")}
    for entry in available:
        snapshot = _cached(entry, settings["cache_ttl"])
        if snapshot is None:
            groups["unknown"].append(entry)
            continue
        spent = {w.label for w in snapshot.windows if w.used_percent is not None
                 and w.used_percent >= settings["exhausted_pct"]}
        tier = next((label for label in ("Monthly", "Weekly", "Session") if label in spent), "healthy")
        groups[tier].append(entry)
    return groups["healthy"] + groups["unknown"] or groups["Session"] or groups["Weekly"]
