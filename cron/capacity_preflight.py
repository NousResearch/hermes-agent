"""Conservative subscription-capacity route selection for new cron fires.

This is account usage, not a per-model throughput meter. A model-family window
only binds the matching Anthropic family; missing/old data is unknown, not green.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from datetime import UTC, datetime
from typing import Callable

from agent.account_usage import AccountUsageSnapshot
from agent.account_usage_cache import snapshot_is_stale

# Cache attempts, including failed/empty fetches. No secrets are retained as keys.
_FETCH_INTERVAL_S = 120.0
_FETCH_DEADLINE_S = 2.0
_lock = threading.Lock()
_attempts: dict[tuple[str, str, str, str], tuple[float, AccountUsageSnapshot | None]] = {}


def _usage_snapshot(runtime: dict) -> AccountUsageSnapshot | None:
    from agent.account_usage import fetch_account_usage
    from agent.deadline import run_bounded_sync
    from hermes_constants import hermes_home_key

    provider = str(runtime.get("provider") or "").lower()
    token = str(runtime.get("api_key") or "")
    if not token:
        return None
    base_url = str(runtime.get("base_url") or "")
    key = (hermes_home_key(), provider, base_url, hashlib.sha256(token.encode()).hexdigest())
    now = time.monotonic()
    with _lock:
        previous = _attempts.get(key)
        if previous and now - previous[0] < _FETCH_INTERVAL_S:
            return previous[1]
        _attempts[key] = (now, None)
    try:
        result = run_bounded_sync(
            lambda: fetch_account_usage(provider, api_key=token, base_url=base_url or None, read_only=True),
            _FETCH_DEADLINE_S, label="cron-capacity-usage")
        snapshot = None if result.timed_out else result.value
    except Exception:
        snapshot = None
    with _lock:
        if _attempts.get(key, (None,))[0] == now:
            _attempts[key] = (now, snapshot)
    return snapshot


def preflight_cron_route(runtime: dict, model: str, chain: list[dict], threshold: float,
                         resolve: Callable[[dict], dict]) -> tuple[dict, str] | None:
    """Resolve candidate credentials locally; only return a known-healthy complete route."""
    resolved: dict[int, dict] = {}

    def snapshot_for(provider: str, candidate_model: str, entry: dict):
        if entry is runtime:
            return _usage_snapshot(runtime)
        try:
            candidate = resolve(entry)
            resolved[id(entry)] = candidate
            return _usage_snapshot(candidate)
        except Exception:
            return None

    entry = select_capacity_route(runtime, model, chain, threshold, snapshot_for)
    if entry is None:
        return None
    return resolved[id(entry)], entry["model"]


def _state(snapshot: AccountUsageSnapshot | None, provider: str, model: str, threshold: float) -> str:
    if snapshot is None or snapshot.provider != provider or snapshot.unavailable_reason or snapshot_is_stale(snapshot):
        return "unknown"
    applicable = []
    account_labels = set()
    for window in snapshot.windows:
        if window.reset_at is not None:
            try:
                if window.reset_at <= datetime.now(UTC):
                    continue  # An old utilization cannot describe the new window.
            except (TypeError, ValueError):
                return "unknown"
        if window.scope == "model":
            if provider != "anthropic":
                continue
            family = "opus" if "opus" in window.label.lower() else "sonnet" if "sonnet" in window.label.lower() else ""
            if not family or family not in model.lower():
                continue
        elif window.scope != "account":
            continue
        else:
            account_labels.add(window.label.lower())
        used = window.used_percent
        if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used) or not 0 <= used <= 100:
            return "unknown"
        applicable.append(used)
    if not applicable:
        return "unknown"
    if any(used >= threshold for used in applicable):
        return "near"
    required = {"current session", "current week"} if provider == "anthropic" else {"session", "weekly"}
    return "healthy" if required <= account_labels else "unknown"


def select_capacity_route(primary: dict, model: str, chain: list[dict], threshold: float,
                          snapshot_for: Callable[[str, str, dict], AccountUsageSnapshot | None]) -> dict | None:
    """Return a complete fallback entry only if primary is near and candidate is known healthy.

    Never infer an unused model's own cap from account-wide windows. Unknown
    usage for either route keeps the original provider+model unchanged.
    """
    provider = str(primary.get("provider") or "").lower()
    if provider not in {"anthropic", "openai-codex"} or not 0 < threshold <= 100:
        return None
    if _state(snapshot_for(provider, model, primary), provider, model, threshold) != "near":
        return None
    for entry in chain:
        candidate = str(entry.get("provider") or "").lower()
        candidate_model = str(entry.get("model") or "")
        if candidate not in {"anthropic", "openai-codex"} or not candidate_model:
            continue
        if _state(snapshot_for(candidate, candidate_model, entry), candidate, candidate_model, threshold) == "healthy":
            return entry
    return None
