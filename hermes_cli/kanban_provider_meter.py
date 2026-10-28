"""Execution-time Kanban provider/model remediation backed by account usage meters.

The dispatcher calls this immediately before spawning a worker.  The policy is
fail-closed: an unavailable or malformed meter never becomes permission to
spend a provider quota.  Nous is intentionally excluded from execution
fallbacks for this policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from agent.account_usage import AccountUsageSnapshot
from agent import account_usage


@dataclass(frozen=True)
class ProviderMeterCandidate:
    provider: str
    model: str


_PROVIDER_PRIORITY = {"anthropic": 0, "openai-codex": 1}

# ---------------------------------------------------------------------------
# Transient-failure cache
# ---------------------------------------------------------------------------
# Keyed by provider name (lowercase). Value: (snapshot, monotonic_time).
# The cache is process-local and reset on each dispatcher restart — that is
# intentional: a fresh dispatcher should re-probe rather than trusting a
# potentially stale reading from a previous run.
_snapshot_cache: dict[str, Tuple[AccountUsageSnapshot, float]] = {}
_SNAPSHOT_CACHE_TTL = 300.0  # seconds

# Substrings that indicate a *transient* unavailability (429 / network).
# Lowercase comparison; NOT the same as quota exhaustion (that shows up in
# the windows' used_percent values, not in unavailable_reason).
_TRANSIENT_REASON_TOKENS = ("429", "rate limit", "network", "timeout", "connection")


def _is_transient_unavailable(snapshot: AccountUsageSnapshot) -> bool:
    """True when available=False because of a transient error (429 / network).

    A transient failure does NOT mean the quota is exhausted.  Treating it
    as exhaustion and switching providers is what caused the original defect
    (openai-codex 401s after Anthropic 429s).
    """
    reason = (snapshot.unavailable_reason or "").lower()
    return any(tok in reason for tok in _TRANSIENT_REASON_TOKENS)


# ---------------------------------------------------------------------------
# Headroom helpers
# ---------------------------------------------------------------------------

def _has_headroom(snapshot: AccountUsageSnapshot | None, minimum_remaining_percent: int) -> bool:
    if snapshot is None or not snapshot.available or not snapshot.windows:
        return False
    numeric = [window.used_percent for window in snapshot.windows]
    if any(value is None for value in numeric):
        return False
    # A provider is usable when at least one concrete usage window has
    # headroom.  Different providers expose different window sets; requiring
    # every window would incorrectly reject Claude when its short window is
    # healthy but a longer informational window is exhausted.
    return any(100.0 - float(value) >= minimum_remaining_percent for value in numeric)


def _has_headroom_all(snapshot: AccountUsageSnapshot | None, minimum_remaining_percent: int) -> bool:
    if snapshot is None or not snapshot.available or not snapshot.windows:
        return False
    numeric = [window.used_percent for window in snapshot.windows]
    if any(value is None for value in numeric):
        return False
    return all(100.0 - float(value) >= minimum_remaining_percent for value in numeric)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def choose_available_candidate(
    candidates: tuple[ProviderMeterCandidate, ...] | list[ProviderMeterCandidate],
    readings: Mapping[str, AccountUsageSnapshot | None],
    *,
    minimum_remaining_percent: int = 10,
) -> ProviderMeterCandidate | None:
    """Choose a metered non-Nous candidate, preferring Claude over Codex."""
    try:
        threshold = int(minimum_remaining_percent)
    except (TypeError, ValueError):
        threshold = 10
    threshold = max(0, min(100, threshold))
    eligible = [
        candidate for candidate in candidates
        if candidate.provider.strip().lower() != "nous"
        and candidate.model.strip()
        and _has_headroom(readings.get(candidate.provider.strip().lower()), threshold)
        and (
            candidate.provider.strip().lower() != "openai-codex"
            or _has_headroom_all(readings.get(candidate.provider.strip().lower()), threshold)
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (_PROVIDER_PRIORITY.get(candidate.provider.strip().lower(), 99),),
    )


def remediate_candidate(
    candidates: tuple[ProviderMeterCandidate, ...] | list[ProviderMeterCandidate],
    *,
    minimum_remaining_percent: int = 10,
    _credentials: Optional[Mapping[str, Tuple[Optional[str], Optional[str]]]] = None,
) -> ProviderMeterCandidate | None:
    """Fetch provider meters and return the execution override, if safe.

    ``_credentials`` is an optional provider → (api_key, base_url) mapping.
    When supplied, each provider's meter is fetched with the given credentials
    instead of the process-default resolver.  This lets the dispatcher query
    each assignee profile's own credentials rather than its own.

    Transient failures (HTTP 429, network errors) are **never** treated as
    quota exhaustion.  When a provider returns a transient failure the last
    successful reading is reused if it is within ``_SNAPSHOT_CACHE_TTL``
    seconds; otherwise that provider is treated as *unknown* and excluded
    from the eligible set (fail-closed).
    """
    readings: dict[str, AccountUsageSnapshot | None] = {}
    now = time.monotonic()
    for provider in {candidate.provider.strip().lower() for candidate in candidates}:
        if provider == "nous":
            continue
        creds = (_credentials or {}).get(provider)
        api_key: Optional[str] = None
        base_url: Optional[str] = None
        if isinstance(creds, (tuple, list)) and len(creds) >= 2:
            api_key = creds[0]
            base_url = creds[1]
        # Resolve through the module at call time so the dispatcher observes
        # refreshed meters and tests/operators can replace the live lookup
        # without re-importing this policy module.
        raw = account_usage.fetch_account_usage(provider, api_key=api_key, base_url=base_url)
        if raw is not None and raw.available:
            # Success — update the cache.
            _snapshot_cache[provider] = (raw, now)
            readings[provider] = raw
        elif raw is not None and _is_transient_unavailable(raw):
            # Transient (429 / network).  Reuse the last good reading if it
            # is still within TTL; otherwise treat as unknown (fail-closed).
            cached = _snapshot_cache.get(provider)
            if cached is not None and (now - cached[1]) < _SNAPSHOT_CACHE_TTL:
                readings[provider] = cached[0]
            else:
                readings[provider] = None
        else:
            # Quota exhausted, auth error, or no data — fail-closed.
            readings[provider] = None
    return choose_available_candidate(
        candidates, readings, minimum_remaining_percent=minimum_remaining_percent
    )
