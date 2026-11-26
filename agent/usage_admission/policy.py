"""Configured Codex usage ceilings: limits, window choice, the decision cache and the checks."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)
_WARNED_INVALID_USAGE_LIMITS: set[tuple[str, str]] = set()
_WARNED_UNMATCHED_USAGE_LIMITS: set[tuple[str, str]] = set()


def get_pool_usage_limits(provider: str) -> dict[str, float]:
    """Return configured soft usage ceilings keyed by credential ID."""
    if provider != "openai-codex":
        return {}
    # credential_pool imports this package, so it is read at call time (and stays patchable there).
    from agent.credential_pool import _load_config_safe

    config = _load_config_safe()
    if not isinstance(config, dict):
        return {}
    all_limits = config.get("credential_pool_usage_limits")
    if not isinstance(all_limits, dict):
        return {}
    provider_limits = all_limits.get(provider)
    if not isinstance(provider_limits, dict):
        return {}
    valid: dict[str, float] = {}
    for credential_id, raw_threshold in provider_limits.items():
        normalized_id = str(credential_id or "").strip()
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = math.nan
        if (
            not normalized_id
            or isinstance(raw_threshold, bool)
            or not math.isfinite(threshold)
            or threshold <= 0
            or threshold > 100
        ):
            warning_key = (provider, normalized_id)
            if warning_key not in _WARNED_INVALID_USAGE_LIMITS:
                _WARNED_INVALID_USAGE_LIMITS.add(warning_key)
                logger.warning(
                    "credential pool: ignoring invalid usage limit for %s/%s",
                    provider,
                    normalized_id or "<empty-id>",
                )
            continue
        valid[normalized_id] = threshold
    return valid


def _select_codex_policy_window(snapshot: Any) -> Optional[Any]:
    """Choose the longest finite Codex usage window from a snapshot."""
    if not getattr(snapshot, "usage_windows_complete", True):
        return None
    usable = []
    duration_candidates = []
    durations_absent = 0
    reset_candidates = []
    fetched_at = getattr(snapshot, "fetched_at", None)
    timestamp_fn = getattr(fetched_at, "timestamp", None)
    try:
        raw_fetched_timestamp: Any = (
            timestamp_fn() if callable(timestamp_fn) else math.nan
        )
        fetched_timestamp = float(raw_fetched_timestamp)
    except (TypeError, ValueError, OverflowError):
        fetched_timestamp = math.nan
    for window in tuple(getattr(snapshot, "windows", ()) or ()):
        raw_used_percent = getattr(window, "used_percent", None)
        if (
            raw_used_percent is None
            or isinstance(raw_used_percent, bool)
            or not isinstance(raw_used_percent, (int, float))
        ):
            return None
        try:
            used_percent = float(raw_used_percent)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(used_percent) or not 0 <= used_percent <= 100:
            return None

        # A reported reset that has already elapsed makes the whole snapshot
        # stale, even when duration metadata is otherwise present.  Choosing
        # that window could preserve a denial from the previous quota cycle.
        reset_at = getattr(window, "reset_at", None)
        reset_timestamp = None
        if reset_at is not None:
            try:
                reset_timestamp = float(reset_at.timestamp())
            except (AttributeError, TypeError, ValueError, OverflowError):
                return None
            if (
                not math.isfinite(fetched_timestamp)
                or not math.isfinite(reset_timestamp)
                or reset_timestamp <= fetched_timestamp
            ):
                return None

        usable.append(window)
        duration = getattr(window, "limit_window_seconds", None)
        if duration is None:
            durations_absent += 1
        else:
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                return None
            try:
                duration_value = float(duration)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(duration_value) or duration_value <= 0:
                return None
            duration_candidates.append((duration_value, window))
        if reset_timestamp is not None:
            reset_candidates.append((reset_timestamp, window))
    if usable and len(duration_candidates) == len(usable):
        longest_duration = max(item[0] for item in duration_candidates)
        longest = [
            window
            for duration, window in duration_candidates
            if duration == longest_duration
        ]
        return longest[0] if len(longest) == 1 else None
    if (
        usable
        and durations_absent == len(usable)
        and len(reset_candidates) == len(usable)
    ):
        farthest_reset = max(item[0] for item in reset_candidates)
        farthest = [
            window
            for reset, window in reset_candidates
            if reset == farthest_reset
        ]
        return farthest[0] if len(farthest) == 1 else None
    return None


_USAGE_ADMISSION_SUCCESS_TTL_SECONDS = 300.0
_USAGE_ADMISSION_ERROR_TTL_SECONDS = 30.0
_USAGE_ADMISSION_PROBE_TIMEOUT_SECONDS = 3.0
_USAGE_ADMISSION_PROBE_WAIT_SECONDS = 3.25
_USAGE_ADMISSION_PROBE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="hermes-codex-usage",
)
_USAGE_ADMISSION_INFLIGHT: dict[
    tuple[str, str, str, str], concurrent.futures.Future[None]
] = {}


@dataclass(frozen=True)
class _UsageAdmissionDecision:
    checked_at: float
    probe_succeeded: bool
    used_percent: Optional[float]
    reset_at: Optional[float]


_USAGE_ADMISSION_CACHE: dict[
    tuple[str, str, str, str], _UsageAdmissionDecision
] = {}
_USAGE_ADMISSION_CACHE_LOCK = threading.Lock()


def _usage_admission_cache_key(
    entry: PooledCredential,
) -> tuple[str, str, str, str]:
    token_fingerprint = hashlib.sha256(
        entry.runtime_api_key.encode("utf-8")
    ).hexdigest()
    return (
        entry.provider,
        entry.id,
        token_fingerprint,
        str(entry.runtime_base_url or "").strip().rstrip("/"),
    )


def _usage_admission_decision_is_fresh(
    decision: _UsageAdmissionDecision,
    now: float,
) -> bool:
    if decision.reset_at is not None:
        try:
            reset_at = float(decision.reset_at)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(reset_at) or reset_at <= time.time():
            return False
    ttl = (
        _USAGE_ADMISSION_SUCCESS_TTL_SECONDS
        if decision.probe_succeeded
        else _USAGE_ADMISSION_ERROR_TTL_SECONDS
    )
    return now - decision.checked_at < ttl


def _usage_admission_decision_denied(
    decision: _UsageAdmissionDecision,
    threshold: float,
    now: float,
) -> bool:
    """Apply current policy to fresh cached telemetry."""
    return bool(
        _usage_admission_decision_is_fresh(decision, now)
        and decision.probe_succeeded
        and decision.used_percent is not None
        and decision.used_percent >= threshold
    )


def usage_admission_policy_denied(provider: str) -> bool:
    """Return whether a fresh configured decision excludes any credential."""
    limits = get_pool_usage_limits(provider)
    if not limits:
        return False
    now = time.monotonic()
    with _USAGE_ADMISSION_CACHE_LOCK:
        return any(
            cache_provider == provider
            and credential_id in limits
            and _usage_admission_decision_denied(
                decision, limits[credential_id], now
            )
            for (
                cache_provider,
                credential_id,
                _token_fingerprint,
                _base_url,
            ), decision in _USAGE_ADMISSION_CACHE.items()
        )


def usage_admission_credential_denied(provider: str, api_key: str) -> bool:
    """Return whether policy denies the credential behind ``api_key``.

    Explicit runtime paths may be the first use of a configured pool token in a
    process.  On a cache miss, match only that exact token to a configured stable
    ID and refresh its pool telemetry before deciding.  Unrelated explicit
    credentials remain outside the pool policy.
    """
    limits = get_pool_usage_limits(provider)
    token = str(api_key or "").strip()
    if not limits or not token:
        return False
    token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _cached_denial() -> bool:
        now = time.monotonic()
        with _USAGE_ADMISSION_CACHE_LOCK:
            return any(
                cache_provider == provider
                and credential_id in limits
                and cached_fingerprint == token_fingerprint
                and _usage_admission_decision_denied(
                    decision, limits[credential_id], now
                )
                for (
                    cache_provider,
                    credential_id,
                    cached_fingerprint,
                    _base_url,
                ), decision in _USAGE_ADMISSION_CACHE.items()
            )

    if _cached_denial():
        return True

    # A cold explicit path has no cache entry yet.  Probe only when the token
    # matches a configured pool credential; otherwise fail open and leave an
    # unrelated explicit credential untouched.
    from agent.credential_pool import load_pool

    try:
        pool = load_pool(provider)
        matching_entry = any(
            entry.id in limits
            and hashlib.sha256(entry.runtime_api_key.encode("utf-8")).hexdigest()
            == token_fingerprint
            for entry in pool.entries()
        )
    except Exception:
        return False
    if not matching_entry:
        return False
    try:
        pool._refresh_usage_admission_cache()
    except Exception:
        return False
    return _cached_denial()
