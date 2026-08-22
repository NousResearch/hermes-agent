"""Secret-free, on-disk cache of provider account usage for delegation routing.

Route selection must be cheap and non-blocking: an orchestrator that fans out
twenty tasks cannot pay a quota HTTP round trip per task, and it must never
stall a delegation behind a provider control plane that is slow or down.  So
selection reads *only* this cache, and refreshes happen out of band.

Freshness contract (thresholds come from ``delegation.routing``):

* ``age <= usage_ttl_seconds`` → **fresh**; used as-is, no refresh.
* ``usage_ttl_seconds < age <= usage_stale_seconds`` → **stale**; the reading
  is still used (a 10-minute-old quota is far better than no signal) and one
  background refresh is scheduled for that provider.
* older, missing, or unavailable → **unknown**; the selector applies the
  configured ``unknown_usage`` policy and one refresh is scheduled.

**What is persisted.** Only a normalized projection: provider slug, fetch
timestamp, source label, and per-window label / used / remaining percent /
reset timestamp.  Never a token, API key, account or profile identity, project
id, request header, credential path, raw error text, or raw provider response
body.  Those fields exist on the snapshot objects this module consumes (a
window ``detail`` can quote an account email; ``unavailable_reason`` can quote
a URL with a query-string key), so the projection is an explicit allowlist
rather than a redaction pass — a new upstream field cannot leak by default.
The file is written atomically at mode 0600 under the active HERMES_HOME.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent.delegation_routing import ProviderUsage, UsageView

logger = logging.getLogger(__name__)

__all__ = [
    "build_usage_view",
    "build_route_usage_view",
    "project_snapshot",
    "read_provider_usage",
    "read_raw",
    "refresh_provider_now",
    "reset_refresh_state",
    "schedule_refresh",
    "store_snapshot",
]

_CACHE_FILENAME = "route_usage.json"
_SCHEMA_VERSION = 1

# Guards both the on-disk merge (read-modify-write) and the in-flight refresh
# set, so concurrent delegations cannot clobber each other's entries or start
# duplicate refreshes for one provider.
_LOCK = threading.RLock()
_INFLIGHT: set[str] = set()

_SAFE_SOURCES: dict[str, frozenset[str]] = {
    "openai-codex": frozenset({"usage_api"}),
    "google-antigravity": frozenset({"quota_summary"}),
}
_SAFE_WINDOW_LABELS: dict[str, frozenset[str]] = {
    "openai-codex": frozenset({"Session", "Weekly"}),
    "google-antigravity": frozenset(
        {
            "Gemini Models (5h)",
            "Gemini Models (weekly)",
            "Claude and GPT models (5h)",
            "Claude and GPT models (weekly)",
        }
    ),
}


def _cache_path() -> Path:
    """Path to the usage cache under the *active* HERMES_HOME/profile."""
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("cache/delegation", "delegation_cache") / _CACHE_FILENAME


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_provider(provider: Optional[str]) -> str:
    """Fold provider aliases onto the slug the cache keys on."""
    value = str(provider or "").strip().lower()
    if value in {"antigravity", "agy", "google-agy"}:
        return "google-antigravity"
    return value


def reset_refresh_state() -> None:
    """Clear in-flight refresh bookkeeping (test seam)."""
    with _LOCK:
        _INFLIGHT.clear()


# ---------------------------------------------------------------------------
# Projection — the allowlist that keeps secrets off disk
# ---------------------------------------------------------------------------


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return None


def _percent(value: Any) -> Optional[float]:
    try:
        pct = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(pct):
        return None
    return max(0.0, min(100.0, pct))


def project_snapshot(snapshot: Any) -> dict[str, Any]:
    """Reduce an ``AccountUsageSnapshot`` to the fields safe to persist.

    Deliberately an allowlist. ``plan``, ``title``, ``details``,
    ``unavailable_reason`` and each window's ``detail`` are dropped: they are
    free-form provider prose that has been observed to carry account emails,
    project ids, and error URLs with embedded keys.
    """
    provider = normalize_provider(getattr(snapshot, "provider", None))
    safe_labels = _SAFE_WINDOW_LABELS.get(provider, frozenset())
    windows: list[dict[str, Any]] = []
    for window in getattr(snapshot, "windows", ()) or ():
        used = _percent(getattr(window, "used_percent", None))
        raw_label = str(getattr(window, "label", "") or "").strip()
        label = raw_label if raw_label in safe_labels else ""
        windows.append(
            {
                "label": label,
                "used_percent": used,
                "remaining_percent": None if used is None else round(100.0 - used, 4),
                "reset_at": _iso(getattr(window, "reset_at", None)),
            }
        )
    if getattr(snapshot, "unavailable_reason", None):
        # An unavailable snapshot carries no trustworthy numbers; record the
        # fetch attempt (so we do not hammer the provider) but no windows and
        # no reason text.
        windows = []
    fetched_at = _iso(getattr(snapshot, "fetched_at", None)) or _iso(_utc_now())
    raw_source = str(getattr(snapshot, "source", "") or "").strip()
    source = raw_source if raw_source in _SAFE_SOURCES.get(provider, frozenset()) else ""
    return {
        "provider": provider,
        "fetched_at": fetched_at,
        "source": source,
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def read_raw() -> dict[str, Any]:
    """Read the whole cache document; ``{}``-shaped on any error."""
    empty: dict[str, Any] = {"version": _SCHEMA_VERSION, "providers": {}}
    try:
        text = _cache_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return empty
    try:
        data = json.loads(text)
    except ValueError:
        # A truncated or hand-edited file is treated as missing rather than
        # fatal: usage is an optimization, never a correctness input.
        return empty
    if not isinstance(data, dict) or not isinstance(data.get("providers"), dict):
        return empty
    return data


def store_snapshot(snapshot: Any) -> None:
    """Merge one provider's projected usage into the cache (atomic, 0600)."""
    record = project_snapshot(snapshot)
    provider = record.get("provider")
    if not provider:
        return
    from utils import atomic_write_text

    with _LOCK:
        data = read_raw()
        providers = dict(data.get("providers") or {})
        providers[provider] = record
        payload = {"version": _SCHEMA_VERSION, "providers": providers}
        try:
            atomic_write_text(
                _cache_path(),
                json.dumps(payload, indent=2, sort_keys=True),
                create_mode=0o600,
            )
        except OSError as exc:
            logger.debug("delegation usage cache write failed: %s", type(exc).__name__)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _worst_remaining(
    windows: Sequence[dict[str, Any]], window_prefixes: Sequence[str]
) -> Optional[float]:
    """Lowest remaining percent across the windows that matter.

    The binding constraint is whichever window is closest to exhaustion, so
    the minimum is the honest reading. ``window_prefixes`` narrows to the
    windows a route actually consumes (an Antigravity account reports Gemini
    and Claude pools separately; a Gemini route must not be blocked by a
    depleted Claude pool).
    """
    candidates: list[float] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        label = str(window.get("label") or "")
        if window_prefixes and not any(label.startswith(p) for p in window_prefixes):
            continue
        remaining = _percent(window.get("remaining_percent"))
        if remaining is not None:
            candidates.append(remaining)
    return min(candidates) if candidates else None


def read_provider_usage(
    provider: str,
    *,
    ttl_seconds: int,
    stale_seconds: int,
    window_prefixes: Sequence[str] = (),
) -> ProviderUsage:
    """Return the cached, normalized usage reading for one provider."""
    normalized = normalize_provider(provider)
    record = (read_raw().get("providers") or {}).get(normalized)
    if not isinstance(record, dict):
        return ProviderUsage(provider=normalized)

    try:
        fetched_at = datetime.fromisoformat(str(record.get("fetched_at")))
    except (TypeError, ValueError):
        return ProviderUsage(provider=normalized)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    age = max(0.0, (_utc_now() - fetched_at).total_seconds())
    if age <= ttl_seconds:
        freshness = "fresh"
    elif age <= stale_seconds:
        freshness = "stale"
    else:
        # Past the stale window the number is no longer trustworthy — report
        # unknown rather than let the selector act on a hours-old quota.
        return ProviderUsage(provider=normalized, freshness="unknown", age_seconds=age)

    remaining = _worst_remaining(record.get("windows") or [], tuple(window_prefixes))
    return ProviderUsage(
        provider=normalized,
        remaining_percent=remaining,
        freshness=freshness if remaining is not None else "unknown",
        age_seconds=age,
    )


def build_usage_view(
    providers: Iterable[str],
    *,
    ttl_seconds: int,
    stale_seconds: int,
    refresh: bool = True,
    window_prefixes_by_provider: Optional[dict[str, Sequence[str]]] = None,
) -> UsageView:
    """Read every provider's cached usage and schedule refreshes as needed.

    Never fetches inline — a stale or unknown reading schedules at most one
    background refresh per provider and returns immediately, so delegation
    latency is independent of provider control-plane health.
    """
    prefixes = window_prefixes_by_provider or {}
    entries: dict[str, ProviderUsage] = {}
    for provider in providers:
        normalized = normalize_provider(provider)
        if not normalized or normalized in entries:
            continue
        entry = read_provider_usage(
            normalized,
            ttl_seconds=ttl_seconds,
            stale_seconds=stale_seconds,
            window_prefixes=tuple(prefixes.get(normalized) or ()),
        )
        entries[normalized] = entry
        # A fresh negative snapshot (unsupported/unavailable usage) is a valid
        # cached probe result.  Keep usage "unknown", but do not hammer the
        # provider again until its TTL expires.
        refresh_due = entry.age_seconds is None or entry.age_seconds > ttl_seconds
        if refresh and entry.freshness != "fresh" and refresh_due:
            _spawn_refresh(normalized)
    return UsageView(entries)


def build_route_usage_view(
    routes: Iterable[Any],
    *,
    ttl_seconds: int,
    stale_seconds: int,
    refresh: bool = True,
) -> UsageView:
    """Build route-scoped readings while refreshing each provider at most once.

    Multiple routes can consume independent windows reported by one provider,
    so unioning their prefixes would incorrectly let a depleted pool block an
    unrelated healthy model family.
    """
    entries: dict[str, ProviderUsage] = {}
    refresh_providers: set[str] = set()
    for route in routes:
        provider = normalize_provider(getattr(route, "provider", None))
        route_id = str(getattr(route, "id", "") or "").strip()
        if not provider or not route_id:
            continue
        if str(getattr(route, "backend", "native")).strip().lower() == "claude-p":
            # Claude Pro/Max exposes no supported machine-readable remaining
            # allowance endpoint. Keep usage explicitly unknown without
            # probing/scraping the CLI or fabricating a cache record.
            entries[route_id] = ProviderUsage(
                provider=provider, freshness="unknown"
            )
            continue
        entry = read_provider_usage(
            provider,
            ttl_seconds=ttl_seconds,
            stale_seconds=stale_seconds,
            window_prefixes=tuple(getattr(route, "usage_window_prefixes", ()) or ()),
        )
        entries[route_id] = entry
        refresh_due = entry.age_seconds is None or entry.age_seconds > ttl_seconds
        if refresh and entry.freshness != "fresh" and refresh_due:
            refresh_providers.add(provider)
    for provider in sorted(refresh_providers):
        _spawn_refresh(provider)
    return UsageView(entries)


# ---------------------------------------------------------------------------
# Refresh (out of band — never on the selection path)
# ---------------------------------------------------------------------------


def _fetch_account_usage(provider: str) -> Any:
    """Indirection seam over PR2's fetcher (mocked in tests)."""
    from agent.account_usage import fetch_account_usage

    return fetch_account_usage(provider)


def _start_thread(fn) -> None:
    """Run *fn* on a daemon thread (test seam)."""
    threading.Thread(target=fn, name="hermes-route-usage-refresh", daemon=True).start()


def refresh_provider_now(provider: str) -> None:
    """Fetch and persist one provider's usage. Never raises.

    A failed refresh leaves the previous entry untouched: a transient control
    plane error should not erase a usable stale reading, and the error text
    itself (which can quote a URL carrying a key) is never persisted or
    logged verbatim.
    """
    normalized = normalize_provider(provider)
    try:
        snapshot = _fetch_account_usage(normalized)
    except Exception as exc:
        logger.debug(
            "delegation usage refresh for %s failed: %s", normalized, type(exc).__name__
        )
        return
    if snapshot is None:
        return
    try:
        store_snapshot(snapshot)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "delegation usage store for %s failed: %s", normalized, type(exc).__name__
        )


def schedule_refresh(provider: str) -> None:
    """Start at most one background refresh per provider at a time."""
    normalized = normalize_provider(provider)
    if not normalized:
        return
    with _LOCK:
        if normalized in _INFLIGHT:
            return
        _INFLIGHT.add(normalized)

    def _worker() -> None:
        try:
            refresh_provider_now(normalized)
        finally:
            with _LOCK:
                _INFLIGHT.discard(normalized)

    try:
        _start_thread(_worker)
    except Exception as exc:  # pragma: no cover - thread creation failure
        with _LOCK:
            _INFLIGHT.discard(normalized)
        logger.debug("could not start usage refresh thread: %s", type(exc).__name__)


def _spawn_refresh(provider: str) -> None:
    """Seam so callers/tests can observe scheduling without threads."""
    schedule_refresh(provider)
