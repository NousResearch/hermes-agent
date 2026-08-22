"""Configurable quota warning engine (Hermes agent, issue #6567).

Builds on :mod:`agent.account_usage` — which already fetches per-provider
account-usage snapshots (``AccountUsageSnapshot`` / ``AccountUsageWindow``) —
and adds:

* configurable percentage thresholds (warning / strong / critical),
* a single highest-level warning line per snapshot,
* a *pre-turn* suppression gate (``quota.suppress_warnings``) for the
  steady-state turns, and a startup variant that always fires,
* a small module-level TTL cache so the pre-turn probe doesn't hit the
  provider network on every turn, plus an in-flight-future registry so a
  stuck probe can never stack up one worker thread per turn.

All threshold logic is pure: ``get_quota_warnings`` takes a snapshot +
thresholds and returns lines.  The config-aware wrappers
(``quota_warning_lines`` / ``startup_warning_lines``) are thin shims over it
so the pure function stays easily unit-testable.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from typing import Any, Optional

# Re-exported so callers (and the test seam) bind ``fetch_account_usage`` on
# *this* module — tests patch ``agent.quota_warnings.fetch_account_usage``.
from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    _format_reset,
    fetch_account_usage,
)

# Design defaults — mirrored by config_defaults.py (Task A owns that file).
_DEFAULT_WARNING = 80.0
_DEFAULT_STRONG = 90.0
_DEFAULT_CRITICAL = 95.0

# Pre-turn probe cadence (issue #6567 design review): reuse a freshly fetched
# snapshot for up to 10 minutes so the warning probe doesn't hammer the
# provider network on every turn.
_DEFAULT_CACHE_TTL = 600.0


@dataclass(frozen=True)
class QuotaThresholds:
    """Percentage thresholds for the quota warning ladder.

    Ordered low→high; a snapshot's peak utilization is compared against all
    three with ``>=`` and mapped to the single highest level it reaches.
    """

    warning: float
    strong: float
    critical: float

    def __post_init__(self) -> None:
        """Validate finiteness, range, and strict ordering.

        Raises :class:`ValueError` when any threshold is non-finite
        (``NaN``/``inf``), outside ``[0, 100]``, or when the levels are not
        strictly ordered ``warning < strong < critical``.
        """
        values = (self.warning, self.strong, self.critical)
        for name, value in zip(("warning", "strong", "critical"), values):
            if not math.isfinite(value):
                raise ValueError(f"{name} threshold must be finite, got {value!r}")
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} threshold must be in [0, 100], got {value!r}")
        if not (self.warning < self.strong < self.critical):
            raise ValueError(
                "thresholds must be strictly ordered warning < strong < critical, "
                f"got warning={self.warning!r} strong={self.strong!r} "
                f"critical={self.critical!r}"
            )


def _quota_section(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return the ``quota`` sub-dict from a config dict, or ``{}`` on mismatch."""
    if not isinstance(config, dict):
        return {}
    section = config.get("quota")
    if not isinstance(section, dict):
        return {}
    return section


def _coerce_threshold(section: dict[str, Any], key: str, default: float) -> float:
    """Read a threshold from the quota section, falling back to ``default``.

    Missing/non-numeric values (str "abc", None, etc.) fall back to the
    default.  Real numbers (int/float, including numeric strings) are coerced
    via ``float()``.  ``bool`` is explicitly rejected before coercion —
    ``float(True)`` would otherwise yield ``1.0``, a degenerate threshold, so
    booleans fall back to the default.
    """
    raw: Any = section.get(key)
    if isinstance(raw, bool):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def quota_thresholds(config: Optional[dict] = None) -> QuotaThresholds:
    """Build :class:`QuotaThresholds` from a config dict.

    Reads ``quota.warning_threshold`` / ``quota.strong_threshold`` /
    ``quota.critical_threshold``; missing or non-numeric values fall back to
    the design defaults 80 / 90 / 95 (``float``-coerced).  Booleans are
    rejected and fall back to the default.

    Raises :class:`ValueError` when the resolved thresholds fail
    :class:`QuotaThresholds` validation — i.e. any value is non-finite, outside
    ``[0, 100]``, or not strictly ordered ``warning < strong < critical``.
    The config-aware wrappers (:func:`quota_warning_lines` /
    :func:`startup_warning_lines`) catch this and return ``[]`` so a
    misconfigured threshold never yields a wrong/mislabeled warning.
    """
    section = _quota_section(config)
    return QuotaThresholds(
        warning=_coerce_threshold(section, "warning_threshold", _DEFAULT_WARNING),
        strong=_coerce_threshold(section, "strong_threshold", _DEFAULT_STRONG),
        critical=_coerce_threshold(section, "critical_threshold", _DEFAULT_CRITICAL),
    )


def _peak_window(snapshot: AccountUsageSnapshot) -> Optional[AccountUsageWindow]:
    """The window with the highest finite ``used_percent``, or ``None``."""
    peak: Optional[AccountUsageWindow] = None
    peak_pct: Optional[float] = None
    for window in snapshot.windows:
        if window.used_percent is None:
            continue
        pct = float(window.used_percent)
        if peak_pct is None or pct > peak_pct:
            peak_pct = pct
            peak = window
    return peak


def get_quota_warnings(
    snapshot: Optional[AccountUsageSnapshot],
    *,
    thresholds: QuotaThresholds,
) -> list[str]:
    """Pure threshold evaluation — one line for the highest level reached.

    * ``None`` snapshot, an unavailable snapshot, or a snapshot with no
      usable ``used_percent`` window → ``[]``.
    * Windows with ``None`` ``used_percent`` are skipped; the *maximum*
      finite percent across the remaining windows drives the comparison.
    * Uses ``>=`` comparisons against the thresholds (80/90/95 by default),
      so a value exactly on a boundary trips that level.

    If the peak window carries a ``reset_at``, a `` — resets <…>`` suffix
    is appended using :func:`agent.account_usage._format_reset`.
    """
    if snapshot is None or not snapshot.available:
        return []

    peak = _peak_window(snapshot)
    if peak is None or peak.used_percent is None:
        return []

    pct = float(peak.used_percent)
    warning, strong, critical = thresholds.warning, thresholds.strong, thresholds.critical

    if pct >= critical:
        line = f"  🚨 Critical quota warning: {pct:.0f}% used (threshold {critical:.0f}%)"
    elif pct >= strong:
        line = f"  ⚠⚠ Strong quota warning: {pct:.0f}% used (threshold {strong:.0f}%)"
    elif pct >= warning:
        line = f"  ⚠ Quota warning: {pct:.0f}% used (threshold {warning:.0f}%)"
    else:
        return []

    if peak.reset_at is not None:
        line += f" — resets {_format_reset(peak.reset_at)}"
    return [line]


def quota_warning_lines(
    snapshot: Optional[AccountUsageSnapshot],
    config: Optional[dict] = None,
) -> list[str]:
    """Pre-turn quota warning lines, honoring ``quota.suppress_warnings``.

    Returns ``[]`` when suppression is enabled (the pre-turn probe is silenced
    for this turn — issue #6567) or when the configured thresholds are invalid.
    Otherwise delegates to :func:`get_quota_warnings` with thresholds parsed
    from ``config``.

    An invalid ``quota`` threshold config (non-finite, out of ``[0, 100]``, or
    not strictly ordered) makes :func:`quota_thresholds` raise
    :class:`ValueError`; the safe failure mode for a warning system is to
    show *no* warning rather than a wrong/mislabeled one, so those are caught
    and ``[]`` is returned.

    Suppression is gated on a strict ``is True`` check — only an explicit
    boolean ``true`` silences the probe, so a YAML string ``"false"`` (which is
    truthy) does *not* suppress.
    """
    if _quota_section(config).get("suppress_warnings") is True:
        return []
    try:
        thresholds = quota_thresholds(config)
    except ValueError:
        # Misconfigured thresholds: fail open (no warning) rather than emit a
        # mislabeled one — issue #6567 design-review requirement.
        return []
    return get_quota_warnings(snapshot, thresholds=thresholds)


def startup_warning_lines(
    snapshot: Optional[AccountUsageSnapshot],
    config: Optional[dict] = None,
) -> list[str]:
    """Startup quota warning lines — always shown, ignoring suppression.

    Per issue #6567 the *first* probe of a session must always surface a
    critical warning to the user even when ``quota.suppress_warnings`` is set,
    so the user is never blinded at session start.

    An invalid ``quota`` threshold config raises :class:`ValueError` from
    :func:`quota_thresholds`; the safe failure mode for a warning system is to
    show *no* warning rather than a wrong/mislabeled one, so those are caught
    and ``[]`` is returned.
    """
    try:
        thresholds = quota_thresholds(config)
    except ValueError:
        # Misconfigured thresholds: fail open (no warning) rather than emit a
        # mislabeled one — issue #6567 design-review requirement.
        return []
    return get_quota_warnings(snapshot, thresholds=thresholds)


# ── TTL cache ─────────────────────────────────────────────────────────────


# Cache: {(provider, base_url, api_key): (timestamp, snapshot)}
_quota_cache: dict[tuple, tuple[float, AccountUsageSnapshot]] = {}
_quota_cache_lock = threading.Lock()

# In-flight probe registry: {(provider, base_url, api_key): (future, executor)}.
# ``fetch_quota_snapshot_bounded`` reuses a still-running future for the same
# credential triple instead of spawning a fresh executor/thread per probe, so
# an unhealthy provider can stack at most one stuck worker (review feedback,
# PR #84946).  Per-key isolation is deliberate: a shared single-worker
# executor would queue every later probe behind one hung fetch.
_in_flight: dict[tuple, tuple[Future, ThreadPoolExecutor]] = {}
_in_flight_lock = threading.Lock()


def fetch_quota_snapshot(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_age: float = _DEFAULT_CACHE_TTL,
) -> Optional[AccountUsageSnapshot]:
    """TTL-cached wrapper around :func:`agent.account_usage.fetch_account_usage`.

    The cache key is ``(provider, base_url, api_key)`` — ``api_key`` is part of
    the key so two credentials for the same provider/base_url (e.g. two Codex
    accounts in the credential pool) never share a cached snapshot,
    eliminating a cross-account stale-data leak (cross-vendor review).
    ``api_key`` never leaves this module: the dict is module-private and is
    not logged or otherwise exposed.  A cached snapshot younger than ``max_age`` seconds (default 600s = 10 min) is returned
    without hitting the network.

    Fail-open: ``fetch_account_usage`` exceptions return ``None`` and are
    *not* cached, so the next call retries rather than serving a stale failure.
    Likewise a ``None`` snapshot (unsupported provider / no creds) is not
    cached.
    """
    key = (provider, base_url, api_key)
    now = monotonic()
    with _quota_cache_lock:
        entry = _quota_cache.get(key)
        if entry is not None:
            cached_ts, cached_snapshot = entry
            if (now - cached_ts) < max_age:
                return cached_snapshot

    # Network I/O happens outside the lock so a slow provider can't stall
    # threads operating on a different cache key.
    try:
        snapshot = fetch_account_usage(provider, base_url=base_url, api_key=api_key)
    except Exception:
        return None

    if snapshot is not None:
        with _quota_cache_lock:
            _quota_cache[key] = (monotonic(), snapshot)
    return snapshot


def clear_quota_cache() -> None:
    """Empty the quota TTL cache and shut down in-flight probe fetches.

    Called at REPL session start so each fresh session re-probes the provider
    instead of reusing the warm cache from a previous session (design-review
    requirement).  In-flight probes are shut down (``wait=False``) so a stuck
    fetch from a previous session cannot linger past its own fetch timeout;
    the next probe for the same key starts fresh.
    """
    with _quota_cache_lock:
        _quota_cache.clear()
    with _in_flight_lock:
        for _, executor in _in_flight.values():
            # cancel_futures=False: these executors never hold pending work
            # (one submit per executor), so there is nothing to cancel — the
            # shutdown only wakes idle workers; a running fetch is never
            # interrupted and exits by itself at its httpx timeout.
            executor.shutdown(wait=False, cancel_futures=False)
        _in_flight.clear()


def fetch_quota_snapshot_bounded(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float,
) -> Optional[AccountUsageSnapshot]:
    """Bounded snapshot fetch for the CLI warning surfaces (advisory probes).

    Runs the TTL-cached fetch on a short-lived single-worker executor and
    waits at most ``timeout`` seconds; returns ``None`` on timeout or error
    (fail-open — never raises into the main thread).

    Thread hygiene (review feedback, PR #84946): at most one in-flight fetch
    per credential triple.  A probe whose previous fetch for the same key is
    still running reuses that future instead of spawning another executor and
    thread, so a slow provider can stack at most one stuck worker, and only
    while a fetch is genuinely in flight.  Once the previous future is done
    its idle worker is shut down (the shutdown wake-up exits it) and a fresh
    executor is created — finished probes leave no lingering threads, and a
    still-running work item is never interrupted (its thread exits by itself
    once the underlying fetch returns; the fetches carry httpx timeouts).

    A shared single-worker executor is deliberately NOT used: one hung fetch
    would queue every later probe behind it and wedge all providers.
    """
    key = (provider, base_url, api_key)
    with _in_flight_lock:
        entry = _in_flight.get(key)
        if entry is not None:
            fut, executor = entry
            if not fut.done():
                pass  # reuse the still-running fetch
            else:
                # Previous fetch finished; its worker is idle.  Shut it down
                # (the wake-up exits the idle worker) and start a fresh one.
                executor.shutdown(wait=False)
                executor = ThreadPoolExecutor(max_workers=1)
                fut = executor.submit(
                    fetch_quota_snapshot, provider,
                    base_url=base_url, api_key=api_key,
                )
                _in_flight[key] = (fut, executor)
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            fut = executor.submit(
                fetch_quota_snapshot, provider,
                base_url=base_url, api_key=api_key,
            )
            _in_flight[key] = (fut, executor)
    try:
        return fut.result(timeout=timeout)
    except Exception:
        return None
