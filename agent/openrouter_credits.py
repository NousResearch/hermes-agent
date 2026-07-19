"""Background-refreshed, per-account OpenRouter credit balance probe.

Shared by the classic prompt-toolkit CLI status bar (``cli.py``) and the Ink
TUI gateway (``tui_gateway/server.py``). Both surfaces need to show the live
OpenRouter credit balance without ever blocking their redraw / event-emit
loops.

Two properties this module guarantees, both required by the PR #57321 review:

1. **Never on the render path.** ``fetch_account_usage()`` makes synchronous
   ``httpx`` calls that can wait up to 10s. The classic status bar redraws
   ~1x/s and the gateway emits ``session.info`` on every turn, so fetching
   inline would stall the UI. ``snapshot()`` only ever returns the last
   *completed* fetch and kicks any refresh onto a daemon thread.

2. **Scoped per account.** The cache is keyed by a non-secret account identity
   (provider + base URL + a hash of the API key). A later session with
   different credentials can never display a prior account's balance, and a
   gateway hosting several accounts keeps a value per account rather than
   thrashing a single slot.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

# Freshness / backoff window for a given account's balance. A completed attempt
# (success *or* failure) starts the clock, so a persistently failing account
# backs off for the full window instead of refetching on every redraw.
CREDITS_TTL_SECONDS = 300.0


def account_identity(
    provider: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> str:
    """Return a non-secret identity string for the resolved account.

    The API key is hashed (truncated SHA-256) — it is never stored in
    cleartext and never rendered. Used as the cache key so a change of
    credentials starts a fresh balance rather than surfacing the old one.
    """
    key = str(api_key or "").strip()
    key_fp = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else "no-key"
    base = str(base_url or "").strip().rstrip("/")
    return f"{provider or ''}|{base}|{key_fp}"


def parse_credits(
    snap: Any,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """Extract ``(balance, label, quota_pct)`` from an ``AccountUsageSnapshot``.

    Pure and hermetic (no network), so the parsing/formatting logic is
    unit-testable without a live fetch. Returns ``(None, None, None)`` when the
    snapshot is missing or has no parseable balance.
    """
    if not snap or not getattr(snap, "details", None):
        return (None, None, None)
    balance: Optional[float] = None
    label: Optional[str] = None
    quota_pct: Optional[float] = None
    match = re.search(r"\$(\d+\.?\d*)", str(snap.details[0]))
    if match:
        balance = float(match.group(1))
        label = f"${balance:,.2f}"
    for window in getattr(snap, "windows", None) or []:
        if (
            getattr(window, "label", "") == "API key quota"
            and getattr(window, "used_percent", None) is not None
        ):
            quota_pct = window.used_percent
            break
    return (balance, label, quota_pct)


class OpenRouterCreditsProbe:
    """Thread-safe, non-blocking reader for OpenRouter credit balances.

    ``snapshot()`` returns the last completed result for the current account
    immediately and schedules a background refresh when the cached value is
    stale. Results are kept per account identity so multiple accounts (e.g. in
    a gateway serving several sessions) never clobber each other.

    ``fetcher`` is injectable for hermetic tests; by default it lazy-imports
    ``agent.account_usage.fetch_account_usage`` (kept lazy because that module
    pulls a heavy SDK chain only needed when credits are actually shown).
    """

    def __init__(
        self,
        ttl: float = CREDITS_TTL_SECONDS,
        fetcher: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._ttl = ttl
        self._fetcher = fetcher
        self._lock = threading.Lock()
        # identity -> {"balance", "label", "quota_pct", "last_attempt"}
        self._entries: Dict[str, Dict[str, Any]] = {}
        # identity -> in-flight refresh thread
        self._threads: Dict[str, threading.Thread] = {}

    def _fetch(
        self,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Any:
        if self._fetcher is not None:
            return self._fetcher(provider, base_url=base_url, api_key=api_key)
        from agent.account_usage import fetch_account_usage

        return fetch_account_usage(provider, base_url=base_url, api_key=api_key)

    def snapshot(
        self,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        *,
        on_update: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return the last completed balance for this account, refreshing in the
        background if stale. Never blocks.

        ``on_update`` (if given) is invoked from the refresh thread once new
        data lands — used by the CLI to repaint the status bar immediately.
        Returns ``{}`` when no balance is known yet for the account.
        """
        identity = account_identity(provider, base_url, api_key)
        self._maybe_refresh(provider, base_url, api_key, identity, on_update)
        with self._lock:
            entry = self._entries.get(identity)
            if entry and entry["balance"] is not None:
                return {
                    "balance": entry["balance"],
                    "label": entry["label"],
                    "quota_pct": entry["quota_pct"],
                }
        return {}

    def refresh_now(
        self,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Dict[str, Any]:
        """Fetch synchronously in the calling thread (no background thread).

        Intended for tests and for one-shot callers (like ``/limits``) that
        already run off the render path. Returns the same shape as
        ``snapshot()``.
        """
        identity = account_identity(provider, base_url, api_key)
        self._worker(provider, base_url, api_key, identity, None)
        with self._lock:
            entry = self._entries.get(identity) or {}
            if entry.get("balance") is not None:
                return {
                    "balance": entry["balance"],
                    "label": entry["label"],
                    "quota_pct": entry["quota_pct"],
                }
        return {}

    def _maybe_refresh(
        self,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        identity: str,
        on_update: Optional[Callable[[], None]],
    ) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(identity)
            if entry is not None and (now - entry["last_attempt"]) <= self._ttl:
                return  # still fresh (or backing off after a failure)
            thread = self._threads.get(identity)
            if thread is not None and thread.is_alive():
                return  # a refresh for this account is already in flight
            thread = threading.Thread(
                target=self._worker,
                args=(provider, base_url, api_key, identity, on_update),
                name="or-credits-refresh",
                daemon=True,
            )
            self._threads[identity] = thread
            thread.start()

    def _worker(
        self,
        provider: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        identity: str,
        on_update: Optional[Callable[[], None]],
    ) -> None:
        balance = label = quota_pct = None
        ok = False
        try:
            snap = self._fetch(provider, base_url, api_key)
            balance, label, quota_pct = parse_credits(snap)
            ok = True
        except Exception:
            # Network / auth / timeout failures must never surface to the UI.
            # Keep the last good value and back off until the next attempt.
            ok = False
        with self._lock:
            entry = self._entries.get(identity)
            if entry is None:
                entry = {
                    "balance": None,
                    "label": None,
                    "quota_pct": None,
                    "last_attempt": 0.0,
                }
                self._entries[identity] = entry
            # Record the attempt time (success or failure) to drive backoff.
            entry["last_attempt"] = time.monotonic()
            if ok:
                entry["balance"] = balance
                entry["label"] = label
                entry["quota_pct"] = quota_pct
        if ok and on_update is not None:
            try:
                on_update()
            except Exception:
                pass


_SHARED_PROBE: Optional[OpenRouterCreditsProbe] = None
_SHARED_PROBE_LOCK = threading.Lock()


def get_shared_probe() -> OpenRouterCreditsProbe:
    """Return a process-wide probe, created on first use.

    Used by the TUI gateway, whose ``_get_usage`` is a module-level function
    with no per-instance state to hang a probe off. Safe for a multi-account
    gateway because the probe caches per account identity.
    """
    global _SHARED_PROBE
    if _SHARED_PROBE is None:
        with _SHARED_PROBE_LOCK:
            if _SHARED_PROBE is None:
                _SHARED_PROBE = OpenRouterCreditsProbe()
    return _SHARED_PROBE
