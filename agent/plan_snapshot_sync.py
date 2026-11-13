"""Keep the cached plan snapshot in step with what the gateway says on each reply.

The pickers read a cached account snapshot (``hermes_cli.nous_account``) and a cached Nous catalog
(``hermes_cli.models_pricing``), so a plan change made in the browser (upgrade, downgrade, cancel) or a
limit filling / resetting would only show after the next session opened. Every Nous reply already
carries the usage-window headers, so this compares them with the snapshot and, only when they
disagree, refreshes both caches once in the background. No extra call on an ordinary turn, and
nothing at all for a reply that is not from Nous.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Both keyed by Hermes home: one process can serve several profiles, each with its own account.
_inflight: set[str] = set()
# What the headers said the last time a refresh ran, and when. The portal can trail the gateway by a few
# minutes (usage is ingested in batches), so the same mismatch is refreshed once per retry interval, not on
# every reply meanwhile, and not never again: a refresh that read the portal before it caught up retries.
_last_attempt: dict[str, tuple[tuple[bool, frozenset[str]], float]] = {}
_RETRY_AFTER_S = 120.0


def _profile_key() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


def _snapshot_full_ids(snapshot: Any) -> Optional[frozenset[str]]:
    """Ids of the full allowances in the cached snapshot; None when it has no window plan."""
    window = getattr(snapshot, "usage_window", None)
    from agent.credits_tracker import _window_seed_fields
    if _window_seed_fields(window) is None:
        return None
    return frozenset(g.id for g in (getattr(window, "gauges", None) or ()) if g.full)


def plan_snapshot_is_stale(state: Any, snapshot: Any) -> bool:
    """True when a reply's credits state contradicts the cached snapshot in a way a picker shows:
    window headers but a non-window snapshot (upgrade), the reverse (downgrade or cancel), or a
    different set of full allowances (a limit filled or reset). No snapshot yet: nothing to fix (the
    session-start seed is fetching it)."""
    if state is None or snapshot is None or getattr(snapshot, "source", None) == "error":
        return False
    on_window = bool(getattr(state, "is_usage_window", False))
    full_ids = _snapshot_full_ids(snapshot)
    if on_window != (full_ids is not None):
        return True
    if not on_window:
        return False
    return frozenset(g.id for g in getattr(state, "gauges", ()) if g.full) != full_ids


def _refresh(plan_changed: bool, key: str) -> None:
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info
        if getattr(get_nous_portal_account_info(force_fresh=True), "source", None) == "error":
            _last_attempt.pop(key, None)  # a failed fetch (reported, not raised) is no attempt either
        if plan_changed:
            # The per-model levels (Large / Free tags) come with the catalog; refetch so they move with the plan.
            from hermes_cli.models_pricing import get_pricing_for_provider
            get_pricing_for_provider("nous", force_refresh=True)
    except Exception:
        logger.debug("plan sync ▸ refresh failed (fail-open)", exc_info=True)
        _last_attempt.pop(key, None)  # a transient portal error must not block the next re-sync
    finally:
        with _lock:
            _inflight.discard(key)


def sync_plan_snapshot(state: Any) -> bool:
    """Start one background refresh when *state* (a fresh ``CreditsState`` from reply headers) says the
    cached plan is out of date. Returns True iff a refresh started. Never raises, never blocks."""
    key = ""
    try:
        key = _profile_key()
        if not getattr(state, "is_usage_window", False):
            # Non-window reply: only a cached window snapshot (a downgrade / cancel) can be stale. Peek the
            # raw cache first so every other account pays no auth-state read per reply.
            from hermes_cli import nous_account
            cached = nous_account._account_info_cache
            if cached is None or _snapshot_full_ids(cached[2]) is None:
                _last_attempt.pop(key, None)
                return False
        from hermes_cli.nous_account import cached_nous_portal_account_info
        snapshot = cached_nous_portal_account_info(any_age=True)
        if not plan_snapshot_is_stale(state, snapshot):
            _last_attempt.pop(key, None)  # in step again: the next mismatch gets its own refresh
            return False
        on_window = bool(getattr(state, "is_usage_window", False))
        seen = (on_window, frozenset(g.id for g in getattr(state, "gauges", ()) if g.full))
        plan_changed = on_window != (_snapshot_full_ids(snapshot) is not None)
        now = time.monotonic()
        with _lock:
            last = _last_attempt.get(key)
            if key in _inflight or (last is not None and last[0] == seen and now - last[1] < _RETRY_AFTER_S):
                return False
            _inflight.add(key)
            _last_attempt[key] = (seen, now)
        # Under this profile's contextvars (its Hermes home and auth store), as the credits seed thread does.
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(_refresh, args=(plan_changed, key), name="nous-plan-sync").start()
        return True
    except Exception:
        logger.debug("plan sync ▸ check failed (fail-open)", exc_info=True)
        with _lock:
            _inflight.discard(key)
        return False
