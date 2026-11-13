"""Credential selection for ``CredentialPool``: ``select`` (rotates and counts) and ``peek`` (reads only)."""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


class CredentialPoolSelectionMixin:
    def select(self, *, model: Optional[str] = None, endpoint_url: Optional[str] = None) -> Optional[PooledCredential]:
        entry, pending_refresh = self._select_under_lock(model=model, endpoint_url=endpoint_url)
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
            # Re-select now that the refreshed entries are back in the pool.
            if entry is None:
                entry, _ = self._select_under_lock(model=model, endpoint_url=endpoint_url)
        if entry is not None:
            self._unmatched_rotation_streak = 0
        return entry

    def _select_under_lock(
        self, *, model: Optional[str] = None, endpoint_url: Optional[str] = None,
    ) -> tuple[Optional[PooledCredential], list[PooledCredential]]:
        with self._lock:
            return self._select_unlocked(model=model, endpoint_url=endpoint_url)

    def _select_unlocked(
        self, *, refresh: bool = True, count: bool = True, model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> tuple[Optional[PooledCredential], list[PooledCredential]]:
        """Select the best available entry; returns ``(entry, pending_refresh)``.

        ``count=False`` skips the ``request_count`` bump for selections that are
        not going to serve a request (a forced-refresh target lookup).
        """
        from agent.credential_pool import (
            STRATEGY_LEAST_USED, STRATEGY_RANDOM, STRATEGY_ROUND_ROBIN, credential_pool_entry_serves_endpoint,
        )
        available, pending_refresh = self._available_entries(clear_expired=True, refresh=refresh, model=model)
        if endpoint_url is not None:
            available = [e for e in available if credential_pool_entry_serves_endpoint(e, endpoint_url)]
        if not available:
            self._current_id = None
            self._log_no_available_entries()
            return None, pending_refresh

        # The pool recovered; re-arm the throttle so a later re-exhaustion
        # logs immediately.
        self._last_no_entries_log_at = None

        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
        elif self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda e: e.request_count)
        else:
            entry = available[0]
        # Count the selection under every strategy. The counter is ``least_used``'s
        # baseline and reaches auth.json on the next persist (exhaustion, rotation,
        # refresh); it used to move only while ``least_used`` was active.
        if count:
            entry = self._adopt(entry, persist=False, request_count=entry.request_count + 1)
        if self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
            self._persist()
            entry = self._find(lambda candidate: candidate.id == entry.id) or entry
        self._current_id = entry.id
        return entry, pending_refresh

    def peek(self, *, model: Optional[str] = None, endpoint_url: Optional[str] = None) -> Optional[PooledCredential]:
        from agent.credential_pool import credential_pool_entry_serves_endpoint
        with self._lock:  # the current entry, else the first available; with ``model``: the current one if available
            current = self._current_unlocked()
            if current is not None and model is None and (
                endpoint_url is None or credential_pool_entry_serves_endpoint(current, endpoint_url)
            ):
                return current
            available = self._available_entries(model=model)[0]
            if endpoint_url is not None:
                available = [e for e in available if credential_pool_entry_serves_endpoint(e, endpoint_url)]
            return next((e for e in available if current and e.id == current.id), next(iter(available), None))
