"""Pool selection under its lock; network preparation belongs to CredentialPool.select."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.credential_pool import CredentialPool, PooledCredential


def select_unlocked(
    self: CredentialPool, *, refresh: bool = True, count: bool = True, model: Optional[str] = None,
) -> tuple[Optional[PooledCredential], list[PooledCredential]]:
    """Select the best available entry; returns ``(entry, pending_refresh)``.

    ``count=False`` skips the ``request_count`` bump for selections that are
    not going to serve a request (a forced-refresh target lookup).
    """
    from agent import credential_pool as pool_mod

    available, pending_refresh = self._available_entries(clear_expired=True, refresh=refresh, model=model)
    if not available:
        self._current_id = None
        self._log_no_available_entries()
        return None, pending_refresh

    # The pool recovered; re-arm the throttle so a later re-exhaustion
    # logs immediately.
    self._last_no_entries_log_at = None

    if self._strategy == pool_mod.STRATEGY_RESET_FIRST:
        from agent.credential_pool_usage import choose_entry
        entry = choose_entry(self.provider, available)
    elif self._strategy == pool_mod.STRATEGY_RANDOM:
        entry = pool_mod.random.choice(available)
    elif self._strategy == pool_mod.STRATEGY_LEAST_USED and len(available) > 1:
        entry = min(available, key=lambda e: e.request_count)
    else:
        entry = available[0]
    # Count the selection under every strategy. The counter is ``least_used``'s
    # baseline and reaches auth.json on the next persist (exhaustion, rotation,
    # refresh); it used to move only while ``least_used`` was active.
    if count:
        entry = self._adopt(entry, persist=False, request_count=entry.request_count + 1)
    if self._strategy in {pool_mod.STRATEGY_ROUND_ROBIN, pool_mod.STRATEGY_RESET_FIRST} and len(available) > 1:
        rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
        rotated.append(replace(entry, priority=len(self._entries) - 1))
        self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
        self._persist()
        entry = self._find(lambda candidate: candidate.id == entry.id) or entry
    self._current_id = entry.id
    return entry, pending_refresh
