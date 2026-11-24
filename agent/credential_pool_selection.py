"""Credential selection for ``CredentialPool``: ``select`` (rotates and counts), ``choose`` plus
``commit`` (the same choice, recorded only once the key is used) and ``peek`` (reads only); and the
forced refresh of the row behind a failed request (one locked lookup, one typed ``RefreshOutcome``)."""
from __future__ import annotations

import enum
import random
from dataclasses import replace
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# Maps a row's stored base URL to the URL a client built from it calls (MiniMax rows store
# ``/anthropic``; a chat-completions client calls the ``/v1`` sibling).
AsCalled = Optional[Callable[[str], str]]


def _serving(entries: list, endpoint_url: Optional[str], as_called: AsCalled = None) -> list:
    """``entries`` that serve ``endpoint_url`` (all of them when it is None); see ``AsCalled``."""
    from agent.credential_pool import credential_pool_entry_serves_endpoint as serves
    if endpoint_url is None:
        return entries
    if as_called is None:
        return [e for e in entries if serves(e, endpoint_url)]
    from hermes_cli.route_identity import normalize_route_base_url as norm
    target = norm(as_called(endpoint_url))
    return [e for e in entries
            if not (url := getattr(e, "runtime_base_url", None) or getattr(e, "base_url", None)) or norm(as_called(url)) == target]


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
        available, pending_refresh = self._available_entries(clear_expired=True, refresh=refresh, model=model)
        available = _serving(available, endpoint_url)
        if not available:
            self._current_id = None
            self._log_no_available_entries()
            return None, pending_refresh

        # The pool recovered; re-arm the throttle so a later re-exhaustion
        # logs immediately.
        self._last_no_entries_log_at = None
        return self._record_pick(self._strategy_pick(available), available, count=count), pending_refresh

    def _strategy_pick(self, available: list[PooledCredential]) -> PooledCredential:
        from agent.credential_pool import STRATEGY_LEAST_USED, STRATEGY_RANDOM
        if self._strategy == STRATEGY_RANDOM:
            return random.choice(available)
        if self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            return min(available, key=lambda e: e.request_count)
        return available[0]

    def _record_pick(self, entry: PooledCredential, available: list[PooledCredential], *, count: bool = True) -> PooledCredential:
        from agent.credential_pool import STRATEGY_ROUND_ROBIN
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
        return entry

    def choose(self, *, model: Optional[str] = None, endpoint_url: Optional[str] = None,
               as_called: AsCalled = None) -> Optional[PooledCredential]:
        """The entry ``select`` would pick now, without counting it, rotating the pool, refreshing a token
        or saving anything; ``commit`` records it once its key is actually used. Under ``random`` a later
        ``select`` may pick another entry."""
        with self._lock:
            available = _serving(self._available_entries(model=model)[0], endpoint_url, as_called)
            return self._strategy_pick(available) if available else None

    def commit(self, credential_id: str, *, model: Optional[str] = None, endpoint_url: Optional[str] = None,
               as_called: AsCalled = None) -> Optional[PooledCredential]:
        """Record a ``choose`` result as used: its request count and the round-robin order move the way
        ``select`` moves them. None when the entry is no longer available."""
        with self._lock:
            available = _serving(self._available_entries(clear_expired=True, model=model)[0], endpoint_url, as_called)
            entry = next((e for e in available if e.id == credential_id), None)
            if entry is None:
                return None
            self._last_no_entries_log_at = None
            return self._record_pick(entry, available)

    def peek(self, *, model: Optional[str] = None, endpoint_url: Optional[str] = None) -> Optional[PooledCredential]:
        from agent.credential_pool import credential_pool_entry_serves_endpoint
        with self._lock:  # the current entry, else the first available; with ``model``: the current one if available
            current = self._current_unlocked()
            if current is not None and model is None and (
                endpoint_url is None or credential_pool_entry_serves_endpoint(current, endpoint_url)
            ):
                return current
            available = _serving(self._available_entries(model=model)[0], endpoint_url)
            return next((e for e in available if current and e.id == current.id), next(iter(available), None))


class RefreshKind(enum.Enum):
    ADOPTED = "adopted"  # the in-lock store re-read found a usable bearer; no refresh token spent
    POSTED = "posted"  # the provider refresh ran
    REFUSED = "refused"  # the row needed a POST that ``may_post`` refused
    NEEDS_ROTATION = "needs_rotation"  # no matching row, a non-OAuth row, or the refresh failed


class RefreshOutcome(NamedTuple):
    kind: RefreshKind
    entry: Optional[PooledCredential] = None
    row: Optional[PooledCredential] = None


NO_REFRESH = RefreshOutcome(RefreshKind.NEEDS_ROTATION)


class CredentialPoolRefreshMixin:
    def try_refresh_current(self) -> Optional[PooledCredential]:
        with self._lock:
            entry = self._current_unlocked()
            refreshed = self._refresh_entry(entry, force=True) if entry is not None else None
            if refreshed is not None:
                self._current_id = refreshed.id
            return refreshed

    def try_refresh_matching(
        self, api_key_hint: Optional[str] = None, credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Force-refresh the entry that supplied the failed request; the id wins over the key."""
        return self.refresh_failed_entry(api_key_hint, credential_id).entry

    def refresh_failed_entry(
        self, api_key_hint: Optional[str] = None, credential_id: Optional[str] = None, *,
        key_wins: bool = False, may_post: Optional[Callable[[PooledCredential], bool]] = None,
    ) -> RefreshOutcome:
        """Under the pool lock: find the failed row, adopt a usable stored bearer, else POST if allowed.

        ``key_wins`` resolves the row like rotation does (#79156) when the key names a pool row. A key
        no row holds is usually this row's own pre-rotation bearer (a stale 401, #138337), so the id
        then still names the row and the refresh adopts its rotated token. With no identity, select
        WITHOUT the proactive refresh: the forced refresh must spend a rotating refresh token exactly once.
        """
        with self._lock:
            by_id, by_key = self._failed_entry_matches(credential_id, api_key_hint)
            id_sent_key = by_id is not None and (not api_key_hint or by_id.runtime_api_key == api_key_hint)
            entry = (by_id if id_sent_key else by_key or by_id) if key_wins else (by_id or by_key)
            if entry is None and not api_key_hint:
                entry = self._current_unlocked() or self._select_unlocked(refresh=False, count=False)[0]
            if entry is None:
                return NO_REFRESH
            self._current_id = entry.id
            outcome = self._refresh_entry_outcome(
                entry, force=True, failed_key=api_key_hint, allow_post=may_post is None or may_post(entry))
            if outcome.entry is not None:
                self._current_id = outcome.entry.id
            return outcome._replace(row=entry)

    def _post_refresh(self, entry: PooledCredential, *, force: bool, allow_post: bool,
                      failed_key: Optional[str] = None) -> RefreshOutcome:
        if not allow_post:
            return RefreshOutcome(RefreshKind.REFUSED)
        refreshed = self._refresh_entry_impl(entry, force=force, failed_key=failed_key)
        return NO_REFRESH if refreshed is None else RefreshOutcome(RefreshKind.POSTED, refreshed)
