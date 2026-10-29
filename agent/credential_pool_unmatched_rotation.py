"""Rotation when a failed request's credential identity matches no pool entry.

An OAuth 401's key hint never matches a pool entry's ``runtime_api_key``, so a
long-lived session's retries all land here. Nothing is marked exhausted (that
would bench an innocent key), the rotation is bounded to one lap (#70401), and
the next entry is selected the way ``select()`` selects: an expiring OAuth token
is refreshed, never handed out as-is (#142473).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger("agent.credential_pool")


class CredentialPoolUnmatchedRotationMixin:
    """Mixed into ``CredentialPool``; uses its lock, entries and selection primitives."""

    def _try_unmatched_rotation(
        self, credential_id: Optional[str], api_key_hint: Optional[str],
    ) -> tuple[bool, Optional[PooledCredential]]:
        """``(True, next_entry)`` when a supplied failed identity matched no entry and was rotated
        here; ``(False, None)`` when the caller should mark the identified entry instead."""
        if not (credential_id or api_key_hint):
            return False, None
        with self._lock:
            if self._identify_failed_entry(credential_id, api_key_hint) is not None:
                return False, None
            tokens_before = self._entry_tokens()
            next_entry, pending = self._rotate_unmatched(tokens_before)
        if pending:
            next_entry = self._finish_unmatched_rotation(next_entry, pending, tokens_before)
        return True, next_entry

    def _rotate_unmatched(
        self, tokens_before: dict[str, tuple[str, str]],
    ) -> tuple[Optional[PooledCredential], list[PooledCredential]]:
        """Rotate without marking anything when the failed identity matches no entry.

        Falling through to current()/_select_unlocked() would bench an
        innocent healthy key for the full TTL. But this must be BOUNDED
        (#70401): with OAuth-token auth the 401's key hint never matches any
        ``runtime_api_key``, so every retry lands here, nothing is marked, and
        the caller retries the same dead token forever (~6/sec, starving the
        event loop). Cap consecutive no-mark rotations at one lap of the
        available entries, then surface the error; no cooldown is written.

        Returns ``(entry, pending_refresh)``. The next entry is selected the
        way ``select()`` would (#142473): an entry whose OAuth token is
        expiring is refreshed, never handed out as-is, or for a single-use-token
        provider returned in *pending_refresh* for the caller to refresh
        outside the lock.
        """
        self._unmatched_rotation_streak += 1
        available_count = len(self._available_entries()[0])
        if self._unmatched_rotation_streak > max(available_count, 1):
            logger.warning(
                "credential pool: failed credential identity matched no "
                "%s entry for %d consecutive rotations (pool size %d) — "
                "surfacing the error instead of rotating again",
                self.provider, self._unmatched_rotation_streak, available_count,
            )
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None, []
        logger.info(
            "credential pool: failed credential identity matched no %s "
            "entry; rotating without marking any credential exhausted",
            self.provider,
        )
        self._current_id = None
        next_entry, pending = self._select_unlocked(refresh=True)
        if next_entry is not None and not self._unmatched_rotation_changes_credential(next_entry, tokens_before):
            # A single-entry pool cannot rotate: returning its only entry would
            # report a recovery without changing the credential, and the
            # caller retries the same 401 indefinitely.
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None, pending
        return next_entry, pending

    def _entry_tokens(self) -> dict[str, tuple[str, str]]:
        return {e.id: (e.access_token or "", e.runtime_api_key or "") for e in self._entries}

    def _unmatched_rotation_changes_credential(
        self, entry: PooledCredential, tokens_before: dict[str, tuple[str, str]],
    ) -> bool:
        """False when *entry* is the pool's only available entry and still carries the token it had
        before this rotation. A refreshed token is a real change, so a sole entry may be returned then."""
        if len(self._available_entries()[0]) != 1:
            return True
        return tokens_before.get(entry.id) != (entry.access_token or "", entry.runtime_api_key or "")

    def _finish_unmatched_rotation(
        self,
        next_entry: Optional[PooledCredential],
        pending: list[PooledCredential],
        tokens_before: dict[str, tuple[str, str]],
    ) -> Optional[PooledCredential]:
        """Refresh the deferred entries of an unmatched rotation outside the lock (as ``select()`` does),
        then re-select if nothing was ready. An entry whose refresh failed stays out of rotation."""
        self._refresh_pending_entries(pending)
        if next_entry is not None:
            return next_entry
        with self._lock:
            next_entry, _pending = self._select_unlocked(refresh=True)
            if next_entry is not None and not self._unmatched_rotation_changes_credential(next_entry, tokens_before):
                self._unmatched_rotation_streak = 0
                self._current_id = None
                return None
            return next_entry
