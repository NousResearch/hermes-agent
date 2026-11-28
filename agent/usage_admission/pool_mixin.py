"""``CredentialPool`` methods that probe, cache and apply Codex usage-admission decisions."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Optional

from agent.usage_admission import policy

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)


class CredentialPoolUsageAdmissionMixin:
    def _probe_usage_admission_entry(
        self,
        entry: PooledCredential,
        cache_key: tuple[str, str, str, str],
    ) -> None:
        """Probe and cache one entry. Network I/O never holds a pool lock."""
        snapshot = None
        try:
            from agent.account_usage import fetch_account_usage

            snapshot = fetch_account_usage(
                "openai-codex",
                base_url=entry.runtime_base_url,
                api_key=entry.runtime_api_key,
                timeout=policy._USAGE_ADMISSION_PROBE_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.debug(
                "credential pool: Codex usage admission probe failed",
                exc_info=True,
            )

        checked_at = time.monotonic()
        window = policy._select_codex_policy_window(snapshot)
        if window is None:
            decision = policy._UsageAdmissionDecision(
                checked_at=checked_at,
                probe_succeeded=False,
                used_percent=None,
                reset_at=None,
            )
        else:
            used_percent = float(window.used_percent)
            reset_at = getattr(window, "reset_at", None)
            try:
                reset_timestamp = (
                    float(reset_at.timestamp()) if reset_at is not None else None
                )
            except (AttributeError, TypeError, ValueError, OverflowError):
                reset_timestamp = None
            decision = policy._UsageAdmissionDecision(
                checked_at=checked_at,
                probe_succeeded=True,
                used_percent=used_percent,
                reset_at=reset_timestamp,
            )

        with policy._USAGE_ADMISSION_CACHE_LOCK:
            # Preserve fresh decisions for other auth identities of the same
            # stable credential ID. A delayed probe for an old OAuth token must
            # not erase the refreshed token's decision.
            stale_keys = [
                key
                for key, cached_decision in policy._USAGE_ADMISSION_CACHE.items()
                if key[:2] == cache_key[:2]
                and key != cache_key
                and not policy._usage_admission_decision_is_fresh(
                    cached_decision, checked_at
                )
            ]
            for stale_key in stale_keys:
                policy._USAGE_ADMISSION_CACHE.pop(stale_key, None)
            policy._USAGE_ADMISSION_CACHE[cache_key] = decision

    def _refresh_usage_admission_cache(
        self,
        candidate_entries: Optional[list[PooledCredential]] = None,
    ) -> None:
        """Refresh configured Codex decisions with one bounded parallel wait."""
        limits = policy.get_pool_usage_limits(self.provider)
        with self._lock:
            self._usage_limits = dict(limits)
            if not limits:
                return
            entry_ids = {entry.id for entry in self._entries}
            source_entries = (
                list(candidate_entries)
                if candidate_entries is not None
                else self._entries
            )
            candidates = [
                entry
                for entry in source_entries
                if entry.id in limits and entry.runtime_api_key
            ]
        for credential_id in limits.keys() - entry_ids:
            warning_key = (self.provider, credential_id)
            with policy._USAGE_ADMISSION_CACHE_LOCK:
                warn = warning_key not in policy._WARNED_UNMATCHED_USAGE_LIMITS
                policy._WARNED_UNMATCHED_USAGE_LIMITS.add(warning_key)
            if warn:
                logger.warning(
                    "credential pool: usage limit configured for missing %s "
                    "credential %s",
                    self.provider,
                    credential_id,
                )
        pending: list[concurrent.futures.Future[None]] = []
        for entry in candidates:
            cache_key = policy._usage_admission_cache_key(entry)
            now = time.monotonic()
            callback_registration = None
            with policy._USAGE_ADMISSION_CACHE_LOCK:
                cached = policy._USAGE_ADMISSION_CACHE.get(cache_key)
                if cached is not None and policy._usage_admission_decision_is_fresh(
                    cached, now
                ):
                    continue
                future = policy._USAGE_ADMISSION_INFLIGHT.get(cache_key)
                if future is None:
                    future = policy._USAGE_ADMISSION_PROBE_EXECUTOR.submit(
                        self._probe_usage_admission_entry,
                        entry,
                        cache_key,
                    )
                    policy._USAGE_ADMISSION_INFLIGHT[cache_key] = future
                    callback_registration = (future, cache_key)
                pending.append(future)

            if callback_registration is not None:
                registered_future, registered_key = callback_registration

                def clear_inflight(
                    completed: concurrent.futures.Future[None],
                    *,
                    key: tuple[str, str, str, str] = registered_key,
                ) -> None:
                    with policy._USAGE_ADMISSION_CACHE_LOCK:
                        if policy._USAGE_ADMISSION_INFLIGHT.get(key) is completed:
                            policy._USAGE_ADMISSION_INFLIGHT.pop(key, None)

                registered_future.add_done_callback(clear_inflight)

        if pending:
            done, _not_done = concurrent.futures.wait(
                set(pending),
                timeout=policy._USAGE_ADMISSION_PROBE_WAIT_SECONDS,
            )
            for future in done:
                try:
                    future.result()
                except Exception:
                    logger.debug(
                        "credential pool: unexpected usage probe worker failure",
                        exc_info=True,
                    )

    def _usage_admission_denied(self, entry: PooledCredential) -> bool:
        if entry.id not in self._usage_limits or not entry.runtime_api_key:
            return False
        cache_key = policy._usage_admission_cache_key(entry)
        now = time.monotonic()
        with policy._USAGE_ADMISSION_CACHE_LOCK:
            decision = policy._USAGE_ADMISSION_CACHE.get(cache_key)
            return bool(
                decision is not None
                and policy._usage_admission_decision_denied(
                    decision, self._usage_limits[entry.id], now
                )
            )

    def _apply_admission_after_forced_refresh(
        self,
        refreshed: Optional[PooledCredential],
    ) -> tuple[Optional[PooledCredential], Optional[str]]:
        """Admit a freshly minted token before it can be assigned.

        Forced OAuth refresh changes the token fingerprint for the same stable
        pool entry.  Probe that exact refreshed token outside the pool lock and
        return the denied stable ID separately so recovery callers can select a
        sibling without misclassifying policy denial as credential exhaustion.
        Telemetry failure retains the policy's fail-open behavior.
        """
        if refreshed is None:
            return None, None
        self._refresh_usage_admission_cache(candidate_entries=[refreshed])
        if self._usage_admission_denied(refreshed):
            logger.info(
                "credential pool: usage admission excludes refreshed %s "
                "credential %s",
                self.provider,
                refreshed.id,
            )
            return None, refreshed.id
        return refreshed, None

    def try_refresh_current_with_admission_status(
        self,
    ) -> tuple[Optional[PooledCredential], Optional[str]]:
        with self._lock:
            refreshed = self._try_refresh_current_unlocked()
        return self._apply_admission_after_forced_refresh(refreshed)

    def try_refresh_current(self) -> Optional[PooledCredential]:
        refreshed, _denied_id = self.try_refresh_current_with_admission_status()
        return refreshed

    def try_refresh_matching(
        self,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        refreshed, _denied_id = self.try_refresh_matching_with_admission_status(
            api_key_hint=api_key_hint,
            credential_id=credential_id,
        )
        return refreshed

    def try_refresh_matching_with_admission_status(
        self,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> tuple[Optional[PooledCredential], Optional[str]]:
        """Force-refresh the entry that supplied the failed request.

        Direct integrations may reload the pool after a request failed, so
        ``current_id`` cannot identify the issuing credential. With no hint,
        select WITHOUT the normal proactive refresh: the forced refresh below
        must consume a rotating refresh token exactly once.
        """
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id) if credential_id else None
            if entry is None:
                if api_key_hint:
                    entry = self._find(lambda e: e.runtime_api_key == api_key_hint)
                else:
                    entry = self._current_unlocked() or self._select_unlocked(refresh=False, count=False)[0]
            if entry is None:
                return None, None
            self._current_id = entry.id
            refreshed = self._try_refresh_current_unlocked(failed_key=api_key_hint)
        return self._apply_admission_after_forced_refresh(refreshed)

