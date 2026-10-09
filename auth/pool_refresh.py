"""Credential-pool synchronization and refresh commit orchestration."""

from __future__ import annotations
from typing import List, Set, Tuple  # noqa: F401 (pool collaborators consume these bindings)

import auth.providers.anthropic as _auth_auth_providers_anthropic

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from auth.credential_pool import PooledCredential


class CredentialPoolRefreshMixin:
    def _sync_anthropic_entry_from_credentials_file(
        self, entry: PooledCredential
    ) -> PooledCredential:
        """Sync a claude_code entry from ~/.claude/.credentials.json if tokens differ."""
        from auth.credential_pool import _CLEAR_STATUS, logger

        if self.provider != "anthropic" or entry.source != "claude_code":
            return entry
        try:
            from auth.providers.anthropic import read_claude_code_credentials

            creds = read_claude_code_credentials(environment=self.environment)
            if not creds:
                return entry
            file_refresh = creds.get("refreshToken", "")
            file_access = creds.get("accessToken", "")
            # Access tokens can be re-issued without a new refresh token, so
            # checking only refresh_token leaves a stale access_token in the
            # pool -> 401 on every request until the exhausted TTL expires.
            if (file_access or file_refresh) and (
                (file_access and file_access != (entry.access_token or ""))
                or (file_refresh and file_refresh != (entry.refresh_token or ""))
            ):
                logger.debug(
                    "Pool entry %s: syncing tokens from credentials file (tokens changed)",
                    entry.id,
                )
                return self._adopt(
                    entry,
                    access_token=file_access or entry.access_token,
                    refresh_token=file_refresh or entry.refresh_token,
                    expires_at_ms=creds.get("expiresAt", 0) or entry.expires_at_ms,
                    **_CLEAR_STATUS,
                )
        except Exception as exc:
            logger.debug("Failed to sync from credentials file: %s", exc)
        return entry

    def _sync_entry_from_pool_store(self, entry: PooledCredential) -> PooledCredential:
        """Adopt a token pair rotated by another pool instance (anthropic, xai-oauth).

        Re-reads the exact persisted row from the credential-pool store while
        the shared cross-process auth-store lock is held. Direct integrations
        load a fresh ``CredentialPool`` per request, so in-memory locks cannot
        protect a single-use refresh token across requests or processes.

        Anthropic borrowed sources (``claude_code``) are excluded: they are
        reference-only rows whose secrets are stripped before reaching
        auth.json, so re-reading yields empty tokens that would be adopted as
        a "rotation" — blanking a usable credential. The singleton file, not
        the pool store, is token authority for those sources; a row with no
        token material at all is refused for the same reason.
        """
        from auth.credential_pool import (
            PooledCredential,
            auth_pool_persistence,
            is_borrowed_credential_source,
            logger,
            plugin_refresh_hook,
            read_credential_pool,
        )

        if (
            self.provider not in ("anthropic", "xai-oauth")
            and plugin_refresh_hook(self.provider) is None
        ):
            return entry
        is_anthropic = self.provider == "anthropic"
        is_xai = self.provider == "xai-oauth"
        display = {"anthropic": "Anthropic", "xai-oauth": "xAI"}.get(
            self.provider, self.provider
        )
        if is_anthropic and is_borrowed_credential_source(entry.source, self.provider):
            return entry
        try:
            persisted = next(
                (
                    p
                    for p in read_credential_pool(self.provider)
                    if isinstance(p, dict) and p.get("id") == entry.id
                ),
                None,
            )
            if not isinstance(persisted, dict):
                return entry
            # Same base policy as _persist/load_pool: a token-less disk row is a known
            # (blank) generation, recorded before the no-token-material bail-out below.
            self._persisted_token_pairs[entry.id] = (
                auth_pool_persistence._credential_token_pair(persisted)
            )
            stored = PooledCredential.from_dict(self.provider, persisted)
            # No token material at all is never a "rotation" (anthropic borrowed rows, a plugin row a
            # peer blanked mid-write): adopting it would replace a usable credential with nothing.
            if (
                not is_xai
                and not (stored.access_token or "").strip()
                and not (stored.refresh_token or "").strip()
            ):
                return entry
            if (
                stored.access_token != entry.access_token
                or stored.refresh_token != entry.refresh_token
            ):
                logger.debug(
                    "Pool entry %s: adopting %s OAuth tokens rotated by another pool instance",
                    entry.id,
                    display,
                )
                self._replace_entry(entry, stored)
                return stored
        except Exception as exc:
            logger.debug(
                "Failed to sync %s OAuth entry from credential pool: %s", display, exc
            )
        return entry

    _sync_anthropic_entry_from_pool_store = _sync_entry_from_pool_store

    def _sync_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync a Codex / xAI device_code entry from auth.json ``providers.<id>.tokens``.

        A fresh ``hermes model`` / ``hermes auth`` login writes new tokens
        under ``_auth_store_lock`` while the pool entry may sit frozen behind
        a ``last_error_reset_at`` hours in the future; without this sync every
        request fails with "no available entries" despite fresh credentials on
        disk. Only singleton-seeded entries apply — env/API-key rows have no
        auth.json shadow.
        """
        from auth.credential_pool import (
            _CLEAR_STATUS,
            _TOKENS_SINGLETON_PROVIDERS,
            _auth_store_lock,
            _codex_entry_tracks_singleton,
            _load_auth_store,
            _load_provider_state,
            _singleton_predates_entry,
            logger,
        )

        spec = _TOKENS_SINGLETON_PROVIDERS.get(self.provider)
        if spec is None:
            return entry
        display = spec[0]
        is_codex = self.provider == "openai-codex"
        sources = (
            ("device_code", "manual:device_code") if is_codex else ("device_code",)
        )
        if entry.source not in sources:
            return entry
        try:
            with _auth_store_lock():
                state = _load_provider_state(_load_auth_store(), self.provider)
            tokens = state.get("tokens") if isinstance(state, dict) else None
            if not isinstance(tokens, dict):
                return entry
            if is_codex and not _codex_entry_tracks_singleton(entry, tokens):
                return entry
            store_access = tokens.get("access_token", "")
            store_refresh = tokens.get("refresh_token", "")
            entry_refresh = entry.refresh_token or ""
            # Adopt when either side differs: a fresh refresh_token from
            # another process means our pair is consumed/stale.
            should_adopt = bool(store_access) and (
                store_access != (entry.access_token or "")
                or (store_refresh and store_refresh != entry_refresh)
            )
            if (
                not should_adopt
                and is_codex
                and store_refresh
                and store_refresh != entry_refresh
                and not store_access
            ):
                # Store has only a refresh_token — another process rotated the
                # pair and the access_token was consumed. Adopt the
                # refresh_token so we don't replay the consumed one.
                logger.info(
                    "Pool entry %s: auth.json has newer refresh_token "
                    "but no access_token; adopting refresh_token to "
                    "avoid replaying consumed token",
                    entry.id,
                )
                should_adopt = True
            if should_adopt and _singleton_predates_entry(state, entry):
                # #106705: manual:* entries never write back to the singleton
                # (#39236), so after a pool-side rotation the singleton sits
                # one chain behind. Adopting it would replay the consumed
                # refresh token. ``last_refresh`` is stamped on every
                # successful rotation on both sides; when either side lacks a
                # parseable stamp this falls through to the historical
                # adopt-on-difference above (#70111).
                logger.info(
                    "Pool entry %s: auth.json singleton predates this entry's "
                    "rotation (last_refresh %s < %s); keeping pool chain to "
                    "avoid replaying the consumed refresh token",
                    entry.id,
                    state.get("last_refresh") if isinstance(state, dict) else None,
                    entry.last_refresh,
                )
                should_adopt = False
            if should_adopt:
                logger.debug(
                    "Pool entry %s: syncing %s tokens from auth.json (refreshed by another process)",
                    entry.id,
                    display,
                )
                field_updates: Dict[str, Any] = {
                    "access_token": store_access or entry.access_token,
                    "refresh_token": store_refresh or entry.refresh_token,
                    **_CLEAR_STATUS,
                }
                if state.get("last_refresh"):
                    field_updates["last_refresh"] = state["last_refresh"]
                return self._adopt(entry, **field_updates)
        except Exception as exc:
            logger.debug("Failed to sync %s entry from auth.json: %s", display, exc)
        return entry

    def _sync_nous_entry_from_auth_store(
        self, entry: PooledCredential
    ) -> PooledCredential:
        """Sync a Nous device_code entry from auth.json ``providers.nous`` if state differs.

        Another process refreshing via ``resolve_nous_runtime_credentials``
        writes fresh tokens under ``_auth_store_lock``; adopting them avoids a
        "refresh token reuse" revocation on the Nous Portal.
        """
        from auth.credential_pool import (
            _CLEAR_STATUS,
            _NOUS_EXTRA_STATE_KEYS,
            _auth_store_lock,
            _load_auth_store,
            _load_provider_state,
            logger,
        )

        if self.provider != "nous" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                state = _load_provider_state(_load_auth_store(), "nous")
            if not state:
                return entry
            comparable = {
                key: state.get(key)
                for key in (
                    "access_token",
                    "refresh_token",
                    "expires_at",
                    "agent_key",
                    "agent_key_expires_at",
                    "inference_base_url",
                )
            }
            if not any(
                v not in (None, "") and getattr(entry, k, None) != v
                for k, v in comparable.items()
            ):
                return entry
            logger.debug("Pool entry %s: syncing Nous state from auth.json", entry.id)
            field_updates: Dict[str, Any] = dict(_CLEAR_STATUS)
            field_updates.update({k: v for k, v in comparable.items() if v})
            extra_updates = dict(entry.extra)
            extra_updates.update({
                k: state[k] for k in _NOUS_EXTRA_STATE_KEYS if state.get(k) is not None
            })
            return self._adopt(entry, extra=extra_updates, **field_updates)
        except Exception as exc:
            logger.debug("Failed to sync Nous entry from auth.json: %s", exc)
        return entry

    def _sync_device_code_entry_to_auth_store(self, entry: PooledCredential) -> None:
        """Write refreshed pool entry tokens back to auth.json ``providers.<id>``.

        Otherwise the next ``load_pool()`` re-seeds the stale singleton state
        over the fresh entry — potentially a consumed single-use refresh
        token. Applies to Nous, OpenAI Codex and xAI OAuth singletons.

        ``set_active=False`` everywhere: a sync-back is a token-rotation side
        effect, not the user choosing a provider; ``_save_provider_state``
        would flip ``active_provider`` to whichever provider refreshed last.

        #74339: decide the root write-through on WHERE the state resolved
        from (``_load_provider_state_with_source``), not on whether the
        profile has a ``providers.<id>`` key — ``_store_provider_state``
        creates that key unconditionally, which self-sealed the check after
        the first refresh. When the grant came from the global root, write
        back to root ONLY and skip the profile store so it never accrues a
        shadowing key that blocks both the fallback and the write-through.
        """
        # Only singleton-seeded entries sync back; ``manual:*`` entries are
        # independent credentials and must not write to the singleton.
        from auth.credential_pool import (
            _TOKENS_SINGLETON_PROVIDERS,
            _auth_store_lock,
            _global_auth_file_path,
            _load_auth_store,
            _load_provider_state_with_source,
            _same_path,
            _save_auth_store,
            _store_provider_state,
            _write_through_provider_state_to_global_root,
            logger,
        )

        if entry.source != "device_code" or self.provider not in (
            "nous",
            *_TOKENS_SINGLETON_PROVIDERS,
        ):
            return
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state, source_path = _load_provider_state_with_source(
                    auth_store, self.provider
                )
                if not isinstance(state, dict):
                    return
                global_root = _global_auth_file_path()
                is_from_root = bool(
                    source_path is not None
                    and global_root is not None
                    and _same_path(source_path, global_root)
                )
                if not self._apply_entry_to_singleton_state(entry, state):
                    return
                if is_from_root:
                    _write_through_provider_state_to_global_root(self.provider, state)
                else:
                    _store_provider_state(
                        auth_store, self.provider, state, set_active=False
                    )
                    _save_auth_store(auth_store)
        except Exception as exc:
            logger.debug(
                "Failed to sync %s pool entry back to auth store: %s",
                self.provider,
                exc,
            )

    def _apply_entry_to_singleton_state(
        self, entry: PooledCredential, state: Dict[str, Any]
    ) -> bool:
        """Copy *entry*'s tokens into the provider's auth.json ``state`` in place."""
        from auth.credential_pool import _NOUS_EXTRA_STATE_KEYS

        if self.provider == "nous":
            state["access_token"] = entry.access_token
            for key in (
                "refresh_token",
                "expires_at",
                "agent_key",
                "agent_key_expires_at",
            ):
                if getattr(entry, key):
                    state[key] = getattr(entry, key)
            for extra_key in _NOUS_EXTRA_STATE_KEYS:
                val = entry.extra.get(extra_key)
                if val is not None:
                    state[extra_key] = val
            if entry.inference_base_url:
                state["inference_base_url"] = entry.inference_base_url
            return True
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            return False
        tokens["access_token"] = entry.access_token
        if entry.refresh_token:
            tokens["refresh_token"] = entry.refresh_token
        if entry.last_refresh:
            state["last_refresh"] = entry.last_refresh
        return True

    # ---- refresh -----------------------------------------------------------

    def _refresh_entry(
        self, entry: PooledCredential, *, force: bool
    ) -> Optional[PooledCredential]:
        from auth.credential_pool import (
            AUTH_TYPE_OAUTH,
            _SINGLE_USE_REFRESH_PROVIDERS,
            _auth_store_lock,
            plugin_refresh_hook,
        )

        if entry.auth_type != AUTH_TYPE_OAUTH or not entry.refresh_token:
            if force:
                self._mark_exhausted(entry, None)
            return None
        # Plugin providers with a ``refresh_credential`` hook are treated as single-use by default:
        # the pool cannot know their grant semantics, and a needless in-lock re-read is cheaper than
        # a ``refresh_token_reused`` login loss. Eligibility comes from the hook, never a name set.
        if (
            self.provider not in _SINGLE_USE_REFRESH_PROVIDERS
            and plugin_refresh_hook(self.provider) is None
        ):
            return self._refresh_entry_impl(entry, force=force)

        # Single-use refresh tokens: sync -> POST -> write-back must be atomic
        # across Hermes processes, or two processes adopt the same on-disk
        # token, both POST it, and the loser gets ``refresh_token_reused`` /
        # ``invalid_grant`` (for Anthropic sources other than claude_code
        # there was no recovery path at all). Serialize through the shared
        # cross-process auth-store flock; a waiter's in-lock re-sync picks up
        # the winner's rotated token and skips the POST.
        with _auth_store_lock(timeout_seconds=self._single_use_refresh_lock_timeout()):
            if self.provider == "openai-codex":
                synced = self._sync_entry_from_auth_store(entry)
                if (
                    synced is not entry
                    and not force
                    and not self._entry_needs_refresh(synced)
                ):
                    return synced
                return self._refresh_entry_impl(synced, force=force)
            synced = self._sync_entry_from_pool_store(entry)
            if self.provider == "anthropic" and synced.source == "claude_code":
                # claude_code entries are NOT profile-owned: the refresh token
                # lives in one shared ~/.claude/.credentials.json (or Keychain)
                # every profile reads. The profile-scoped lock above only covers
                # THIS profile's auth.json, so take the dedicated shared-file
                # lock (inner, per the ordering invariant on ``_auth_store_lock``)
                # and re-read that authoritative file before any
                # adopt-and-return shortcut fires. The official ``claude`` CLI
                # rotating out-of-band is handled by the sync-and-retry-once
                # fallback in ``_recover_failed_refresh``.
                with self._claude_code_credentials_lock():
                    synced = self._sync_anthropic_entry_from_credentials_file(synced)
                    if synced.refresh_token != entry.refresh_token:
                        return synced
                    return self._refresh_entry_impl(synced, force=force)
            if (
                synced.access_token != entry.access_token
                or synced.refresh_token != entry.refresh_token
            ):
                return synced
            return self._refresh_entry_impl(synced, force=force)

    def _claude_code_credentials_lock(self):
        """Cross-process lock keyed to the shared claude_code credentials file.

        Unlike the per-profile ``_auth_store_lock()`` this serializes every
        profile and process that might refresh a ``claude_code`` entry.
        """
        from auth.credential_pool import _auth_store_lock
        from auth.providers.anthropic import claude_code_credentials_path

        return _auth_store_lock(
            timeout_seconds=self._single_use_refresh_lock_timeout(),
            target_path=claude_code_credentials_path(),
        )

    def _fail_closed_unpersisted_rotation(
        self,
        entry: PooledCredential,
        exc: BaseException,
        *,
        store: str,
    ) -> None:
        """Quarantine an entry whose rotated pair never reached its store.

        For ``claude_code`` / ``hermes_pkce`` the singleton file — not
        auth.json — is authoritative: ``_seed_from_singletons()`` re-reads it
        on every ``load_pool()``. When the refresh POST succeeded but the
        singleton write failed, the replacement pair exists only in memory
        while the consumed pair survives on disk and would be re-seeded over
        any row we persisted; the next refresh would replay the spent token.
        So never expose or persist the rotated pair; mark the entry terminally
        so it surfaces as an explicit re-auth requirement.
        """
        from auth.credential_pool import CREDENTIAL_PERSIST_FAILED_REASON, logger

        logger.error(
            "Anthropic %s refresh rotated the single-use token but could not commit it "
            "to %s (%s) — failing closed and quarantining the credential; "
            "re-authenticate to recover",
            entry.source,
            store,
            exc,
        )
        try:
            from auth.providers.anthropic import (
                mark_rotation_consumed_uncommitted,
                spent_rotation_source_path,
            )

            # The singleton still holds the spent pair and load_pool() re-seeds
            # it, so record the fingerprints — persisted to the shared source's
            # sidecar registry (we hold its path-keyed lock here) so OTHER
            # processes/profiles adopt the terminal verdict instead of leasing
            # the stale pair or re-POSTing the spent refresh token.
            mark_rotation_consumed_uncommitted(
                entry.access_token,
                entry.refresh_token,
                source_path=spent_rotation_source_path(entry.source),
            )
        except Exception:  # pragma: no cover - never block the quarantine
            logger.debug(
                "Failed to record consumed rotation fingerprints", exc_info=True
            )
        self._mark_exhausted(
            entry,
            None,
            {
                "reason": CREDENTIAL_PERSIST_FAILED_REASON,
                "message": f"rotated credential was not durably written to {store}: {exc}",
            },
        )
        return None

    def _single_use_refresh_lock_timeout(self) -> float:
        """Configured refresh POST timeout plus margin, so a slow token endpoint cannot starve the flock."""
        from auth.credential_pool import _REFRESH_TIMEOUT_ENV_VARS, auth_storage

        env_var = _REFRESH_TIMEOUT_ENV_VARS.get(
            self.provider, "HERMES_ANTHROPIC_REFRESH_TIMEOUT_SECONDS"
        )
        from utils import env_float

        refresh_timeout_seconds = env_float(env_var, 20)
        return max(
            float(auth_storage.AUTH_LOCK_TIMEOUT_SECONDS),
            float(refresh_timeout_seconds) + 5.0,
        )

    def _commit_anthropic_rotation(
        self, entry: PooledCredential, refreshed: Dict[str, Any]
    ) -> None:
        """Write a rotated Anthropic pair to its authoritative singleton, or fail closed.

        claude_code -> ~/.claude/.credentials.json (so the fallback resolver
        and other profiles see it). hermes_pkce -> ~/.hermes/.anthropic_oauth.json
        (``_seed_from_singletons`` re-seeds it every load; a borrowed row commits
        to the ROOT's file, never a new profile-local copy, #100339). Not
        ``endswith``: manual:hermes_pkce is pool-owned and a singleton for it
        would be a second authority for the same refresh-token family.
        """
        from auth.credential_pool import _RefreshDone, _singleton_target_for_entry

        if entry.source == "claude_code":
            store = "~/.claude/.credentials.json"
        elif entry.source == "hermes_pkce":
            store = "~/.hermes/.anthropic_oauth.json"
        else:
            return
        try:
            args = (
                refreshed["access_token"],
                refreshed["refresh_token"],
                refreshed["expires_at_ms"],
            )
            if entry.source == "claude_code":
                _auth_auth_providers_anthropic._write_claude_code_credentials(
                    *args, spent_refresh_token=entry.refresh_token or ""
                )
            else:
                _auth_auth_providers_anthropic._write_hermes_oauth_credentials(
                    *args, target=_singleton_target_for_entry(self, entry)
                )
        except Exception as wexc:
            # Authoritative commit failed: do not mark, persist or return the
            # rotation as successful, and bypass the re-POST recovery path —
            # there is nothing left to retry with.
            raise _RefreshDone(
                self._fail_closed_unpersisted_rotation(entry, wexc, store=store)
            )

    def _refresh_anthropic(self, entry: PooledCredential) -> PooledCredential:
        """POST the Anthropic refresh, commit to the singleton, return the rotated (unpersisted) entry."""
        from auth.credential_pool import _RefreshDone, replace
        from auth.providers.anthropic import (
            is_rotation_consumed_uncommitted,
            refresh_anthropic_oauth_pure,
            spent_rotation_source_path,
        )

        # Never POST a refresh token another process already spent: the
        # durable sidecar verdict is what a fresh interpreter sees here.
        source_path = spent_rotation_source_path(entry.source)
        if is_rotation_consumed_uncommitted(
            entry.refresh_token, source_path=source_path
        ) or (
            is_rotation_consumed_uncommitted(
                entry.access_token, source_path=source_path
            )
        ):
            raise _RefreshDone(
                self._fail_closed_unpersisted_rotation(
                    entry,
                    RuntimeError(
                        "credential pair was rotated by another process but the "
                        "rotation never committed (spent-rotation sidecar verdict)"
                    ),
                    store=str(source_path or "credential store"),
                )
            )
        refreshed = refresh_anthropic_oauth_pure(
            entry.refresh_token, use_json=entry.source.endswith("hermes_pkce")
        )
        updated = replace(
            entry,
            access_token=refreshed["access_token"],
            refresh_token=refreshed["refresh_token"],
            expires_at_ms=refreshed["expires_at_ms"],
        )
        self._commit_anthropic_rotation(entry, refreshed)
        return updated

    def _post_tokens_refresh(self, entry: PooledCredential) -> PooledCredential:
        """Codex / xAI: POST the refresh and return the rotated (unpersisted) entry."""
        from auth.credential_pool import _TOKENS_SINGLETON_PROVIDERS, replace

        refresh_fn_name = _TOKENS_SINGLETON_PROVIDERS[self.provider][2]
        refreshed = self._provider_hooks.refresh_tokens(
            entry.access_token, entry.refresh_token
        )
        return replace(
            entry,
            access_token=refreshed["access_token"],
            refresh_token=refreshed["refresh_token"],
            last_refresh=refreshed.get("last_refresh"),
        )

    def _refresh_entry_impl(
        self, entry: PooledCredential, *, force: bool
    ) -> Optional[PooledCredential]:
        # Single-use-token providers adopt fresher tokens from their store
        # BEFORE spending the refresh_token; ``entry`` is rebound to the synced
        # row so the failure path below recovers against the pair we POSTed.
        from auth.credential_pool import (
            _MARK_OK,
            _RefreshDone,
            _TOKENS_SINGLETON_PROVIDERS,
            apply_plugin_refresh_result,
            logger,
            plugin_refresh_hook,
            replace,
        )

        try:
            if self.provider == "anthropic":
                updated = self._refresh_anthropic(entry)
            elif self.provider in _TOKENS_SINGLETON_PROVIDERS:
                entry = self._sync_entry_from_auth_store(entry)
                updated = self._post_tokens_refresh(entry)
            elif (plugin_refresh := plugin_refresh_hook(self.provider)) is not None:
                rotated = plugin_refresh(entry)
                if not rotated:
                    # ``None``/empty = the plugin could not rotate: bench like a failed refresh POST, never
                    # report the stale row as refreshed (the loop would replay the dead bearer).
                    raise RuntimeError(
                        "provider refresh_credential returned no rotated fields"
                    )
                updated = apply_plugin_refresh_result(entry, rotated)
            elif self.provider == "nous":
                stale_key = (
                    entry.runtime_api_key or entry.agent_key or entry.access_token
                )
                synced = self._sync_nous_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    # A peer already rotated and persisted a usable key: adopt
                    # it without consuming the single-use refresh token again.
                    if (
                        force
                        and entry.runtime_api_key
                        and entry.runtime_api_key != stale_key
                    ):
                        logger.debug(
                            "Nous entry %s: adopting peer-rotated token, skipping refresh",
                            entry.id,
                        )
                        return entry
                self._provider_hooks.resolve_credentials(
                    force_refresh=force, stale_access_token=stale_key or None
                )
                updated = self._sync_nous_entry_from_auth_store(entry)
            else:
                return entry
        except _RefreshDone as done:
            return done.result
        except Exception as exc:
            logger.debug(
                "Credential refresh failed for %s/%s: %s", self.provider, entry.id, exc
            )
            return self._recover_failed_refresh(entry, exc)

        updated = replace(updated, **_MARK_OK)
        self._replace_entry(entry, updated)
        # Declare the cleared id: a borrowed row carries no access_token on disk, so
        # the merge's token-change bypass cannot apply and a plain persist would copy
        # the still-binding cooldown back over this success.
        self._persist(status_cleared_ids=[updated.id])
        # Sync back so _seed_from_singletons() on the next load_pool() sees
        # fresh state instead of re-seeding consumed tokens.
        self._sync_device_code_entry_to_auth_store(updated)
        return updated

    def _recover_failed_refresh(
        self, entry: PooledCredential, exc: Exception
    ) -> Optional[PooledCredential]:
        """After a failed refresh POST: adopt a peer's rotation, quarantine a dead grant, or bench.

        Another process may have consumed the refresh token between our
        pre-POST sync and the HTTP call; re-read the provider's token
        authority once more and adopt fresher tokens before giving up.
        """
        from auth.credential_pool import (
            STATUS_OK,
            _MARK_OK,
            _RefreshDone,
            _TOKENS_SINGLETON_PROVIDERS,
            logger,
            plugin_refresh_hook,
            recover_failed_plugin_refresh,
        )

        if self.provider == "anthropic":
            if entry.source == "claude_code":
                synced = self._sync_anthropic_entry_from_credentials_file(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "Retrying refresh with synced token from credentials file"
                    )
                    try:
                        from auth.providers.anthropic import (
                            refresh_anthropic_oauth_pure,
                        )

                        refreshed = refresh_anthropic_oauth_pure(
                            synced.refresh_token,
                            use_json=synced.source.endswith("hermes_pkce"),
                        )
                        # Commit to the authoritative singleton BEFORE marking or
                        # persisting the pool row, or a failed write leaves an
                        # "ok" row that the next load_pool() re-seeds over.
                        self._commit_anthropic_rotation(synced, refreshed)
                        return self._adopt(
                            synced,
                            access_token=refreshed["access_token"],
                            refresh_token=refreshed["refresh_token"],
                            expires_at_ms=refreshed["expires_at_ms"],
                            last_status=STATUS_OK,
                            last_status_at=None,
                            last_error_code=None,
                        )
                    except _RefreshDone as done:
                        return done.result
                    except Exception as retry_exc:
                        logger.debug("Retry refresh also failed: %s", retry_exc)
                elif not self._entry_needs_refresh(synced):
                    logger.debug(
                        "Credentials file has valid token, using without refresh"
                    )
                    return synced
            else:
                # Backstop for pool-owned sources (hermes_pkce, manual:dashboard_pkce):
                # the winner may have persisted between our pre-check and our POST.
                synced = self._sync_entry_from_pool_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "Anthropic OAuth refresh failed but pool store has newer tokens — adopting"
                    )
                    return self._adopt(synced, **_MARK_OK)
            from auth.providers.anthropic import is_terminal_anthropic_refresh_error

            if is_terminal_anthropic_refresh_error(exc):
                # A dead grant is not "exhausted": benching it for a TTL replays the dead token every
                # hour at DEBUG, so the lost login left no trace (#113023). Never touch the external
                # CLI's credentials file here — only Hermes' own row goes DEAD.
                logger.warning(
                    "Anthropic OAuth refresh token for %s is terminally invalid (%s); the credential "
                    "leaves rotation. Re-run 'hermes auth add anthropic' to sign in again.",
                    entry.label or entry.id[:8],
                    exc,
                )
                self._mark_dead_refresh_grant(entry, exc)
                return None
        elif self.provider in _TOKENS_SINGLETON_PROVIDERS:
            _, display, _, terminal_fn_name = _TOKENS_SINGLETON_PROVIDERS[self.provider]
            synced = self._sync_entry_from_auth_store(entry)
            if synced.refresh_token != entry.refresh_token:
                logger.debug(
                    "%s OAuth refresh failed but auth.json has newer tokens — adopting",
                    display,
                )
                return self._adopt(synced, **_MARK_OK)
            # Terminal error with no newer tokens: the stored refresh_token is
            # dead. Clear it from auth.json so the next session does not
            # re-seed the revoked credentials, and drop singleton-seeded
            # entries from the pool (mirrors the Nous quarantine path).
            if self._provider_hooks.terminal_error(exc):
                # WARNING, not debug: this is the moment a login is lost. At the default log level a
                # silent quarantine looked like "I logged in once and Hermes keeps failing" (#113023).
                logger.warning(
                    "%s OAuth refresh token is terminally invalid (%s); clearing local token state. "
                    "Re-run 'hermes auth add %s' to sign in again.",
                    display,
                    exc,
                    self.provider,
                )
                self._clear_terminal_tokens_state(entry, exc)
                self._quarantine_sources(entry, {"device_code"})
                self._mark_dead_refresh_grant(entry, exc)
                return None
        elif self.provider == "nous":
            synced = self._sync_nous_entry_from_auth_store(entry)
            if synced.refresh_token != entry.refresh_token:
                logger.debug(
                    "Nous refresh failed but auth.json has newer tokens — adopting"
                )
                updated = self._adopt(synced, **_MARK_OK)
                self._sync_device_code_entry_to_auth_store(updated)
                return updated
            if isinstance(exc, TimeoutError):
                # Lost the auth-store lock race under heavy fan-out. That says
                # nothing about the credential — benching it here emptied the
                # pool for ~120 sessions ("matched no nous entry ... pool size
                # 0"). The caller's retry re-syncs once the winner persisted.
                logger.debug(
                    "Nous refresh skipped: auth store lock busy; not benching entry"
                )
                return entry
            if self._provider_hooks.terminal_error(exc):
                logger.warning(
                    "Nous refresh token is terminally invalid (%s); clearing local token state. "
                    "Re-run 'hermes auth add nous' to sign in again.",
                    exc,
                )
                self._clear_terminal_nous_state(entry, exc)
                self._quarantine_sources(
                    entry,
                    {"device_code", "manual:device_code"},
                )
                self._mark_dead_refresh_grant(entry, exc)
                return None
        elif plugin_refresh_hook(self.provider) is not None:
            handled, result = recover_failed_plugin_refresh(self, entry, exc)
            if handled:
                return result
        self._mark_exhausted(entry, None)
        return None

    def _mark_dead_refresh_grant(self, entry: PooledCredential, exc: Exception) -> None:
        """Mark a row whose refresh token was terminally rejected DEAD, if the quarantine kept it.

        ``_quarantine_sources`` drops only singleton-seeded rows; an independent ``manual:*`` login
        (``hermes auth add``) survives, and an unmarked survivor re-enters rotation and re-fires the
        terminal WARNING on every later refresh attempt. DEAD leaves rotation until a write-side
        re-auth sync clears it (never via TTL).
        """
        from auth.credential_pool import STATUS_DEAD, time

        with self._lock:
            current = next(
                (item for item in self._entries if item.id == entry.id), None
            )
            if current is None or current.last_status == STATUS_DEAD:
                return
            self._adopt(
                current,
                last_status=STATUS_DEAD,
                last_status_at=time.time(),
                last_error_code=None,
                last_error_reason=str(getattr(exc, "code", None) or "invalid_grant"),
                last_error_message=str(exc),
            )

    def _clear_terminal_tokens_state(
        self, entry: PooledCredential, exc: Exception
    ) -> None:
        """Drop the dead Codex/xAI token pair from auth.json unless a peer already rotated it."""
        from auth.credential_pool import (
            _TOKENS_SINGLETON_PROVIDERS,
            _auth_store_lock,
            _load_auth_store,
            _load_provider_state,
            _save_auth_store,
            _save_provider_state,
            datetime,
            logger,
            timezone,
        )

        display = _TOKENS_SINGLETON_PROVIDERS[self.provider][1]
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, self.provider) or {}
                tokens = (
                    (state.get("tokens") or {}) if isinstance(state, dict) else None
                )
                if isinstance(tokens, dict):
                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                    if (
                        not store_refresh
                        or store_refresh == str(entry.refresh_token or "").strip()
                    ):
                        tokens.pop("access_token", None)
                        tokens.pop("refresh_token", None)
                        state["tokens"] = tokens
                        state["last_auth_error"] = {
                            "provider": self.provider,
                            "code": getattr(exc, "code", "unknown"),
                            "message": str(exc),
                            "reason": "credential_pool_refresh_failure",
                            "relogin_required": True,
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                        _save_provider_state(auth_store, self.provider, state)
                        _save_auth_store(auth_store)
        except Exception as clear_exc:
            logger.debug(
                "Failed to clear terminal %s OAuth state: %s", display, clear_exc
            )

    def _clear_terminal_nous_state(
        self, entry: PooledCredential, exc: Exception
    ) -> None:
        from auth.credential_pool import (
            _auth_store_lock,
            _load_auth_store,
            _load_provider_state,
            _save_auth_store,
            _save_provider_state,
            logger,
        )

        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "nous") or {
                    "client_id": entry.client_id,
                    "portal_base_url": entry.portal_base_url,
                    "inference_base_url": entry.inference_base_url,
                    "token_type": entry.token_type,
                    "scope": entry.scope,
                    "tls": entry.tls,
                }
                store_refresh = str(state.get("refresh_token") or "").strip()
                if (
                    not store_refresh
                    or store_refresh == str(entry.refresh_token or "").strip()
                ):
                    self._provider_hooks.quarantine_state(
                        state, exc, reason="credential_pool_refresh_failure"
                    )
                    self._provider_hooks.quarantine_pool(
                        auth_store, exc, reason="credential_pool_refresh_failure"
                    )
                    _save_provider_state(auth_store, "nous", state)
                    _save_auth_store(auth_store)
        except Exception as clear_exc:
            logger.debug("Failed to clear terminal Nous OAuth state: %s", clear_exc)

    def _codex_quota_restored_upstream(self, entry: PooledCredential) -> bool:
        """Live-check whether an exhausted Codex entry's quota reset early.

        A Codex 429 persists a ``last_error_reset_at`` that can be days out
        (weekly windows), but the window can reopen before then (redeemed
        reset, plan upgrade, OpenAI reset) — issue #43747. Only fires for
        429/quota-shaped errors; the probe is throttled per token (5 min) so
        it is safe on the hot selection path.
        """
        from auth.credential_pool import STATUS_EXHAUSTED, logger

        if self.provider != "openai-codex" or entry.last_status != STATUS_EXHAUSTED:
            return False
        if not self._provider_hooks.quota_shaped(
            entry.last_error_code,
            entry.last_error_reason,
            entry.last_error_message,
        ):
            return False
        token = entry.access_token or ""
        if not token:
            return False
        try:
            # An exhausted entry is skipped by the refresh chain, so its stored token is usually
            # expired by probe time (401 -> None -> cooldown kept, #89415): refresh it first.
            fresh = self._provider_hooks.refresh_probe_token(token, entry.refresh_token)
            if fresh:
                # Persist the rotated pair on both sides the way ``_refresh_entry`` does:
                # ``last_refresh`` plus the singleton write-back, or the next selection's
                # auth-store sync re-adopts the consumed pair from ``providers.openai-codex``
                # and clears the cooldown with it.
                entry = self._adopt(
                    entry,
                    access_token=fresh["access_token"],
                    refresh_token=fresh["refresh_token"],
                    last_refresh=fresh.get("last_refresh") or entry.last_refresh,
                )
                self._sync_device_code_entry_to_auth_store(entry)
                token = entry.access_token or token
            # The row keeps the canonical URL; a gateway key belongs to its route host (#121486).
            return bool(
                self._provider_hooks.quota_probe(
                    token, base_url=self._provider_hooks.route_base_url(entry.base_url)
                )
            )
        except Exception:
            logger.debug("Codex quota-restored probe failed", exc_info=True)
            return False

    def _entry_needs_refresh(self, entry: PooledCredential) -> bool:
        from auth.credential_pool import AUTH_TYPE_OAUTH, time

        if entry.auth_type != AUTH_TYPE_OAUTH:
            return False
        if self.provider == "anthropic":
            if entry.expires_at_ms is None:
                return False
            return int(entry.expires_at_ms) <= int(time.time() * 1000) + 120_000
        if self.provider == "openai-codex":
            return self._provider_hooks.token_expiring(entry.access_token)
        if self.provider == "xai-oauth":
            return self._provider_hooks.token_expiring(entry.access_token)
        # Nous refresh can require network access and happens when runtime
        # credentials are actually resolved, not on enumeration/selection.
        return False
