"""Entry-owned xAI pool refreshes: bind one store before spending a rotating grant."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from hermes_cli import auth

if TYPE_CHECKING:
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger("agent.credential_pool")


@dataclass(frozen=True)
class _GrantOwner:
    path: Path
    persisted_row: bool


@contextmanager
def _entry_transaction(pool: CredentialPool, entry: PooledCredential) -> Iterator[_GrantOwner]:
    """Pool-row provenance outranks the provider singleton's (possibly unrelated) owner."""
    from hermes_cli.auth_xai import _load_xai_oauth_state_with_source

    timeout = pool._single_use_refresh_lock_timeout()
    with auth._auth_store_lock(timeout):
        active_path = auth._auth_file_path()
        store = auth._load_auth_store()
        if entry.id in pool._borrowed_root_ids:
            path = auth._global_auth_file_path()
            if path is None:
                raise RuntimeError("Borrowed xAI credential has no root owner")
        elif entry.id in auth._entry_ids(store.get("credential_pool", {}).get("xai-oauth", [])):
            path = active_path
        elif entry.source == "device_code":
            _, path = _load_xai_oauth_state_with_source(store)
            path = path or active_path
        else:
            path = active_path
        owner = _GrantOwner(path, entry.id in pool._persisted_token_pairs)
        # Profile -> root is the established lock order. Keep BOTH through POST and commit.
        with auth._auth_store_lock(timeout, target_path=path):
            yield owner


def _sync_entry(pool: CredentialPool, entry: PooledCredential, owner: _GrantOwner) -> PooledCredential | None:
    from agent.credential_pool import PooledCredential, _singleton_predates_entry

    store = auth._load_auth_store(owner.path)
    row = auth._entry_ids(store.get("credential_pool", {}).get("xai-oauth", [])).get(entry.id)
    if owner.persisted_row and row is None:
        # A peer quarantined or removed it while this pool waited for the owner's lock.
        pool._quarantine_sources(entry, set(), matching_tokens=True)
        return None
    updated = entry
    if row is not None:
        pool._persisted_token_pairs[entry.id] = auth._credential_token_pair(row)
        updated = PooledCredential.from_dict("xai-oauth", row)
    state = auth._provider_state_in(store, "xai-oauth") or {}
    tokens = state.get("tokens") or {}
    if entry.source == "device_code" and tokens.get("access_token") and not _singleton_predates_entry(state, updated):
        updated = replace(updated, access_token=tokens["access_token"],
                          refresh_token=tokens.get("refresh_token") or updated.refresh_token,
                          last_refresh=state.get("last_refresh") or updated.last_refresh)
    if (updated.access_token, updated.refresh_token) != (entry.access_token, entry.refresh_token):
        pool._replace_entry(entry, updated)
        return updated
    return entry


def _commit_rotation(
    pool: CredentialPool, entry: PooledCredential, updated: PooledCredential, owner: _GrantOwner,
) -> PooledCredential:
    from agent.credential_pool import _MARK_OK
    from agent.credential_pool_sync import apply_entry_to_singleton_state

    updated = replace(updated, **_MARK_OK)
    store = auth._load_auth_store(owner.path)
    rows = store.get("credential_pool", {}).get("xai-oauth", [])
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("id") == entry.id:
            rows[index] = updated.to_dict()
            break
    else:
        if entry.source != "device_code" and not owner.persisted_row:
            auth._store_section(store, "credential_pool").setdefault("xai-oauth", []).append(updated.to_dict())
    if entry.source == "device_code":
        state = auth._provider_state_in(store, "xai-oauth") or {"tokens": {}}
        state.setdefault("tokens", {})
        apply_entry_to_singleton_state("xai-oauth", updated, state)
        auth._store_provider_state(store, "xai-oauth", state, set_active=False)
    # Update only this grant in the captured owner. Never re-resolve a provider-wide fallback.
    auth._save_auth_store(store, target_path=owner.path)
    pool._replace_entry(entry, updated)
    if entry.id in auth._entry_ids(store.get("credential_pool", {}).get("xai-oauth", [])):
        pool._persisted_token_pairs[entry.id] = (updated.access_token, updated.refresh_token)
    return updated


def _recover_refresh(
    pool: CredentialPool, entry: PooledCredential, exc: Exception, owner: _GrantOwner,
) -> PooledCredential | None:
    from hermes_cli.auth_xai import _quarantine_xai_oauth_tokens

    synced = _sync_entry(pool, entry, owner)
    if synced is None:
        return None
    if synced.refresh_token != entry.refresh_token:
        return _commit_rotation(pool, entry, synced, owner)
    if auth._is_terminal_xai_oauth_refresh_error(exc):
        logger.warning(
            "xAI OAuth refresh token is terminally invalid (%s); clearing its owning store. "
            "Re-run 'hermes auth add xai-oauth' to sign in again.", exc,
        )
        store = auth._load_auth_store(owner.path)
        _quarantine_xai_oauth_tokens(
            exc, source=(store, auth._provider_state_in(store, "xai-oauth"), owner.path),
            failed_tokens={"access_token": entry.access_token, "refresh_token": entry.refresh_token},
            reason="credential_pool_refresh_failure",
        )
        # A source-wide device_code prune would also remove the independent profile singleton.
        pool._quarantine_sources(entry, set(), matching_tokens=True)
    else:
        pool._mark_exhausted(entry, None)
    return None


def sync_xai_pool_entry(pool: CredentialPool, entry: PooledCredential) -> PooledCredential:
    """Cooldown recovery must consult the row's owner, not the profile singleton."""
    from agent.credential_pool import STATUS_DEAD

    with _entry_transaction(pool, entry) as owner:
        synced = _sync_entry(pool, entry, owner)
        if synced is None:
            # Already removed from the live pool; do not select the loop's stale snapshot.
            return replace(entry, last_status=STATUS_DEAD)
        if synced is not entry:
            return _commit_rotation(pool, entry, synced, owner)
        return entry


def refresh_xai_pool_entry(pool: CredentialPool, entry: PooledCredential) -> PooledCredential | None:
    """Refresh or adopt this entry under its owner's lock, preserving other grants."""
    with _entry_transaction(pool, entry) as owner:
        synced = _sync_entry(pool, entry, owner)
        if synced is None:
            return None
        if synced is not entry:
            return _commit_rotation(pool, entry, synced, owner)
        try:
            updated = pool._post_tokens_refresh(entry)
        except Exception as exc:
            logger.debug("xAI pool refresh failed for %s", entry.id, exc_info=True)
            return _recover_refresh(pool, entry, exc, owner)
        return _commit_rotation(pool, entry, updated, owner)
