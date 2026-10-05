"""Multi-profile root write-through for single-use OAuth pools.

A named profile with no rows of its own for a single-use-refresh provider
(``SINGLE_USE_REFRESH_POOL_PROVIDERS``) reads the global root's rows through the
``read_credential_pool`` fallback ("borrowing", #100339). Everything that must then
land back in ROOT rather than materialise a profile-local fork lives here: the pytest
seat belt on the root path, ownership checks, the UPDATE-only root row merge, and the
rotated provider-state write-through. Sibling of ``agent/credential_pool.py``; it
late-imports the facade only where a symbol would otherwise be a module-level cycle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import hermes_cli.auth as auth_mod
from hermes_cli.auth import (
    SINGLE_USE_REFRESH_POOL_PROVIDERS,
    _auth_store_lock,
    _global_auth_file_path,
    _load_auth_store,
    _save_auth_store,
    write_credential_pool,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger(__name__)


def _guarded_global_root(global_path: Optional[Path]) -> Optional[Path]:
    """Apply the pytest seat belt to a resolved global-root auth.json path.

    ``None`` means classic mode (profile == root) or "refuse": under pytest,
    never write the real user's ``~/.hermes/auth.json`` even when HERMES_HOME
    points at a profile path (mirrors the read-side guard in
    ``_load_global_auth_store``). Uses the unmodified HOME env, not
    ``Path.home()`` which fixtures may monkeypatch.
    """
    if global_path is None:
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                # Comparing the guard path must not probe the real auth store.
                if os.path.normcase(os.path.abspath(global_path)) == os.path.normcase(os.path.abspath(real_root)):
                    return None
            except Exception:
                return None
    return global_path


def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> None:
    """Persist a rotated OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard: nous,
    openai-codex, and xai-oauth rotate the refresh_token on refresh, so when
    a profile pool refresh rotates a grant it resolved from the root fallback,
    the rotated chain must land back in root. Otherwise root keeps a revoked
    refresh token and every other profile dies with ``refresh_token_reused``
    / ``invalid_grant`` once its access token expires.

    Only updates ``providers.<provider_id>`` in the root store; never touches
    the profile store (the caller already saved that). Swallows all errors —
    a failed write-through degrades to root-stale and must never break the
    profile's own successful save. Mirrors
    ``hermes_cli.auth._write_through_xai_oauth_to_global_root``.

    See #48415.
    """
    try:
        global_path = _guarded_global_root(auth_mod._global_auth_file_path())
    except Exception:
        return
    if global_path is None:
        return
    try:
        auth_mod._persist_provider_state_to_store(provider_id, state, global_path, set_active=False)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("%s pool refresh: write-through to global root failed: %s", provider_id, exc)


def _singleton_target_for_entry(pool: "CredentialPool", entry: "PooledCredential") -> Optional[Path]:
    """Root ``.anthropic_oauth.json`` when *entry* is a borrowed hermes_pkce row, else None."""
    if entry.source != "hermes_pkce" or entry.id not in getattr(pool, "_borrowed_root_ids", ()):
        return None
    try:
        from agent.anthropic_credentials import _root_hermes_oauth_file
        return _root_hermes_oauth_file()
    except Exception:
        return None


def _store_owns_pool_provider(auth_store: Dict[str, Any], provider: str) -> bool:
    """True when an already-loaded *auth_store* has its own rows for *provider*."""
    pool = auth_store.get("credential_pool")
    entries = pool.get(provider) if isinstance(pool, dict) else None
    return isinstance(entries, list) and bool(entries)


def _profile_owns_pool_provider(provider: str) -> bool:
    """True when the ACTIVE auth.json has its own rows for *provider*.

    Named profiles with no local rows read the provider through the
    ``read_credential_pool`` global-root fallback ("borrowing").
    """
    # Classic mode (profile == root) has no root fallback, so the answer is always "owns";
    # skip the per-call auth.json re-read on this hot load_pool path.
    if auth_mod._global_auth_file_path() is None:
        return True
    try:
        auth_store = _load_auth_store()
    except Exception:
        return True  # unreadable store: assume ownership, keep legacy path
    return _store_owns_pool_provider(auth_store, provider)


def _borrowed_single_use_pool_root() -> Optional[Path]:
    """Global-root auth.json when persisting a BORROWED single-use pool, else None.

    ``None`` means "persist to the active store as usual": classic mode
    (profile == root), or the profile owns its own rows for this provider.
    """
    try:
        return _guarded_global_root(_global_auth_file_path())
    except Exception:
        return None


def _update_root_pool_rows(
    provider: str, payloads: List[Dict[str, Any]], global_path: Path,
    *, status_cleared_ids: Optional[Iterable[str]] = None,
    token_bases: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> List[Dict[str, Any]]:
    """UPDATE-ONLY merge of *payloads* into the root store's rows for *provider*.

    A borrower may refresh the root's rows (rotation, cooldown state) but
    never add or delete them — the root owns their lifecycle. In particular a
    profile's singleton-prune (it has no ``.anthropic_oauth.json`` of its own)
    must not delete the root grant, so ``removed_ids`` is ignored by callers.
    """
    with _auth_store_lock(target_path=global_path):
        store = _load_auth_store(global_path)
        pool = store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            store["credential_pool"] = pool
        existing = pool.get(provider)
        existing_list = existing if isinstance(existing, list) else []
        incoming_by_id = auth_mod._entry_ids(payloads)
        cleared = {cid for cid in (status_cleared_ids or ()) if cid}
        bases = token_bases or {}
        merged: List[Dict[str, Any]] = []
        changed = False
        for disk_entry in existing_list:
            did = disk_entry.get("id") if isinstance(disk_entry, dict) else None
            incoming = incoming_by_id.get(did) if did else None
            if incoming is None:
                merged.append(disk_entry)
                continue
            updated = auth_mod._merge_pool_row_generation(
                incoming, disk_entry, provider,
                base_pair=bases.get(did), status_cleared=did in cleared,
            )
            if updated != disk_entry:
                changed = True
            merged.append(updated)
        if changed:
            pool[provider] = merged
            _save_auth_store(store, target_path=global_path)
        return merged


def persist_pool_entries(
    provider: str,
    payloads: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
    status_cleared_ids: Optional[Iterable[str]] = None,
    token_bases: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Persist a provider's pool rows to the store that OWNS them.

    A named profile that sees a single-use-refresh provider (see
    ``SINGLE_USE_REFRESH_POOL_PROVIDERS``) only through the global-root fallback must not
    materialize a local ``credential_pool.<provider>`` copy: that copy forks
    the single-use refresh token, the first profile to rotate commits the new
    pair only to its own file, and root plus every sibling die with
    ``invalid_grant`` (#100339). Such rows are written back to the root store
    (under the root lock); everything else goes to the active store.
    """
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS and not _profile_owns_pool_provider(provider):
        global_path = _borrowed_single_use_pool_root()
        if global_path is not None:
            try:
                return _update_root_pool_rows(
                    provider, payloads, global_path,
                    status_cleared_ids=status_cleared_ids, token_bases=token_bases,
                )
            except Exception as exc:
                # Fail closed on the FORK, not on the save: never fall back to
                # writing a local copy (that IS the bug). The in-memory pool
                # still holds the rotated pair for this process.
                logger.warning(
                    "%s pool: write-through of borrowed root grant failed (%s); "
                    "not materializing a profile-local copy",
                    provider, exc,
                )
            return None
    return write_credential_pool(
        provider, payloads, removed_ids=removed_ids, status_cleared_ids=status_cleared_ids,
        token_bases=token_bases,
    )
