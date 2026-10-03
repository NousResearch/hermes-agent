"""MiniMax OAuth credential-pool refresh against its authoritative auth store."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional

import hermes_cli.auth as auth

logger = logging.getLogger(__name__)

_REFRESH_TIMEOUT_SECONDS = 15.0


def prune_removed_borrowed_source(entry_ids: set[str]) -> bool:
    """Delete root pool references only after both stores confirm the source is gone."""
    if not entry_ids:
        return False
    with auth._auth_store_lock():
        active_store = auth._load_auth_store()
        active_state = (active_store.get("providers") or {}).get("minimax-oauth")
        if isinstance(active_state, dict) and active_state.get("access_token"):
            return False
        root_path = auth._global_auth_file_path()
        if root_path is None or auth._same_path(root_path, auth._auth_file_path()):
            return False
        with auth._auth_store_lock(target_path=root_path):
            root_store = auth._load_auth_store(root_path)
            root_state = (root_store.get("providers") or {}).get("minimax-oauth")
            if isinstance(root_state, dict) and root_state.get("access_token"):
                return False
            pool = root_store.get("credential_pool")
            if not isinstance(pool, dict):
                return False
            rows = pool.get("minimax-oauth")
            if not isinstance(rows, list):
                return False
            kept = [
                row
                for row in rows
                if not (
                    isinstance(row, dict)
                    and row.get("id") in entry_ids
                    and row.get("source") == "oauth"
                )
            ]
            if len(kept) == len(rows):
                return False
            if kept:
                pool["minimax-oauth"] = kept
            else:
                pool.pop("minimax-oauth", None)
            auth._save_auth_store(root_store, target_path=root_path)
            return True


def _expiry_ms(state: Dict[str, Any]) -> Optional[int]:
    try:
        raw = state.get("expires_at")
        return int(datetime.fromisoformat(raw).timestamp() * 1000) if raw else None
    except Exception:
        return None


def entry_needs_refresh(entry: Any) -> bool:
    """Whether a MiniMax pool row is inside the proactive refresh skew."""
    try:
        expires_at = datetime.fromisoformat(entry.expires_at or "").timestamp()
    except Exception:
        return True
    return (expires_at - time.time()) <= auth.MINIMAX_OAUTH_REFRESH_SKEW_SECONDS


def _entry_from_state(pool: Any, entry: Any, state: Dict[str, Any]) -> Any:
    """Adopt authoritative flat MiniMax state without persisting it yet."""
    extra = dict(entry.extra)
    for key in ("client_id", "portal_base_url", "obtained_at", "expires_in"):
        if state.get(key) is not None:
            extra[key] = state[key]
    updates = {
        "access_token": state.get("access_token") or "",
        "refresh_token": state.get("refresh_token") or None,
        "expires_at": state.get("expires_at"),
        "expires_at_ms": _expiry_ms(state),
        "base_url": str(state.get("inference_base_url") or entry.base_url or "").rstrip(
            "/"
        ),
        "extra": extra,
        "last_status": None,
        "last_status_at": None,
        "last_error_code": None,
        "last_error_reason": None,
        "last_error_message": None,
        "last_error_reset_at": None,
    }
    updated = replace(entry, **updates)
    if updated != entry:
        pool._replace_entry(entry, updated)
    return updated


def sync_entry_from_auth_store(pool: Any, entry: Any) -> Any:
    """Adopt a peer rotation for an exhausted live pool instance."""
    if pool.provider != "minimax-oauth" or entry.source != "oauth":
        return entry
    try:
        with auth._provider_state_transaction("minimax-oauth") as (
            _auth_store,
            state,
            _source_path,
        ):
            if not isinstance(state, dict) or not state.get("access_token"):
                return entry
            return _entry_from_state(pool, entry, state)
    except Exception:
        logger.debug(
            "Failed to sync MiniMax OAuth entry from auth store", exc_info=True
        )
        return entry


def _quarantine_source(
    pool: Any,
    entry: Any,
    auth_store: Dict[str, Any],
    state: Dict[str, Any],
    source_path: Any,
    exc: Any,
) -> None:
    quarantined = dict(state)
    auth._minimax_oauth_quarantine_on_terminal_refresh(quarantined, exc, persist=False)
    try:
        auth._save_provider_state_to_source(
            auth_store, "minimax-oauth", quarantined, source_path
        )
    except Exception:
        logger.error(
            "MiniMax OAuth terminal quarantine could not be persisted",
            exc_info=True,
        )
    borrowed_ids = {
        str(item.id) for item in pool._entries if item.source == "oauth" and item.id
    }
    pool._quarantine_sources(entry, {"oauth"})
    try:
        prune_removed_borrowed_source(borrowed_ids)
    except Exception:
        logger.error(
            "MiniMax OAuth quarantined state but could not remove stale root pool rows",
            exc_info=True,
        )


def _fail_closed_after_write_error(
    pool: Any,
    entry: Any,
    auth_store: Dict[str, Any],
    state: Dict[str, Any],
    source_path: Any,
    save_exc: Exception,
) -> None:
    """Never expose a rotated pair whose authoritative write did not commit."""
    persist_error = auth.AuthError(
        "MiniMax OAuth rotated its refresh token but the replacement could not "
        "be persisted; please re-login.",
        provider="minimax-oauth",
        code="credential_persist_failed",
        relogin_required=True,
    )
    logger.error(
        "MiniMax OAuth refresh committed upstream but not to %s (%s); "
        "quarantining the grant",
        source_path,
        save_exc,
    )
    _quarantine_source(pool, entry, auth_store, state, source_path, persist_error)


def refresh_entry(pool: Any, entry: Any, *, force: bool) -> Any:
    """Serialize source re-read, refresh POST, and write-back across profiles."""
    lock_timeout = max(
        float(auth.AUTH_LOCK_TIMEOUT_SECONDS), _REFRESH_TIMEOUT_SECONDS + 5.0
    )
    try:
        with auth._provider_state_transaction(
            "minimax-oauth", timeout_seconds=lock_timeout
        ) as (auth_store, source_state, source_path):
            if not (
                isinstance(source_state, dict)
                and source_state.get("access_token")
                and source_state.get("refresh_token")
            ):
                logger.info(
                    "MiniMax OAuth source disappeared while waiting for its lock; "
                    "dropping the stale pool row"
                )
                stale_ids = {
                    str(item.id)
                    for item in pool._entries
                    if item.source == "oauth" and item.id
                }
                pool._quarantine_sources(entry, {"oauth"})
                try:
                    prune_removed_borrowed_source(stale_ids)
                except Exception:
                    logger.warning(
                        "MiniMax OAuth source vanished but stale root rows could not be removed",
                        exc_info=True,
                    )
                return None

            stale_pair = (entry.access_token, entry.refresh_token)
            synced = _entry_from_state(pool, entry, source_state)
            authoritative_pair = (synced.access_token, synced.refresh_token)
            if authoritative_pair != stale_pair:
                # A lock winner already rotated this grant. Persist/adopt that
                # generation and never replay the consumed refresh token.
                pool._persist(status_cleared_ids=[synced.id])
                return pool._find(lambda item: item.id == synced.id) or synced
            if not force and not entry_needs_refresh(synced):
                return synced

            try:
                refreshed = auth.refresh_minimax_oauth_pure(
                    source_state, timeout_seconds=_REFRESH_TIMEOUT_SECONDS
                )
            except Exception as exc:
                if auth._is_terminal_minimax_oauth_refresh_error(exc):
                    _quarantine_source(
                        pool, synced, auth_store, source_state, source_path, exc
                    )
                    return None
                pool._mark_exhausted(synced, None)
                return None

            new_state = dict(source_state)
            new_state.update(refreshed)
            updated = _entry_from_state(pool, synced, new_state)
            try:
                # The singleton is authoritative. Commit it before the pool row
                # so a pool-write failure is healed from this state on reload.
                auth._save_provider_state_to_source(
                    auth_store, "minimax-oauth", new_state, source_path
                )
            except Exception as save_exc:
                # Put the in-memory row back on the pre-rotation generation; the
                # newly issued bearer must never escape without a durable owner.
                pool._replace_entry(updated, synced)
                _fail_closed_after_write_error(
                    pool, synced, auth_store, source_state, source_path, save_exc
                )
                return None

            pool._replace_entry(synced, updated)
            pool._persist(status_cleared_ids=[updated.id])
            return pool._find(lambda item: item.id == updated.id) or updated
    except TimeoutError:
        logger.debug("MiniMax OAuth refresh skipped: auth store lock busy")
        return entry
