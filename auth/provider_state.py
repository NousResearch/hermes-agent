"""Provider-state transactions and writes to the store owning the grant."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from auth import store


def _provider_state_in(
    store: Dict[str, Any], provider_id: str
) -> Optional[Dict[str, Any]]:
    """Shallow copy of ``store["providers"][provider_id]`` when it is a dict, else None."""
    providers = store.get("providers") if store else None
    state = providers.get(provider_id) if isinstance(providers, dict) else None
    return dict(state) if isinstance(state, dict) else None


def _load_provider_state_with_source(
    auth_store: Dict[str, Any],
    provider_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Provider state plus the auth.json path it came from (profile first, then the global root).

    Refresh paths that rotate single-use OAuth refresh tokens must write the updated chain back to
    the same store they read."""
    state = _provider_state_in(auth_store, provider_id)
    if state is not None:
        return state, store._auth_file_path()
    global_state = _provider_state_in(store._load_global_auth_store(), provider_id)
    return (
        (global_state, store._global_auth_file_path())
        if global_state is not None
        else (None, None)
    )


def _load_provider_state(
    auth_store: Dict[str, Any], provider_id: str
) -> Optional[Dict[str, Any]]:
    """Provider state; in profile mode falls back to the global-root ``auth.json`` per provider (same
    shadowing as ``read_credential_pool``), so profile workers see globally-authed providers."""
    return _load_provider_state_with_source(auth_store, provider_id)[0]


@contextmanager
def _provider_state_transaction(
    provider_id: str, timeout_seconds: float = store.AUTH_LOCK_TIMEOUT_SECONDS
):
    """Lock the active auth store and any global fallback source, in that order.

    Re-reading the source after its lock is acquired prevents stale refreshes and whole-file lost
    updates without inverting the documented auth -> shared lock order. ``timeout_seconds`` applies
    to BOTH locks: a transaction that spans a network call must let waiters outlive that call."""
    with store._auth_store_lock(timeout_seconds):
        auth_store = store._load_auth_store()
        state, source_path = _load_provider_state_with_source(auth_store, provider_id)
        if source_path is None or store._same_path(
            source_path, store._auth_file_path()
        ):
            yield auth_store, state, source_path
            return
        with store._auth_store_lock(timeout_seconds, target_path=source_path):
            yield (
                auth_store,
                _provider_state_in(store._load_auth_store(source_path), provider_id),
                source_path,
            )


def _store_provider_state(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    *,
    set_active: bool = True,
) -> None:
    store._store_section(auth_store, "providers")[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id


def _save_provider_state(
    auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]
) -> None:
    """Write *state* under ``providers`` and make *provider_id* the active provider."""
    _store_provider_state(auth_store, provider_id, state, set_active=True)


def _save_active_provider_state(provider_id: str, state: Dict[str, Any]) -> Path:
    """Lock, load, write *state* as the active provider, save. Returns the auth store path."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        _save_provider_state(auth_store, provider_id, state)
        return store._save_auth_store(auth_store)


def _persist_provider_state_to_store(
    provider_id: str,
    state: Dict[str, Any],
    target_path: Path,
    *,
    set_active: bool = False,
) -> Path:
    """Merge one provider into a specific auth store under that store's lock."""
    with store._auth_store_lock(target_path=target_path):
        auth_store = store._load_auth_store(target_path)
        _store_provider_state(
            auth_store, provider_id, dict(state), set_active=set_active
        )
        return store._save_auth_store(auth_store, target_path=target_path)


def _save_provider_state_to_source(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    source_path: Optional[Path],
) -> None:
    """Persist provider state back to the auth store it was read from.

    A token refresh rewrites credentials, not the user's choice of provider: ``active_provider`` is
    left as it is (a Nous free-tier identity refreshed for a connector call must not become the
    inference provider of an install that has its own key)."""
    if source_path is None or store._same_path(source_path, store._auth_file_path()):
        _store_provider_state(auth_store, provider_id, state, set_active=False)
        store._save_auth_store(auth_store)
    else:
        _persist_provider_state_to_store(
            provider_id, state, source_path, set_active=False
        )


def mark_provider_active_if_unset(provider_id: str) -> None:
    """Set ``active_provider`` only when none is set yet: the first ``hermes auth add`` credential must
    make its provider active (else setup reports "No inference provider configured"); later adds
    leave the user's choice untouched."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        if not (auth_store.get("active_provider") or "").strip():
            auth_store["active_provider"] = provider_id
            store._save_auth_store(auth_store)


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Persisted auth state for a provider (profile first, global-root fallback), or None."""
    return _load_provider_state(store._load_auth_store(), provider_id)


def get_active_provider() -> Optional[str]:
    """Return the currently active provider ID from auth store."""
    return store._load_auth_store().get("active_provider")


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """Clear auth state for a provider (the active one when *provider_id* is None). Used by
    ``hermes logout``. Returns True if something was cleared."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False
        cleared = False
        for section in ("providers", "credential_pool"):
            entries = store._store_section(auth_store, section)
            if target in entries:
                del entries[target]
                cleared = True
        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True
        if cleared:
            store._save_auth_store(auth_store)
        return cleared


def deactivate_provider() -> None:
    """Clear active_provider without deleting credentials: used when the user switches to a non-OAuth
    provider (OpenRouter, custom) so auto-resolution doesn't keep picking the OAuth provider."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        auth_store["active_provider"] = None
        store._save_auth_store(auth_store)

def mark_provider_active(provider_id: str) -> None:
    """Select an already chosen provider without changing its credentials."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        auth_store["active_provider"] = provider_id
        store._save_auth_store(auth_store)


def save_provider_auth_state(
    provider_id: str, state: Dict[str, Any], *, set_active: bool = False
) -> Path:
    """Persist a login result without changing the inference route."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        _store_provider_state(auth_store, provider_id, state, set_active=set_active)
        return store._save_auth_store(auth_store)


def logout_provider_auth(provider_id: str, *, configured: bool = False) -> bool:
    """Clear a login and its shared adoption source before reporting logout."""
    if not (clear_provider_auth(provider_id) or configured):
        return False
    if provider_id == "nous":
        from auth.providers.nous_store import _clear_shared_nous_state
        _clear_shared_nous_state("logout")
    return True


def restore_active_provider(prior_active_provider: Any) -> None:
    """Undo the ``active_provider="nous"`` that ``_save_provider_state`` wrote during login."""
    from auth.store import _auth_store_lock, _load_auth_store, _save_auth_store
    with _auth_store_lock():
        auth_store = _load_auth_store()
        if prior_active_provider:
            auth_store["active_provider"] = prior_active_provider
        else:
            auth_store.pop("active_provider", None)
        _save_auth_store(auth_store)

def update_provider_auth_state(provider_id: str, changes: Dict[str, Any]) -> None:
    """Merge provider metadata under the store lock without changing the active route."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        state = _load_provider_state(auth_store, provider_id) or {}
        state.update(changes)
        _store_provider_state(auth_store, provider_id, state, set_active=False)
        store._save_auth_store(auth_store)
