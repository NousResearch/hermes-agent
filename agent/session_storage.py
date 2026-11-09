"""The session storage a scope's agent uses, selected by the ``sessions.storage`` config key.

:func:`session_storage` is the one resolver production callers use to reach the session storage
ports (``hermes_state_port``). ``sessions.storage: local`` (the default) is today's store: the
profile's own ``SessionDB`` the caller already holds, wrapped by the local backend, behaviour
unchanged. Any other name is a session storage provider plugin, resolved through
:func:`hermes_state_port.resolve_session_storage`, and it must prove it isolates tenants
(``assert_isolating``) before its ports are opened — a provider that raises is refused, never
replaced by the local store.

The backend name is read from the config of the store's own profile — the home that holds the
``state.db`` — never the ambient one, so under a multiplexed gateway a served profile's own
choice wins; with no local store (a hosted agent behind a provider) it is the bound profile's
config.

The local backend pins every operation to the store's own profile
(``hermes_state_port_local``). A caller that runs under another profile — a multiplexed owner
draining a served profile's FIFO with the launch profile ambient — gets the store's profile
bound around the operation, so the pin holds exactly where today's unpinned calls ran; a
caller that already runs as the store's profile binds nothing, and a hosted scope is left for
the store to refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

from hermes_state_port import SessionStorage, SessionStorageProvider, SessionStore, TurnAdmission

__all__ = ["session_storage"]

# Providers resolved and isolation-proven once per (name, plugin directory): a provider instance
# is stateless (its connections come from a shared pool, pinned per transaction), and two homes
# naming the same word may carry different plugin files, so the directory is part of the key.
_PROVIDERS: dict[tuple[str, str], SessionStorageProvider] = {}


def _storage_name(home: Path) -> str:
    """The ``sessions.storage`` backend *home*'s config names, "local" when it names none."""
    from hermes_cli.config_effective import load_user_config_effective
    sessions = load_user_config_effective(home / "config.yaml").get("sessions")
    name = sessions.get("storage") if isinstance(sessions, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else "local"


def _named_provider(name: str) -> SessionStorageProvider:
    """The provider *name* names, isolation-proven (``assert_isolating``) before first open; a
    provider that raises is refused here, never replaced by the local store."""
    from plugins.session_storage import find_provider_dir
    from hermes_state_port import resolve_session_storage
    directory = find_provider_dir(name)
    key = (name, str(directory) if directory is not None else "")
    provider = _PROVIDERS.get(key)
    if provider is None:
        provider = resolve_session_storage(name)
        provider.assert_isolating()
        _PROVIDERS[key] = provider
    return provider


def _pin_matches(home: Path) -> bool:
    """Whether the local store's pin passes as the caller runs (``hermes_state_port_local``): the
    bound scope, or with none bound the ambient profile, already names *home*."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    scope = current_scope()
    if scope is not None and not scope.is_local:
        return False  # a hosted pin is the store's to refuse; this never rebinds it
    agent = scope.agent if scope is not None else get_hermes_home()
    return hermes_home_key(agent) == hermes_home_key(home)


class _ProfilePinned:
    """A local port with the store's own profile bound around each operation when the caller runs
    under another profile (today these callers reached the store with no pin at all); the bind is
    the store's whole profile scope, so the pin sees the profile the store serves."""

    __slots__ = ("_home", "_port")

    def __init__(self, port: Any, home: Path) -> None:
        self._port, self._home = port, home

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._port, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            from hermes_scope import current_scope
            scope = current_scope()
            if scope is not None and not scope.is_local:
                return attr(*args, **kwargs)  # a hosted pin is the store's to refuse, never rebound
            if _pin_matches(self._home):
                return attr(*args, **kwargs)
            from agent.runtime_scope import bind_profile_scope
            with bind_profile_scope(self._home, hydrate_secrets=False):
                return attr(*args, **kwargs)

        return call


def session_storage(db: Any = None) -> Optional[SessionStorage]:
    """The session storage of the store *db* (the caller's ``SessionDB``), or of the bound profile
    when there is no local store: the profile's own ``state.db`` behind the ports when its config
    names ``local`` (the default), else the named provider's ports. ``None`` only for the local
    backend with no store — nothing to persist to, today's behaviour."""
    path = getattr(db, "db_path", None)
    home = Path(path).resolve().parent if path is not None else None
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    name = _storage_name(home)
    if name == "local":
        if db is None:
            return None
        from hermes_state_port_local import LocalSessionStorageProvider
        storage = LocalSessionStorageProvider().open(db)
        return SessionStorage(sessions=cast(SessionStore, _ProfilePinned(storage.sessions, home)),
                              admission=cast(TurnAdmission, _ProfilePinned(storage.admission, home)),
                              close=storage.close)
    return _named_provider(name).open()
