"""The local secret store backend: today's profile ``.env`` and ``auth.json`` behind the port.

:class:`LocalSecretStore` is one profile home's credentials. Every operation is the same call the
code makes today — reads through ``owner_secret_scope`` (a profile's own ``.env`` plus its external
sources and any managed values; the launch profile's environment-only credentials, the ones
``systemd`` or ``op run`` inject with no file to rebuild from, for the process's own home), writes
through ``save_env_value`` / ``remove_env_value`` and the ``hermes_cli.auth`` store — after one
check of the pin: the bound scope (``hermes_scope.current_scope()``) must be a local one whose
agent is this store's own profile home. A local install holds one agent's credentials only, so any
other pin (another tenant, another profile) is refused with :class:`SecretPinRefused` before
anything is read or written. With no scope bound, the pin is the ambient profile
(``get_hermes_home()``), as every local caller has it today, and the local person may administer.

The store holds the agent level only and says so (:attr:`~SecretStoreProvider.levels`): a write
naming the owner-wide or tenant-wide level is refused with :class:`SecretLevelRefused`, a clear
error, never a silent fallback onto the agent level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_scope import LOCAL_TENANT, current_scope
from hermes_secret_port import (
    LEVELS, SecretAdminRefused, SecretLevelRefused, SecretPinRefused, SecretStore, SecretStoreProvider,
    fresh_write_error,
)


class LocalSecretStore(SecretStore):
    """One profile home's secrets and auth records, pinned to that home.

    As the port's contract requires, a failed credential write holds no value on any error
    path — this backend never mutates a frame it does not own: the whole write runs in one
    inner helper frame that binds the value (or record), and on any failure (a refusal, a
    validation error, a mid-write I/O error from today's writer) the method deletes its own
    value binding and raises a fresh port error outside the except block with the chain slots
    cleared (:func:`hermes_secret_port.fresh_write_error`) — the error's traceback holds only
    this method's frame and the caller's, and every frame that held the plaintext is gone.
    """

    def __init__(self, home: str | Path) -> None:
        from hermes_constants import hermes_home_key
        self.home = Path(home)
        self._home_key = hermes_home_key(self.home)

    def _pin(self, *, administering: bool = False) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise SecretPinRefused(f"a local secret store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise SecretPinRefused("the bound agent is not this secret store's profile")
        if administering and scope is not None and not scope.can_administer:
            # No scope bound is the local person's own call, administering, as every local
            # writer works today; a bound scope states its right explicitly.
            raise SecretAdminRefused("this scope may not administer this agent's credentials")

    def _require_agent_level(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"not a secret level: {level!r}")
        if level != "agent":
            raise SecretLevelRefused("the local secret store holds the agent level only")

    def _resolved(self) -> dict[str, str]:
        """The profile's whole credential mapping, exactly as a bound turn's secret scope builds
        it: its own ``.env`` plus external sources and managed values, with the launch profile's
        frozen environment included when this home is the process's own."""
        from tui_gateway.launch_profile_policy import owner_secret_scope
        return owner_secret_scope(self.home)

    def get(self, name: str) -> Optional[str]:
        self._pin()
        return self._resolved().get(name)

    def names(self) -> list[str]:
        self._pin()
        return sorted(self._resolved())

    def set(self, name: str, value: str, *, level: str) -> None:
        failure = None
        try:
            self._write_secret(name, value, level)
        except BaseException as exc:  # health: allow BLE001 -- only the fresh error below survives
            failure = fresh_write_error(exc)
        del value
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    def _write_secret(self, name: str, value: str, level: str) -> None:
        """The whole write in one helper frame, so no frame binding the value survives a
        failure (see :meth:`set`): every refusal and validation condition — the pin, the
        level, today's write lock — checked before the value is passed to today's writer,
        then the write itself at today's path (``get_env_path()`` honors the profile-home
        override the pin just matched): the managed-scope write lock refuses by raising, the
        file gets the same serialized line every local writer produces, and the live process
        is published to exactly as it is today."""
        from hermes_cli.config import require_env_writable, save_env_value
        self._pin(administering=True)
        self._require_agent_level(level)
        require_env_writable(name, "set")
        save_env_value(name, value)

    def delete(self, name: str, *, level: str) -> bool:
        self._pin(administering=True)
        self._require_agent_level(level)
        from hermes_cli.config import remove_env_value, require_env_writable
        require_env_writable(name, "remove")
        return remove_env_value(name)

    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        self._pin()
        from hermes_cli import auth
        # Today's reader: the profile's own auth.json first, then the global root per provider.
        return auth._load_provider_state_with_source(auth._load_auth_store(), provider)[0]

    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        failure = None
        try:
            self._write_auth_record(provider, record)
        except BaseException as exc:  # health: allow BLE001 -- only the fresh error below survives
            failure = fresh_write_error(exc)
        del record
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    def _write_auth_record(self, provider: str, record: dict[str, Any]) -> None:
        """As :meth:`_write_secret`: the record's whole write in one helper frame — today's
        transaction, the store's own advisory lock and the same 0600 atomic write, writing the
        record without changing the install's choice of active provider."""
        from hermes_cli import auth
        self._pin(administering=True)
        with auth._auth_store_lock():
            store = auth._load_auth_store()
            auth._store_provider_state(store, provider, dict(record), set_active=False)
            auth._save_auth_store(store)


class LocalSecretStoreProvider(SecretStoreProvider):
    """Today's secrets: one profile home's ``.env`` and ``auth.json``, holding that agent's
    credentials only."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, home: Any = None) -> SecretStore:
        """The store for *home* (the bound profile's home when omitted). The home is frozen here
        once; the pin is checked by every operation."""
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return LocalSecretStore(home)

    def assert_isolating(self) -> None:
        """A local store is one profile's files and refuses every pin but its own; there is no
        shared role that could bypass that. The files are the owner's own machine, not a tenant
        boundary: what isolates tenants here is the pin, which this backend never skips."""
        return
