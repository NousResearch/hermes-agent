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
    path — this backend never mutates a frame it does not own: every load and validation that
    can fail (the pin, the level, today's write lock, the auth store a record's write merges
    into) runs before the value (or record) is bound in any frame below the write method, and
    the write itself calls nothing that logs with the live exception — today's loaders do log
    their failures, so the auth store loads before the record reaches any frame. On any
    failure the method deletes its own value binding and raises outside the except block with
    the chain slots cleared: the port's own refusal as it raised it, and any other failure as
    the fresh error :func:`hermes_secret_port.fresh_write_error` built — the error's traceback
    holds only this method's frame and the caller's, and every frame that held the plaintext
    is gone.
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
            # The caller's text never rides in a port-authored message and never survives in
            # a frame this refusal's traceback keeps: the valid levels are all the error names.
            del level
            raise ValueError("not a secret level; the secret levels are 'agent', 'owner' and 'tenant'")
        if level != "agent":
            del level
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
        refusal = None
        try:
            # Every check and validation that can fail — the pin, the level, today's write
            # lock — runs before the value is bound in any frame below this one: each keeps
            # the message this port wrote for it, and no frame it unwound ever held the value.
            # The level is caller-supplied text that can alias the value, so it leaves this
            # frame before any error does too.
            self._pin(administering=True)
            self._require_agent_level(level)
            from hermes_cli.config import require_env_writable
            require_env_writable(name, "set")
        except BaseException as exc:  # health: allow BLE001 -- the port's own refusal, re-raised below
            refusal = exc
        failure = None
        if refusal is None:
            try:
                self._write_secret(name, value)
            except BaseException as exc:  # health: allow BLE001 -- the fresh error below is all that survives
                failure = fresh_write_error(exc)
        del value
        del level
        if refusal is not None:
            raise refusal
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    def _write_secret(self, name: str, value: str) -> None:
        """The write itself — the only code that binds the value (see :meth:`set`): today's
        writer, at today's path (``get_env_path()`` honors the profile-home override the pin
        just matched), with every check and validation behind it. Today's writer logs nothing
        with the live exception, so no failure it raises is ever retained by a log handler
        with the value in a frame of its traceback; the fresh error :meth:`set` raises in its
        place names only the failing exception's class and errno."""
        from hermes_cli.config import save_env_value
        save_env_value(name, value)

    def delete(self, name: str, *, level: str) -> bool:
        # As set: the level is caller-supplied text that can alias a value, so it leaves this
        # frame before any error does, and the port's own refusal keeps the message the port
        # wrote for it.
        refusal = None
        try:
            self._pin(administering=True)
            self._require_agent_level(level)
            from hermes_cli.config import require_env_writable
            require_env_writable(name, "remove")
        except BaseException as exc:  # health: allow BLE001 -- re-raised below
            refusal = exc
        del level
        if refusal is not None:
            raise refusal
        from hermes_cli.config import remove_env_value
        return remove_env_value(name)

    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        self._pin()
        from hermes_cli import auth
        # Today's reader: the profile's own auth.json first, then the global root per provider.
        return auth._load_provider_state_with_source(auth._load_auth_store(), provider)[0]

    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        refusal = None
        try:
            self._pin(administering=True)
        except BaseException as exc:  # health: allow BLE001 -- the port's own refusal, re-raised below
            refusal = exc
        failure = None
        store = None
        if refusal is None:
            from hermes_cli import auth
            try:
                with auth._auth_store_lock():
                    # The store loads and parses before the record is bound in any frame below
                    # this one: today's loader logs a load failure with the live exception, and
                    # a log handler that keeps that record renders its traceback frames later —
                    # no frame it unwound holds the record (this frame deletes its own binding
                    # before any error leaves it, and drops the loaded store before any raise).
                    store = auth._load_auth_store()
                    # The write itself: the only code that binds the record.
                    self._write_auth_record(store, provider, record)
            except BaseException as exc:  # health: allow BLE001 -- the fresh error below is all that survives
                failure = fresh_write_error(exc)
            store = None
        del record
        if refusal is not None:
            raise refusal
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    def _write_auth_record(self, store: dict[str, Any], provider: str, record: dict[str, Any]) -> None:
        """The record's merge and atomic replace — today's transaction tail, byte for byte: the
        same store lock (held by :meth:`set_auth` across the load and this write), the same
        merge into the loaded store, the same 0600 atomic write, writing the record without
        changing the install's choice of active provider. The only code that binds the record,
        and it calls nothing that logs with the live exception (today's auth writer logs
        none), so no observer-kept exception ever unwinds a record-holding frame."""
        from hermes_cli import auth
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
