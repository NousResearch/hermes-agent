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
    write_failure,
)


class LocalSecretStore(SecretStore):
    """One profile home's secrets and auth records, pinned to that home.

    As the port's contract requires, a failed credential write holds no value on any error
    path — this backend never mutates a frame it does not own. Every check and validation
    that can fail (the pin, the level, today's write lock) runs before the value (or record)
    is bound in any frame below the write method, and the write itself — the only code that
    binds the value — runs through :meth:`_run_writer`, the one boundary this class puts
    around every writer it calls. The boundary's ``try`` sits inside any context manager the
    write holds (today's auth store lock, held across the load, the merge and the save), so
    a writer's failure is caught before the lock's machinery unwinds it: an ``__exit__``
    hands the exception it is given to the generator it wraps and assigns its
    ``__traceback__``, and a hostile writer's attribute hook would otherwise run right there,
    while the record is still bound in the frames being unwound. The boundary records only
    that the write failed — it never reads the writer's exception, whose every attribute is
    text the failing code chose and a channel a hostile writer can carry the value through
    (an errno property that raises with the value it saw, an errno too large for Python to
    format, and reading one must never be able to preempt the cleanup) — and once its except
    block has exited, dropping the failure and its traceback with it, it drops its own
    bindings of the writer and the value (or record) they carried and raises the port's
    fixed error (:func:`hermes_secret_port.write_failure`) with ``from None``. The write
    method's own bindings leave its frame as the error unwinds it, so every frame that held
    the plaintext is gone: the error's traceback holds only the boundary's frame, the write
    method's and the caller's — none of them binding anything.
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

    def _run_writer(self, operation: str, key: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """The one boundary every writer call crosses — today's ``save_env_value`` /
        ``remove_env_value``, the ``hermes_cli.auth`` store's load, merge and save, the copy
        of the record a write files: the only method that calls a writer, always from inside
        any lock the write holds, so a writer's failure is caught before it reaches a context
        manager's ``__exit__`` — a hostile exception's attribute hook must never be evaluated
        while the value (or record) is still bound in a frame below this one. The handler
        records only that the write failed: the writer's exception — its message, type,
        errno, notes, group members, every attribute writer text — is never read and is gone
        with its traceback when the block exits. The value (or record) reached this frame
        only through *fn* and *args*, which are dropped before anything is raised, so the
        fixed error :func:`hermes_secret_port.write_failure` builds — naming only the
        operation and the key — is raised ``from None`` out of a frame binding nothing; the
        write method's own cleanup finishes as the error unwinds it.
        """
        try:
            result = fn(*args, **kwargs)
        except Exception:  # health: allow BLE001 -- the writer's failure is never inspected or kept
            result = None
            failed = True
        else:
            failed = False
        del fn, args, kwargs
        if failed:
            raise write_failure(operation, key) from None
        return result

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
        del level
        if refusal is not None:
            del value
            raise refusal
        from hermes_cli.config import save_env_value
        try:
            # The write itself — the only code that binds the value (the writer runs inside
            # _run_writer, below this frame), and this frame's own binding leaves it as any
            # error unwinds it, however the write fails.
            self._run_writer("set", name, save_env_value, name, value)
        finally:
            del value

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
        # The delete is a write the same way a set is: today's writer reads the file, drops
        # the key from the lines it holds and replaces the file atomically, so its failure —
        # a disk that fills at the replace, with the stored secrets still bound in the
        # writer's own frames — is caught inside the same boundary and replaced by the
        # port's fixed error, never read and never the error the caller sees.
        return self._run_writer("delete", name, remove_env_value, name)

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
        if refusal is not None:
            del record
            raise refusal
        from hermes_cli import auth
        try:
            with auth._auth_store_lock():
                # Every step that touches the record's bytes runs inside the writer boundary,
                # inside the lock — the load first, so the store parses before the record is
                # bound in any frame below this one (today's loader logs a load failure with
                # the live exception, and a log handler that keeps that record renders its
                # traceback frames later — no frame it unwound holds the record), then the
                # record's own copy, the merge into the loaded store and the atomic save. The
                # boundary catches each step's failure before the lock's context managers
                # unwind it: their ``__exit__`` never evaluates a writer's exception while the
                # record is still bound in a frame below this one.
                store = self._run_writer("set_auth", provider, auth._load_auth_store)
                record_copy = self._run_writer("set_auth", provider, dict, record)
                self._run_writer("set_auth", provider, auth._store_provider_state,
                                 store, provider, record_copy, set_active=False)
                self._run_writer("set_auth", provider, auth._save_auth_store, store)
        finally:
            # This frame's own bindings leave it as any error unwinds it: the record the
            # caller asked to store, the copy the write filed, and the loaded store holding
            # the merged copy of it.
            del record
            record_copy = None
            store = None


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
