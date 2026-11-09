"""An in-memory secret backend that holds every ownership level, for the resolution tests.

The local backend is today's files and holds the agent level only, so the owner-wide and
tenant-wide resolution invariants need a backend that declares and stores them. This one keys
every entry on the ownership chain the bound scope names — never on anything a caller supplies —
and serves exactly one tenant, refusing any other pin like a hosted backend serving its own.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

from hermes_conformance.secrets import Backend
from hermes_scope import Group, Owner, Principal, Scope, current_scope
from hermes_secret_port import (
    LEVELS, SecretAdminRefused, SecretPinRefused, SecretStore, SecretStoreProvider,
    fresh_write_error,
)


TENANT = "t"
#: The key the managed-key hook pins, as an administrator would.
MANAGED_KEY = "ZZ_ADMINISTRATOR_PINNED_KEY"
#: The agents the fixture serves, and each one's owner: a principal owning two, a group owning two.
OWNERS: dict[str, Owner] = {
    "a": Principal("p1"), "b": Principal("p1"), "c": Group("g2"), "d": Group("g2"),
}
#: How the fixture's injected mid-write failure fails at each named failure point: the write
#: itself and the atomic replace with a plain ``OSError``, the serialization with a
#: ``TypeError``. The in-memory store has one write step, so every named point fails there —
#: the nearest point it has to each.
_INJECTED = {"write": OSError, "encode": TypeError, "replace": OSError}


def _owner_key(owner: Owner) -> tuple[str, str]:
    return (type(owner).__name__, owner.id)


class MemorySecretStore(SecretStore):
    """Every level, keyed by what the bound scope names. Serves one tenant, and no pin at all.

    As the port's contract requires, a failed write holds no value on any error path: every
    check and validation that can fail (the placement checks, the fixture's pinned-key write
    lock) runs before the value (or record) is bound in any frame below the write method, and
    the write itself is the only code that binds it. On any failure the method deletes its own
    value binding and raises outside the except block with the chain slots cleared: the
    port's own refusal as it raised it, and any other failure as the fresh error
    (:func:`hermes_secret_port.fresh_write_error`) — so the error's traceback holds only this
    frame and the caller's, and the frame that held the value is gone.
    """

    levels = frozenset(LEVELS)

    def __init__(self, state: dict, agent: str, provider: MemorySecretStoreProvider) -> None:
        self.state, self.agent, self.provider = state, agent, provider

    def _pin(self, *, administering: bool = False) -> Scope:
        scope = current_scope()
        if scope is None:
            raise SecretPinRefused("no scope is bound; this store serves none by default")
        if scope.tenant != TENANT:
            raise SecretPinRefused(f"this store serves tenant {TENANT!r} only")
        if scope.agent != self.agent:
            raise SecretPinRefused("the bound agent is not this store's")
        if administering and not scope.can_administer:
            raise SecretAdminRefused("this scope may not administer this agent's credentials")
        return scope

    def _levels(self, scope: Scope):
        """The scope's resolution chain, most specific first: the agent, its owner, the tenant."""
        return (("agent", scope.agent), ("owner", _owner_key(scope.owner)), ("tenant", None))

    def get(self, name: str) -> Optional[str]:
        scope = self._pin()
        for level, key in self._levels(scope):
            if name in self.state[level].setdefault(key, {}):
                return self.state[level][key][name]
        return None

    def names(self) -> list[str]:
        scope = self._pin()
        found = set()
        for level, key in self._levels(scope):
            found.update(self.state[level].get(key, {}))
        return sorted(found)

    def _place(self, level: str) -> tuple[str, Any]:
        if level not in LEVELS:
            # The caller's text never rides in a port-authored message and never survives in
            # a frame this refusal's traceback keeps: the valid levels are all the error names.
            del level
            raise ValueError("not a secret level; the secret levels are 'agent', 'owner' and 'tenant'")
        if level == "agent":
            return "agent", self._pin(administering=True).agent
        if level == "owner":
            return "owner", _owner_key(self._pin(administering=True).owner)
        self._pin(administering=True)
        return "tenant", None

    def set(self, name: str, value: str, *, level: str) -> None:
        refusal = None
        where, key = None, None
        try:
            # Every check and validation that can fail — the placement (the level, the pin and
            # the administration right), the fixture's pinned-key write lock — runs before the
            # value is bound in any frame below this one, each keeping its own message. The
            # level is caller-supplied text that can alias the value, so it leaves this frame
            # before any error does too.
            where, key = self._place(level)
            if name in self.provider.locked_keys:
                # The managed-key write lock: an administrator pinned this key, and the store
                # refuses changing it with a plain error, not the store's refusal.
                raise ValueError("the key is pinned by an administrator; this store refuses to change it")
        except BaseException as exc:  # health: allow BLE001 -- the port's own refusal, re-raised below
            refusal = exc
        failure = None
        if refusal is None:
            try:
                self._write_secret(where, key, name, value)
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

    def _write_secret(self, where, key, name: str, value: str) -> None:
        """The write itself — the only code that binds the value (see :meth:`set`), with the
        fixture's injected mid-write failure ahead of it. *where* and *key* are the placement
        the checks already resolved (``None`` only when the write never runs)."""
        if self.provider.fail_writes:
            # The fixture's injected mid-write failure at its named point: the caller's value
            # is already bound in this frame, and must not outlive the error in any frame
            # below the caller.
            raise _INJECTED[self.provider.fail_writes]("the store's write path failed part-way through")
        self.state[where].setdefault(key, {})[name] = value

    def delete(self, name: str, *, level: str) -> bool:
        # As set: the level is caller-supplied text that can alias a value, so it leaves this
        # frame before any error does.
        refusal = None
        where = key = None
        try:
            where, key = self._place(level)
        except BaseException as exc:  # health: allow BLE001 -- re-raised below
            refusal = exc
        del level
        if refusal is not None:
            raise refusal
        return self.state[where].setdefault(key, {}).pop(name, None) is not None

    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        scope = self._pin()
        for level, key in self._levels(scope):
            record = self.state["auth"][level].get(key, {}).get(provider)
            if record is not None:
                return dict(record)
        return None

    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        refusal = None
        key = None
        try:
            # The placement check (the pin and the administration right) runs before the
            # record is bound in any frame below this one, keeping its own message.
            key = self._pin(administering=True).agent
        except BaseException as exc:  # health: allow BLE001 -- the port's own refusal, re-raised below
            refusal = exc
        failure = None
        if refusal is None:
            try:
                self._write_auth_record(key, provider, record)
            except BaseException as exc:  # health: allow BLE001 -- the fresh error below is all that survives
                failure = fresh_write_error(exc)
        del record
        if refusal is not None:
            raise refusal
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            raise failure

    def _write_auth_record(self, key, provider: str, record: dict[str, Any]) -> None:
        """The write itself — the only code that binds the record (see :meth:`set_auth`), with
        the fixture's injected mid-write failure ahead of it."""
        if self.provider.fail_writes:
            # As _write_secret: the fixture's injected mid-write failure at its named point,
            # record bound here.
            raise _INJECTED[self.provider.fail_writes]("the store's write path failed part-way through")
        self.state["auth"]["agent"].setdefault(key, {})[provider] = dict(record)


class MemorySecretStoreProvider(SecretStoreProvider):
    """Serves the fixture's agents: a fresh one owns a fresh empty state."""

    levels = frozenset(LEVELS)

    def __init__(self, state: Optional[dict] = None, agents: Optional[dict[str, str]] = None):
        self.state = state or {"agent": {}, "owner": {}, "tenant": {}, "auth": {"agent": {}, "owner": {}, "tenant": {}}}
        self.agents = agents or {}
        #: The failure point the fixture's ``fail_writes`` hook names while it fails every
        #: write (``None``: writes succeed).
        self.fail_writes: Optional[str] = None
        #: The keys an administrator has pinned (the fixture's ``managed_key`` hook): the
        #: store's own write lock refuses changing them with a plain error.
        self.locked_keys: set[str] = set()

    @property
    def name(self) -> str:
        return "memory"

    def open(self, home: Any = None) -> SecretStore:
        agent = self.agents.get(str(home), str(home))
        return MemorySecretStore(self.state, agent, self)

    def assert_isolating(self) -> None:
        return


def reference_backend() -> Backend:
    """The suite's :class:`Backend` over one shared store: agents ``a``/``b`` owned by one
    principal, ``c``/``d`` by one group, all of one tenant."""
    provider = MemorySecretStoreProvider(agents={name: name for name in OWNERS})

    @contextmanager
    def bind_scope(agent, *, actor, administer):
        scope = Scope(TENANT, agent, actor, owner=OWNERS[agent], can_administer=administer)
        from agent.runtime_scope import bind_state
        with bind_state(scope):
            yield

    def bind(agent):
        owner = OWNERS[agent]
        actor = owner if isinstance(owner, Principal) else Principal("group-admin")
        return bind_scope(agent, actor=actor, administer=True)

    def bind_member(agent):
        return bind_scope(agent, actor=Principal("member"), administer=False)

    def foreign_tenant(agent):
        from agent.runtime_scope import bind_state
        return bind_state(Scope("acme", agent, Principal("u"), owner=Group("g"), can_administer=False))

    @contextmanager
    def unbound():
        yield  # a hosted backend binds nothing: no scope must be a refusal, not a default

    @contextmanager
    def fail_writes(kind: str):
        """While active, every write fails part-way through at the failure point *kind* names
        (``hermes_conformance.secrets.suite.WRITE_FAILURES``): the in-memory store has one
        write step, so each named point fails there — the nearest point it has — with its
        kind of error (a plain ``OSError`` for the write and the replace, a ``TypeError`` for
        the encode), with the caller's value (or record) already bound on the store's own
        frame."""
        provider.fail_writes = kind
        try:
            yield
        finally:
            provider.fail_writes = None

    @contextmanager
    def managed_key():
        """While active, the yielded key is pinned by an administrator (the store's own
        write lock): writing it fails with a plain ``ValueError`` part-way through the
        write — not the store's refusal."""
        provider.locked_keys.add(MANAGED_KEY)
        try:
            yield MANAGED_KEY
        finally:
            provider.locked_keys.discard(MANAGED_KEY)

    return Backend(open=lambda agent: provider.open(agent), bind=bind, bind_member=bind_member,
                   foreign_tenant=foreign_tenant, unbound=unbound, managed_key=managed_key,
                   fail_writes=fail_writes, levels=provider.levels)
