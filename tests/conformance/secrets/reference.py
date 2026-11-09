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
)

TENANT = "t"
#: The agents the fixture serves, and each one's owner: a principal owning two, a group owning two.
OWNERS: dict[str, Owner] = {
    "a": Principal("p1"), "b": Principal("p1"), "c": Group("g2"), "d": Group("g2"),
}


def _owner_key(owner: Owner) -> tuple[str, str]:
    return (type(owner).__name__, owner.id)


class MemorySecretStore(SecretStore):
    """Every level, keyed by what the bound scope names. Serves one tenant, and no pin at all."""

    levels = frozenset(LEVELS)

    def __init__(self, state: dict, agent: str) -> None:
        self.state, self.agent = state, agent

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
            raise ValueError(f"not a secret level: {level!r}")
        if level == "agent":
            return "agent", self._pin(administering=True).agent
        if level == "owner":
            return "owner", _owner_key(self._pin(administering=True).owner)
        self._pin(administering=True)
        return "tenant", None

    def set(self, name: str, value: str, *, level: str) -> None:
        try:
            where, key = self._place(level)
        except PermissionError:
            # The refusal propagates with the value bound in no frame this store owns
            # (error reporters render traceback locals): clear it before re-raising.
            del value
            raise
        self.state[where].setdefault(key, {})[name] = value

    def delete(self, name: str, *, level: str) -> bool:
        where, key = self._place(level)
        return self.state[where].setdefault(key, {}).pop(name, None) is not None

    def get_auth(self, provider: str) -> Optional[dict[str, Any]]:
        scope = self._pin()
        for level, key in self._levels(scope):
            record = self.state["auth"][level].get(key, {}).get(provider)
            if record is not None:
                return dict(record)
        return None

    def set_auth(self, provider: str, record: dict[str, Any]) -> None:
        try:
            key = self._pin(administering=True).agent
        except PermissionError:
            # As set(): the record is cleared from this frame before the refusal re-raises.
            del record
            raise
        self.state["auth"]["agent"].setdefault(key, {})[provider] = dict(record)


class MemorySecretStoreProvider(SecretStoreProvider):
    """Serves the fixture's agents: a fresh one owns a fresh empty state."""

    levels = frozenset(LEVELS)

    def __init__(self, state: Optional[dict] = None, agents: Optional[dict[str, str]] = None):
        self.state = state or {"agent": {}, "owner": {}, "tenant": {}, "auth": {"agent": {}, "owner": {}, "tenant": {}}}
        self.agents = agents or {}

    @property
    def name(self) -> str:
        return "memory"

    def open(self, home: Any = None) -> SecretStore:
        agent = self.agents.get(str(home), str(home))
        return MemorySecretStore(self.state, agent)

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

    return Backend(open=lambda agent: provider.open(agent), bind=bind, bind_member=bind_member,
                   foreign_tenant=foreign_tenant, unbound=unbound, levels=provider.levels)
