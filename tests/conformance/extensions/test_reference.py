"""The extension registry conformance suite against the reference backend.

The reference backend (``hermes_conformance.extensions.reference``) holds one tenant's whole
registry in one in-memory store, so this fixture can vary the owners a local install cannot:
agents ``a1`` and ``a2`` belong to one principal, ``b`` and ``m`` to two others, and ``g`` to a
group. Every class of the suite runs here, the ownership levels included.
"""

from contextlib import nullcontext

import pytest

from agent.runtime_scope import bind_state
from hermes_conformance.extensions import (
    AdminConformance, Backend, LevelBackend, LevelConformance, PinningConformance,
    ProvisioningConformance, SwitchConformance,
)
from hermes_conformance.extensions.reference import ReferenceExtensionRegistry, TenantStore
from hermes_conformance.extensions.suite import install
from hermes_extension_registry_port import RegistryAdminRefused
from hermes_scope import Group, Principal, Scope, System

TENANT = "t1"
P, Q, M = Principal("p"), Principal("q"), Principal("member")
TEAM = Group("team")


def _registry(store, agent, owner):
    return ReferenceExtensionRegistry(store, agent=agent, owner=owner)


def _scope(agent, owner, actor=None, *, can_administer=True):
    # A group's agents are administered by one of the tenant's principals (an org admin), so a
    # group owner never becomes the actor.
    if actor is None:
        actor = owner if isinstance(owner, Principal) else Principal(f"{owner.id}-admin")
    return bind_state(Scope(TENANT, agent, actor, owner=owner, can_administer=can_administer))


def reference_backend():
    """The reference backend's :class:`Backend`: agents ``a`` (owned by P) and ``b`` (owned by
    Q) over one tenant's store."""
    store = TenantStore(TENANT)
    owners = {"a": P, "b": Q}

    def connect(agent):
        return _registry(store, f"agent-{agent}", owners[agent])

    def install_tenant(reg, name):
        # The tenant's provisioning actor alone may install a tenant-wide entry.
        with _scope(reg.agent, reg.owner, System("provisioning")):
            return install(reg, name=name, level="tenant")

    def install_shared(reg):
        install_tenant(reg, "shared")

    return Backend(
        open=connect,
        bind=lambda agent: _scope(f"agent-{agent}", owners[agent]),
        bind_noadmin=lambda agent: _scope(f"agent-{agent}", owners[agent], Principal("teammate"),
                                          can_administer=False),
        other_owner=lambda agent: _scope(f"agent-{agent}", Q),
        foreign_tenant=lambda agent: bind_state(Scope("acme", f"agent-{agent}", owners[agent],
                                                      owner=owners[agent], can_administer=True)),
        unbound=lambda: nullcontext(),  # a hosted backend with no scope bound refuses everything
        install_shared=install_shared,
    )


def reference_level():
    """The reference backend's :class:`LevelBackend`: two agents of P, two other principals'
    agents, and a group's agent."""
    store = TenantStore(TENANT)
    owners = {"a1": P, "a2": P, "b": Q, "m": M, "g": TEAM}

    def connect(agent):
        return _registry(store, agent, owners[agent])

    return LevelBackend(
        open=connect,
        bind=lambda agent: _scope(agent, owners[agent]),
        # A non-administering turn on the same agent: a teammate, or (for the group's agent) one
        # of its members.
        bind_member=lambda agent: _scope(agent, owners[agent],
                                         Principal("member") if agent == "g" else Principal("friend"),
                                         can_administer=False),
        # The tenant's provisioning actor, the one writer of tenant-wide entries.
        bind_provisioning=lambda agent: _scope(agent, owners[agent], System("provisioning")),
    )


@pytest.fixture
def backend(monkeypatch):
    from tests.conformance.extensions.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    return reference_backend()


@pytest.fixture
def level(monkeypatch):
    from tests.conformance.extensions.mutants import apply_mutant
    apply_mutant(monkeypatch)
    return reference_level()


class TestReferencePinning(PinningConformance):
    pass


class TestReferenceAdmin(AdminConformance):
    pass


class TestReferenceSwitch(SwitchConformance):
    pass


class TestReferenceLevel(LevelConformance):
    pass


class TestReferenceProvisioning(ProvisioningConformance):
    pass


def test_every_administration_refusal_happens_before_anything_is_looked_up():
    """A scope that may not administer is refused for every write before the target is looked
    up, so the refusal is the same for an entry the registry holds, a foreign one and a missing
    one — and nothing is read on the way."""
    reg = _registry(TenantStore(TENANT), "agent-a", P)

    def unexpected(*args, **kwargs):
        raise AssertionError("the target was looked up before the administration gate")
    reg._resolve = unexpected
    with _scope("agent-a", P, Principal("teammate"), can_administer=False):
        for name in ("present", "missing", "foreign"):
            for call in (lambda n=name: reg.remove(n),
                         lambda n=name: reg.set_enabled(n, True),
                         lambda n=name: reg.set_agent_enabled(n, True),
                         lambda n=name: install(reg, name=n)):
                with pytest.raises(RegistryAdminRefused):
                    call()
