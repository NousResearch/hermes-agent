"""Conformance suite for user extension registry backends (``hermes_extension_registry_port``).

Core runs it against the local backend (``tests/conformance/extensions/test_local.py``) and
against the in-memory reference backend (``test_reference.py``). A backend plugin runs the same
suite by importing the conformance classes into its own test module and supplying fixtures that
return a :class:`Backend` (and a :class:`LevelBackend`) for its registry::

    from hermes_conformance.extensions import AdminConformance, Backend, LevelBackend

    class TestMyBackend(AdminConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips.

Covered:

- pinning: every operation of the port is refused for another agent's pin, another tenant
  naming the same agent, another owner naming it, and no pin at all, and a refusal changes
  nothing the agent observes through the port (:func:`observe`);
- administration: installing, removing, disabling and switching on are refused for a scope that
  may not administer the agent, with a refusal that never reveals whether the entry exists (the
  same message as for a missing name), while every read still works for such a scope;
- the tenant's provisioning: a tenant-wide entry is written by the provisioning actor alone, not
  by an agent's administrator (:class:`ProvisioningConformance`, against a backend that holds a
  whole tenant's registry);
- per-agent switches: an entry switched on for one agent is absent from another agent's — of any
  owner — offered list, and answers there exactly like a missing one; a level-disabled entry is
  offered to no one even with its switch on; removing an entry takes its switches with it;
- the ownership levels (:class:`LevelConformance`, against a backend that holds a whole tenant's
  registry): an owner-wide entry resolves on every agent of its owner and on no other, a
  tenant-wide one on every principal of the tenant, a group's entries resolve on its agents for
  its members while a member's own never reach them, one name at both levels resolves by owner,
  and the per-agent switch is one agent's alone.

NOT covered. A backend that passes this suite may still not:

- check an entry's files against its ``content_hash`` before a run, or refuse a failed scan
  (the caller's, later surfaces: the port stores ``scan_result`` as data);
- isolate tenants in storage itself (``ExtensionRegistryProvider.assert_isolating``; only that
  the method exists and refuses a backend that has not checked is tested elsewhere);
- survive a crash between two writes (durability of the backend's own storage);
- shape results beyond the fields the tests read.

The local backend holds one agent per profile file and one owner (the local person), so it
cannot express several agents of one owner, another owner of the tenant, or a group's agents:
that is what :class:`LevelConformance` is for, and the reference backend
(:mod:`hermes_conformance.extensions.reference`) exists so those rules are tested in core today.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable

import pytest

from hermes_extension_registry_port import (
    ExtensionRegistry, RegistryAdminRefused, RegistryPinRefused,
)


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's registry. Agents are ``"a"`` and ``"b"``; ``a`` holds the
      suite's world, and ``b`` is a second agent the world's entries must not reach.
    - ``bind(agent)``: bind the agent's administering scope (its owner, may administer).
    - ``bind_noadmin(agent)``: bind a scope on the same agent that may not administer it (a
      teammate's turn; a group member, for a group-owned agent).
    - ``other_owner(agent)``: bind a scope of ANOTHER owner that names the same agent.
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing.
    - ``install_shared(registry)``: install the world's ``shared`` entry — at the tenant level
      under the tenant's provisioning scope where the backend stores one, else the backend's
      stand-in for it (the local backend's owner-wide row, the registry's local exception).
    """

    open: Callable[[str], ExtensionRegistry]
    bind: Callable[[str], AbstractContextManager]
    bind_noadmin: Callable[[str], AbstractContextManager]
    other_owner: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    install_shared: Callable[[ExtensionRegistry], None]


@dataclass
class LevelBackend:
    """How the suite drives a backend that holds a whole tenant's registry, for the ownership
    levels. Agents:

    - ``a1``, ``a2``: two agents of one principal owner;
    - ``b``: an agent of another principal;
    - ``m``: a third principal's own agent;
    - ``g``: an agent owned by a group.

    ``bind(agent)`` binds its administering scope; ``bind_member(agent)`` binds a
    non-administering one (a teammate; for ``g``, a group member's turn); ``bind_provisioning``
    binds the tenant's provisioning actor on the agent — the only scope that may write a
    tenant-wide entry (it may administer).
    """

    open: Callable[[str], ExtensionRegistry]
    bind: Callable[[str], AbstractContextManager]
    bind_member: Callable[[str], AbstractContextManager]
    bind_provisioning: Callable[[str], AbstractContextManager]


def _refused(call: Callable[[], object]) -> str:
    """The refusal *call* raises (its message), asserting it is the registry's pin refusal."""
    with pytest.raises(RegistryPinRefused) as refused:
        call()
    return str(refused.value)


def _admin_refused(call: Callable[[], object]) -> str:
    with pytest.raises(RegistryAdminRefused) as refused:
        call()
    return str(refused.value)


def install(registry: ExtensionRegistry, *, name: str, level: str = "owner",
            kind: str = "plugin", version: str = "1.0") -> dict:
    """Record one entry through *registry*, under the bound scope."""
    return registry.install(name=name, kind=kind, version=version, content_hash=f"hash-{name}-{version}",
                            manifest={"name": name, "version": version}, scan_result={"passed": True},
                            level=level)


# --- the world every test runs against ------------------------------------------------------

@dataclass
class World:
    """Agent ``a``'s registry holds: ``weather`` (a plugin, enabled, switched on for ``a``),
    ``auth`` (an MCP server, withdrawn at its level, switched on) and ``shared`` (a tenant-wide
    plugin, enabled, switched off)."""

    registry: ExtensionRegistry


def make_world(backend: Backend) -> World:
    registry = backend.open("a")
    with backend.bind("a"):
        install(registry, name="weather")
        registry.set_agent_enabled("weather", True)
        install(registry, name="auth", kind="mcp_server")
        registry.set_enabled("auth", False)
        registry.set_agent_enabled("auth", True)
    backend.install_shared(registry)
    return World(registry)


def observe(backend: Backend, registry: ExtensionRegistry) -> dict:
    """Everything about agent ``a`` the port exposes. A refused operation must leave all of it
    unchanged."""
    with backend.bind("a"):
        return {"listed": sorted(r["name"] for r in registry.list()),
                "rows": registry.list(),
                "get_weather": registry.get("weather"),
                "get_auth": registry.get("auth"),
                "enabled": sorted(r["name"] for r in registry.enabled_extensions()),
                "switch_weather": registry.agent_enabled("weather"),
                "switch_auth": registry.agent_enabled("auth"),
                "switch_shared": registry.agent_enabled("shared")}


def _names(rows: list) -> list:
    return sorted(r["name"] for r in rows)


#: Every read, naming rows agent ``a`` holds in a :class:`World`.
READS: dict[str, Callable] = {
    "get": lambda r: r.get("weather"),
    "get_missing": lambda r: r.get("absent"),
    "list": lambda r: r.list(),
    "list_kind": lambda r: r.list(kind="mcp_server"),
    "agent_enabled": lambda r: r.agent_enabled("weather"),
    "agent_enabled_missing": lambda r: r.agent_enabled("absent"),
    "enabled_extensions": lambda r: r.enabled_extensions(),
    "enabled_extensions_kind": lambda r: r.enabled_extensions(kind="mcp_server"),
}

#: Every write, with arguments that succeed in a :class:`World` under agent ``a``'s own
#: administering scope. A tenant-wide install is not among them: an agent's administrator may
#: not make one (:class:`ProvisioningConformance`).
WRITES: dict[str, Callable] = {
    "install": lambda r: install(r, name="nova"),
    "install_mcp": lambda r: install(r, name="server", kind="mcp_server"),
    "set_enabled": lambda r: r.set_enabled("weather", False),
    "set_agent_enabled_off": lambda r: r.set_agent_enabled("weather", False),
    "set_agent_enabled_on": lambda r: r.set_agent_enabled("shared", True),
    "remove": lambda r: r.remove("weather"),
}

#: The operations that target an entry by name, for the same-answer tests (the refusal for an
#: entry the registry holds must equal the refusal for a missing name).
NAMED_WRITES: dict[str, Callable] = {
    "set_enabled": lambda r, name: r.set_enabled(name, False),
    "set_agent_enabled": lambda r, name: r.set_agent_enabled(name, True),
    "remove": lambda r, name: r.remove(name),
}


def _foreign_pins(backend: Backend) -> dict:
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "other_owner": lambda: backend.other_owner("a"), "unbound": backend.unbound}


class PinningConformance:
    """Every operation is pinned to its agent. Inherit and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        world = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: READS[operation](world.registry))
        with backend.bind("a"):
            READS[operation](world.registry)

    @pytest.mark.parametrize("operation", sorted(WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "other_owner", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(
            self, backend, operation, pin):
        world = make_world(backend)
        before = observe(backend, world.registry)
        with _foreign_pins(backend)[pin]():
            _refused(lambda: WRITES[operation](world.registry))
        assert observe(backend, world.registry) == before

    @pytest.mark.parametrize("operation", sorted(READS))
    def test_every_read_succeeds_for_its_own_agent(self, backend, operation):
        world = make_world(backend)
        with backend.bind("a"):
            READS[operation](world.registry)

    @pytest.mark.parametrize("operation", sorted(WRITES))
    def test_every_write_succeeds_for_its_own_administering_scope(self, backend, operation):
        world = make_world(backend)
        before = observe(backend, world.registry)
        with backend.bind("a"):
            WRITES[operation](world.registry)
        assert observe(backend, world.registry) != before


class AdminConformance:
    """Installing and switching are administration. Inherit and supply a ``backend`` fixture."""

    def test_every_read_works_for_a_scope_that_may_not_administer(self, backend):
        world = make_world(backend)
        with backend.bind_noadmin("a"):
            assert _names(world.registry.list()) == ["auth", "shared", "weather"]
            assert world.registry.get("weather")["version"] == "1.0"
            assert world.registry.agent_enabled("weather") is True
            assert _names(world.registry.enabled_extensions()) == ["weather"]

    @pytest.mark.parametrize("operation", sorted(WRITES))
    def test_every_write_is_refused_without_administration_and_changes_nothing(self, backend, operation):
        world = make_world(backend)
        before = observe(backend, world.registry)
        with backend.bind_noadmin("a"):
            _admin_refused(lambda: WRITES[operation](world.registry))
        assert observe(backend, world.registry) == before

    @pytest.mark.parametrize("operation", sorted(NAMED_WRITES))
    def test_a_refusal_never_reveals_whether_the_entry_exists(self, backend, operation):
        """The refusal for an entry the registry holds is the refusal for a missing name, byte
        for byte, for each operation that targets one."""
        world = make_world(backend)
        write = NAMED_WRITES[operation]
        with backend.bind_noadmin("a"):
            first = _admin_refused(lambda: write(world.registry, "weather"))
            second = _admin_refused(lambda: write(world.registry, "absent"))
        assert first == second

    def test_installation_is_refused_identically_whatever_the_name(self, backend):
        world = make_world(backend)
        with backend.bind_noadmin("a"):
            first = _admin_refused(lambda: install(world.registry, name="weather"))
            second = _admin_refused(lambda: install(world.registry, name="absent"))
        assert first == second


class SwitchConformance:
    """The per-agent switch is one agent's alone. Inherit and supply a ``backend`` fixture."""

    def test_a_switched_on_entry_is_enabled_for_its_agent_and_absent_from_every_other(self, backend):
        world = make_world(backend)
        theirs = backend.open("b")
        with backend.bind("a"):
            assert _names(world.registry.enabled_extensions()) == ["weather"]
            assert world.registry.agent_enabled("weather") is True
        with backend.bind("b"):
            # Another agent's registry answers exactly as if the entry had never existed.
            assert _names(theirs.enabled_extensions()) == []
            assert theirs.agent_enabled("weather") is False
            assert theirs.agent_enabled("absent") is False
            assert theirs.get("weather") is None
            assert theirs.get("absent") is None
            assert theirs.set_agent_enabled("weather", True) is False
            assert theirs.set_agent_enabled("absent", True) is False
            assert theirs.remove("weather") is False
            assert theirs.remove("absent") is False
            assert theirs.set_enabled("weather", False) is False
            assert theirs.set_enabled("absent", False) is False

    def test_an_entry_withdrawn_at_its_level_is_offered_to_no_one_even_with_its_switch_on(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.registry.get("auth")["enabled"] is False
            assert world.registry.agent_enabled("auth") is True  # the switch is still on
            assert "auth" not in _names(world.registry.enabled_extensions())
            assert world.registry.set_enabled("auth", True) is True
            assert "auth" in _names(world.registry.enabled_extensions())

    def test_removing_an_entry_takes_its_switch_with_it(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.registry.remove("weather") is True
            assert world.registry.get("weather") is None
            assert world.registry.agent_enabled("weather") is False
            assert world.registry.enabled_extensions() == []
            install(world.registry, name="weather")  # a fresh entry starts switched off
            assert world.registry.agent_enabled("weather") is False


class LevelConformance:
    """The ownership levels, against a backend that holds a whole tenant's registry. Inherit and
    supply a ``level`` fixture."""

    def _world(self, level):
        """``a1`` installed ``team-tool`` for its owner; the tenant's provisioning installed
        ``org-tool`` for the tenant; ``m`` installed ``member-tool`` for itself."""
        a1, m = level.open("a1"), level.open("m")
        with level.bind("a1"):
            install(a1, name="team-tool")
        with level.bind_provisioning("a1"):
            install(a1, name="org-tool", level="tenant")
        with level.bind("m"):
            install(m, name="member-tool")
        return a1, m

    def test_an_owner_wide_entry_resolves_on_every_agent_of_its_owner_and_on_no_other(self, level):
        self._world(level)
        with level.bind("a2"):
            assert level.open("a2").get("team-tool") is not None
        for agent in ("b", "m", "g"):
            with level.bind(agent):
                assert level.open(agent).get("team-tool") is None
                assert "team-tool" not in _names(level.open(agent).list())
        with level.bind("b"):
            theirs = level.open("b")
            assert theirs.set_agent_enabled("team-tool", True) is False
            assert theirs.set_agent_enabled("absent", True) is False  # foreign answers like missing

    def test_a_tenant_wide_entry_resolves_for_every_principal_of_the_tenant(self, level):
        self._world(level)
        for agent in ("a1", "a2", "b", "m", "g"):
            with level.bind(agent):
                assert level.open(agent).get("org-tool") is not None, agent
        with level.bind_member("g"):
            assert level.open("g").get("org-tool") is not None  # a member's turn too

    def test_a_members_own_entries_never_reach_a_group_agent(self, level):
        self._world(level)
        with level.bind_member("g"):
            theirs = level.open("g")
            assert theirs.get("member-tool") is None
            assert "member-tool" not in _names(theirs.list())
            assert theirs.agent_enabled("member-tool") is False
            assert _names(theirs.enabled_extensions()) == []

    def test_a_groups_entries_reach_its_agents_for_its_members(self, level):
        with level.bind("g"):
            install(level.open("g"), name="group-tool")
        with level.bind_member("g"):
            theirs = level.open("g")
            assert theirs.get("group-tool") is not None
        for agent in ("a1", "a2", "b", "m"):
            with level.bind(agent):
                assert level.open(agent).get("group-tool") is None

    def test_one_name_at_both_levels_resolves_by_owner(self, level):
        a1 = level.open("a1")
        with level.bind_provisioning("a1"):
            install(a1, name="dup", level="tenant")
        with level.bind("a1"):
            install(a1, name="dup")
            assert a1.get("dup")["level"] == "owner"
        with level.bind("a2"):
            assert level.open("a2").get("dup")["level"] == "owner"
        with level.bind("b"):
            assert level.open("b").get("dup")["level"] == "tenant"

    def test_removing_one_owners_entry_leaves_another_owners_of_the_same_name(self, level):
        """A switch row belongs to one entry — its level, owner and name — so removing an
        entry drops its own switches alone: another owner's entry of the same name, and this
        agent's switch for it, are untouched."""
        a1, b = level.open("a1"), level.open("b")
        with level.bind("a1"):
            install(a1, name="dup")
            assert a1.set_agent_enabled("dup", True) is True
        with level.bind("b"):
            install(b, name="dup")
            assert b.remove("dup") is True  # its own entry alone
        with level.bind("a1"):
            assert a1.get("dup") is not None
            assert a1.agent_enabled("dup") is True
            assert _names(a1.enabled_extensions()) == ["dup"]

    def test_an_entry_never_inherits_the_switch_of_another_level_of_its_name(self, level):
        a1 = level.open("a1")
        with level.bind_provisioning("a1"):
            install(a1, name="dup", level="tenant")
        with level.bind("a1"):
            assert a1.set_agent_enabled("dup", True) is True  # the tenant-wide one's switch
            install(a1, name="dup")                           # the owner-wide one shadows it
            assert a1.agent_enabled("dup") is False           # a fresh entry starts switched off

    def test_the_offered_list_resolves_one_entry_per_name_as_get_does(self, level):
        """``enabled_extensions`` offers one entry per name, by the same precedence as ``get``:
        a name at both levels — both switched on — is offered as its owner-wide entry to its
        owner, and as its tenant-wide entry to everyone else, never as both."""
        a1 = level.open("a1")
        with level.bind_provisioning("a1"):
            install(a1, name="dup", level="tenant")
        with level.bind("a1"):
            assert a1.set_agent_enabled("dup", True) is True  # the tenant-wide one's switch
            install(a1, name="dup")
            assert a1.set_agent_enabled("dup", True) is True  # the owner-wide one's switch too
            offered = a1.enabled_extensions()
            assert [r["level"] for r in offered] == ["owner"]
            assert offered[0] == a1.get("dup")
        with level.bind("b"):
            theirs = level.open("b")
            assert theirs.set_agent_enabled("dup", True) is True
            offered = theirs.enabled_extensions()
            assert [r["level"] for r in offered] == ["tenant"]
            assert offered[0] == theirs.get("dup")

    def test_pinning_an_agent_never_widens_the_tenant(self, level):
        """A scope naming the same owner's OTHER agent is still a foreign pin on this one's
        registry: it is refused, and the refusal says nothing about the rows."""
        self._world(level)
        a1 = level.open("a1")
        with level.bind("a2"):
            for call in (lambda: a1.get("team-tool"), lambda: a1.list(),
                         lambda: a1.enabled_extensions(),
                         lambda: a1.set_agent_enabled("team-tool", True)):
                _refused(call)
        with level.bind("a1"):
            assert a1.get("team-tool") is not None

    def test_the_per_agent_switch_is_one_agent_alone(self, level):
        self._world(level)
        with level.bind("a1"):
            a1 = level.open("a1")
            assert a1.set_agent_enabled("team-tool", True) is True
            assert _names(a1.enabled_extensions()) == ["team-tool"]
        for agent, bind in (("a2", level.bind), ("b", level.bind), ("m", level.bind),
                           ("g", level.bind_member)):
            with bind(agent):
                theirs = level.open(agent)
                if theirs.get("team-tool") is None:
                    continue  # an agent that cannot see the entry is tested by the pin matrix
                assert theirs.agent_enabled("team-tool") is False
                assert "team-tool" not in _names(theirs.enabled_extensions())


class ProvisioningConformance:
    """Tenant-wide entries are written by the tenant's provisioning actor alone. The agent's
    administration right reaches the agent's switches and its owner's owner-wide entries, never
    the tenant's (no tenant-administration claim exists yet); the provisioning actor is the one
    other writer, and it answers for the whole tenant. Inherit and supply a ``level`` fixture
    whose ``bind_provisioning`` binds that actor."""

    def test_a_principal_administrator_cannot_install_a_tenant_wide_entry(self, level):
        a1 = level.open("a1")
        with level.bind("a1"):
            first = _admin_refused(lambda: install(a1, name="org-tool", level="tenant"))
            second = _admin_refused(lambda: install(a1, name="absent-tool", level="tenant"))
        assert first == second  # the refusal never says whether the name is taken
        with level.bind_member("g"):
            _admin_refused(lambda: install(level.open("g"), name="org-tool", level="tenant"))
        with level.bind("b"):
            assert level.open("b").get("org-tool") is None  # and nothing changed for anyone

    def test_the_provisioning_actor_installs_tenant_wide_entries_for_the_whole_tenant(self, level):
        with level.bind_provisioning("a1"):
            row = install(level.open("a1"), name="org-tool", level="tenant")
        assert row["level"] == "tenant"
        for agent, bind in (("a1", level.bind), ("a2", level.bind), ("b", level.bind),
                            ("m", level.bind), ("g", level.bind_member)):
            with bind(agent):
                assert level.open(agent).get("org-tool") is not None, agent

    def test_a_principal_administrator_cannot_disable_or_remove_a_tenant_wide_entry(self, level):
        a1 = level.open("a1")
        with level.bind_provisioning("a1"):
            install(a1, name="org-tool", level="tenant")
        with level.bind("a1"):
            for call in (lambda: a1.set_enabled("org-tool", False),
                         lambda: a1.remove("org-tool")):
                _admin_refused(call)
            # the switch is the agent's own row: an owner may still switch a tenant-wide entry on
            assert a1.set_agent_enabled("org-tool", True) is True
            assert _names(a1.enabled_extensions()) == ["org-tool"]
        with level.bind("b"):
            theirs = level.open("b")
            assert theirs.get("org-tool")["enabled"] is True  # untouched by the other owner's admin
            assert _names(theirs.enabled_extensions()) == []  # its own switch is off

    def test_the_provisioning_actor_administers_tenant_wide_entries(self, level):
        a1 = level.open("a1")
        with level.bind_provisioning("a1"):
            install(a1, name="org-tool", level="tenant")
            assert a1.set_enabled("org-tool", False) is True
            assert a1.set_enabled("org-tool", True) is True
            assert a1.remove("org-tool") is True
        with level.bind("b"):
            assert level.open("b").get("org-tool") is None
