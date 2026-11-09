"""The skills index conformance suite: isolation and level invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks
to the port only (``hermes_skills_port``); the fixture says how to bind each agent's scope and
open its index. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every read (:data:`ALL_READS`) and every write (:data:`ALL_WRITES`) is refused for
  another agent, another tenant and no pin, and a refusal changes nothing the agent observes
  through the port (:func:`observe`: the resolved list, one row per skill, every body and
  supporting file);
- levels: a write naming a level the backend's provider does not store (or one no skills index
  stores at all) is refused before anything is written, and a refusal changes nothing;
- isolation: another agent's skill is indistinguishable from a missing one through every read
  and every delete — the row is ``None``, the body ``None``, the delete ``False``, the list
  silent, with no error naming what exists elsewhere;
- resolution (``ResolutionConformance``, over a backend that stores every level): the agent's
  own skill wins over its owner's and the tenant's, an owner-wide skill resolves on every agent
  of that owner and no other, a tenant-wide skill resolves last, and another tenant's skill is
  indistinguishable from a missing one.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- who may write owner-wide and tenant-wide skills (administration): the signed claim decides it;
  the port takes the level a write names, and the suite tests what each level holds, not who
  may name it;
- the offer-time gates (platform, environment, app availability, the disabled list): callers
  apply them to what the index returns;
- where skill bodies live (``read_body`` is opaque text: a blob store behind it changes nothing
  the suite can see);
- the cross-root tier resolution of a local install (project, create-dir and external skills
  directories), which the port leaves with today's callers;
- the shape of rows beyond the fields the tests read.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from hermes_skills_port import (
    AGENT_LEVEL, SKILL_LEVELS, SkillsIndexing, SkillsLevelRefused, StoragePinRefused,
)


def skill_md(name: str, description: str) -> str:
    """A ``SKILL.md`` the port's write accepts: frontmatter naming the skill."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nInstructions."


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's index.
    - ``bind(agent)``: bind the agent's scope (a non-turn activity of its owner).
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing (and its backend
      refuses to serve).
    - ``levels``: the ownership levels the backend's provider stores; writes at any other level
      must be refused.
    """

    open: Callable[[str], SkillsIndexing]
    bind: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    levels: frozenset = frozenset({AGENT_LEVEL})


@dataclass
class World:
    """Agent ``a``: skill ``alpha`` (no category, holding ``references/notes.md``) and skill
    ``beta`` (category ``ops``)."""

    storage: SkillsIndexing


def make_world(backend: Backend) -> World:
    storage = backend.open("a")
    with backend.bind("a"):
        storage.index.put("alpha", skill_md("alpha", "The alpha skill."))
        storage.index.put("beta", skill_md("beta", "The beta skill."), category="ops")
        storage.index.put_file("alpha", "references/notes.md", "alpha notes")
    return World(storage)


#: Every name a test observes, so a refusal that creates or destroys any of them shows.
NAMES = ("alpha", "beta", "gamma", "delta", "missing")
#: Every supporting file a test observes.
FILES = ("references/notes.md", "scripts/run.py")
_ROW_FIELDS = ("name", "description", "category", "status", "load_name", "level")


def observe(backend: Backend, storage: SkillsIndexing) -> dict[str, Any]:
    """Everything about agent ``a`` the port exposes: its resolved list, one row per name, every
    body and supporting file. A refused operation must leave all of it unchanged."""
    with backend.bind("a"):
        state: dict[str, Any] = {"listed": sorted((r["name"], r["level"]) for r in storage.index.list())}
        for name in NAMES:
            row = storage.index.get(name)
            state[f"get:{name}"] = None if row is None else {k: row[k] for k in _ROW_FIELDS}
            state[f"body:{name}"] = storage.index.read_body(name)
            for file_path in FILES:
                state[f"file:{name}:{file_path}"] = storage.index.read_body(name, file_path)
    return state


#: Every write, with arguments that succeed in a :class:`World` at the agent level; *level* is
#: the level the test writes at.
ALL_WRITES: dict[str, Callable[[SkillsIndexing, World, str], Any]] = {
    "put_new": lambda st, w, level: st.index.put("gamma", skill_md("gamma", "A new skill."), level=level),
    "put_replace": lambda st, w, level: st.index.put("alpha", skill_md("alpha", "Rewritten."), level=level),
    "put_category": lambda st, w, level: st.index.put(
        "delta", skill_md("delta", "A new skill."), category="ops", level=level),
    "put_file": lambda st, w, level: st.index.put_file("alpha", "references/notes.md", "changed", level=level),
    "put_file_new": lambda st, w, level: st.index.put_file("alpha", "scripts/run.py", "print(1)", level=level),
    "delete": lambda st, w, level: st.index.delete("beta", level=level),
    "delete_file": lambda st, w, level: st.index.delete_file("alpha", "references/notes.md", level=level),
}

#: Every read, naming rows agent ``a`` holds in a :class:`World` (and some it does not).
ALL_READS: dict[str, Callable[[SkillsIndexing, World], Any]] = {
    "list": lambda st, w: st.index.list(),
    "get": lambda st, w: st.index.get("alpha"),
    "get_missing": lambda st, w: st.index.get("missing"),
    "read_body": lambda st, w: st.index.read_body("alpha"),
    "read_file": lambda st, w: st.index.read_body("alpha", "references/notes.md"),
}


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(StoragePinRefused) as refused:
        call()
    return str(refused.value)


def _foreign_pins(backend: Backend) -> dict[str, Callable[[], AbstractContextManager]]:
    """Every pin but agent ``a``'s own: another agent, another tenant naming ``a``, no pin."""
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "unbound": backend.unbound}


class SkillsConformance:
    """Every operation is pinned to its agent, and every write names a level it stores. Inherit
    and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(ALL_READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        world = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: ALL_READS[operation](world.storage, world))
        with backend.bind("a"):  # and its own pin still reads
            ALL_READS[operation](world.storage, world)

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(
            self, backend, operation, pin):
        world = make_world(backend)
        before = observe(backend, world.storage)
        with _foreign_pins(backend)[pin]():
            _refused(lambda: ALL_WRITES[operation](world.storage, world, AGENT_LEVEL))
        assert observe(backend, world.storage) == before

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_every_write_succeeds_for_its_own_agent(self, backend, operation):
        """The positive control for the refusal tests: each call is one the agent may make, and
        it changes what the agent observes."""
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            ALL_WRITES[operation](world.storage, world, AGENT_LEVEL)
        assert observe(backend, world.storage) != before

    def test_with_no_scope_pinned_reads_see_nothing(self, backend):
        world = make_world(backend)
        with backend.unbound():
            for operation in ALL_READS.values():
                _refused(lambda operation=operation: operation(world.storage, world))

    def test_another_agent_never_sees_the_agents_skills(self, backend):
        world = make_world(backend)
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.index.get("alpha") is None
            assert theirs.index.read_body("alpha") is None
            assert theirs.index.list() == []
            _refused(lambda: world.storage.index.list())  # a's own index, a foreign pin
            _refused(lambda: world.storage.index.get("alpha"))
            _refused(lambda: world.storage.index.delete("alpha"))
        with backend.bind("a"):  # and nothing b did touched a's skills
            assert world.storage.index.get("alpha") is not None

    def test_a_foreign_skill_is_indistinguishable_from_a_missing_one(self, backend):
        make_world(backend)
        theirs = backend.open("b")
        with backend.bind("b"):
            for read in (lambda: theirs.index.get("alpha"), lambda: theirs.index.get("missing"),
                         lambda: theirs.index.read_body("alpha"), lambda: theirs.index.read_body("missing"),
                         lambda: theirs.index.read_body("alpha", "references/notes.md")):
                assert read() is None
            assert theirs.index.delete("alpha") == theirs.index.delete("missing") is False
            assert theirs.index.delete_file("alpha", "references/notes.md") is False
            assert [r["name"] for r in theirs.index.list()] == []

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_a_write_at_a_level_the_backend_does_not_store_is_refused_whole(self, backend, operation):
        """``participant`` no skills index stores; the others are refused by a backend whose
        provider declares it does not store them. A backend that stores every level runs the
        resolution tests instead; every level this test names is refused, never skipped."""
        refused_levels = ["participant", *sorted(SKILL_LEVELS - backend.levels)]
        assert refused_levels, "a skills index never stores every level this test refuses"
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            for level in refused_levels:
                with pytest.raises(SkillsLevelRefused):
                    ALL_WRITES[operation](world.storage, world, level)
        assert observe(backend, world.storage) == before


class ResolutionConformance:
    """Reads resolve agent level → owner-wide → tenant-wide, and a foreign tenant's skill is
    indistinguishable from a missing one. Inherit and supply a ``resolver`` fixture whose
    backend stores every level (:data:`SKILL_LEVELS`)."""

    def test_every_level_writes_and_reads_for_its_own_agent(self, resolver):
        """The positive control: one agent's own writes at every level read back."""
        storage = resolver.open("a")
        with resolver.bind("a"):
            storage.index.put("a-skill", skill_md("a-skill", "Agent level."))
            storage.index.put("o-skill", skill_md("o-skill", "Owner level."), level="owner")
            storage.index.put("t-skill", skill_md("t-skill", "Tenant level."), level="tenant")
            listed = {r["name"]: r["level"] for r in storage.index.list()}
        assert listed == {"a-skill": "agent", "o-skill": "owner", "t-skill": "tenant"}
        with resolver.bind("a"):
            for name in listed:
                assert storage.index.read_body(name) is not None

    def test_the_agents_own_skill_wins_over_the_owners_and_the_tenants(self, resolver):
        storage = resolver.open("a")
        with resolver.bind("a"):
            storage.index.put("shared", skill_md("shared", "Tenant level."), level="tenant")
            storage.index.put("shared", skill_md("shared", "Owner level."), level="owner")
            storage.index.put("shared", skill_md("shared", "Agent level."))
            assert storage.index.get("shared")["level"] == "agent"
            assert "Agent level." in storage.index.read_body("shared")
            assert storage.index.delete("shared")  # the agent's own write goes first
            assert storage.index.get("shared")["level"] == "owner"
            assert storage.index.delete("shared", level="owner")
            assert storage.index.get("shared")["level"] == "tenant"

    def test_an_owner_wide_skill_resolves_on_every_agent_of_that_owner_and_no_other(self, resolver):
        with resolver.bind("a"):
            resolver.open("a").index.put("fleet", skill_md("fleet", "Every agent of the owner."),
                                         level="owner")
        for agent in ("a", "c"):  # one owner: both resolve it
            with resolver.bind(agent):
                assert resolver.open(agent).index.get("fleet")["level"] == "owner"
        with resolver.bind("b"):  # another owner: the skill is missing, and stays missing
            theirs = resolver.open("b")
            assert theirs.index.get("fleet") is None
            assert theirs.index.delete("fleet", level="owner") is False
            assert [r["name"] for r in theirs.index.list()] == []

    def test_a_tenant_wide_skill_resolves_after_the_owner(self, resolver):
        storage = resolver.open("a")
        with resolver.bind("a"):
            storage.index.put("portal", skill_md("portal", "Tenant level."), level="tenant")
            storage.index.put("portal", skill_md("portal", "Owner level."), level="owner")
            assert storage.index.get("portal")["level"] == "owner"
            storage.index.delete("portal", level="owner")
            assert storage.index.get("portal")["level"] == "tenant"

    def test_another_tenants_skill_is_indistinguishable_from_a_missing_one(self, resolver):
        with resolver.foreign_tenant("a"):
            resolver.open("a").index.put("t2-only", skill_md("t2-only", "The other tenant."),
                                         level="tenant")
        storage = resolver.open("a")
        with resolver.bind("a"):
            assert storage.index.get("t2-only") is None
            assert storage.index.get("t2-only") == storage.index.get("missing")
            assert storage.index.read_body("t2-only") is None
            assert storage.index.delete("t2-only") is False
            assert "t2-only" not in [r["name"] for r in storage.index.list()]
        with resolver.foreign_tenant("a"):  # while the other tenant still holds it
            assert resolver.open("a").index.get("t2-only") is not None


__all__ = [
    "ALL_READS",
    "ALL_WRITES",
    "NAMES",
    "Backend",
    "ResolutionConformance",
    "SkillsConformance",
    "StoragePinRefused",
    "World",
    "make_world",
    "observe",
    "skill_md",
]
