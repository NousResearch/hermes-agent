"""The skills index conformance suite against the local backend (today's per-profile ``skills/``
directory), and its resolution part against the suite's all-levels reference backend.

Agents are profile homes under ``tmp_path``. The local backend stores the agent level only, so
the suite's level tests exercise its refusals here; the resolution order (agent → owner-wide →
tenant-wide) is exercised against the reference backend, which stores every level. A plugin
storing every level runs both parts against its own backend.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_conformance.skills import (
    Backend, ResolutionConformance, SkillsConformance, reference_backend,
)
from hermes_scope import Group, Principal, Scope
from hermes_skills_port_local import LocalSkillsIndexProvider


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`: one profile home per agent."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)

    def open(agent):
        return LocalSkillsIndexProvider().open(homes[agent] / "skills")

    @contextmanager
    def unbound():
        # No scope: a local index's pin is the ambient profile, here one that owns no index.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"),
                                owner=Group("g"), can_administer=False))

    return Backend(open=open, bind=lambda agent: bind_profile_scope(homes[agent]),
                   foreign_tenant=foreign_tenant, unbound=unbound, levels=frozenset({"agent"}))


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.skills.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


@pytest.fixture
def resolver(monkeypatch):
    from tests.conformance.skills.mutants import apply_mutant
    apply_mutant(monkeypatch)
    yield reference_backend()


class TestLocalSkills(SkillsConformance):
    pass


class TestReferenceResolution(ResolutionConformance):
    pass
