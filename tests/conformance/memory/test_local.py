"""The memory conformance suite against the local backend (today's ``MEMORY.md`` / ``USER.md``).

Agents are profile homes under ``tmp_path``; a replica is a second handle on the same profile's
files (another process's store). The suite's list of invariants it does NOT cover applies here too
(``hermes_conformance.memory.suite``): the local store holds the agent level only, and the fixture
cannot vary the participant or owner.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_conformance.memory import Backend, MemoryConformance
from hermes_memory_port_local import LocalMemoryStorageProvider
from hermes_scope import Group, Principal, Scope


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir()

    def connect(agent):
        return LocalMemoryStorageProvider().open(homes[agent])

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    def foreign_tenant(agent):
        """A scope of another tenant naming the same agent, with the ambient home naming the same
        profile too: only the pin's tenant half can refuse it (the agent and the home both match)."""
        @contextmanager
        def bound():
            with monkeypatch.context() as mp:
                mp.setenv("HERMES_HOME", str(homes[agent]))
                with bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"),
                                      can_administer=False)):
                    yield
        return bound()

    return Backend(open=connect, replica=connect, bind=lambda agent: bind_profile_scope(homes[agent]),
                   home=lambda agent: homes[agent], foreign_tenant=foreign_tenant, unbound=unbound)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.memory.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


class TestLocalMemory(MemoryConformance):
    pass
