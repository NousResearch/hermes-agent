"""The kanban conformance suite against the local backend (today's ``kanban.db`` boards).

Agents are kanban roots under ``tmp_path`` (today's ``HERMES_KANBAN_HOME``), each with its own
profile home to bind. Opening a store points the environment at that agent's root and leaves it
there: a store serves the root it was opened for, never the ambient one, and the suite relies
on the leftover environment to catch a backend that re-resolves per call. A replica is a second
store on the same root (another process's handle). The suite's list of invariants it does NOT
cover applies here too (``hermes_conformance.kanban.suite``).
"""

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_conformance.kanban import Backend, IsolationConformance
from hermes_kanban_port_local import LocalKanbanStorageProvider
from hermes_scope import Group, Principal, Scope


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`."""
    roots = {name: tmp_path / f"kanban-{name}" for name in ("a", "b")}
    homes = {name: tmp_path / f"home-{name}" for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    provider = LocalKanbanStorageProvider()
    # A worker of this test process carries no board pin; the roots above are all there is.
    for pin in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(pin, raising=False)

    def open_store(agent):
        # Today's root resolution, once per open: the environment names this agent's root and
        # stays wherever the last open pointed.
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(roots[agent]))
        return provider.open().kanban

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"),
                                can_administer=False))

    def unbound():
        import contextlib
        return contextlib.nullcontext()

    return Backend(open=open_store, replica=open_store,
                   bind=lambda agent: bind_profile_scope(homes[agent]),
                   foreign_tenant=foreign_tenant, unbound=unbound, serves_unbound=True)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.kanban.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


# The suite's two parts run as separate modules (pinning in test_local_pinning.py) so each file
# stays well inside the per-file runner budget.
class TestLocalIsolation(IsolationConformance):
    pass
