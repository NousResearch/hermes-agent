"""The cron conformance suite against the local backend (today's per-profile ``jobs.json`` and
``executions.db``).

Agents are profile homes under ``tmp_path``; a replica is a second store handle on the same home
(another process's view of the same files). The suite's list of invariants it does NOT cover
applies here too (``hermes_conformance.cron.suite``): the local store holds the agent level
only, and the fixture cannot vary the owner or the administration right.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from cron_store_port_local import LocalCronStorageProvider
from hermes_conformance.cron import (
    Backend, ExecutionConformance, FireClaimConformance, TickConformance,
)
from hermes_scope import LOCAL_TENANT, Group, Principal, Scope


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)
    provider = LocalCronStorageProvider()

    def open_store(agent):
        return provider.open(homes[agent])

    @contextmanager
    def bind(agent, actor=None):
        home = homes[agent]
        if actor is None:
            with bind_profile_scope(home):
                yield
        else:  # a named principal acting on the same agent (the suite varies who creates a job)
            with bind_state(Scope(LOCAL_TENANT, str(home), Principal(actor))):
                yield

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"),
                                can_administer=False))

    return Backend(open=open_store, replica=open_store, bind=bind,
                   foreign_tenant=foreign_tenant, unbound=unbound)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.cron.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


# The suite's parts run as separate modules (pinning in test_local_pinning.py) so each file stays
# well inside the per-file runner budget.
class TestLocalTicks(TickConformance):
    pass


class TestLocalFires(FireClaimConformance):
    pass


class TestLocalExecutions(ExecutionConformance):
    pass
