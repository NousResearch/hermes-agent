"""The extension registry conformance suite against the local backend.

Agents are profile homes under ``tmp_path``; each profile's registry is its own
``extension_registry.json``. The local backend holds one agent per file and one owner (the
local person), so it cannot express several agents of one owner, another principal of the
tenant, or a group's agents: the ownership-level rules run against the reference backend
(``test_reference.py``), and what this fixture covers is listed in
``hermes_conformance.extensions.suite``.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_conformance.extensions import (
    AdminConformance, Backend, PinningConformance, SwitchConformance,
)
from hermes_extension_registry_port_local import LocalExtensionRegistryProvider
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, Principal, Scope, push_state


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`: two profile homes, and a third for the ambient pin."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)

    def connect(agent):
        return LocalExtensionRegistryProvider().open(homes[agent])

    @contextmanager
    def unbound():
        # No scope: a local registry's pin is the ambient profile, here one that owns no registry.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    # Variant scopes bind the scope the way a routed activity binds it (``push_state``): only the
    # pin is under test, so no profile state is rebuilt around it.
    backend = Backend(
        open=connect,
        bind=lambda agent: bind_profile_scope(homes[agent]),
        bind_noadmin=lambda agent: push_state(scope=Scope(
            LOCAL_TENANT, str(homes[agent]), LOCAL_OWNER, owner=LOCAL_OWNER, can_administer=False)),
        other_owner=lambda agent: push_state(scope=Scope(
            LOCAL_TENANT, str(homes[agent]), LOCAL_OWNER, owner=Principal("rival"), can_administer=True)),
        # A hosted tenant naming this very path as its agent: only the tenant differs (the owner
        # is the same person), so the refusal under test is the tenant's alone.
        foreign_tenant=lambda agent: push_state(scope=Scope(
            "acme", str(homes[agent]), Principal("u"), owner=LOCAL_OWNER, can_administer=True)),
        unbound=unbound,
    )
    return backend


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.extensions.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


class TestLocalPinning(PinningConformance):
    pass


class TestLocalAdmin(AdminConformance):
    pass


class TestLocalSwitch(SwitchConformance):
    pass
