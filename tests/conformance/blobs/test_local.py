"""The blob conformance suite against the local backend (today's files under the profile home).

Agents are profile homes under ``tmp_path``. The local store holds the agent level only (the
documented local exception), so the participant-level invariants run elsewhere, against the
two-level reference backend (``test_reference.py``); here the level-refusal conformance proves
a participant write is refused instead.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_blob_port_local import LocalBlobStore, LocalBlobStoreProvider
from hermes_conformance.blobs import Backend, BlobLevelRefusalConformance, PinningConformance
from hermes_scope import Group, Principal, Scope


def local_backend(tmp_path, monkeypatch):
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir()

    def open_store(agent):
        return LocalBlobStoreProvider().open(homes[agent])

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    def key_path(agent, key):
        store = LocalBlobStore(homes[agent])
        _root, path = store._locate(key)
        return path

    return Backend(open=open_store, bind=lambda agent: bind_profile_scope(homes[agent]),
                   foreign_tenant=lambda agent: bind_state(
                       Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"),
                             can_administer=False)),
                   unbound=unbound, key_path=key_path)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.blobs.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield local_backend(tmp_path, monkeypatch)


class TestLocalBlobs(PinningConformance):
    pass


class TestLocalBlobLevelRefusal(BlobLevelRefusalConformance):
    pass
