"""The blob conformance suite against a two-level reference backend.

Today's local backend stores the agent level only, so the suite's participant-level
invariants — a member of a group agent cannot read another member's conversation file, the
participant level resolves first, members still read the agent's own files — run against this
hosted-shaped reference backend (``reference.py``), which stores both levels with the
ownership key prefixed onto the key.
"""

from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_state
from hermes_conformance.blobs import Backend, BlobConformance
from hermes_scope import Group, Principal, Scope, System
from tests.conformance.blobs.reference import ReferenceBlobStore, key_path as reference_key_path

TENANT = "t1"


def reference_backend(tmp_path):
    root = tmp_path / "blobs"

    @contextmanager
    def unbound():
        # A hosted-style backend serves no ambient profile: with nothing bound, every
        # operation must refuse rather than fall back to a default.
        yield

    return Backend(
        open=lambda agent: ReferenceBlobStore(root, TENANT, agent),
        bind=lambda agent: bind_state(
            Scope(TENANT, agent, Principal("owner"), owner=Principal("owner"), can_administer=True)),
        bind_member=lambda agent, member: bind_state(
            Scope(TENANT, agent, Principal(member), owner=Group("g1"), can_administer=False)),
        bind_system=lambda agent: bind_state(
            Scope(TENANT, agent, System("cron"), owner=Group("g1"), can_administer=False)),
        foreign_tenant=lambda agent: bind_state(
            Scope("acme", agent, Principal("u"), owner=Group("g"), can_administer=False)),
        unbound=unbound,
        key_path=lambda agent, key: reference_key_path(root, TENANT, agent, key),
    )


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.blobs.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield reference_backend(tmp_path)


class TestReferenceBlobs(BlobConformance):
    pass
