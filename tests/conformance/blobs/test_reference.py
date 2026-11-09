"""The blob conformance suite against a two-level reference backend.

Today's local backend stores the agent level only, so the suite's participant-level
invariants — a member of a group agent cannot read another member's conversation file, the
participant level resolves first, members still read the agent's own files — run against this
hosted-shaped reference backend (``reference.py``), which stores both levels with the
ownership key prefixed onto the key. Conversation files are session-owned data, so their
writes run under the writer's claimed turn, and the reference backend checks the turn's fence
against the session's lease (``reference.SessionLeases``) inside the write: the fixture's
``claim`` / ``turn`` / ``supersede`` hooks drive that lease the way the turn-admission port
drives a hosted one.
"""

from contextlib import contextmanager, nullcontext

import pytest

from agent.runtime_scope import bind_state, bind_turn, outlive_turn
from hermes_conformance.blobs import Backend, BlobConformance
from hermes_scope import Group, Principal, Scope, System
from tests.conformance.blobs.reference import ReferenceBlobStore, SessionLeases
from tests.conformance.blobs.reference import key_path as reference_key_path

TENANT = "t1"


def reference_backend(tmp_path):
    root = tmp_path / "blobs"
    leases: dict[str, SessionLeases] = {}

    def leases_of(agent):
        return leases.setdefault(agent, SessionLeases())

    @contextmanager
    def unbound():
        # A hosted-style backend serves no ambient profile: with nothing bound, every
        # operation must refuse rather than fall back to a default.
        yield

    @contextmanager
    def turn(agent, member, session, fence, *, outlives=False):
        # A member's claimed turn: the scope a conversation-file write runs under, bound as
        # the turn's TurnScope (outlives: work designed to outlive it, keeping the fence's
        # owner-epoch half).
        with bind_state(Scope(TENANT, agent, Principal(member), owner=Group("g1"), can_administer=False)):
            with bind_turn(session, "t1", fence):
                with outlive_turn() if outlives else nullcontext():
                    yield

    def supersede(agent, session, how):
        lease = leases_of(agent)
        if how == "owner_epoch":
            lease.take_over()
        elif how == "next_claim":
            lease.claim(session)
        else:
            raise ValueError(f"unknown lease move {how!r}")

    return Backend(
        open=lambda agent: ReferenceBlobStore(root, TENANT, agent, leases_of(agent)),
        bind=lambda agent: bind_state(
            Scope(TENANT, agent, Principal("owner"), owner=Principal("owner"), can_administer=True)),
        bind_member=lambda agent, member: bind_state(
            Scope(TENANT, agent, Principal(member), owner=Group("g1"), can_administer=False)),
        bind_system=lambda agent: bind_state(
            Scope(TENANT, agent, System("cron"), owner=Group("g1"), can_administer=False)),
        foreign_tenant=lambda agent: bind_state(
            Scope("acme", agent, Principal("u"), owner=Group("g"), can_administer=False)),
        unbound=unbound,
        claim=lambda agent, session: leases_of(agent).claim(session),
        turn=turn,
        supersede=supersede,
        key_path=lambda agent, key: reference_key_path(root, TENANT, agent, key),
    )


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.blobs.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    yield reference_backend(tmp_path)


class TestReferenceBlobs(BlobConformance):
    pass
