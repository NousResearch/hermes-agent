"""The store a turn writes to is resolved once, when the turn's handle is first used, and reused for every write.

A hosted scope's turn-path writes (the transcript flush, under the claimed turn's fence) reach the pinned store of the
configured provider and no local ``state.db``, not even one at the ambient home. Once bound, the handle never asks the
assembly again: neither a later change of the ambient profile nor an assembly that could no longer build the backend
moves or fails the turn's writes, and the gateway authority claims through the handle it was built with.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import persistence_assembly as pa
from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn, claim_fence
from gateway.config import Platform
from hermes_scope import Principal, Scope
from persistence import TurnFenceSuperseded
from persistence.conversation.sessions import NewSession
from tests.contracts._reference import InMemoryBackend
from tests.contracts._scopes import pinned_scope
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority
from tests.hosted_persistence._turn_rows import flush
from tests.hosted_persistence.test_turn_fence_lifetime import S, _contents, _turn, claimed


def _unassemblable(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("the turn's store was resolved again")


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryBackend]:
    """The configured provider: the in-memory reference backend (it serves pins); the process caches start empty."""
    backend = InMemoryBackend()
    monkeypatch.setattr(pa, "_assemble", lambda provider, stores, **_: backend)
    monkeypatch.setattr(pa, "_shared", {})
    monkeypatch.setattr(pa, "_pinned_backends", {})
    try:
        yield backend
    finally:
        backend.close()


@contextmanager
def hosted(tenant: str = "tenant-a", agent: str = "agent-x") -> Iterator[None]:
    with bind_state(Scope(tenant, agent, Principal("member"), owner=Principal("member"), can_administer=False)):
        yield


def _claimed_hosted_turn(store: Any) -> Any:
    store.sessions.create(NewSession(session_id=S, source="cli"))
    epoch = store.runtime.begin_epoch(instance_id="one")
    store.admissions.admit(epoch=epoch, principal_id="u", session_id=S, request_id="r1", payload={})
    return store.admissions.claim(epoch=epoch, session_id=S)


def test_a_hosted_turns_flush_reaches_its_pinned_store_and_no_local_store(provider, monkeypatch) -> None:
    home = Path(os.environ["HERMES_HOME"])
    with hosted():
        store = pa.open_persistence(pa.profile_scope())
        row = _claimed_hosted_turn(store)
        agent = SimpleNamespace(persistence=store, _session_db=store, session_id=S)
        with bind_turn(S, row.admission_id, claim_fence(row), store=store.scope):
            flush(agent, "hosted reply")
            # Bound once: an assembly that could no longer build the backend does not reach the turn's writes.
            monkeypatch.setattr(pa, "_assemble", _unassemblable)
            monkeypatch.setattr(pa, "open_persistence", _unassemblable)
            flush(agent, "second reply")
            store.runtime.begin_epoch(instance_id="two")
            with pytest.raises(TurnFenceSuperseded):
                flush(agent, "superseded reply")
        store.close()
    pinned = provider.open(pinned_scope("tenant-a", "agent-x"))
    assert [m.content for m in pinned.messages.read_transcript(S)] == ["hosted reply", "second reply"]
    assert not (home / "state.db").exists()
    assert not list(home.rglob("state.db"))


def test_a_turns_writes_stay_on_its_store_when_the_ambient_profile_moves(claimed, tmp_path, monkeypatch) -> None:
    store, _epoch, row = claimed
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    agent = SimpleNamespace(persistence=store, _session_db=store, session_id=S)
    with _turn(store, row):
        flush(agent, "first")
        monkeypatch.setattr(pa, "_assemble", _unassemblable)
        monkeypatch.setattr(pa, "open_persistence", _unassemblable)
        with bind_profile_scope(elsewhere):
            flush(agent, "after the ambient profile moved")
    assert _contents(store) == ["first", "after the ambient profile moved"]
    assert not (elsewhere / "state.db").exists()


@pytest.mark.asyncio
async def test_the_authority_claims_through_the_handle_it_was_built_with(tmp_path, monkeypatch) -> None:
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    built = authority.persistence
    built.admissions.admit(epoch=authority.epoch, principal_id="u", session_id="s", request_id="r1", payload={"text": "hi"})
    monkeypatch.setattr(pa, "_assemble", _unassemblable)
    monkeypatch.setattr(pa, "open_persistence", _unassemblable)
    monkeypatch.setattr(pa, "profile_scope", _unassemblable)
    monkeypatch.setattr("gateway.session_api_turn.check_api_turn", lambda *_a: None)
    with db:
        row, first = await authority._claim_next(REF, authority.sessions["s"])
        assert first is not None and row and row["admission_id"] == first["admission_id"]
        assert built.admissions.get(admission_id=first["admission_id"]).status == "started"
        assert authority.persistence is built
