"""``admit`` and ``claim`` on SQLite raise the owner's ``GateRefusal`` inside the write transaction (``admit`` after
the queue cap, ``claim`` right after the epoch fence): a refused write leaves the file unchanged, and an exact replay
never reaches the gate."""

from __future__ import annotations

import pytest

from persistence.authority.admissions import GateRefusal
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._support import dump
from tests.persistence_sqlite.authority._admission_support import EPOCH, admit

DRAINING = GateRefusal("runtime_draining")


def _queued(db, epoch):
    db.create_session("s1", source="cli")
    return admit(db, "s1", "r1")


def _admit(store, request_id, gate):
    return store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="s1", request_id=request_id,
                                  payload={"text": request_id}, gate=gate)


def test_admit_without_a_gate_admits(seeded):
    _path, store, _ = seeded(_queued)
    row, created = _admit(store, "r2", None)
    assert created and row.status == "queued"


def test_admit_refused_by_the_gate_writes_nothing(seeded):
    path, store, _ = seeded(_queued)
    before = dump(path)
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        _admit(store, "r2", DRAINING)
    assert dump(path) == before


def test_exact_replay_does_not_reach_the_gate(seeded):
    path, store, first = seeded(_queued)
    before = dump(path)
    row, created = _admit(store, "r1", DRAINING)
    assert not created and row.admission_id == first["admission_id"]
    assert dump(path) == before


def test_claim_refused_by_the_gate_writes_nothing(seeded):
    path, store, first = seeded(_queued)
    before = dump(path)
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        store.admissions.claim(epoch=EPOCH, session_id="s1", gate=DRAINING)
    assert dump(path) == before
    claimed = store.admissions.claim(epoch=EPOCH, session_id="s1", gate=None)
    assert claimed.admission_id == first["admission_id"] and claimed.status == "started"
