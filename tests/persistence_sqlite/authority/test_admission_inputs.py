"""Input validation of the SQLite admission verbs, with the legacy store's behaviour: every refusal writes nothing,
and values the legacy store accepts are accepted."""

from __future__ import annotations

import sqlite3
from collections import UserDict
from types import MappingProxyType

import pytest

from persistence import FenceStale, PermanentStoreError
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._support import dump
from tests.persistence_sqlite.authority._admission_support import EPOCH, admit, claim


@pytest.mark.parametrize(
    "session_id,payload_factory,stale,reason",
    [
        pytest.param(42, dict, False, "invalid_params", id="integer-id"),
        pytest.param(None, dict, False, "invalid_params", id="none-id"),
        pytest.param(True, dict, False, "invalid_params", id="boolean-id"),
        pytest.param(4.2, dict, False, "invalid_params", id="float-id"),
        pytest.param(b"42", dict, False, "invalid_params", id="bytes-id"),
        pytest.param(["42"], dict, False, "invalid_params", id="list-id"),
        pytest.param({"id": "42"}, dict, False, "invalid_params", id="dict-id"),
        pytest.param(42, dict, True, "invalid_params", id="integer-before-stale-epoch"),
        pytest.param("42", dict, True, "stale_epoch", id="string-stale-epoch"),
        pytest.param("42", UserDict, False, "invalid_params", id="user-dict-payload"),
        pytest.param("42", MappingProxyType, False, "invalid_params", id="mapping-proxy-payload"),
    ],
)
def test_admit_preserves_input_validation(seeded, session_id, payload_factory, stale, reason):
    path, store, _ = seeded(lambda db, epoch: db.create_session(str(session_id), source="test"))
    before = dump(path)
    error_type = FenceStale if reason == "stale_epoch" else AuthorityRefusal
    with pytest.raises(error_type) as refused:
        store.admissions.admit(epoch=0 if stale else EPOCH, principal_id="owner", session_id=session_id,
                               request_id="request", payload=payload_factory({"text": "hello"}))
    assert refused.value.reason == reason
    assert dump(path) == before
    assert store.admissions.list(session_id=str(session_id), pending_only=False) == []


@pytest.mark.parametrize("result_factory", [UserDict, MappingProxyType, dict],
                         ids=["user-dict", "mapping-proxy", "ordinary-dict"])
def test_settle_accepts_a_mapping_result_as_the_legacy_store_does(seeded, result_factory):
    # The legacy store copies a non-dict mapping with dict() in compact_result and settles.
    def seed(db, epoch):
        db.create_session("session", source="test")
        admit(db, "session", "request", principal_id="owner", payload={"text": "hello"})
        return claim(db, "session")

    path, store, claimed = seeded(seed)
    before = dump(path)
    settled = store.admissions.settle(epoch=EPOCH, admission_id=claimed["admission_id"],
                                      generation=claimed["generation"], outcome="completed",
                                      result=result_factory({"text": "finished"}))
    assert (settled.status, settled.outcome) == ("terminal", "completed")
    assert store.admissions.get(admission_id=claimed["admission_id"]) == settled
    saved = store.admissions.settled_result(admission_id=claimed["admission_id"])
    assert saved.record == settled and saved.result == {"text": "finished"}
    assert dump(path) != before


def _textual_ids(db, epoch):
    for sid in ("1", "True", "None"):
        db.create_session(sid, source="cli")
        row = admit(db, sid, sid, payload={"sid": sid})
        if sid == "None":
            db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET admission_id='None' WHERE admission_id=?", (row["admission_id"],)))


@pytest.mark.parametrize("stale", [False, True])
def test_admission_payload_refusal_comes_before_the_epoch(seeded, stale):
    # The legacy store refuses invalid_params before any read.
    path, store, _ = seeded(_textual_ids)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.admit(epoch=0 if stale else EPOCH, principal_id="p", session_id="1", request_id="new",
                               payload=[])
    assert refused.value.reason == "invalid_params"
    assert dump(path) == before
    assert len(store.admissions.list(session_id="1")) == 1


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("outcome,result", [("completed", []), ("cancelled", {})])
def test_settlement_validation_matches_the_legacy_store(seeded, stale, outcome, result):
    # The legacy store refuses a cancelled outcome invalid_params
    # before the epoch, and settles a list result as the empty dict compact_result makes of it.
    def seed(db, epoch):
        _textual_ids(db, epoch)
        return claim(db, "1")

    path, store, current = seeded(seed)
    before = dump(path)
    call = lambda: store.admissions.settle(epoch=0 if stale else EPOCH, admission_id=current["admission_id"],
                                           generation=current["generation"], outcome=outcome, result=result)
    if outcome == "completed" and not stale:
        assert call().status == "terminal"
        assert store.admissions.settled_result(admission_id=current["admission_id"]).result == {}
        return
    with pytest.raises(AuthorityRefusal) as refused:
        call()
    assert refused.value.reason == ("invalid_params" if outcome == "cancelled" else "stale_epoch")
    assert dump(path) == before
    assert store.admissions.get(admission_id=current["admission_id"]).status == "started"


def test_invalid_string_intent_is_refused_before_any_read(seeded):
    # The legacy store takes the intent string and refuses an unknown one.
    path, store, _ = seeded(_textual_ids)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="1", request_id="new", payload={},
                               intent="bad")
    assert refused.value.reason == "invalid_params"
    assert dump(path) == before


@pytest.mark.parametrize("verb", ["get", "cancel"])
def test_none_does_not_address_the_textual_admission_id(seeded, verb):
    _path, store, _ = seeded(_textual_ids)
    kwargs = {"admission_id": None}
    if verb == "cancel":
        kwargs["epoch"] = EPOCH
    with pytest.raises(TypeError, match="can only concatenate str"):
        getattr(store.admissions, verb)(**kwargs)
    assert store.admissions.get(admission_id="None").status == "queued"


def test_unsupported_lookup_retains_the_sqlite_refusal(seeded):
    _path, store, _ = seeded(_textual_ids)
    with pytest.raises(PermanentStoreError) as caught:
        store.admissions.get(admission_id=[])
    assert isinstance(caught.value.__cause__, sqlite3.ProgrammingError)
    assert str(caught.value.__cause__) == "Error binding parameter 1: type 'list' is not supported"


def test_noncontiguous_lookup_buffer_is_not_stringified(seeded):
    _path, store, _ = seeded(_textual_ids)
    with pytest.raises(BufferError, match="memoryview: underlying buffer is not C-contiguous"):
        store.admissions.list(session_id=memoryview(b"123")[::2], pending_only=False)


def test_numeric_session_lookup_uses_sqlite_affinity(seeded):
    _path, store, _ = seeded(_textual_ids)
    assert [row.target_session_id for row in store.admissions.list(session_id=True)] == ["1"]
    assert [row.target_session_id for row in store.admissions.list(session_id="True")] == ["True"]
