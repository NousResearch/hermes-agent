"""``AdmissionRepository`` verbs that reach the conversation family's bodies (the retired-session tombstones, the runtime
lineage walk, the discarded-turn closer's transcript write): one transaction per verb that a failure rolls back, the
owner gate raised right after the epoch fence and before every other refusal, an exact replay of a resolved row, and the
stored id, payload and intent."""

from __future__ import annotations

import time

import pytest

from persistence.authority.admissions import ApiTargetClaim, GateRefusal, QueuedReplacement, TurnBoundary
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._support import counted_transactions, dump
from tests.persistence_sqlite.authority._admission_support import (
    EPOCH, admit, assert_rolls_back, begin_epoch, claim, recover, settle,
)

DRAINING = GateRefusal("runtime_draining")


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)


# ── seeds ──────────────────────────────────────────────────────────────────────


def _ledger(db, epoch):
    db.create_session("s1", source="cli")
    db.create_session("s2", source="cli")
    first = admit(db, "s1", "r1")
    second = admit(db, "s1", "r2")
    db.create_session("gone", source="cli")
    retired = admit(db, "gone", "rg", principal_id="api", payload={"text": "gone"})
    settle(db, claim(db, "gone"))
    db.delete_session("gone")
    return {"first": first, "second": second, "retired": retired}


# ── admit ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("extra", ["replaces_started", "supersedes_missing", "api_target_conflict"])
def test_admit_gate_comes_before_every_extra(seeded, extra):
    def seed(db, epoch):
        values = _ledger(db, epoch)
        values["started"] = claim(db, "s1")
        admit(db, "s2", "api-req", principal_id="api", payload={"text": "a"})
        return values

    path, store, v = seeded(seed)
    extras = {"replaces_started": {"replaces": QueuedReplacement(v["started"]["admission_id"])},
              "supersedes_missing": {"supersedes": "absent"},
              "api_target_conflict": {"api_target": ApiTargetClaim(settings={}, client_key=True)}}
    request_id = "api-req" if extra == "api_target_conflict" else "fresh"
    before = dump(path)
    for gate, reason in ((DRAINING, "runtime_draining"),
                         (None, {"replaces_started": "admission_started", "supersedes_missing": "stale_generation",
                                 "api_target_conflict": "admission_conflict"}[extra])):
        with pytest.raises(AuthorityRefusal) as refused:
            store.admissions.admit(epoch=EPOCH, principal_id="api", session_id="s1", request_id=request_id,
                                   payload={"text": "fresh"}, gate=gate, **extras[extra])
        assert refused.value.reason == reason
        assert dump(path) == before


def test_admit_reads_the_tombstones_then_writes_once(seeded):
    path, store, _ = seeded(_ledger)
    call = lambda: store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="s1", request_id="r5",
                                          payload={"text": "r5"})
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 1}


# ── claim ──────────────────────────────────────────────────────────────────────


def _claim_seed(db, epoch):
    values = _ledger(db, epoch)
    db.create_session("s3", source="cli")
    admit(db, "s3", "lost")
    claim(db, "s3")
    db.create_session("empty", source="cli")
    values["epoch"] = begin_epoch(db, "owner-2")
    recover(db, values["epoch"])
    return values


def test_claim_gate_comes_right_after_the_epoch_fence(seeded):
    path, store, v = seeded(_claim_seed)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.claim(epoch=EPOCH, session_id="s1", gate=DRAINING)
    assert refused.value.reason == "stale_epoch"
    with pytest.raises(AuthorityRefusal) as refused:  # the session is never read: not not_found
        store.admissions.claim(epoch=v["epoch"], session_id="absent", gate=DRAINING)
    assert refused.value.reason == "runtime_draining"
    with pytest.raises(AuthorityRefusal) as refused:  # nor the FIFO: not unknown_execution
        store.admissions.claim(epoch=v["epoch"], session_id="s3", gate=DRAINING)
    assert refused.value.reason == "runtime_draining"
    assert dump(path) == before


def test_claim_is_one_write_and_rolls_back(seeded):
    path, store, v = seeded(_claim_seed)
    call = lambda: store.admissions.claim(epoch=v["epoch"], session_id="s1")
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        assert call().admission_id == v["first"]["admission_id"]
    assert counts == {"write": 1, "read": 0}


# ── settle with the discarded-turn closer, resolve_unknown ─────────────────────


def _open_turn(db, epoch):
    values = _ledger(db, epoch)
    db.append_message("s1", role="user", content="r1", timestamp=1_700_000_000.0)
    values["claimed"] = claim(db, "s1")
    db.append_message("s1", role="assistant", content="Send it again", display_kind="failed_turn",
                      timestamp=1_700_000_001.0)
    return values


def _unknown(db, epoch):
    values = _open_turn(db, epoch)
    values["epoch"] = begin_epoch(db, "owner-2")
    recover(db, values["epoch"])
    return values


def test_resolve_unknown_replays_a_resolved_row_and_is_one_write(seeded, frozen_clock):
    path, store, v = seeded(_unknown)
    row, epoch = v["claimed"], v["epoch"]
    call = lambda: store.admissions.resolve_unknown(epoch=epoch, admission_id=row["admission_id"],
                                                    generation=row["generation"], notice="n")
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        resolved = call()
    assert counts == {"write": 1, "read": 0}
    after = dump(path)
    assert call() == resolved
    assert dump(path) == after


def test_settle_with_a_boundary_is_one_write_and_rolls_back(seeded, frozen_clock):
    path, store, v = seeded(_open_turn)
    row = v["claimed"]
    call = lambda: store.admissions.settle(
        epoch=EPOCH, admission_id=row["admission_id"], generation=row["generation"], outcome="interrupted",
        boundary=TurnBoundary(notice="Stopped"))
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 0}


# ── input cases that reach the conversation bodies ─────────────


def test_admit_accepts_a_textual_id_and_dict_payload(seeded):
    path, store, _ = seeded(lambda db, epoch: db.create_session("42", source="test"))
    before = dump(path)
    admitted, created = store.admissions.admit(epoch=EPOCH, principal_id="owner", session_id="42",
                                               request_id="request", payload={"text": "hello"})
    assert created and admitted.status == "queued" and admitted.target_session_id == "42"
    assert admitted.payload == {"text": "hello"}
    assert store.admissions.list(session_id="42", pending_only=False) == [admitted]
    assert dump(path) != before


@pytest.mark.parametrize("intent", ["queue", "steer", "redirect"])
def test_admit_takes_the_stored_intent_strings(seeded, intent):
    # The intent is stored as the string given.
    _path, store, _ = seeded(lambda db, epoch: db.create_session("1", source="cli"))
    admitted, _ = store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="1", request_id="new", payload={},
                                         intent=intent)
    assert admitted.intent == intent
