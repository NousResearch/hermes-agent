"""``return_claim`` on SQLite: it undoes a claim to the exact ledger the claim's gate refusal leaves, replays
idempotently, refuses every stale claim token with ``stale_generation``, and races the next owner's handover
(``begin_epoch`` then ``recover`` on another thread) to one of the two outcomes the handover allows."""

from __future__ import annotations

import threading

import pytest

from persistence import FenceStale
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._support import copy_db, counted_transactions, dump
from tests.persistence_sqlite.authority._admission_support import (
    EPOCH, admit, assert_rolls_back, begin_epoch, claim, get, record_dict, recover, register, settle, steer,
)


def _claimed(db, epoch):
    """s1 holds r1 (queued under epoch 1, then claimed under epoch 2 by the same owner restarted) and r2 queued."""
    db.create_session("s1", source="cli")
    first = admit(db, "s1", "r1")
    second = admit(db, "s1", "r2")
    current = begin_epoch(db, "owner-2")
    queued = get(db, first["admission_id"])
    before = dump(db.db_path)
    claimed = claim(db, "s1", epoch=current)
    return {"epoch": current, "queued": queued, "claimed": claimed, "second": second, "before": before}


def _return(store, v, **overrides):
    kwargs = dict(epoch=v["epoch"], admission_id=v["claimed"]["admission_id"], generation=v["claimed"]["generation"],
                  queued_owner_epoch=v["queued"]["owner_epoch"])
    kwargs.update(overrides)
    return store.admissions.return_claim(**kwargs)


def test_return_claim_leaves_the_ledger_a_refused_claim_leaves(seeded):
    path, store, v = seeded(_claimed)
    assert v["queued"]["owner_epoch"] == EPOCH and v["claimed"]["owner_epoch"] == v["epoch"] == EPOCH + 1
    returned = _return(store, v)
    assert record_dict(returned) == v["queued"]
    assert returned.seq == v["queued"]["seq"] and returned.generation is None
    assert dump(path) == v["before"]


def test_return_claim_replays_a_returned_row_unchanged(seeded):
    path, store, v = seeded(_claimed)
    first = _return(store, v)
    after = dump(path)
    assert _return(store, v) == first
    assert dump(path) == after
    # A row that was never claimed is already queued with no generation: the same answer.
    second = v["second"]
    assert record_dict(_return(store, v, admission_id=second["admission_id"], queued_owner_epoch=None)) == second


def _settled(db, v):
    settle(db, v["claimed"], epoch=v["epoch"])


def _recovered(db, v):
    _next_owner(db, v)
    recover(db, v["epoch"])


def _next_owner(db, v):
    v["epoch"] = begin_epoch(db, "owner-3")


def _session_moved(db, v):
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id='s1'"))


def _worker_bound(db, v):
    register(db, "s1", v["claimed"]["generation"], epoch=v["epoch"])


def _steer_bound(db, v):
    steer(db, v["second"], v["claimed"], epoch=v["epoch"])


#: How each stale case moves the store past the claim token before ``return_claim`` runs.
_STALE = {"terminal": _settled, "unknown": _recovered, "other_owner": _next_owner, "session_generation": _session_moved,
          "worker": _worker_bound, "steer": _steer_bound}


def _stale(db, v, how):
    if how in _STALE:
        _STALE[how](db, v)


@pytest.mark.parametrize("how,overrides", [
    ("terminal", {}), ("unknown", {}), ("other_owner", {}), ("session_generation", {}), ("worker", {}), ("steer", {}),
    ("generation", {"generation": 2}), ("generation_text", {"generation": "1"}), ("generation_none", {"generation": None}),
])
def test_return_claim_refuses_every_stale_claim_token(seeded, how, overrides):
    def seed(db, epoch):
        v = _claimed(db, epoch)
        _stale(db, v, how)
        return v

    path, store, v = seeded(seed)
    before = dump(path)
    with pytest.raises(FenceStale) as refused:
        _return(store, v, **overrides)
    assert refused.value.reason == "stale_generation"
    assert dump(path) == before


@pytest.mark.parametrize("overrides,reason", [({"epoch": EPOCH}, "stale_epoch"), ({"epoch": "2"}, "stale_epoch"),
                                              ({"admission_id": "absent"}, "not_found")])
def test_return_claim_checks_the_epoch_then_the_row(seeded, overrides, reason):
    path, store, v = seeded(_claimed)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        _return(store, v, **overrides)
    assert refused.value.reason == reason
    assert dump(path) == before


def test_return_claim_is_one_write_and_rolls_back(seeded):
    path, store, v = seeded(_claimed)
    assert_rolls_back(path, store.holder(), lambda: _return(store, v))
    with counted_transactions(store.holder()) as counts:
        _return(store, v)
    assert counts == {"write": 1, "read": 0}


# ── the handover race ──────────────────────────────────────────────────────────


def _handover(store, results):
    """The next owner on its own thread: a new epoch, then boot recovery."""
    epoch = store.runtime.begin_epoch(instance_id="next-owner")
    results["recovered"] = store.admissions.recover(epoch=epoch)


def _returner(store, v, results):
    try:
        results["returned"] = _return(store, v)
    except Exception as exc:  # the outcome under test
        results["returned"] = exc


def _race(store, v, first=None):
    """Run the return and the handover on two threads; ``first`` names the one that commits before the other starts
    (``None``: both start together)."""
    results, gate, start = {}, threading.Event(), threading.Barrier(2)

    def run(name, fn):
        start.wait()
        if first is not None and name != first:
            gate.wait(30)
        try:
            fn()
        finally:
            if name == first:
                gate.set()

    threads = [threading.Thread(target=run, args=("return", lambda: _returner(store, v, results))),
               threading.Thread(target=run, args=("handover", lambda: _handover(store, results)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads)
    return results, store.admissions.get(admission_id=v["claimed"]["admission_id"])


def _assert_returned_first(results, row, v):
    assert record_dict(results["returned"]) == v["queued"]
    assert results["recovered"] == 0
    assert (row.status, row.seq, row.owner_epoch, row.generation) == (
        "queued", v["queued"]["seq"], v["queued"]["owner_epoch"], None)


def _assert_handed_over_first(results, row, v):
    assert isinstance(results["returned"], FenceStale) and results["returned"].reason == "stale_epoch"
    assert results["recovered"] == 1
    assert (row.status, row.owner_epoch, row.generation) == ("unknown", v["epoch"], v["claimed"]["generation"])


def test_return_claim_before_the_handover_requeues_the_input(seeded):
    _path, store, v = seeded(_claimed)
    results, row = _race(store, v, first="return")
    _assert_returned_first(results, row, v)


def test_handover_before_return_claim_marks_the_claim_unknown(seeded):
    _path, store, v = seeded(_claimed)
    results, row = _race(store, v, first="handover")
    _assert_handed_over_first(results, row, v)


def test_return_claim_racing_the_handover_lands_one_of_the_two_outcomes(seeded):
    path, store, v = seeded(_claimed)
    seed = path.with_name("race-seed.db")
    copy_db(path, seed)
    seen = set()
    for _ in range(20):
        copy_db(seed, path)
        results, row = _race(store, v)
        if isinstance(results["returned"], Exception):
            _assert_handed_over_first(results, row, v)
            seen.add("handover")
        else:
            _assert_returned_first(results, row, v)
            seen.add("return")
    assert seen
