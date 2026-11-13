"""Caller-history delegation delivery claims are exactly-once and releasable."""
import pytest

from hermes_state import SessionDB


def _deliver(db, sid, deleg_id):
    db.append_delegation_delivery(sid, f"[ASYNC DELEGATION COMPLETE — {deleg_id}]", {"delegation_id": deleg_id})


@pytest.fixture
def db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="api_server")
        yield db
    finally:
        db.close()


def test_released_claim_is_claimable_again(db):
    _deliver(db, "sid", "deleg_a")
    claimed = db.claim_caller_history_deliveries("sid")
    assert [r["display_metadata"]["delegation_id"] for r in claimed] == ["deleg_a"]
    assert db.claim_caller_history_deliveries("sid") == []

    assert db.release_caller_history_deliveries("sid", [r["id"] for r in claimed]) == 1
    again = db.claim_caller_history_deliveries("sid")
    assert [r["id"] for r in again] == [r["id"] for r in claimed]


def test_release_is_scoped_to_the_session_lineage(db):
    for sid in ("mine", "other"):
        db.create_session(sid, source="api_server")
        _deliver(db, sid, f"deleg_{sid}")
    other_rows = db.claim_caller_history_deliveries("other")
    assert db.release_caller_history_deliveries("mine", [r["id"] for r in other_rows]) == 0
    assert db.claim_caller_history_deliveries("other") == []
    assert db.release_caller_history_deliveries("mine", []) == 0


def test_expired_reservation_is_reclaimable(db):
    _deliver(db, "sid", "deleg_a")
    first = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
    assert len(first) == 1 and first[0]["reservation_token"] and first[0]["delivery_id"] == first[0]["id"]
    assert first[0]["session_id"] == "sid" and "deleg_a" in first[0]["content"]
    assert db.reserve_caller_history_deliveries("sid", "other", ttl_seconds=60) == []
    assert db.claim_caller_history_deliveries("sid") == []  # fold skips live leases
    expired = db.reserve_caller_history_deliveries("sid", "webui2", ttl_seconds=-1)
    assert expired == []  # still held by first lease
    db.release_caller_history_deliveries(reservation_token=first[0]["reservation_token"], owner="webui")
    short = db.reserve_caller_history_deliveries("sid", "crashy", ttl_seconds=-1)  # already expired
    retry = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
    assert [r["id"] for r in retry] == [short[0]["id"]]
    assert db.commit_caller_history_deliveries(short[0]["reservation_token"], "crashy") == 0  # taken over


def test_commit_prevents_redelivery(db):
    _deliver(db, "sid", "deleg_a")
    rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
    assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "intruder") == 0
    assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "webui") == 1
    assert db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60) == []
    assert db.claim_caller_history_deliveries("sid") == []
    assert db.release_caller_history_deliveries(reservation_token=rows[0]["reservation_token"], owner="webui") == 0


def test_release_returns_row_and_commit_by_ids(db):
    _deliver(db, "sid", "deleg_a")
    rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
    assert db.release_caller_history_deliveries(reservation_token=rows[0]["reservation_token"], owner="webui") == 1
    again = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60, limit=5)
    assert [r["id"] for r in again] == [rows[0]["id"]]
    assert db.commit_caller_history_deliveries([again[0]["id"]], "webui") == 1


def test_fold_consumed_row_cannot_be_reserved(db):
    _deliver(db, "sid", "deleg_a")
    assert len(db.claim_caller_history_deliveries("sid")) == 1
    assert db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60) == []


def test_token_holder_commits_after_its_lease_expired(db):
    _deliver(db, "sid", "deleg_a")
    rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=-1)  # lease already over
    assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "intruder") == 0
    assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "webui") == 1
    assert db.claim_caller_history_deliveries("sid") == []


def test_rewound_delivery_row_is_neither_reserved_nor_claimed(db):
    """A row soft-deleted by a rewind (active=0) is no longer history: it must not be re-delivered."""
    _deliver(db, "sid", "deleg_old")
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET active = 0 WHERE session_id = 'sid'"))
    assert db.reserve_caller_history_deliveries("sid", "webui", 60) == []
    assert db.claim_caller_history_deliveries("sid") == []
