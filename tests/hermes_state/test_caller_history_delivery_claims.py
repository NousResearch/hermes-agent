"""Caller-history delegation delivery claims are exactly-once and releasable."""
from hermes_state import SessionDB


def _deliver(db, sid, deleg_id):
    db.append_delegation_delivery(sid, f"[ASYNC DELEGATION COMPLETE — {deleg_id}]", {"delegation_id": deleg_id})


def test_released_claim_is_claimable_again(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="api_server")
        _deliver(db, "sid", "deleg_a")
        claimed = db.claim_caller_history_deliveries("sid")
        assert [r["display_metadata"]["delegation_id"] for r in claimed] == ["deleg_a"]
        assert db.claim_caller_history_deliveries("sid") == []

        assert db.release_caller_history_deliveries("sid", [r["id"] for r in claimed]) == 1
        again = db.claim_caller_history_deliveries("sid")
        assert [r["id"] for r in again] == [r["id"] for r in claimed]
    finally:
        db.close()


def test_release_is_scoped_to_the_session_lineage(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for sid in ("mine", "other"):
            db.create_session(sid, source="api_server")
            _deliver(db, sid, f"deleg_{sid}")
        other_rows = db.claim_caller_history_deliveries("other")
        assert db.release_caller_history_deliveries("mine", [r["id"] for r in other_rows]) == 0
        assert db.claim_caller_history_deliveries("other") == []
        assert db.release_caller_history_deliveries("mine", []) == 0
    finally:
        db.close()


def _db_with_row(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="api_server")
    _deliver(db, "sid", "deleg_a")
    return db


def test_expired_reservation_is_reclaimable(tmp_path):
    db = _db_with_row(tmp_path)
    try:
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
    finally:
        db.close()


def test_commit_prevents_redelivery(tmp_path):
    db = _db_with_row(tmp_path)
    try:
        rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
        assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "intruder") == 0
        assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "webui") == 1
        assert db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60) == []
        assert db.claim_caller_history_deliveries("sid") == []
        assert db.release_caller_history_deliveries(reservation_token=rows[0]["reservation_token"], owner="webui") == 0
    finally:
        db.close()


def test_release_returns_row_and_commit_by_ids(tmp_path):
    db = _db_with_row(tmp_path)
    try:
        rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60)
        assert db.release_caller_history_deliveries(reservation_token=rows[0]["reservation_token"], owner="webui") == 1
        again = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60, limit=5)
        assert [r["id"] for r in again] == [rows[0]["id"]]
        assert db.commit_caller_history_deliveries([again[0]["id"]], "webui") == 1
    finally:
        db.close()


def test_fold_consumed_row_cannot_be_reserved(tmp_path):
    db = _db_with_row(tmp_path)
    try:
        assert len(db.claim_caller_history_deliveries("sid")) == 1
        assert db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=60) == []
    finally:
        db.close()


def test_token_holder_commits_after_its_lease_expired(tmp_path):
    db = _db_with_row(tmp_path)
    try:
        rows = db.reserve_caller_history_deliveries("sid", "webui", ttl_seconds=-1)  # lease already over
        assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "intruder") == 0
        assert db.commit_caller_history_deliveries(rows[0]["reservation_token"], "webui") == 1
        assert db.claim_caller_history_deliveries("sid") == []
    finally:
        db.close()
