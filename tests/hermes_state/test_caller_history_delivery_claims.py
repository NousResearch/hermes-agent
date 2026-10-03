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
