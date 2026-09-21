import time
from contextlib import closing
import pytest
from hermes_state import SessionDB


def test_prune_sessions_respects_touch_session_activity(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        now = time.time()
        old_time = now - 100 * 86400  # 100 days ago
        recent_time = now - 2 * 86400  # 2 days ago

        # Session with older messages and started_at, but recent heartbeat / activity
        db.create_session("active_by_heartbeat", source="cli")
        db.append_message("active_by_heartbeat", "user", "hello", timestamp=old_time)
        db.end_session("active_by_heartbeat", "done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
            (old_time, old_time, "active_by_heartbeat"),
        )
        db._conn.commit()
        db.touch_session_activity("active_by_heartbeat", ts=recent_time)

        # Truly old session
        db.create_session("truly_old", source="cli")
        db.append_message("truly_old", "user", "old message", timestamp=old_time)
        db.end_session("truly_old", "done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
            (old_time, old_time, "truly_old"),
        )
        db._conn.commit()

        # Check dry-run candidates
        candidates = db.list_prune_candidates(older_than_days=10)
        candidate_ids = [c["id"] for c in candidates]
        assert "active_by_heartbeat" not in candidate_ids
        assert "truly_old" in candidate_ids

        candidate_map = {c["id"]: c for c in db.list_prune_candidates(older_than_days=None)}
        assert candidate_map["active_by_heartbeat"]["last_active"] == pytest.approx(recent_time, abs=1.0)

        # Pruning sessions older than 10 days should spare active_by_heartbeat
        pruned = db.prune_sessions(older_than_days=10)
        assert pruned == 1
        assert db.get_session("truly_old") is None
        assert db.get_session("active_by_heartbeat") is not None
