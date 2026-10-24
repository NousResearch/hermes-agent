"""Generation checks and mutations hold one SQLite writer reservation."""
import os
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("action", ["begin", "open", "resolve", "finish"])
def test_generation_and_lease_fence_share_the_mutation_transaction(tmp_path, monkeypatch, action):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        assert callable(getattr(db, "begin_session_observation", None)), "durable observation producer missing"
        db.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=transaction"
        assert db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        request = db.open_session_attention("s", turn, "approval", holder=holder)
        assert turn and request
        import hermes_state_observations as observations
        original = observations._live_lease
        checked = []

        def lease_under_reservation(conn, sid, now):
            lease = original(conn, sid, now)
            # An independent writer cannot replace the lease between this read
            # and publication. Zero busy timeout makes the assertion event-free.
            with sqlite3.connect(path, timeout=0) as contender:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    contender.execute("BEGIN IMMEDIATE")
            checked.append(sid)
            return lease

        monkeypatch.setattr(observations, "_live_lease", lease_under_reservation)
        operations = {
            "begin": lambda: db.begin_session_observation("s", holder),
            "open": lambda: db.open_session_attention("s", turn, "question", holder=holder),
            "resolve": lambda: db.resolve_session_attention("s", turn, request, holder=holder),
            "finish": lambda: db.finish_session_observation("s", holder, turn, "complete"),
        }
        assert operations[action]()
        assert checked == ["s"], "producer skipped its transactional lease fence"


def test_failed_publication_rolls_back_without_consuming_generation(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        assert callable(getattr(db, "begin_session_observation", None)), "durable observation producer missing"
        db.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=rollback"
        assert db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        before = dict(db._read_one("SELECT * FROM session_observations"))
        db._write_sql("""CREATE TRIGGER reject_observation BEFORE UPDATE ON session_observations
                          BEGIN SELECT RAISE(ABORT, 'synthetic observation failure'); END""", ())
        with pytest.raises(sqlite3.IntegrityError, match="synthetic observation failure"):
            db.finish_session_observation("s", holder, turn, "complete")
        assert dict(db._read_one("SELECT * FROM session_observations")) == before
        db._write_sql("DROP TRIGGER reject_observation", ())
        assert db.finish_session_observation("s", holder, turn, "complete")
        assert db.read_session_observations(["s"], profile="default")[0]["last_result"]["turn_id"] == turn
