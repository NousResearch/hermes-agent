"""Terminal proof may not be replaced by a duplicate or guessed from prose."""
import os
from hermes_state import SessionDB


def test_terminal_result_sealed_while_lease_remains_live(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", profile_name="default")
        holder = f"pid={os.getpid()}:turn=original"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        db.finish_session_observation("s", holder, turn, "complete")
        assert not db.finish_session_observation("s", holder, turn, "error"), "terminal result is sealed"


def test_reacquired_holder_and_replaced_generation_fence_all_producers(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as old, SessionDB(path) as successor:
        old.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=reused"
        assert old.try_acquire_session_turn_lease("s", holder)
        first = old.begin_session_observation("s", holder)
        request = old.open_session_attention("s", first, "approval", holder=holder)
        assert request
        acquired = old._read_one("SELECT acquired_at FROM session_turn_leases")[0]
        assert old.refresh_session_turn_lease("s", holder)
        assert old._read_one("SELECT acquired_at FROM session_turn_leases")[0] == acquired
        old.release_session_turn_lease("s", holder)
        assert successor.try_acquire_session_turn_lease("s", holder)
        # An identical holder with a different acquisition cannot use the old generation.
        assert not old.finish_session_observation("s", holder, first, "error")
        assert not old.open_session_attention("s", first, "approval", holder=holder)
        assert not old.resolve_session_attention("s", first, request, holder=holder)
        second = successor.begin_session_observation("s", holder)
        assert second != first
        assert not old.finish_session_observation("s", holder, first, "complete")
        assert not old.resolve_session_attention("s", first, request, holder=holder)
        assert successor.finish_session_observation("s", holder, second, "interrupted")
        successor.release_session_turn_lease("s", holder)
        row = old.read_session_observations(["s"], profile="default")[0]
        assert row["execution"] == "idle"
        assert row["turn_id"] == row["last_result"]["turn_id"] == second
        assert row["last_result"]["status"] == "interrupted"
        assert row["attention"]["kind"] == "none"
