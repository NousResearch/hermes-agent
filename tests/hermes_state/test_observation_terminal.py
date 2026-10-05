"""Terminal proof may not be replaced by a duplicate or guessed from prose."""
import os
from types import SimpleNamespace
from hermes_state import SessionDB


def test_terminal_result_sealed_while_lease_remains_live(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", profile_name="default")
        holder = f"pid={os.getpid()}:turn=original"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        db.finish_session_observation("s", holder, turn, "complete")
        assert not db.finish_session_observation("s", holder, turn, "error"), "terminal result is sealed"
