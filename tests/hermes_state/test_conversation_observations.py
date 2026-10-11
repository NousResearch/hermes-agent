"""Content-free observations are proofs, not guesses from absent leases."""
import os
import json

from hermes_state import SessionDB


def test_execution_requires_current_lease_or_fenced_terminal(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        assert callable(getattr(db, "read_session_observations", None)), "native observation reader missing"
        db.create_session("s", "cli", profile_name="default")
        assert db.read_session_observations(["s", "absent"], profile="default")[0]["execution"] == "unknown"
        holder = f"pid={os.getpid()}:turn=synthetic"
        assert db.try_acquire_session_turn_lease("s", holder)
        legacy = db.read_session_observations(["s"], profile="default")[0]
        assert (legacy["execution"], legacy["provenance"], legacy["attention"]["kind"]) == ("running", "lease", "unknown")
        turn = db.begin_session_observation("s", holder)
        row = db.read_session_observations(["s"], profile="default")[0]
        assert row["turn_id"] == turn and row["execution"] == "running" and row["attention"]["kind"] == "none"
        db.release_session_turn_lease("s", holder)  # crash: no terminal producer
        assert db.read_session_observations(["s"], profile="default")[0]["execution"] == "unknown"
        assert not db.finish_session_observation("s", holder, turn, "error")
        successor = holder + "-new"
        assert db.try_acquire_session_turn_lease("s", successor)
        new_turn = db.begin_session_observation("s", successor)
        assert not db.finish_session_observation("s", holder, turn, "complete")
        assert db.finish_session_observation("s", successor, new_turn, "interrupted")
        db.release_session_turn_lease("s", successor)
        terminal = db.read_session_observations(["s"], profile="default")[0]
        assert terminal["execution"] == "idle" and terminal["last_result"]["status"] == "interrupted"
        # A legacy new producer trumps the earlier native result.
        assert db.try_acquire_session_turn_lease("s", holder)
        legacy = db.read_session_observations(["s"], profile="default")[0]
        assert legacy["provenance"] == "lease" and legacy["last_result"] is None
        assert holder not in json.dumps(legacy)


def test_validation_survives_compression_restart_and_profile_roundtrip(tmp_path):
    assert callable(getattr(SessionDB, "open_session_attention", None)), "explicit durable attention missing"
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, profile in zip(homes, ["A", "B"]):
        with SessionDB(home / "state.db") as db:
            db.create_session("s", "tui", profile_name=profile)
            holder = f"pid={os.getpid()}:turn={profile}"
            assert db.try_acquire_session_turn_lease("s", holder)
            turn = db.begin_session_observation("s", holder)
            request = db.open_session_attention("s", turn, "validation")
            assert request
            db.publish_compression_child(parent_session_id="s", child_session_id="tip", source="tui",
                messages=[{"role": "user", "content": "synthetic handoff"}], require_compression_lease=False)
            assert db.finish_session_observation("tip", holder, turn, "complete")
            db.release_session_turn_lease("tip", holder)
    for home, profile in [(homes[0], "A"), (homes[1], "B"), (homes[0], "A")]:
        with SessionDB(home / "state.db", read_only=True) as db:
            root_row, tip_row = db.read_session_observations(["s", "tip"], profile=profile)
            assert root_row["lineage_tip_id"] == "tip" and root_row["turn_id"] == tip_row["turn_id"]
            assert root_row["execution"] == "idle" and root_row["last_result"]["status"] == "complete"
            assert root_row["attention"]["kind"] == "validation"
            assert db.read_session_observations(["s"], profile="foreign")[0]["lineage_tip_id"] is None
    with SessionDB(homes[0] / "state.db") as db:
        old = db.read_session_observations(["s"], profile="A")[0]
        holder = f"pid={os.getpid()}:turn=next"
        assert db.try_acquire_session_turn_lease("tip", holder)
        turn = db.begin_session_observation("tip", holder)
        assert not db.resolve_session_attention("tip", old["turn_id"], old["attention"]["request_id"])
        assert db.resolve_session_attention("tip", turn, old["attention"]["request_id"], request_turn_id=old["turn_id"])
        assert db.read_session_observations(["s"], profile="A")[0]["attention"]["kind"] == "none"


def test_legacy_successor_cannot_resurrect_an_older_idle_proof(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", profile_name="default")
        holder = f"pid={os.getpid()}:turn=original"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        db.finish_session_observation("s", holder, turn, "complete")
        db.release_session_turn_lease("s", holder)
        successor = holder + "-legacy"
        db.try_acquire_session_turn_lease("s", successor)
        assert db.read_session_observations(["s"], profile="default")[0]["provenance"] == "lease"
        db.release_session_turn_lease("s", successor)
        row = db.read_session_observations(["s"], profile="default")[0]
        assert row["execution"] == "unknown" and row["last_result"] is None, "legacy successor erased terminal certainty"
