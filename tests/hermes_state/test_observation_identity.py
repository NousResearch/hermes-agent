"""Read shape excludes content even from malformed legacy observations."""
import json
import os
from hermes_state import SessionDB
from hermes_state_observations import read_session_observations_at_path


def test_attention_wire_is_a_fixed_metadata_projection(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        holder = f"pid={os.getpid()}:turn=metadata"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        assert db.open_session_attention("s", turn, "validation")
        attention = db.read_session_observations(["s"], profile="default")[0]["attention"]
        attention["prompt"] = "synthetic-private-sentinel"
        db._write_sql("UPDATE session_observations SET attention_json=?", (json.dumps([attention]),))
        wire = db.read_session_observations(["s"], profile="default")[0]
        assert set(wire["attention"]) == {"kind", "request_id", "turn_id", "opened_at"}, "content escaped metadata projection"
        assert "synthetic-private-sentinel" not in json.dumps(wire)


def test_ambiguous_compression_does_not_select_newest_sibling(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("root", "desktop", profile_name="default")
        db.create_session("left", "desktop", parent_session_id="root", profile_name="default")
        db.create_session("right", "desktop", parent_session_id="root", profile_name="default")
        db._write_sql("UPDATE sessions SET end_reason='compression' WHERE id='root'", ())
        holder = f"pid={os.getpid()}:turn=metadata"
        db.try_acquire_session_turn_lease("left", holder)
        db.begin_session_observation("left", holder)
        assert all(r["execution"] == "unknown" and r["lineage_tip_id"] is None
                   for r in db.read_session_observations(["root", "left", "right"], profile="default"))


def test_last_active_is_activity_not_read_time_or_terminal_age(tmp_path):
    import time
    now = time.time()
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "desktop", profile_name="default")
        db._write_sql("UPDATE sessions SET last_activity_at=? WHERE id='s'", (now - 600,))
        db._write_sql("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                      ("s", "assistant", "synthetic-private-sentinel", now - 42))
        row = db.read_session_observations(["s"], profile="default")[0]
        assert row.get("last_active") == now - 42, "metadata must renew the listing's effective activity time"
        assert "synthetic-private-sentinel" not in json.dumps(row)
        assert db.read_session_observations(["s"], profile="foreign")[0].get("last_active") is None


def test_unverifiable_model_config_is_unknown_without_poisoning_siblings(tmp_path):
    path = tmp_path / "state.db"
    configs = ["[]", "null", "true", "42", '"synthetic"', "{bad"]
    with SessionDB(path) as db:
        db.create_session("good", "desktop", profile_name="default", model_config={})
        holder = f"pid={os.getpid()}:turn=identity"
        assert db.try_acquire_session_turn_lease("good", holder)
        turn = db.begin_session_observation("good", holder)
        assert db.finish_session_observation("good", holder, turn, "complete")
        db.release_session_turn_lease("good", holder)
        for i, cfg in enumerate(configs):
            sid = f"bad{i}"
            db.create_session(sid, "desktop", profile_name="default")
            db._write_sql("UPDATE sessions SET model_config=? WHERE id=?", (cfg, sid))
    before = path.read_bytes()
    ids = ["good", *(f"bad{i}" for i in range(len(configs)))]
    rows = read_session_observations_at_path(path, ids, profile="default")
    assert rows[0]["last_result"]["status"] == "complete", "invalid sibling erased healthy proof"
    assert [r["session_id"] for r in rows] == ids
    assert all(r["execution"] == "unknown" and r["lineage_tip_id"] is None and r.get("last_active") is None
               for r in rows[1:]), "unverifiable model config gained observation identity"
    assert path.read_bytes() == before
