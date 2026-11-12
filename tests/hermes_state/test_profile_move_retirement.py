"""Every profile import respects the target's permanent deletion fence."""

import pytest

from hermes_state import SessionDB
from hermes_state_mutation_retirement import RETIRED_PREFIX
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize("moved", [False, True])
def test_profile_imports_never_restore_a_deleted_target_id(tmp_path, monkeypatch, moved):
    home = tmp_path / "hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor = SessionDB(home / "state.db")
    target = SessionDB(profile / "state.db")
    sid = "deleted-profile-import"
    try:
        donor.create_session(sid, "tui")
        donor.append_message(sid, "user", "keep source history")
        target._conn.execute("INSERT INTO state_meta(key,value) VALUES (?,?)", (RETIRED_PREFIX + sid, "{}"))
        target._conn.commit()
        if moved:
            payload = donor.export_session_for_move(sid)
            with pytest.raises(RuntimeStoreError, match="session_deleted"):
                target.import_moved_session(payload, profile_name="science", transfer_from=donor.db_path)
        else:
            result = target.import_sessions([donor.export_session(sid)], transfer_from=donor.db_path)
            assert result["ok"] is False
        assert target.get_session(sid) is None
        assert [message["content"] for message in donor.get_messages(sid)] == ["keep source history"]
        assert donor.get_session(sid)["end_reason"] is None
    finally:
        donor.close()
        target.close()
