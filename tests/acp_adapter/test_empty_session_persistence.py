"""Only new, contentless ACP sessions are ephemeral; existing state remains durable."""
import json
from types import SimpleNamespace

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def test_new_session_persists_only_when_content_exists(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture"))
    state = manager.create_session(cwd=str(tmp_path))
    assert db.get_session(state.session_id) is None
    manager.update_cwd(state.session_id, str(tmp_path / "moved"))
    manager.save_session(state.session_id)
    empty_fork = manager.fork_session(state.session_id)
    assert db.get_session(state.session_id) is None
    assert db.get_session(empty_fork.session_id) is None
    state.history.append({"role": "user", "content": "kept content"})
    manager.save_session(state.session_id)
    fork = manager.fork_session(state.session_id)
    for sid in (state.session_id, fork.session_id):
        assert db.get_session(sid)["source"] == "acp"
        assert db.get_messages_as_conversation(sid)[0]["content"] == "kept content"
    db.close()


def test_existing_empty_history_still_updates_metadata(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture"))
    # An old, unprompted ACP client may still own its row: source is not liveness.
    db.create_session(session_id="existing", source="acp", model="original")
    state = manager.get_session("existing")
    assert state is not None
    assert not state.history
    state.model = "selected-model"
    manager.update_cwd(state.session_id, str(tmp_path / "selected"))
    row = db.get_session(state.session_id)
    assert row["model"] == "selected-model"
    assert json.loads(row["model_config"])["cwd"] == state.cwd
    assert manager.get_session(state.session_id) is state
    assert db.list_never_active_keyed_sessions(older_than_days=0) == []
    db.close()
