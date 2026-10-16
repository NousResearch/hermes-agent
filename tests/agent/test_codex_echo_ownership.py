"""The accepted input owns its row; transport echoes do not own new rows."""
import threading
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _persist_projected_messages
from agent.message_metadata import append_message
from agent.session_persistence import SessionPersistenceMixin
from agent.transports.codex_event_projector import CodexEventProjector
from hermes_state import SessionDB


@pytest.mark.parametrize("platform_id", ["2146", None])
@pytest.mark.parametrize("projection", ["echo", "assistant_only", "different", "later_equal"])
def test_only_submitted_leading_echo_is_excluded(tmp_path, platform_id, projection):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session(session_id="echo", source="telegram", model="codex")
        agent = SessionPersistenceMixin()
        agent.session_id = "echo"
        agent._session_db = db
        agent._session_db_created = True
        agent._last_flushed_db_idx = 0
        agent._session_persist_lock = threading.RLock()
        messages = []
        # Two independently accepted identical inputs must both survive.
        for turn_index in range(2):
            append_message(messages, {"role": "user", "content": "accepted", "platform_message_id": platform_id})
            assert agent._flush_messages_to_session_db(messages)
            projector = CodexEventProjector()
            items = []
            if projection != "assistant_only":
                items.append({"type": "userMessage", "id": f"u{turn_index}", "content": [
                    {"type": "text", "text": "different" if projection == "different" else "wire caption"}]})
            items.append({"type": "agentMessage", "id": f"a{turn_index}", "text": "reply"})
            if projection == "later_equal":
                items.append({"type": "userMessage", "id": f"s{turn_index}", "content": [
                    {"type": "text", "text": "wire caption"}]})
            projected = []
            for item in items:
                projected.extend(projector.project({"method": "item/completed", "params": {"item": item}}).messages)
            _persist_projected_messages(agent, SimpleNamespace(
                projected_messages=projected, submitted_user_text="wire caption"), messages)
            assert agent._flush_messages_to_session_db(messages)
        users = [row["content"] for row in db.get_messages_as_conversation("echo") if row["role"] == "user"]
        extra = {"different": ["different"], "later_equal": ["wire caption"]}.get(projection, [])
        assert users == (["accepted"] + extra) * 2
    finally:
        db.close()
