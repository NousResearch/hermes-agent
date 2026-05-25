from datetime import datetime, timezone

from hermes_state import SessionDB
from agent.conversation_turn_policy import build_message_time_context


def test_message_time_context_formats_current_and_previous_times():
    history = [
        {"role": "user", "content": "avant", "timestamp": 1779471675.972964},
        {"role": "assistant", "content": "reponse", "timestamp": 1779482355.972964},
    ]

    block = build_message_time_context(
        current_timestamp=1779482475.972964,
        history=history,
        now=1779482475.972964,
    )

    assert "Current message: 2026-05-22T20:41:15.972964Z / 22:41 Europe/Paris" in block
    assert "Previous user message: 3h ago" in block
    assert "Previous assistant message: 2min ago" in block


def test_get_messages_as_conversation_restores_sqlite_timestamp(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="s1", source="telegram", user_id="u1")
    db.append_message(
        session_id="s1",
        role="user",
        content="salut",
        timestamp=datetime.fromtimestamp(1779482475.972964, tz=timezone.utc),
    )

    messages = db.get_messages_as_conversation("s1")

    assert messages == [
        {
            "role": "user",
            "content": "salut",
            "timestamp": 1779482475.972964,
        }
    ]
