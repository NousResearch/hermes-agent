from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agent.memory_manager import sanitize_context, sanitize_transcript_context
from hermes_state import SessionDB
from run_agent import AIAgent


class _CaptureSessionDB:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def append_message(self, **kwargs):
        self.rows.append(kwargs)
        return len(self.rows)


def _agent_with_capture_db() -> tuple[AIAgent, _CaptureSessionDB]:
    agent = cast(Any, AIAgent.__new__(AIAgent))
    capture_db = _CaptureSessionDB()
    agent._session_db = capture_db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent.session_id = "native-image-session"
    agent._apply_persist_user_message_override = lambda messages: None
    return cast(AIAgent, agent), capture_db


def test_session_replay_strips_legacy_native_image_attachment_markers(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    sid = db.create_session("legacy-native-image-session", source="gateway")
    db.append_message(
        session_id=sid,
        role="user",
        content=(
            "Where do I scan this?\n\n"
            "[Image attached at: /tmp/hermes-home/image_cache/img_ccf883cb57da.jpg]\n"
            "[screenshot]"
        ),
    )

    replay = db.get_messages_as_conversation(sid)

    assert len(replay) == 1
    assert replay[0]["role"] == "user"
    assert replay[0]["content"] == "Where do I scan this?"


def test_transcript_sanitizer_strips_remote_image_attachment_markers():
    content = (
        "Please compare this.\n\n"
        "[Image attached: https://example.com/image.png]\n"
        "[inline image]"
    )

    assert sanitize_transcript_context(content).strip() == "Please compare this."


def test_base_context_sanitizer_preserves_literal_screenshot_text():
    content = "Please write the literal token [screenshot] in the docs."

    assert sanitize_context(content) == content
    assert sanitize_transcript_context(content) == content


def test_flush_multimodal_user_message_does_not_persist_image_path_markers():
    agent, capture_db = _agent_with_capture_db()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Where do I scan this?\n\n"
                        "[Image attached at: /tmp/hermes-home/image_cache/img_ccf883cb57da.jpg]"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                },
            ],
        }
    ]

    agent._flush_messages_to_session_db(messages)

    assert capture_db.rows[0]["content"] == "Where do I scan this?"
