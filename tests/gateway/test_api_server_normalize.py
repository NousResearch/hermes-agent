"""Tests for _normalize_chat_content in the API server adapter."""

from gateway.platforms import api_server
from gateway.platforms.api_server import _normalize_chat_content


class TestNormalizeChatContent:
    """Content normalization converts array-based content parts to plain text."""

    def test_none_returns_empty_string(self):
        assert _normalize_chat_content(None) == ""

    def test_plain_string_returned_as_is(self):
        assert _normalize_chat_content("hello world") == "hello world"


    def test_text_content_part(self):
        content = [{"type": "text", "text": "hello"}]
        assert _normalize_chat_content(content) == "hello"

    def test_input_text_content_part(self):
        content = [{"type": "input_text", "text": "user input"}]
        assert _normalize_chat_content(content) == "user input"

    def test_output_text_content_part(self):
        content = [{"type": "output_text", "text": "assistant output"}]
        assert _normalize_chat_content(content) == "assistant output"


    def test_empty_text_parts_filtered(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "actual"},
            {"type": "text", "text": ""},
        ]
        assert _normalize_chat_content(content) == "actual"




class TestSessionResponse:
    """_session_response exposes a stable, client-safe set of session fields."""

    def test_session_binding_fields_returned(self):
        """session_key and the structured chat binding are client-safe
        metadata: external consumers use them to map a live session to the
        chat/group/thread it originated from."""
        from gateway.platforms.api_server import APIServerAdapter

        session = {
            "id": "sess_1",
            "user_id": "u1",
            "session_key": "agent:main:telegram:group:-1001234567890:1",
            "chat_id": "-1001234567890",
            "chat_type": "group",
            "thread_id": "1",
        }
        payload = APIServerAdapter._session_response(session)
        assert payload["session_key"] == "agent:main:telegram:group:-1001234567890:1"
        assert payload["chat_id"] == "-1001234567890"
        assert payload["chat_type"] == "group"
        assert payload["thread_id"] == "1"

    def test_binding_fields_omitted_when_absent(self):
        """Keys missing from the source row are not fabricated."""
        from gateway.platforms.api_server import APIServerAdapter

        payload = APIServerAdapter._session_response({"id": "sess_1", "user_id": "u1"})
        for key in ("session_key", "chat_id", "chat_type", "thread_id"):
            assert key not in payload

    def test_unsafe_keys_stay_stripped(self):
        """Sensitive snapshots are not echoed back, only existence flags."""
        from gateway.platforms.api_server import APIServerAdapter

        session = {
            "id": "sess_1",
            "session_key": "key-abc",
            "system_prompt": "secret prompt",
            "model_config": {"k": "v"},
        }
        payload = APIServerAdapter._session_response(session)
        assert "system_prompt" not in payload
        assert "model_config" not in payload
        assert payload["has_system_prompt"] is True
        assert payload["has_model_config"] is True
        assert payload["session_key"] == "key-abc"
