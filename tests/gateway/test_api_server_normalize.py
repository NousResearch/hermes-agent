"""Tests for _normalize_chat_content in the API server adapter."""

from gateway.platforms import api_server
from gateway.platforms.api_server import _normalize_chat_content
from gateway.platforms.api_server import APIServerAdapter


class TestNormalizeChatContent:
    """Content normalization converts array-based content parts to plain text."""

    def test_none_returns_empty_string(self):
        assert _normalize_chat_content(None) == ""

    def test_plain_string_returned_as_is(self):
        assert _normalize_chat_content("hello world") == "hello world"

    def test_empty_string_returned_as_is(self):
        assert _normalize_chat_content("") == ""

    def test_text_content_part(self):
        content = [{"type": "text", "text": "hello"}]
        assert _normalize_chat_content(content) == "hello"

    def test_input_text_content_part(self):
        content = [{"type": "input_text", "text": "user input"}]
        assert _normalize_chat_content(content) == "user input"

    def test_output_text_content_part(self):
        content = [{"type": "output_text", "text": "assistant output"}]
        assert _normalize_chat_content(content) == "assistant output"

    def test_multiple_text_parts_joined_with_newline(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _normalize_chat_content(content) == "first\nsecond"

    def test_mixed_string_and_dict_parts(self):
        content = ["plain string", {"type": "text", "text": "dict part"}]
        assert _normalize_chat_content(content) == "plain string\ndict part"

    def test_image_url_parts_silently_skipped(self):
        content = [
            {"type": "text", "text": "check this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        assert _normalize_chat_content(content) == "check this:"

    def test_integer_content_converted(self):
        assert _normalize_chat_content(42) == "42"

    def test_boolean_content_converted(self):
        assert _normalize_chat_content(True) == "True"

    def test_deeply_nested_list_respects_depth_limit(self):
        """Nesting beyond max_depth returns empty string."""
        content = [[[[[[[[[[[["deep"]]]]]]]]]]]]
        result = _normalize_chat_content(content)
        # The deep nesting should be truncated, not crash
        assert isinstance(result, str)

    def test_large_list_capped(self):
        """Lists beyond MAX_CONTENT_LIST_SIZE are truncated."""
        content = [{"type": "text", "text": f"item{i}"} for i in range(2000)]
        result = _normalize_chat_content(content)
        # Should not contain all 2000 items
        assert result.count("item") <= 1000

    def test_oversized_string_truncated(self):
        """Strings beyond 64KB are truncated."""
        huge = "x" * 100_000
        result = _normalize_chat_content(huge)
        assert len(result) == 65_536

    def test_empty_text_parts_filtered(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "actual"},
            {"type": "text", "text": ""},
        ]
        assert _normalize_chat_content(content) == "actual"

    def test_dict_without_type_skipped(self):
        content = [{"foo": "bar"}, {"type": "text", "text": "real"}]
        assert _normalize_chat_content(content) == "real"

    def test_empty_list_returns_empty(self):
        assert _normalize_chat_content([]) == ""

    def test_many_small_parts_normalize_without_quadratic_rescan(self, monkeypatch):
        """Large content arrays should normalize in linear time."""
        content = [{"type": "text", "text": "x"} for _ in range(1000)]
        sum_calls = 0

        def counting_sum(values):
            nonlocal sum_calls
            sum_calls += 1
            return sum(values)

        monkeypatch.setattr(api_server, "sum", counting_sum, raising=False)
        result = _normalize_chat_content(content)

        assert result.count("x") == 1000
        assert sum_calls == 0


class TestSessionResponse:
    """_session_response exposes a stable, client-safe set of session fields."""

    def test_session_key_returned(self):
        """session_key is metadata the client API should surface."""
        session = {"id": "sess_1", "user_id": "u1", "session_key": "key-abc"}
        payload = APIServerAdapter._session_response(session)
        assert payload["session_key"] == "key-abc"

    def test_session_key_omitted_when_absent(self):
        """Keys missing from the source row are not fabricated."""
        session = {"id": "sess_1", "user_id": "u1"}
        payload = APIServerAdapter._session_response(session)
        assert "session_key" not in payload

    def test_unsafe_keys_stripped(self):
        """Sensitive snapshots are not echoed back, only existence flags."""
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
