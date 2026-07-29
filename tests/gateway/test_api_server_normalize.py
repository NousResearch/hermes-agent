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


