"""Tests for ``_normalize_stream_delta_content`` (issue #63734).

Mistral and NVIDIA custom providers can return ``delta.content`` as a
list of content-block dicts on the streaming request follows a completed
tool call. The downstream ``_fire_stream_delta`` /
``_record_streamed_assistant_text`` expects a plain string. This suite
pins the normalisation helper that prevents the
``AttributeError: 'list' object has no attribute 'strip'`` crash.
"""

from agent.chat_completion_helpers import _normalize_stream_delta_content


class TestNormalizeStreamDeltaContent:
    """``_normalize_stream_delta_content`` contract."""

    def test_string_passthrough(self):
        """Plain string returns unchanged."""
        assert _normalize_stream_delta_content("hello") == "hello"
        assert _normalize_stream_delta_content("") == ""

    def test_list_of_text_blocks(self):
        """List of content-block dicts with ``text`` keys -> joined text."""
        chunks = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _normalize_stream_delta_content(chunks) == "HelloWorld"

    def test_list_mixed_types(self):
        """Non-dict items in list are str()-converted."""
        chunks = [
            {"type": "text", "text": "Score: "},
            42,
            {"type": "text", "text": " points"},
        ]
        assert _normalize_stream_delta_content(chunks) == "Score: 42 points"

    def test_list_missing_text_key(self):
        """Dicts without ``text`` key contribute empty string."""
        chunks = [
            {"type": "text", "text": "A"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "B"},
        ]
        assert _normalize_stream_delta_content(chunks) == "AB"

    def test_list_empty(self):
        """Empty list gives empty string."""
        assert _normalize_stream_delta_content([]) == ""

    def test_none_converted_to_string(self):
        """``None`` is str()-converted (defensive)."""
        assert _normalize_stream_delta_content(None) == "None"

    def test_integer_converted_to_string(self):
        """Integer is str()-converted (defensive)."""
        assert _normalize_stream_delta_content(42) == "42"
