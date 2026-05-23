"""Tests for content normalization in the API server adapter."""

from gateway.platforms.api_server import _normalize_chat_content, _normalize_multimodal_content


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


class TestNormalizeMultimodalContent:
    """Multimodal normalization preserves safe image filename metadata."""

    def test_image_filename_injected_before_image_part(self):
        content = [
            {"type": "text", "text": "make listing"},
            {
                "type": "image_url",
                "filename": "D22946.jpg",
                "image_url": {"url": "https://example.com/D22946.jpg", "detail": "high"},
            },
        ]

        result = _normalize_multimodal_content(content)

        assert result == [
            {"type": "text", "text": "make listing"},
            {"type": "text", "text": "Attached image filename: D22946.jpg"},
            {"type": "image_url", "image_url": {"url": "https://example.com/D22946.jpg", "detail": "high"}},
        ]

    def test_input_image_nested_filename_supported(self):
        content = [
            {
                "type": "input_image",
                "image_url": {"url": "data:image/png;base64,abc123", "name": "D22947_3strip.png"},
            },
        ]

        result = _normalize_multimodal_content(content)

        assert result[0] == {"type": "text", "text": "Attached image filename: D22947_3strip.png"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_filename_metadata_is_basename_sanitized(self):
        content = [
            {
                "type": "image_url",
                "file_name": "/Users/Matt/Pictures/D22948\n.jpg",
                "image_url": "https://example.com/img.png",
            },
        ]

        result = _normalize_multimodal_content(content)

        assert result[0] == {"type": "text", "text": "Attached image filename: D22948 .jpg"}
        assert "Users" not in result[0]["text"]

    def test_image_without_filename_preserves_existing_shape(self):
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]

        assert _normalize_multimodal_content(content) == [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]
