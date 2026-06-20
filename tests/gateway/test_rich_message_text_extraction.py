"""Tests for _extract_text_from_rich_message and _collect_block_text."""
from types import MappingProxyType
from gateway.platforms.telegram import _extract_text_from_rich_message, _collect_block_text


class TestExtractTextFromRichMessage:
    """Tests for _extract_text_from_rich_message."""

    def test_none_returns_none(self):
        assert _extract_text_from_rich_message(None) is None

    def test_plain_dict_with_blocks(self):
        rich_msg = {
            "blocks": [
                {"type": "paragraph", "text": "Hello world"},
                {"type": "heading", "text": "Title", "size": 2},
            ]
        }
        result = _extract_text_from_rich_message(rich_msg)
        assert result == "Hello world\nTitle"

    def test_mappingproxy_with_blocks(self):
        """MappingProxyType must NOT fail (isinstance(dict) is False)."""
        rich_msg = MappingProxyType({
            "blocks": [
                {"type": "paragraph", "text": "Proxy text"},
            ]
        })
        result = _extract_text_from_rich_message(rich_msg)
        assert result == "Proxy text"

    def test_blocks_with_nested_text_list(self):
        """Blocks where text is a list of inline elements (bold, etc.)."""
        rich_msg = {
            "blocks": [
                {
                    "type": "paragraph",
                    "text": [
                        "Hello ",
                        {"type": "bold", "text": "world"},
                    ],
                },
            ]
        }
        result = _extract_text_from_rich_message(rich_msg)
        assert result == "Hello \nworld"

    def test_blocks_with_list_items(self):
        """List items have label + nested blocks."""
        rich_msg = {
            "blocks": [
                {
                    "type": "list",
                    "items": [
                        {"label": "•", "blocks": [{"type": "text", "text": "Item 1"}]},
                        {"label": "•", "blocks": [{"type": "text", "text": "Item 2"}]},
                    ],
                }
            ]
        }
        result = _extract_text_from_rich_message(rich_msg)
        assert result is not None
        assert "Item 1" in result
        assert "Item 2" in result

    def test_fallback_to_markdown(self):
        """When no blocks, fall back to markdown field."""
        rich_msg = {"markdown": "**Bold** text with `code`"}
        result = _extract_text_from_rich_message(rich_msg)
        assert result == "**Bold** text with `code`"

    def test_empty_blocks_falls_back_to_markdown(self):
        rich_msg = {"blocks": [], "markdown": "fallback text"}
        result = _extract_text_from_rich_message(rich_msg)
        assert result == "fallback text"

    def test_non_dict_returns_none(self):
        assert _extract_text_from_rich_message("not a dict") is None
        assert _extract_text_from_rich_message(42) is None

    def test_empty_dict_returns_none(self):
        assert _extract_text_from_rich_message({}) is None

    def test_blocks_with_only_empty_text(self):
        rich_msg = {"blocks": [{"type": "paragraph", "text": ""}]}
        # Empty text should not produce output; falls through to markdown
        result = _extract_text_from_rich_message(rich_msg)
        assert result is None


class TestCollectBlockText:
    """Tests for _collect_block_text."""

    def test_string_elements(self):
        out = []
        _collect_block_text(["hello", "world"], out)
        assert out == ["hello", "world"]

    def test_mixed_string_and_dict(self):
        out = []
        _collect_block_text(["plain", {"type": "bold", "text": "bold text"}], out)
        assert out == ["plain", "bold text"]

    def test_nested_text_list(self):
        out = []
        block = {"type": "paragraph", "text": ["a", {"type": "bold", "text": "b"}]}
        _collect_block_text([block], out)
        assert out == ["a", "b"]

    def test_empty_input(self):
        out = []
        _collect_block_text([], out)
        assert out == []
