"""Tests for Telegram MessageEntity extraction and truncation."""

import pytest

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(
        enabled=True,
        token="dummy",
    )
    return TelegramAdapter(config)


class TestExtractEntities:
    def test_empty_content(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("")
        assert text == ""
        assert entities == []

    def test_plain_text(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("hello world")
        assert text == "hello world"
        assert entities == []

    def test_bold(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("**hello**")
        assert text == "hello"
        assert len(entities) == 1
        assert entities[0]["type"] == "bold"
        assert entities[0]["offset"] == 0
        assert entities[0]["length"] == 5

    def test_italic(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("*hello*")
        assert text == "hello"
        assert len(entities) == 1
        assert entities[0]["type"] == "italic"
        assert entities[0]["offset"] == 0
        assert entities[0]["length"] == 5

    def test_inline_code(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("`hello`")
        assert text == "hello"
        assert len(entities) == 1
        assert entities[0]["type"] == "code"
        assert entities[0]["offset"] == 0
        assert entities[0]["length"] == 5

    def test_link(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("[click](https://example.com)")
        assert text == "click"
        assert len(entities) == 1
        assert entities[0]["type"] == "text_link"
        assert entities[0]["url"] == "https://example.com"

    def test_strikethrough(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("~~hello~~")
        assert text == "hello"
        assert len(entities) == 1
        assert entities[0]["type"] == "strikethrough"

    def test_spoiler(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("||hello||")
        assert text == "hello"
        assert len(entities) == 1
        assert entities[0]["type"] == "spoiler"

    def test_header_becomes_bold(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("### Title")
        assert text == "Title"
        assert len(entities) == 1
        assert entities[0]["type"] == "bold"

    def test_fenced_code_block(self):
        adapter = _make_adapter()
        md = "```python\nprint(1)\n```"
        text, entities = adapter._extract_entities(md)
        assert text == "print(1)\n"
        assert len(entities) == 1
        assert entities[0]["type"] == "pre"
        assert entities[0]["language"] == "python"

    def test_blockquote(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("> quote")
        assert text == "quote"
        assert len(entities) == 1
        assert entities[0]["type"] == "blockquote"

    def test_mixed_formatting(self):
        adapter = _make_adapter()
        text, entities = adapter._extract_entities("Hello **world** and `code`")
        assert text == "Hello world and code"
        types = [e["type"] for e in entities]
        assert "bold" in types
        assert "code" in types

    def test_code_block_does_not_leak_internal_markdown(self):
        adapter = _make_adapter()
        md = "```\n**not bold**\n```"
        text, entities = adapter._extract_entities(md)
        assert text == "**not bold**\n"
        assert len(entities) == 1
        assert entities[0]["type"] == "pre"


class TestTruncateWithEntities:
    def test_no_truncation(self):
        adapter = _make_adapter()
        text = "short"
        entities = [{"type": "bold", "offset": 0, "length": 5}]
        chunks = adapter._truncate_with_entities(text, entities, 100)
        assert len(chunks) == 1
        assert chunks[0][0] == "short"
        assert chunks[0][1][0]["type"] == "bold"
        assert chunks[0][1][0]["offset"] == 0
        assert chunks[0][1][0]["length"] == 5

    def test_truncation_splits_entities(self):
        adapter = _make_adapter()
        text = "x" * 5000
        entities = [{"type": "bold", "offset": 10, "length": 4980}]
        # Telegram max is 4096 UTF-16 units; with reserve we get ~4086
        chunks = adapter._truncate_with_entities(text, entities, 4096)
        assert len(chunks) > 1
        # Verify each chunk ends with indicator
        for i, (chunk_text, _chunk_entities) in enumerate(chunks):
            assert f"({i + 1}/{len(chunks)})" in chunk_text
        # First chunk should contain a trimmed bold entity
        first_entities = chunks[0][1]
        assert len(first_entities) >= 1
        assert first_entities[0]["type"] == "bold"
        # Second chunk should start with the rest of the bold entity
        second_entities = chunks[1][1]
        assert len(second_entities) >= 1
        assert second_entities[0]["type"] == "bold"
        assert second_entities[0]["offset"] == 0

    def test_entities_outside_chunk_are_shifted(self):
        adapter = _make_adapter()

        # Build text so that the entity starts just after the code-point split
        # boundary (budget = 4096 - 10 = 4086).  The italic entity should land
        # entirely in the second chunk with an offset that accounts for the
        # single character carried over from the split.
        text = "a" * 4087 + "b" * 10
        entities = [
            {"type": "italic", "offset": 4087, "length": 10},
        ]
        chunks = adapter._truncate_with_entities(text, entities, 4096)
        assert len(chunks) == 2
        # First chunk should have no entities
        assert chunks[0][1] == []
        # Second chunk should have italic at offset 1 (skipping the trailing 'a')
        second_types = {e["type"]: e["offset"] for e in chunks[1][1]}
        assert "italic" in second_types
        assert second_types["italic"] == 1

    def test_empty_entities_after_truncation(self):
        adapter = _make_adapter()
        text = "x" * 5000
        chunks = adapter._truncate_with_entities(text, [], 4096)
        assert len(chunks) > 1
        for _chunk_text, chunk_entities in chunks:
            assert chunk_entities == []


class TestExtractAndTruncateIntegration:
    def test_full_pipeline_with_markdown(self):
        adapter = _make_adapter()
        md = "**Hello** " * 1000  # long bold repeated text (6000 chars plain)
        plain, entities = adapter._extract_entities(md)
        chunks = adapter._truncate_with_entities(plain, entities, 4096)
        assert len(chunks) > 1
        for chunk_text, chunk_entities in chunks:
            # Verify no chunk exceeds Telegram limit (in UTF-16)
            from gateway.platforms.base import utf16_len

            assert utf16_len(chunk_text) <= 4096
            # Verify all entity offsets are valid (code-point space for ASCII)
            for ent in chunk_entities:
                assert ent["offset"] >= 0
                assert ent["offset"] + ent["length"] <= len(chunk_text)
