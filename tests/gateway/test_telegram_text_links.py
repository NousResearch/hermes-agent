"""Tests for Telegram text_link entity expansion (#31071).

Covers: _expand_text_links static method that converts Telegram text_link
entities into Markdown [text](url) syntax in incoming messages.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    sys.modules.setdefault("telegram", mod)
    sys.modules.setdefault("telegram.ext", MagicMock())
    sys.modules.setdefault("telegram.constants", MagicMock())


_ensure_telegram_mock()
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


class TestExpandTextLinks:
    """Verify _expand_text_links converts text_link entities to Markdown."""

    @staticmethod
    def _entity(offset: int, length: int, url: str, entity_type: str = "text_link"):
        return SimpleNamespace(type=entity_type, offset=offset, length=length, url=url)

    def test_single_text_link(self):
        """Single text_link entity → [text](url)."""
        text = "Check this article"
        entities = [self._entity(0, 5, "https://example.com")]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "[Check](https://example.com) this article"

    def test_multiple_text_links(self):
        """Multiple text_link entities all expanded."""
        text = "read foo and bar"
        entities = [
            self._entity(5, 3, "https://foo.com"),
            self._entity(13, 3, "https://bar.com"),
        ]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "read [foo](https://foo.com) and [bar](https://bar.com)"

    def test_chinese_text_link(self):
        """CJK text_link entity from #31071.

        CJK characters in BMP (U+4E00-U+9FFF) are 1 UTF-16 code unit each,
        same as ASCII. Only emoji/supplementary chars need special handling.
        """
        text = "财联社"
        entities = [self._entity(0, 3, "https://mp.weixin.qq.com/s/abc123")]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "[财联社](https://mp.weixin.qq.com/s/abc123)"

    def test_mixed_ascii_cjk_text_link(self):
        """Mixed ASCII + CJK text with text_link on CJK portion."""
        text = "看这篇财联社报道"
        # CJK chars are 1 UTF-16 unit each, so '财联社' is at offset=3, length=3
        entities = [self._entity(3, 3, "https://mp.weixin.qq.com/s/xyz")]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "看这篇[财联社](https://mp.weixin.qq.com/s/xyz)报道"

    def test_emoji_text_link(self):
        """Emoji (outside BMP) uses 2 UTF-16 code units per character."""
        text = "😀点击"
        # 😀 = U+1F600, 2 UTF-16 units; '点击' = 2 BMP chars, 1 unit each
        # '点击' starts at UTF-16 offset 2, length 2
        entities = [self._entity(2, 2, "https://example.com")]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "😀[点击](https://example.com)"

    def test_non_text_link_entities_ignored(self):
        """bold, italic, mention etc. should not be modified."""
        text = "hello @user"
        entities = [
            SimpleNamespace(type="bold", offset=0, length=5),
            SimpleNamespace(type="mention", offset=6, length=5),
        ]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "hello @user"

    def test_entity_without_url_skipped(self):
        """text_link entity with no url attribute → skip."""
        text = "no url here"
        entities = [SimpleNamespace(type="text_link", offset=0, length=2, url=None)]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "no url here"

    def test_empty_text_returns_unchanged(self):
        """Empty or None text → unchanged."""
        assert TelegramAdapter._expand_text_links("", []) == ""
        assert TelegramAdapter._expand_text_links("", None) == ""

    def test_no_entities_returns_unchanged(self):
        """Text without entities → unchanged."""
        assert TelegramAdapter._expand_text_links("hello", []) == "hello"
        assert TelegramAdapter._expand_text_links("hello", None) == "hello"

    def test_overlapping_offsets_replaced_correctly(self):
        """Two adjacent text_link entities expanded independently."""
        text = "ab"
        entities = [
            self._entity(0, 1, "https://a.com"),
            self._entity(1, 1, "https://b.com"),
        ]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "[a](https://a.com)[b](https://b.com)"

    def test_text_link_in_middle(self):
        """text_link not at start of message."""
        text = "visit Google for more"
        entities = [self._entity(6, 6, "https://google.com")]
        result = TelegramAdapter._expand_text_links(text, entities)
        assert result == "visit [Google](https://google.com) for more"
