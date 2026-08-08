"""Tests for Telegram ``text_link`` entity expansion in inbound messages.

Telegram delivers a URL attached to a word (e.g. "тут" -> github.com) as a
``text_link`` entity. The visible text carries no URL, so without expansion the
model only ever sees the bare word and cannot fetch the link. ``_expand_link_entities``
inlines the real URL right after its anchor for both ``text`` and ``caption``.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


class _Entity:
    def __init__(self, type, offset, length, url=None):
        self.type = type
        self.offset = offset
        self.length = length
        self.url = url


class _Message:
    def __init__(self, text=None, caption=None, entities=None, caption_entities=None):
        self.text = text
        self.caption = caption
        self.entities = entities
        self.caption_entities = caption_entities


@pytest.fixture
def adapter():
    # _expand_link_entities only uses getattr on the message, so an unbound
    # instance is enough.
    return TelegramAdapter.__new__(TelegramAdapter)


def test_hidden_link_in_word_is_inlined(adapter):
    msg = _Message(
        text="Ссылка: тут\n#tag",
        entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")],
    )
    out = adapter._expand_link_entities(msg)
    assert "https://github.com/Cysharp/R3" in out
    assert out.startswith("Ссылка: тут (https://github.com/Cysharp/R3)")


def test_plain_text_without_entities_is_unchanged(adapter):
    msg = _Message(text="просто текст без ссылок")
    assert adapter._expand_link_entities(msg) == "просто текст без ссылок"


def test_caption_link_on_media_is_inlined(adapter):
    msg = _Message(
        caption="Смотри тут проект",
        caption_entities=[_Entity("text_link", 7, 3, "https://example.com/x")],
    )
    assert adapter._expand_link_entities(msg) == "Смотри тут (https://example.com/x) проект"


def test_utf16_offset_is_respected_after_emoji(adapter):
    # Telegram entity offsets are measured in UTF-16 code units. The emoji is
    # two units, so the visible anchor starts at offset 3, not Python index 2.
    msg = _Message(
        text="🔥 тут",
        entities=[_Entity("text_link", 3, 3, "https://example.com/emoji")],
    )
    assert adapter._expand_link_entities(msg) == "🔥 тут (https://example.com/emoji)"


def test_expansion_is_idempotent(adapter):
    msg = _Message(
        text="Ссылка: тут\n#tag",
        entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")],
    )
    out = adapter._expand_link_entities(msg)
    repeat = _Message(text=out, entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")])
    assert adapter._expand_link_entities(repeat) == out


def test_multiple_distinct_links(adapter):
    msg = _Message(
        text="a b",
        entities=[
            _Entity("text_link", 0, 1, "https://one.com"),
            _Entity("text_link", 2, 1, "https://two.com"),
        ],
    )
    out = adapter._expand_link_entities(msg)
    assert out == "a (https://one.com) b (https://two.com)"


def test_non_text_link_entities_are_ignored(adapter):
    msg = _Message(text="жирный текст", entities=[_Entity("bold", 0, 6)])
    assert adapter._expand_link_entities(msg) == "жирный текст"


def test_anchor_repeats_later_in_text(adapter):
    msg = _Message(
        text="тут и ещё тут",
        entities=[_Entity("text_link", 0, 3, "https://x.com")],
    )
    out = adapter._expand_link_entities(msg)
    assert out.startswith("тут (https://x.com) и ещё тут")


@pytest.mark.parametrize(
    "entity",
    [
        _Entity("text_link", 1, 99, "https://example.com/past-end"),
        _Entity("text_link", 0, 1, 123),
        _Entity("text_link", "invalid", 1, "https://example.com/bad-offset"),
    ],
)
def test_malformed_link_entities_are_ignored(adapter, entity):
    msg = _Message(text="abc", entities=[entity])
    assert adapter._expand_link_entities(msg) == "abc"


def test_text_does_not_use_caption_entities(adapter):
    msg = _Message(
        text="plain text",
        caption="linked caption",
        caption_entities=[_Entity("text_link", 0, 6, "https://example.com/caption")],
    )
    assert adapter._expand_link_entities(msg) == "plain text"


def test_offset_inside_utf16_surrogate_pair_is_ignored(adapter):
    msg = _Message(
        text="🔥 link",
        entities=[_Entity("text_link", 1, 1, "https://example.com/mid-surrogate")],
    )
    assert adapter._expand_link_entities(msg) == "🔥 link"
