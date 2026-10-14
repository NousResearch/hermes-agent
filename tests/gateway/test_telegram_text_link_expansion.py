"""Tests for Telegram ``text_link`` entity expansion in inbound messages.

Telegram delivers a URL attached to a word (e.g. "тут" -> github.com) as a
``text_link`` entity. The visible text carries no URL, so without expansion the
model only ever sees the bare word and cannot fetch the link. ``expand_link_entities``
inlines the real URL right after its anchor for both ``text`` and ``caption``.
"""

import pytest

from plugins.platforms.telegram.telegram_entities import expand_link_entities


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


def test_hidden_link_in_word_is_inlined():
    msg = _Message(
        text="Ссылка: тут\n#tag",
        entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")],
    )
    out = expand_link_entities(msg)
    assert "https://github.com/Cysharp/R3" in out
    assert out.startswith("Ссылка: тут (https://github.com/Cysharp/R3)")



def test_caption_link_on_media_is_inlined():
    msg = _Message(
        caption="Смотри тут проект",
        caption_entities=[_Entity("text_link", 7, 3, "https://example.com/x")],
    )
    assert expand_link_entities(msg) == "Смотри тут (https://example.com/x) проект"


def test_utf16_offset_is_respected_after_emoji():
    # Telegram entity offsets are measured in UTF-16 code units. The emoji is
    # two units, so the visible anchor starts at offset 3, not Python index 2.
    msg = _Message(
        text="🔥 тут",
        entities=[_Entity("text_link", 3, 3, "https://example.com/emoji")],
    )
    assert expand_link_entities(msg) == "🔥 тут (https://example.com/emoji)"


def test_expansion_is_idempotent():
    msg = _Message(
        text="Ссылка: тут\n#tag",
        entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")],
    )
    out = expand_link_entities(msg)
    repeat = _Message(text=out, entities=[_Entity("text_link", 8, 3, "https://github.com/Cysharp/R3")])
    assert expand_link_entities(repeat) == out





@pytest.mark.parametrize(
    "entity",
    [
        _Entity("text_link", 1, 99, "https://example.com/past-end"),
        _Entity("text_link", 0, 1, 123),
        _Entity("text_link", "invalid", 1, "https://example.com/bad-offset"),
    ],
)
def test_malformed_link_entities_are_ignored(entity):
    msg = _Message(text="abc", entities=[entity])
    assert expand_link_entities(msg) == "abc"


def test_text_does_not_use_caption_entities():
    msg = _Message(
        text="plain text",
        caption="linked caption",
        caption_entities=[_Entity("text_link", 0, 6, "https://example.com/caption")],
    )
    assert expand_link_entities(msg) == "plain text"


def test_offset_inside_utf16_surrogate_pair_is_ignored():
    msg = _Message(
        text="🔥 link",
        entities=[_Entity("text_link", 1, 1, "https://example.com/mid-surrogate")],
    )
    assert expand_link_entities(msg) == "🔥 link"


def test_anchor_that_is_already_the_url_is_not_duplicated():
    url = "https://example.com/self"
    msg = _Message(text=f"see {url} now", entities=[_Entity("text_link", 4, len(url), url)])
    assert expand_link_entities(msg) == f"see {url} now"
