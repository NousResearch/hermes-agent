"""Regression tests for inbound Telegram hidden-link expansion."""

from types import SimpleNamespace

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


def _entity(entity_type="text_link", offset=0, length=1, url=None):
    return SimpleNamespace(type=entity_type, offset=offset, length=length, url=url)


def _message(text=None, caption=None, entities=None, caption_entities=None):
    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities,
        caption_entities=caption_entities,
    )


@pytest.fixture
def adapter():
    return TelegramAdapter.__new__(TelegramAdapter)


def test_ramp_hidden_link_is_exposed(adapter):
    text = "Read the Ramp paper and others"
    msg = _message(
        text=text,
        entities=[_entity(offset=9, length=10, url="https://ramp.com/data/ai-jobs-impact/paper")],
    )
    assert adapter._expand_link_entities(msg) == (
        "Read the Ramp paper (https://ramp.com/data/ai-jobs-impact/paper) and others"
    )


def test_caption_link_is_exposed(adapter):
    msg = _message(
        caption="Смотри тут проект",
        caption_entities=[_entity(offset=7, length=3, url="https://example.com/x")],
    )
    assert adapter._expand_link_entities(msg) == "Смотри тут (https://example.com/x) проект"


def test_utf16_offset_after_emoji(adapter):
    msg = _message(
        text="🔥 тут",
        entities=[_entity(offset=3, length=3, url="https://example.com/emoji")],
    )
    assert adapter._expand_link_entities(msg) == "🔥 тут (https://example.com/emoji)"


def test_multiple_links(adapter):
    entities = [
        _entity(offset=0, length=1, url="https://one.example"),
        _entity(offset=2, length=1, url="https://two.example"),
    ]
    expanded = adapter._expand_link_entities(_message(text="a b", entities=entities))
    assert expanded == "a (https://one.example) b (https://two.example)"


def test_single_link_expansion_is_idempotent(adapter):
    entity = _entity(offset=0, length=1, url="https://one.example")
    expanded = adapter._expand_link_entities(_message(text="a", entities=[entity]))
    assert expanded == "a (https://one.example)"
    assert adapter._expand_link_entities(_message(text=expanded, entities=[entity])) == expanded


@pytest.mark.parametrize(
    "entity",
    [
        _entity("bold", 0, 1),
        _entity(offset=1, length=99, url="https://example.com/past-end"),
        _entity(offset=0, length=1, url=123),
        _entity(offset="bad", length=1, url="https://example.com/bad-offset"),
        _entity(offset=1, length=1, url="https://example.com/mid-surrogate"),
    ],
)
def test_irrelevant_or_malformed_entities_are_ignored(adapter, entity):
    text = "🔥 link" if entity.url == "https://example.com/mid-surrogate" else "abc"
    assert adapter._expand_link_entities(_message(text=text, entities=[entity])) == text


def test_visible_url_without_text_link_is_unchanged(adapter):
    text = "https://example.com"
    assert adapter._expand_link_entities(
        _message(text=text, entities=[_entity("url", 0, len(text))])
    ) == text
