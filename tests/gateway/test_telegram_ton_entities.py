"""Bot API 10.4 rich TON address / bank card number preservation in inbound Telegram text.

``RichTextTonAddress`` and ``RichTextBankCardNumber`` keep the machine-readable value in ``address`` /
``bank_card_number`` beside a display text that may be a descriptive label, masked, or empty. The
agent-visible flattening must keep the actual value; the addressing views (wake words, ``?``/``/``
routing) must not. Ordinary ``MessageEntity`` spans of those types already carry the value in the
message text, so they are left untouched instead of being annotated a second time.
"""

from types import MappingProxyType, SimpleNamespace

import pytest

from plugins.platforms.telegram.telegram_entities import (
    _rich_inline,
    addressing_sources,
    addressing_text,
    expand_link_entities,
    expand_quote_link_entities,
    rich_message_text,
)

TON = "EQBvW8Z5huBkMJYdnfAEM5JqTNkuWX3diqYENkWsIL0XggGG"
CARD = "4111111111113456"


class _Entity:
    def __init__(self, type, offset, length, url=None):
        self.type = type
        self.offset = offset
        self.length = length
        self.url = url


def _message(text=None, caption=None, entities=None, caption_entities=None):
    return SimpleNamespace(text=text, caption=caption, entities=entities,
                           caption_entities=caption_entities, api_kwargs=None)


def _rich(blocks):
    return SimpleNamespace(api_kwargs={"rich_message": {"blocks": blocks}})


def _paragraph(text):
    return {"type": "paragraph", "text": text}


def _ton(text, address=TON):
    return {"type": "ton_address", "text": text, "address": address}


# --- ordinary MessageEntity spans stay exactly as sent -----------------------------------------

def test_ordinary_ton_address_entity_leaves_text_untouched():
    # The emoji is two UTF-16 code units, so the entity starts at 7 in Telegram's units.
    msg = _message(text=f"🔥 pay {TON} now", entities=[_Entity("ton_address", 7, len(TON))])
    assert expand_link_entities(msg) == f"🔥 pay {TON} now"


def test_ordinary_bank_card_entity_is_not_annotated_again():
    card = "4111 1111 1111 1111"
    msg = _message(caption=f"card: {card}", caption_entities=[_Entity("bank_card_number", 6, len(card))])
    assert expand_link_entities(msg) == f"card: {card}"


# --- rich TON address --------------------------------------------------------------------------

def test_rich_ton_address_label_keeps_the_actual_address():
    msg = _rich([_paragraph([_ton("my wallet")])])
    assert rich_message_text(msg) == f"my wallet ({TON})"
    # The agent's event text for a text-less rich message comes through ``expand_link_entities``.
    assert expand_link_entities(msg) == f"my wallet ({TON})"


def test_rich_ton_address_text_already_the_address_is_not_duplicated():
    msg = _rich([_paragraph([_ton(TON)])])
    assert rich_message_text(msg) == TON


def test_rich_ton_address_already_inside_its_label_is_not_repeated():
    assert _rich_inline([_ton(f"pay to {TON}")]) == f"pay to {TON}"


@pytest.mark.parametrize("text", ["", [], None])
def test_rich_ton_address_with_empty_text_keeps_the_address(text):
    node = {"type": "ton_address", "address": TON}
    if text is not None:
        node["text"] = text
    assert rich_message_text(_rich([_paragraph([node])])) == TON


def test_nested_rich_ton_address_keeps_the_address():
    node = _ton("wallet")
    msg = _rich([_paragraph([{"type": "bold", "children": [
        {"type": "italic", "children": ["pay ", node]}]}])])
    assert rich_message_text(msg) == f"pay wallet ({TON})"


# --- rich bank card number ---------------------------------------------------------------------

def test_rich_bank_card_masked_text_keeps_the_documented_number():
    msg = _rich([_paragraph([{"type": "bank_card_number", "text": "•••• 3456",
                             "bank_card_number": CARD}])])
    assert rich_message_text(msg) == f"•••• 3456 ({CARD})"


def test_rich_bank_card_number_without_text_keeps_the_number():
    assert _rich_inline([{"type": "bank_card_number", "bank_card_number": CARD}]) == CARD


def test_rich_bank_card_formatted_text_does_not_duplicate_the_number():
    formatted = "4111 1111 1111 3456"
    msg = _rich([_paragraph([{"type": "bank_card_number", "text": formatted,
                             "bank_card_number": CARD}])])
    assert rich_message_text(msg) == formatted


# --- addressing gate: payloads are not wake words ----------------------------------------------

def test_addressing_views_do_not_read_the_hidden_ton_address():
    msg = _rich([_paragraph(["pay ", _ton("wallet")])])
    assert rich_message_text(msg) == f"pay wallet ({TON})"
    assert rich_message_text(msg, urls=False) == "pay wallet"
    assert addressing_text(msg) == "pay wallet"


def test_hidden_ton_payload_cannot_wake_the_bot():
    node = _ton("wallet", address="@hermes_bot")
    msg = _rich([_paragraph([node])])
    assert rich_message_text(msg) == "wallet (@hermes_bot)"
    assert addressing_text(msg) == "wallet"
    for text, entities in addressing_sources(msg):
        assert "@hermes_bot" not in text
        assert entities == []


def test_empty_labelled_ton_address_is_not_a_wake_word():
    msg = _rich([_paragraph([_ton("", address="@hermes_bot")])])
    assert rich_message_text(msg) == "@hermes_bot"
    assert addressing_text(msg) == ""
    assert addressing_sources(msg)[-1] == ("", [])


# --- quotes read the selection only -------------------------------------------------------------

def test_quote_does_not_inherit_the_replied_ton_address_entity():
    replied = _message(text=f"old {TON}", entities=[_Entity("ton_address", 4, len(TON))])
    quote = SimpleNamespace(text="old", position=0)
    assert expand_quote_link_entities(quote, replied) == "old"


def test_only_the_selected_rich_segment_is_flattened():
    selected = [_ton("wallet")]
    assert _rich_inline(selected) == f"wallet ({TON})"


# --- safe parsing -------------------------------------------------------------------------------

def test_malformed_rich_payloads_degrade_to_the_display_text():
    text = [
        {"type": "ton_address", "text": "a", "address": 123},
        {"type": "bank_card_number", "text": "1234", "bank_card_number": ["n"]},
        {"type": ["ton_address"], "text": "b"},
        "c",
    ]
    assert rich_message_text(_rich([_paragraph(text)])) == "a1234bc"


def test_rich_message_is_read_from_a_mapping_proxy_api_kwargs():
    msg = SimpleNamespace(api_kwargs=MappingProxyType(
        {"rich_message": {"blocks": [_paragraph([_ton("wallet")])]}}))
    assert rich_message_text(msg) == f"wallet ({TON})"
