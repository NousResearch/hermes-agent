"""The stored-content decoder (``session_rules.turns.decode_content``) and its prefix (``persistence.conversation.messages.CONTENT_JSON_PREFIX``).

A prefixed cell decodes as JSON, a prefixed cell that is not JSON logs one warning and comes back as the same string, and anything else comes back as the same object. The warning logs under ``session_rules.turns``.
"""

from __future__ import annotations

import json
import logging

import pytest

from persistence.conversation import messages
from session_rules import turns

WARNING = "Failed to decode JSON-encoded message content; returning raw string"


def test_the_prefix_has_one_home():
    assert messages.CONTENT_JSON_PREFIX == "\x00json:"
    assert "CONTENT_JSON_PREFIX" in messages.__all__
    assert turns.CONTENT_JSON_PREFIX is messages.CONTENT_JSON_PREFIX
    assert turns._CONTENT_JSON_PREFIX is messages.CONTENT_JSON_PREFIX
    assert turns._decode_content is turns.decode_content


@pytest.mark.parametrize("value", [[{"type": "text", "text": "hi"}], {"a": 1}, "inner text", 5, None])
def test_a_prefixed_json_cell_decodes(value, caplog):
    with caplog.at_level(logging.WARNING):
        assert turns.decode_content(messages.CONTENT_JSON_PREFIX + json.dumps(value)) == value
    assert caplog.records == []


@pytest.mark.parametrize("body", ['[{"type": "text"} EXTRA', "", "{bad"])
def test_a_prefixed_cell_that_is_not_json_comes_back_raw_with_one_warning(body, caplog):
    cell = messages.CONTENT_JSON_PREFIX + body
    with caplog.at_level(logging.WARNING):
        assert turns.decode_content(cell) is cell
    assert [(r.name, r.levelname, r.getMessage()) for r in caplog.records] == [("session_rules.turns", "WARNING", WARNING)]


@pytest.mark.parametrize("value", ["plain text", "", "x\x00json:[]", None, 5, 2.5, [{"type": "text"}], {"k": "v"}])
def test_anything_else_comes_back_as_the_same_object(value):
    assert turns.decode_content(value) is value
