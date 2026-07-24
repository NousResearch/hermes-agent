"""Contracts for outbound user-visible response extraction."""

from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.response_normalization import extract_visible_response_text


@pytest.mark.parametrize(
    "response",
    [
        "ordinary prose",
        '{"status":"ok"}',
        '```json\n{"status":"ok","summary":"example"}\n```',
        "[SimpleNamespace(type='output_text', text='code example')]",
        "  whitespace is part of the answer  ",
    ],
)
def test_visible_response_strings_pass_through_exactly(response):
    assert extract_visible_response_text(response) == response


def test_recognized_typed_blocks_and_message_wrappers_extract_visible_prose():
    response = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "first"},
            SimpleNamespace(type="input_text", text="second"),
            {"type": "output_text", "text": "third"},
            SimpleNamespace(type="summary_text", text="fourth"),
        ],
    }

    assert extract_visible_response_text(response) == "first\nsecond\nthird\nfourth"

    object_wrapper = SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text="object wrapper")],
    )
    assert extract_visible_response_text(object_wrapper) == "object wrapper"


def test_unknown_structures_are_not_suppressed_or_inferred_from_generic_keys():
    response = {
        "status": "ok",
        "summary": "not a typed block",
        "metadata": {"created_at": "tomorrow"},
    }

    assert extract_visible_response_text(response) == str(response)
    assert extract_visible_response_text([]) == "[]"
    assert extract_visible_response_text(["one", "two"]) == "['one', 'two']"
    assert extract_visible_response_text({"type": "output_text"}) == (
        "{'type': 'output_text'}"
    )
    assert extract_visible_response_text({"type": "output_text", "text": None}) == (
        "{'type': 'output_text', 'text': None}"
    )
    assert extract_visible_response_text({
        "type": "custom",
        "text": "do not infer",
    }) == ("{'type': 'custom', 'text': 'do not infer'}")


def test_unprintable_unknown_structure_returns_an_explicit_fallback():
    class Unprintable:
        def __str__(self):
            raise RuntimeError("cannot render")

    assert extract_visible_response_text(Unprintable()) == (
        "[Unsupported response content: Unprintable]"
    )


def test_responses_api_output_item_uses_visible_text_extraction():
    items = APIServerAdapter._extract_output_items({
        "messages": [],
        "final_response": {
            "type": "message",
            "content": [{"type": "output_text", "text": "visible answer"}],
        },
    })

    assert items[-1]["content"] == [{"type": "output_text", "text": "visible answer"}]
