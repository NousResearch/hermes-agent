"""Regression tests for #144569: reasoning delivered as ``content[].reasoning_text`` is not dropped.

A provider whose Responses wire returns the reasoning item with ``summary: null`` and ``text: null``
but ``content: [{"type": "reasoning_text", "text": ...}]`` had that text discarded: the collector only
read ``summary`` and ``text``. Reasoning tokens were billed and the ``💭`` block never rendered.
"""

import types

from agent.codex_responses_adapter import _OutputScan, _extract_responses_reasoning_text

# The live wire shape reported in #144569 (raw JSON from a Responses bridge).
WIRE_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "role": None,
    "summary": None,
    "content": [{"type": "reasoning_text", "text": "classic cognitive reflection test"}],
    "text": None,
}


def _object_item(**overrides):
    fields = {"type": "reasoning", "summary": None, "text": None, "content": None}
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_content_parts_are_read_from_the_wire_shape():
    assert _extract_responses_reasoning_text(WIRE_ITEM) == "classic cognitive reflection test"


def test_content_parts_are_read_from_attribute_style_items():
    item = _object_item(content=[types.SimpleNamespace(type="reasoning_text", text="step one")])
    assert _extract_responses_reasoning_text(item) == "step one"


def test_summary_still_wins_over_content():
    item = _object_item(
        summary=[types.SimpleNamespace(type="summary_text", text="summarized")],
        content=[types.SimpleNamespace(type="reasoning_text", text="raw")])
    assert _extract_responses_reasoning_text(item) == "summarized"


def test_text_still_wins_over_content():
    item = _object_item(text="plain text", content=[{"type": "reasoning_text", "text": "raw"}])
    assert _extract_responses_reasoning_text(item) == "plain text"


def test_multiple_content_parts_are_joined_in_order():
    item = _object_item(content=[
        {"type": "reasoning_text", "text": "first"},
        {"type": "output_text", "text": "not reasoning"},
        {"type": "reasoning_text", "text": "second"},
    ])
    assert _extract_responses_reasoning_text(item) == "first\nsecond"


def test_empty_and_foreign_shapes_yield_no_text():
    for item in (
        {"type": "reasoning", "summary": None, "text": None, "content": None},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": ""}]},
        {"type": "reasoning", "content": [{"type": "output_text", "text": "assistant text"}]},
        {"type": "reasoning", "content": "not-a-list"},
    ):
        assert _extract_responses_reasoning_text(item) == ""


def test_output_scan_keeps_reasoning_from_content_parts():
    scan = _OutputScan("completed")
    scan.scan([types.SimpleNamespace(**WIRE_ITEM)], issuer_kind="other", issuer_model="glm-5.3")
    assert scan.saw_reasoning_item is True
    assert scan.reasoning_parts == ["classic cognitive reflection test"]
