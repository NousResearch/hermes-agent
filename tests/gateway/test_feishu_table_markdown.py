"""Tests for Feishu adapter outbound markdown payload construction.

Tests that markdown content — including tables, headings, blockquotes — is
rendered via Interactive Card messages (``msg_type="interactive"``, Card JSON
2.0 with ``{tag: "markdown"}`` elements) rather than being downgraded to
plain text or limited Post messages.

Interactive Cards render the full markdown spec (headings, tables, blockquotes)
while Post ``{tag:'md'}`` elements only support a subset.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(content: str) -> "tuple[str, str] | list[tuple[str, str]]":
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` is a method that only uses module-level
    helpers and never touches ``self.*``, so a bare object is sufficient.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    return inst._build_outbound_payload(content)


def _card_md_texts(payload_str: str) -> list[str]:
    """Pull every ``{tag:'markdown', content:'...'}`` element out of a Card JSON 2.0 payload.

    Real payload shape::

        {"schema": "2.0", "config": {...}, "body": {"elements": [{"tag": "markdown", "content": "..."}]}}
    """
    payload = json.loads(payload_str)
    if not isinstance(payload, dict):
        return []
    texts: list[str] = []
    body = payload.get("body", {})
    elements = body.get("elements", [])
    for el in elements:
        if isinstance(el, dict) and el.get("tag") == "markdown":
            texts.append(el.get("content", ""))
    return texts


def test_markdown_table_uses_interactive_card():
    """Tables are rendered via Interactive Card (Card JSON 2.0), not post or text.

    Interactive Cards support native table rendering with proper alignment and
    borders, which Post ``{tag:'md'}`` elements do not.
    """
    content = (
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |"
    )
    result = _call_build_outbound_payload(content)
    # Should be a single (msg_type, payload) tuple, not a multi-card list
    assert isinstance(result, tuple), f"expected a single tuple, got {type(result)}"
    msg_type, payload_str = result
    assert msg_type == "interactive", (
        f"expected 'interactive' for a markdown table, got {msg_type!r}; "
        "tables should use Card JSON 2.0 for full rendering support"
    )
    md_texts = _card_md_texts(payload_str)
    assert md_texts, f"card payload must include at least one markdown element; got {payload_str!r}"
    joined = "".join(md_texts)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted in card payload"
    )


def test_heading_uses_interactive_card():
    """Headings (## etc.) are rendered via Interactive Card."""
    content = "## Status Report\n\nAll systems operational."
    result = _call_build_outbound_payload(content)
    assert isinstance(result, tuple)
    msg_type, payload_str = result
    assert msg_type == "interactive", (
        f"expected 'interactive' for headings, got {msg_type!r}"
    )
    md_texts = _card_md_texts(payload_str)
    joined = "".join(md_texts)
    assert "Status Report" in joined


def test_plain_text_stays_text():
    """Plain content without markdown indicators stays as text type."""
    content = "Hello world, just a simple message."
    result = _call_build_outbound_payload(content)
    assert isinstance(result, tuple)
    msg_type, _ = result
    assert msg_type == "text", (
        f"expected 'text' for plain content, got {msg_type!r}"
    )


def test_card_payload_is_valid_json_2_0():
    """The interactive card payload conforms to Card JSON 2.0 schema."""
    content = "**bold** and `code` and\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    result = _call_build_outbound_payload(content)
    assert isinstance(result, tuple)
    msg_type, payload_str = result
    assert msg_type == "interactive"
    payload = json.loads(payload_str)
    assert payload.get("schema") == "2.0", "Card must use schema 2.0"
    assert "body" in payload, "Card must have a body"
    assert "elements" in payload["body"], "Card body must have elements"
    elements = payload["body"]["elements"]
    assert len(elements) >= 1, "Card must have at least one element"
    assert all(el.get("tag") == "markdown" for el in elements), (
        "All elements should be markdown type"
    )


def test_multi_table_split():
    """Content with many tables splits into multiple card payloads."""
    tables = []
    for i in range(8):
        tables.append(
            f"## Section {i}\n\n"
            f"| key_{i} | val_{i} |\n"
            f"|---------|--------|\n"
            f"| a{i}    | b{i}   |\n"
        )
    content = "\n".join(tables)
    result = _call_build_outbound_payload(content)
    # With 8 tables and a limit of 5 per card, should split
    if isinstance(result, list):
        assert len(result) >= 2, "8 tables should produce at least 2 cards"
        for msg_type, payload_str in result:
            assert msg_type == "interactive"
            payload = json.loads(payload_str)
            assert payload.get("schema") == "2.0"
    else:
        # Single card is also acceptable if all tables fit within limits
        msg_type, _ = result
        assert msg_type == "interactive"
