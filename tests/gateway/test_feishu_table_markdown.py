"""Tests for Feishu adapter outbound markdown payload construction.

Reproduces the bug tracked in hermes-agent issue #52786:
`_build_outbound_payload` was force-downgrading any message containing a
markdown pipe table to ``msg_type=text``, so Feishu clients rendered the raw
pipe-and-dash source instead of a table.  The fix uses Card JSON 2.0
interactive messages that render markdown (including tables) natively.

These tests guard the fix.  They invoke the real adapter via the project's
plugin-loader helper so that no ``sys.path`` / ``sys.modules`` games are
needed.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(content: str):
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` is a method that only uses module-level
    helpers and never touches ``self.*``, so a bare object is sufficient.

    Returns the raw result — either a (msg_type, payload) tuple or a list
    of (msg_type, payload) tuples for multi-card messages.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    return inst._build_outbound_payload(content)


def _card_md_contents(payload_str: str) -> list[str]:
    """Pull every markdown element's content from a Card JSON 2.0 payload.

    Real payload shape::

        {"schema": "2.0", "body": {"elements": [{"tag": "markdown", "content": "..."}]}}
    """
    payload = json.loads(payload_str)
    if not isinstance(payload, dict):
        return []
    elements = payload.get("body", {}).get("elements", [])
    return [e.get("content", "") for e in elements if e.get("tag") == "markdown"]


def test_markdown_table_uses_interactive_not_text():
    """Regression test for issue #52786 (and its older sibling #23938).

    A message whose only markdown is a table must take the ``interactive``
    (Card JSON 2.0) path, not be downgraded to plain text.
    """
    content = (
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |"
    )
    result = _call_build_outbound_payload(content)
    if isinstance(result, list):
        msg_type, payload_str = result[0]
    else:
        msg_type, payload_str = result
    assert msg_type == "interactive", (
        f"expected 'interactive' for a markdown table (issue #52786), got {msg_type!r}; "
        "the table-downgrade branch in _build_outbound_payload has been re-introduced"
    )
    md_contents = _card_md_contents(payload_str)
    assert md_contents, f"card payload must include at least one markdown element; got {payload_str!r}"
    joined = "".join(md_contents)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted when switching from text to interactive"
    )


def test_plain_text_without_markdown_still_uses_text():
    """Negative control: a message with no markdown hints and no table must
    still go to plain text.  Guards against accidentally promoting everything
    to ``interactive``."""
    result = _call_build_outbound_payload("just a plain sentence with no markup")
    if isinstance(result, list):
        msg_type, _ = result[0]
    else:
        msg_type, _ = result
    assert msg_type == "text"


def test_existing_markdown_heading_still_uses_interactive():
    """Sanity: the existing markdown path (heading / list / code / bold /
    link) must use Card JSON 2.0 after the upgrade."""
    result = _call_build_outbound_payload("# hello world\n")
    if isinstance(result, list):
        msg_type, payload_str = result[0]
    else:
        msg_type, payload_str = result
    assert msg_type == "interactive"
    md_contents = _card_md_contents(payload_str)
    assert md_contents, f"expected at least one markdown element; got {payload_str!r}"
    assert any("hello world" in t for t in md_contents), (
        f"expected 'hello world' in markdown elements; got {md_contents!r}"
    )


def test_table_combined_with_other_markdown_does_not_downgrade():
    """A message that mixes a table with surrounding markdown must also
    take the ``interactive`` path.

    The old ``_MARKDOWN_TABLE_RE`` branch returned ``text`` unconditionally
    and stripped all the surrounding markdown formatting, so a Feishu
    reader saw literal pipes and lost the prose framing the table.
    """
    content = (
        "Here is the data:\n\n"
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |\n\n"
        "Let me know."
    )
    result = _call_build_outbound_payload(content)
    if isinstance(result, list):
        msg_type, payload_str = result[0]
    else:
        msg_type, payload_str = result
    assert msg_type == "interactive"
    md_contents = _card_md_contents(payload_str)
    joined = "\n".join(md_contents)
    assert "Here is the data" in joined, (
        "leading prose was lost when downgrading a mixed-table message"
    )
    assert "col A" in joined, "table header was lost"
    assert "Let me know" in joined, "trailing prose was lost"


def test_card_payload_has_schema_2_0():
    """Verify the Card JSON 2.0 structure is correct."""
    content = "**Bold** and `code`"
    result = _call_build_outbound_payload(content)
    if isinstance(result, list):
        _, payload_str = result[0]
    else:
        _, payload_str = result
    payload = json.loads(payload_str)
    assert payload.get("schema") == "2.0"
    assert payload.get("config", {}).get("wide_screen_mode") is True
    assert "elements" in payload.get("body", {})
