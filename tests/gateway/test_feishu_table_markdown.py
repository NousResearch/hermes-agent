"""Tests for Feishu adapter outbound markdown payload construction.

Reproduces the bug tracked in hermes-agent issue #52786:
`_build_outbound_payload` was force-downgrading any message containing a
markdown pipe table to ``msg_type=text``, so Feishu clients rendered the raw
pipe-and-dash source instead of a table.  Empirically current Feishu clients
render ``post``+``md`` tables natively, so the downgrade branch must be removed.

These tests guard the fix.  They invoke the real adapter via the project's
plugin-loader helper so that no ``sys.path`` / ``sys.modules`` games are
needed.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(content: str) -> tuple[str, str]:
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` is a method that only uses module-level
    helpers (``_MARKDOWN_TABLE_RE``, ``_MARKDOWN_HINT_RE``,
    ``_build_markdown_post_payload``) and never touches ``self.*``, so a bare
    object is sufficient.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    return inst._build_outbound_payload(content)


def _md_texts_from_post_payload(payload_str: str) -> list[str]:
    """Pull every ``{tag:'md', text:'...'}`` element out of a Feishu post payload.

    Real payload shape::

        {"zh_cn": {"content": [[{"tag": "md", "text": "..."}], ...]}}

    Helpers and tests need to introspect the ``md`` blocks regardless of
    locale, so we walk the structure generically.
    """
    payload = json.loads(payload_str)
    if not isinstance(payload, dict):
        return []
    texts: list[str] = []
    for lang_val in payload.values():
        if not isinstance(lang_val, dict):
            continue
        content = lang_val.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, list):
                candidates = block
            else:
                candidates = [block]
            for el in candidates:
                if isinstance(el, dict) and el.get("tag") == "md":
                    texts.append(el.get("text", ""))
    return texts


def test_markdown_table_uses_post_not_text():
    """Regression test for issue #52786 (and its older sibling #23938).

    A message whose only markdown is a table must take the ``post`` path,
    not be downgraded to plain text.
    """
    content = (
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |"
    )
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "post", (
        f"expected 'post' for a markdown table (issue #52786), got {msg_type!r}; "
        "the table-downgrade branch in _build_outbound_payload has been re-introduced"
    )
    md_texts = _md_texts_from_post_payload(payload_str)
    assert md_texts, f"post payload must include at least one md element; got {payload_str!r}"
    joined = "".join(md_texts)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted when switching from text to post"
    )


def test_plain_text_without_markdown_still_uses_text():
    """Negative control: a message with no markdown hints and no table must
    still go to plain text.  Guards against accidentally promoting everything
    to ``post``."""
    msg_type, _ = _call_build_outbound_payload("just a plain sentence with no markup")
    assert msg_type == "text"


def test_existing_markdown_heading_still_uses_post():
    """Sanity: the existing ``post`` path (heading / list / code / bold /
    link) must still work after the table downgrade is removed."""
    msg_type, payload_str = _call_build_outbound_payload("# hello world\n")
    assert msg_type == "post"
    md_texts = _md_texts_from_post_payload(payload_str)
    assert md_texts, f"expected at least one md element; got {payload_str!r}"
    assert any("hello world" in t for t in md_texts), (
        f"expected 'hello world' in md elements; got {md_texts!r}"
    )


def test_table_combined_with_other_markdown_does_not_downgrade():
    """A message that mixes a table with surrounding markdown must also
    take the ``post`` path.

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
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "post"
    md_texts = _md_texts_from_post_payload(payload_str)
    joined = "\n".join(md_texts)
    assert "Here is the data" in joined, (
        "leading prose was lost when downgrading a mixed-table message"
    )
    assert "col A" in joined, "table header was lost"
    assert "Let me know" in joined, "trailing prose was lost"
