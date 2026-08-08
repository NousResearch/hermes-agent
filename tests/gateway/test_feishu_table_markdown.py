"""Tests for Feishu adapter outbound markdown payload construction.

Reproduces the bug tracked in hermes-agent issue #52786:
`_build_outbound_payload` was force-downgrading any message containing a
markdown pipe table to ``msg_type=text``, so Feishu clients rendered the raw
pipe-and-dash source instead of a table.

This file also guards the Card-Kit table rendering path (default
``table_mode="card"``): tables are sent as ``interactive`` cards with a
Card-Kit 2.0 native ``table`` element — the only stable table rendering
path in Feishu (post-type ``md`` elements drop tables silently). With
``table_mode="ascii"`` the same content goes out as a ``post`` with an
ASCII code-fence table. Either way, a table-shaped message must never be
downgraded to plain ``text``.

These tests invoke the real adapter via the project's plugin-loader helper
so that no ``sys.path`` / ``sys.modules`` games are needed.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(
    content: str, table_mode: str = "card"
) -> tuple[str, str]:
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` only uses module-level helpers
    (``_MARKDOWN_TABLE_RE``, ``_MARKDOWN_HINT_RE``,
    ``_build_markdown_post_payload``) plus the ``_table_mode`` attribute
    (read defensively with a ``card`` default), so a bare object with the
    attribute set is sufficient.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    inst._table_mode = table_mode
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


_TABLE_CONTENT = (
    "| col A | col B |\n"
    "| ----- | ----- |\n"
    "| 1     | 2     |"
)


def test_markdown_table_uses_interactive_card_not_text():
    """Regression test for issue #52786 (and its older sibling #23938).

    With the default ``table_mode="card"`` a table-shaped message must take
    the Card-Kit ``interactive`` path — never be downgraded to plain text.
    """
    msg_type, payload_str = _call_build_outbound_payload(_TABLE_CONTENT)
    assert msg_type == "interactive", (
        f"expected 'interactive' card for a markdown table, got {msg_type!r}; "
        "the table-downgrade branch in _build_outbound_payload has been re-introduced"
    )
    payload = json.loads(payload_str)
    assert payload["schema"] == "2.0"
    elements = payload["body"]["elements"]
    assert elements and elements[0]["tag"] == "table", (
        "card payload must include a native table element"
    )
    assert [c["name"] for c in elements[0]["columns"]] == ["col A", "col B"]


def test_markdown_table_uses_post_not_text_in_ascii_mode():
    """With ``table_mode="ascii"`` a table must go out as ``post`` (ASCII
    code-fence), not plain text."""
    msg_type, payload_str = _call_build_outbound_payload(
        _TABLE_CONTENT, table_mode="ascii"
    )
    assert msg_type == "post", (
        f"expected 'post' for a markdown table in ascii mode, got {msg_type!r}"
    )
    md_texts = _md_texts_from_post_payload(payload_str)
    assert md_texts, f"post payload must include at least one md element; got {payload_str!r}"
    joined = "".join(md_texts)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted when switching from text to post"
    )
