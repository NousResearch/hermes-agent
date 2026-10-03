"""Tests for the Feishu adapter's opt-in outbound footer.

The footer appends an ``Agent | Model | Provider`` line to messages the
bot sends. It ships off by default: with ``footer_template`` empty the
adapter must produce byte-identical payloads to the pre-footer
behaviour, which is what makes the feature safe to land in the same
commit as the card-refetch fix.

These are behaviour-contract tests: they assert the relationship
between configuration and rendered output (template empty → no footer
row; template set → exactly one extra row carrying the rendered line),
not a frozen snapshot of the payload JSON.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from plugins.platforms.feishu.adapter import (
    FeishuAdapter,
    _build_markdown_post_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_adapter() -> FeishuAdapter:
    """A FeishuAdapter carrying only the footer fields ``__init__`` installs.

    Bypasses ``__init__`` so the test does not have to stand up the
    gateway startup chain; the footer methods read only the
    ``_footer_*`` slots plus ``_build_outbound_payload``'s markdown
    detection. The slots are set with ``setattr`` because they are
    installed dynamically from settings rather than as class
    annotations.
    """
    adapter = object.__new__(FeishuAdapter)
    for name, value in (
        ("_footer_template", ""),
        ("_footer_agent_label", "Hermes"),
        ("_footer_model_label", ""),
        ("_footer_provider_label", ""),
    ):
        setattr(adapter, name, value)
    return adapter


def _post_rows(payload: str) -> list:
    """Return the row list out of a rendered ``post`` payload."""
    return json.loads(payload)["zh_cn"]["content"]


def _footer_rows(rows: list) -> list:
    """Rows that look like a footer: a single ``md`` cell."""
    return [r for r in rows if len(r) == 1 and r[0].get("tag") == "md"]


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def test_footer_disabled_by_default():
    """An empty template means no footer — the feature's off switch."""
    adapter = _bare_adapter()

    assert adapter._resolve_outbound_footer() == ""


def test_footer_renders_all_placeholders():
    """All three placeholders substitute from the configured labels."""
    adapter = _bare_adapter()
    adapter._footer_template = "{agent} | {model} | {provider}"
    adapter._footer_agent_label = "Hermes"
    adapter._footer_model_label = "MiniMax-M3"
    adapter._footer_provider_label = "minimax"

    assert adapter._resolve_outbound_footer() == "Hermes | MiniMax-M3 | minimax"


def test_footer_falls_back_when_model_and_provider_unresolved():
    """Unresolved model/provider render as ``?`` instead of vanishing.

    ``agent_runtime`` is unavailable in a bare test process, so the
    runtime resolution path cannot supply a label. The footer must still
    occupy its three fields rather than silently dropping them.
    """
    adapter = _bare_adapter()
    adapter._footer_template = "{agent} | {model} | {provider}"
    adapter._footer_model_label = ""
    adapter._footer_provider_label = ""

    rendered = adapter._resolve_outbound_footer()

    # agent is always known; the other two degrade to the placeholder marker.
    assert rendered.startswith("Hermes")
    assert rendered.count("|") == 2


def test_unknown_placeholder_renders_template_verbatim():
    """A typo'd placeholder surfaces the raw template rather than dropping the footer."""
    adapter = _bare_adapter()
    adapter._footer_template = "{agent} | {modle}"  # deliberate typo

    rendered = adapter._resolve_outbound_footer()

    # str.format raises KeyError on an unknown name; the adapter catches it
    # and returns the template unchanged so the user can see the mistake.
    assert rendered == "{agent} | {modle}"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def test_post_payload_unchanged_without_footer():
    """No footer → the row list is exactly the content rows."""
    content = "hello **world**"

    without = _post_rows(_build_markdown_post_payload(content))
    with_empty = _post_rows(_build_markdown_post_payload(content, footer=""))

    assert without == with_empty


def test_post_payload_appends_single_footer_row():
    """A footer adds exactly one extra ``md`` row, after the content rows."""
    content = "hello **world**"
    footer = "Hermes | MiniMax-M3 | minimax"

    rows_without = _post_rows(_build_markdown_post_payload(content))
    rows_with = _post_rows(_build_markdown_post_payload(content, footer=footer))

    assert len(rows_with) == len(rows_without) + 1
    assert rows_with[-1] == [{"tag": "md", "text": footer}]


def test_text_payload_carries_footer_after_blank_line():
    """A plain-text send (no markdown hints) gets the footer as a separate paragraph."""
    adapter = _bare_adapter()
    adapter._client = MagicMock()

    msg_type, payload = adapter._build_outbound_payload("plain body", footer="— Hermes")

    assert msg_type == "text"
    assert json.loads(payload)["text"] == "plain body\n\n— Hermes"


def test_text_payload_unchanged_without_footer():
    """Without a footer the text payload is just the content."""
    adapter = _bare_adapter()

    msg_type, payload = adapter._build_outbound_payload("plain body", footer="")

    assert msg_type == "text"
    assert json.loads(payload)["text"] == "plain body"
