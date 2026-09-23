"""Behavior-contract tests for merge_forward expansion and button-URL extraction.

Regression: Feishu merge_forward messages arrive with fixed body content
"Merged and Forwarded Message"; the real sub-messages are only retrievable via
GET /im/v1/messages/{id} (items[0] = the merge_forward, items[1:] = forwarded
originals). Before the fix the gateway showed only "[Merged forward message]".
"""

import json

import pytest

from plugins.platforms.feishu.adapter import (
    _collect_action_urls,
    _normalize_interactive_message,
    normalize_feishu_message,
)


def test_merge_forward_with_submessages_uses_placeholder_when_payload_has_none():
    """merge_forward payload without entries falls back to the placeholder text."""
    normalized = normalize_feishu_message(message_type="merge_forward", raw_content="")
    assert normalized.text_content == "[Merged forward message]"
    assert normalized.metadata == {"entry_count": 0, "title": ""}


def test_merge_forward_extracts_entries_from_payload():
    payload = {
        "title": "转发",
        "messages": [{"sender_name": "Alice", "text": "hello"}],
    }
    normalized = normalize_feishu_message(
        message_type="merge_forward", raw_content=json.dumps(payload)
    )
    assert "转发" in normalized.text_content
    assert "Alice: hello" in normalized.text_content
    assert normalized.metadata["entry_count"] == 1


def test_collect_action_urls_extracts_multi_url_buttons():
    card = {
        "elements": [
            [{"tag": "button", "text": "打开文档", "type": "primary",
              "multi_url": {"url": "https://example.com/doc"}}],
        ]
    }
    assert _collect_action_urls(card) == ["打开文档: https://example.com/doc"]


def test_collect_action_urls_extracts_plain_url_buttons():
    card = {"elements": [[{"tag": "button", "text": "查看", "url": "https://example.com/x"}]]}
    assert _collect_action_urls(card) == ["查看: https://example.com/x"]


def test_collect_action_urls_ignores_callback_only_buttons():
    """Callback buttons carry no URL field — nothing to extract, no false links."""
    card = {"elements": [[{"tag": "button", "text": "查看原文", "type": "primary"}]]}
    assert _collect_action_urls(card) == []


def test_interactive_card_with_url_button_appends_link_line():
    card = {
        "elements": [
            [{"tag": "button", "text": "打开文档", "type": "primary",
              "multi_url": {"url": "https://example.com/doc"}}],
        ]
    }
    normalized = _normalize_interactive_message("interactive", card)
    assert "链接: 打开文档: https://example.com/doc" in normalized.text_content


def test_interactive_callback_button_has_no_link_line():
    card = {"elements": [[{"tag": "button", "text": "查看原文", "type": "primary"}]]}
    normalized = _normalize_interactive_message("interactive", card)
    assert "链接:" not in normalized.text_content
    assert "查看原文" in normalized.text_content
