"""Tests for computer_use screenshot eviction on the chat-completions path.

Covers ``agent.chat_completion_helpers._evict_old_computer_use_screenshots``,
the non-Anthropic counterpart to ``anthropic_adapter._evict_old_screenshots``.
"""

import copy

import pytest

from agent.chat_completion_helpers import _evict_old_computer_use_screenshots

PLACEHOLDER = {"type": "text", "text": "[screenshot removed to save context]"}


def _multimodal_screenshot(n: int) -> dict:
    """A computer_use tool result carrying one screenshot, in the
    ``_multimodal`` envelope shape produced by tools/computer_use/tool.py."""
    return {
        "role": "tool",
        "tool_call_id": f"call_{n}",
        "content": {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": f"screenshot {n}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,AAAA{n}"},
                },
            ],
            "text_summary": f"screenshot {n}",
            "meta": {},
        },
    }


def _plain_list_screenshot(n: int) -> dict:
    """A vision message whose content is a bare list of parts (no envelope)."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"look {n}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,BBBB{n}"}},
        ],
    }


def _image_parts(content) -> list:
    parts = content["content"] if isinstance(content, dict) else content
    return [p for p in parts if isinstance(p, dict) and p.get("type") == "image_url"]


def test_keeps_only_last_three_screenshots():
    messages = [{"role": "user", "content": "hi"}]
    messages += [_multimodal_screenshot(i) for i in range(5)]

    out = _evict_old_computer_use_screenshots(messages)

    kept = [m for m in out if _image_parts(m["content"])]
    # screenshots 2, 3, 4 keep their image; 0 and 1 are stripped.
    assert len(kept) == 3
    assert [m["tool_call_id"] for m in kept] == ["call_2", "call_3", "call_4"]


def test_old_screenshots_replaced_with_placeholder():
    messages = [_multimodal_screenshot(i) for i in range(4)]

    out = _evict_old_computer_use_screenshots(messages)

    stripped = out[0]["content"]["content"]
    assert PLACEHOLDER in stripped
    assert not _image_parts(stripped)
    # The text part of the stripped message is preserved.
    assert {"type": "text", "text": "screenshot 0"} in stripped


def test_under_budget_is_identity():
    messages = [_multimodal_screenshot(i) for i in range(3)]
    out = _evict_old_computer_use_screenshots(messages)
    assert out is messages


def test_plain_list_content_is_evicted_too():
    messages = [_plain_list_screenshot(i) for i in range(4)]

    out = _evict_old_computer_use_screenshots(messages)

    kept = [m for m in out if _image_parts(m["content"])]
    assert len(kept) == 3
    assert PLACEHOLDER in out[0]["content"]


def test_input_is_not_mutated():
    messages = [_multimodal_screenshot(i) for i in range(5)]
    snapshot = copy.deepcopy(messages)

    _evict_old_computer_use_screenshots(messages)

    assert messages == snapshot


def test_non_screenshot_messages_untouched():
    messages = [
        {"role": "system", "content": "you are helpful"},
        _multimodal_screenshot(0),
        {"role": "assistant", "content": "ok"},
        _multimodal_screenshot(1),
        _multimodal_screenshot(2),
        _multimodal_screenshot(3),
    ]

    out = _evict_old_computer_use_screenshots(messages)

    # Only screenshot 0 is over budget; the text messages pass through by identity.
    assert out[0] is messages[0]
    assert out[2] is messages[2]
    assert not _image_parts(out[1]["content"])


def test_custom_keep_budget():
    messages = [_multimodal_screenshot(i) for i in range(4)]
    out = _evict_old_computer_use_screenshots(messages, keep=1)
    kept = [m for m in out if _image_parts(m["content"])]
    assert len(kept) == 1
    assert kept[0]["tool_call_id"] == "call_3"
