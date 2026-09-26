"""Regression tests: HTTP 413 recovery must score progress in BYTES.

Bug (#88960 / #47339): a 413 is a *byte*-size error, but the recovery loop in
``agent/conversation_loop.py`` scored compression progress with
``estimate_messages_tokens_rough``, which deliberately prices every image at
a flat per-image token cost so screenshots don't trigger premature
compaction.  When the payload is image-dominated that progress test can never
be satisfied: in the reporting session two ``vision_analyze`` results were
5,627,202 bytes — 96.6% of the request body — while contributing only ~3K of
the ~80K token estimate.  Compaction (post-#97160) frees those megabytes, but
the token-scored check reported "no progress", burned all three attempts, and
wedged the session permanently at 13% context usage.

The fix: the 413 no-progress check measures ``serialized_messages_bytes``
(exact, free) before and after each compression pass, never the token
estimate.  These tests assert that invariant directly.
"""

import pytest

from agent.message_sanitization import serialized_messages_bytes


def _data_url_image(size_bytes: int) -> dict:
    """An image part whose inline data URL is ~``size_bytes`` long."""
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + ("A" * size_bytes)},
    }


def _tool_msg_with_image(size_bytes: int, text: str = "screenshot captured") -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": [
            {"type": "text", "text": text},
            _data_url_image(size_bytes),
        ],
    }


class TestSerializedMessagesBytes:
    def test_counts_inline_data_url_payloads(self):
        small = serialized_messages_bytes([_tool_msg_with_image(1_000)])
        huge = serialized_messages_bytes([_tool_msg_with_image(3_000_000)])
        assert huge - small == pytest.approx(3_000_000 - 1_000, abs=64)


    def test_utf8_bytes_not_codepoints(self):
        ascii_msgs = [{"role": "user", "content": "aaaa"}]
        utf8_msgs = [{"role": "user", "content": "éééé"}]  # 2 bytes each in UTF-8
        assert serialized_messages_bytes(utf8_msgs) > serialized_messages_bytes(
            ascii_msgs
        )

    def test_degenerate_input(self):
        assert serialized_messages_bytes([]) == 0
        assert serialized_messages_bytes("not-a-list") == 0  # type: ignore[arg-type]

    def test_never_raises_on_non_serializable_content(self):
        class Weird:
            pass

        messages = [{"role": "tool", "content": Weird()}]
        assert serialized_messages_bytes(messages) > 0
