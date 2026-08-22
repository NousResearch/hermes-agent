"""Regression tests: HTTP 413 recovery when the payload is image-dominated.

Bug: a 413 is a *byte*-size error, but ``estimate_messages_tokens_rough``
prices every image at a flat per-image token cost so that screenshots don't
trigger premature compaction.  That makes text compression structurally
unable to clear an image-dominated 413 — two ~3MB base64 screenshots can be
>95% of the request body while contributing only ~3K to the token estimate,
so every compression attempt reports "no progress", the attempt budget is
burned, and the session wedges permanently (images are re-sent from stored
history on every subsequent turn, so /compress and /retry can't help either).

These tests assert the invariant that actually matters: recovery must be
driven by measured payload BYTES, not by the token estimate.
"""

import pytest

from agent.message_sanitization import (
    _image_part_payload_bytes,
    strip_oversized_image_parts,
)
from agent.model_metadata import estimate_messages_tokens_rough


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


class TestImagePartPayloadBytes:
    def test_measures_inline_data_url_length(self):
        part = _data_url_image(1000)
        assert _image_part_payload_bytes(part) >= 1000

    def test_remote_url_costs_only_its_reference(self):
        part = {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
        assert _image_part_payload_bytes(part) < 100

    def test_non_image_parts_cost_nothing(self):
        assert _image_part_payload_bytes({"type": "text", "text": "x" * 5000}) == 0
        assert _image_part_payload_bytes("not-a-dict") == 0
        assert _image_part_payload_bytes({"type": "image_url"}) == 0


class TestTokenEstimateIsBlindToImageBytes:
    """The root cause, asserted directly.

    This is why compression can never clear an image-dominated 413: the
    estimate barely moves regardless of how many megabytes are on the wire.
    """

    def test_estimate_barely_moves_as_image_bytes_explode(self):
        small = [_tool_msg_with_image(1_000)]
        huge = [_tool_msg_with_image(3_000_000)]

        small_tokens = estimate_messages_tokens_rough(small)
        huge_tokens = estimate_messages_tokens_rough(huge)

        # ~3000x more bytes on the wire...
        assert len(huge[0]["content"][1]["image_url"]["url"]) > 2_000_000

        # ...but the token estimate is essentially unchanged, so the
        # "did compression make progress?" test (new < original * 0.95)
        # can never be satisfied by summarizing text.
        assert huge_tokens < small_tokens * 2


class TestStripOversizedImageParts:
    def test_strips_image_exceeding_the_byte_budget(self):
        messages = [_tool_msg_with_image(3_000_000)]
        changed, reclaimed = strip_oversized_image_parts(
            messages, max_image_bytes=256 * 1024
        )
        assert changed is True
        assert reclaimed > 2_000_000

    def test_preserves_small_images(self):
        messages = [_tool_msg_with_image(1_000)]
        changed, reclaimed = strip_oversized_image_parts(
            messages, max_image_bytes=256 * 1024
        )
        assert changed is False
        assert reclaimed == 0
        assert any(
            p.get("type") == "image_url" for p in messages[0]["content"]
        ), "small image must survive"

    def test_preserves_text_alongside_a_stripped_image(self):
        messages = [_tool_msg_with_image(3_000_000, text="the important finding")]
        strip_oversized_image_parts(messages, max_image_bytes=256 * 1024)
        remaining = messages[0]["content"]
        assert any(
            isinstance(p, dict) and p.get("text") == "the important finding"
            for p in remaining
        ), "text must survive image stripping"

    def test_tool_message_stripped_to_nothing_keeps_its_slot(self):
        """Deleting it would orphan the assistant's tool_call_id -> HTTP 400."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                "content": [_data_url_image(3_000_000)],
            }
        ]
        changed, _ = strip_oversized_image_parts(
            messages, max_image_bytes=256 * 1024
        )
        assert changed is True
        assert len(messages) == 1, "tool message must not be deleted"
        assert messages[0]["tool_call_id"] == "call_xyz"
        assert isinstance(messages[0]["content"], str)

    def test_protect_last_n_shields_the_current_turn(self):
        """An image the user just attached must not vanish mid-turn."""
        messages = [
            _tool_msg_with_image(3_000_000, text="old"),
            _tool_msg_with_image(3_000_000, text="recent"),
        ]
        changed, _ = strip_oversized_image_parts(
            messages, max_image_bytes=256 * 1024, protect_last_n=1
        )
        assert changed is True
        # Older one stripped, most recent preserved.
        assert isinstance(messages[0]["content"], list)
        assert not any(
            p.get("type") == "image_url" for p in messages[0]["content"]
        )
        assert any(
            p.get("type") == "image_url" for p in messages[1]["content"]
        ), "protected recent image must survive"

    def test_reclaims_enough_to_clear_a_realistic_413(self):
        """End-to-end sizing: the real-world shape that wedged a session."""
        messages = [{"role": "user", "content": "text turn"} for _ in range(190)]
        messages.insert(50, _tool_msg_with_image(2_756_000))
        messages.insert(80, _tool_msg_with_image(2_871_000))

        before = sum(
            _image_part_payload_bytes(p)
            for m in messages
            if isinstance(m.get("content"), list)
            for p in m["content"]
        )
        assert before > 5_000_000

        changed, reclaimed = strip_oversized_image_parts(
            messages, max_image_bytes=256 * 1024, protect_last_n=4
        )
        assert changed is True
        assert reclaimed > 5_000_000

    def test_no_op_on_degenerate_input(self):
        assert strip_oversized_image_parts([], max_image_bytes=1024) == (False, 0)
        assert strip_oversized_image_parts(
            [_tool_msg_with_image(3_000_000)], max_image_bytes=0
        ) == (False, 0)
        assert strip_oversized_image_parts(
            "not-a-list",  # type: ignore[arg-type]
            max_image_bytes=1024,
        ) == (False, 0)
