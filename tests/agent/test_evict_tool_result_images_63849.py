"""Tests for ``_evict_old_tool_result_images`` (issue #63849).

Tool-result images (browser_vision / vision_analyze screenshots) accumulated
forever on the OpenAI-compatible request path because the only eviction lived
in the Anthropic adapter. This test suite pins the per-call eviction that
replaces old tool-role image parts with text placeholders.
"""

import copy

from agent.context_compressor import _evict_old_tool_result_images


def _make_tool_msg(content_type: str = "image_url") -> dict:
    """Helper: a tool-role message with one image part."""
    return {
        "role": "tool",
        "content": [
            {"type": content_type, "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


def _make_user_msg(content_type: str = "image_url") -> dict:
    """Helper: a user-role message with one image part."""
    return {
        "role": "user",
        "content": [
            {"type": content_type, "image_url": {"url": "data:image/png;base64,BBBB"}},
        ],
    }


def _make_tool_msg_no_image() -> dict:
    """Helper: a tool-role message with text-only content."""
    return {
        "role": "tool",
        "content": [
            {"type": "text", "text": "command output"},
        ],
    }


class TestEvictOldToolResultImages:
    """Regression suite for _evict_old_tool_result_images."""

    def test_no_images_returns_unchanged(self):
        """No image-bearing messages => returns the same list reference."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": [{"type": "text", "text": "output"}]},
        ]
        result = _evict_old_tool_result_images(messages)
        assert result is messages, "Should return the same list when no eviction"

    def test_empty_list_returns_empty(self):
        """Empty input => empty output."""
        assert _evict_old_tool_result_images([]) == []

    def test_messages_below_threshold_unchanged(self):
        """≤3 tool screenshots stays unchanged (same reference)."""
        messages = [_make_tool_msg() for _ in range(3)]
        messages.insert(0, {"role": "user", "content": "look at this"})
        result = _evict_old_tool_result_images(messages)
        assert result is messages, "≤3 tool images should not trigger eviction"

    def test_keeps_last_n_images(self):
        """5 tool screenshots → keep the newest 3, evict the oldest 2."""
        messages = [_make_tool_msg() for _ in range(5)]
        result = _evict_old_tool_result_images(messages, max_keep=3)

        # We should get 5 messages back (same length)
        assert len(result) == 5

        # The first 2 (oldest) tool messages should have their image parts
        # replaced with text placeholders.
        for i in range(2):
            parts = result[i]["content"]
            assert isinstance(parts, list)
            for p in parts:
                assert p["type"] == "text", f"Msg {i}: expected text placeholder"
                assert "screenshot removed" in p["text"]

        # The last 3 (newest) tool messages should keep their original images.
        for i in range(2, 5):
            parts = result[i]["content"]
            assert isinstance(parts, list)
            for p in parts:
                assert p["type"] != "text" or "screenshot" not in p.get("text", ""), (
                    f"Msg {i}: newest should keep image parts"
                )

    def test_does_not_mutate_input(self):
        """Input list and its messages must not be mutated."""
        messages = [_make_tool_msg() for _ in range(5)]
        original = copy.deepcopy(messages)

        _evict_old_tool_result_images(messages, max_keep=3)

        # Input list should be unchanged
        assert messages == original, "Input list was mutated"

    def test_input_not_mutated_when_count_exceeds(self):
        """Even when eviction triggers, original list is unmodified."""
        messages = [_make_tool_msg() for _ in range(5)]
        original = copy.deepcopy(messages)

        _evict_old_tool_result_images(messages, max_keep=3)

        assert messages == original, (
            "Input list was mutated even though eviction occurred"
        )
        # Verify the content parts are still image type
        for msg in messages:
            for p in msg["content"]:
                # OpenAI tool-image parts are type "image_url"
                assert p["type"] != "text" or "screenshot" not in p.get("text", "")

    def test_user_images_not_touched(self):
        """User-role images should never be evicted."""
        messages = [
            _make_user_msg(),
            _make_tool_msg(),
            _make_user_msg(),
            _make_tool_msg(),
            _make_tool_msg(),
            _make_tool_msg(),
            _make_tool_msg(),
        ]
        result = _evict_old_tool_result_images(messages, max_keep=2)

        # User messages should keep their images
        assert result[0]["content"] == messages[0]["content"]
        assert result[2]["content"] == messages[2]["content"]

        # Tool messages beyond max_keep should have placeholders
        for i in [1, 3]:  # tool messages at positions 1 and 3
            parts = result[i]["content"]
            assert parts[0]["type"] == "text"
            assert "screenshot removed" in parts[0]["text"]

    def test_anthropic_image_type_also_evicted(self):
        """Anthropic 'image' type inside tool messages is also evicted.

        This covers the scenario where an Anthropic native image block
        ends up in an OpenAI-format tool message (rare but possible
        during format conversion).
        """
        messages = []
        for _ in range(5):
            messages.append({
                "role": "tool",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    }},
                ],
            })
        result = _evict_old_tool_result_images(messages, max_keep=3)

        # First 2 should be replaced
        for i in range(2):
            parts = result[i]["content"]
            assert parts[0]["type"] == "text"
            assert "screenshot removed" in parts[0]["text"]

        # Last 3 should keep original
        for i in range(2, 5):
            assert result[i]["content"] == messages[i]["content"]

    def test_mixed_text_and_image_in_tool_message(self):
        """A tool message with both text and image parts — only the image
        part is replaced; text parts survive."""
        multi_msg = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "stdout: done"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCCC"}},
                {"type": "text", "text": "stderr: none"},
            ],
        }
        # multi_msg is the oldest, so it gets evicted (max_keep=1).
        messages = [multi_msg, _make_tool_msg(), _make_tool_msg(), _make_tool_msg()]
        result = _evict_old_tool_result_images(messages, max_keep=1)

        # The evicted message (the oldest beyond max_keep)
        evicted = result[0]
        parts = evicted["content"]
        assert len(parts) == 3
        assert parts[0]["type"] == "text" and parts[0]["text"] == "stdout: done"
        assert parts[1]["type"] == "text" and "screenshot removed" in parts[1]["text"]
        assert parts[2]["type"] == "text" and parts[2]["text"] == "stderr: none"

        # The other evicted (pos 1, 2) are single-image → 1 placeholder part
        for i in (1, 2):
            parts_i = result[i]["content"]
            assert len(parts_i) == 1
            assert parts_i[0]["type"] == "text"
            assert "screenshot removed" in parts_i[0]["text"]

    def test_input_image_type_recognized(self):
        """OpenAI Responses API 'input_image' type is also evicted."""
        messages = []
        for _ in range(5):
            messages.append({
                "role": "tool",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,DDDD"},
                ],
            })
        result = _evict_old_tool_result_images(messages, max_keep=2)

        assert len(result) == 5
        # First 3 should be replaced
        for i in range(3):
            assert result[i]["content"][0]["type"] == "text"
        # Last 2 keep original
        for i in range(3, 5):
            assert result[i]["content"] == messages[i]["content"]

    def test_tool_messages_without_images_not_counted(self):
        """Tool messages that have no image parts don't count toward max_keep."""
        messages = [
            _make_tool_msg(),           # image-bearing (counted)
            _make_tool_msg_no_image(),  # not image-bearing (not counted)
            _make_tool_msg(),           # image-bearing (counted)
            _make_tool_msg_no_image(),  # not image-bearing (not counted)
            _make_tool_msg(),           # image-bearing (counted)
            _make_tool_msg(),           # image-bearing (exceeds max_keep)
        ]
        result = _evict_old_tool_result_images(messages, max_keep=3)

        # Position 0 (oldest image) should be evicted
        assert result[0]["content"][0]["type"] == "text"
        assert "screenshot removed" in result[0]["content"][0]["text"]

        # Non-image-bearing messages at 1 and 3 should be untouched
        assert result[1] is messages[1]
        assert result[3] is messages[3]

        # Newest images at 2, 4, 5 should be untouched
        assert result[2]["content"] == messages[2]["content"]
        assert result[4]["content"] == messages[4]["content"]
        assert result[5]["content"] == messages[5]["content"]
