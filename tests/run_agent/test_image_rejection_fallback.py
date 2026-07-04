"""Tests for the image-rejection fallback in run_agent.

When a server rejects image content (e.g. text-only endpoints), the agent
strips image parts from message history and retries text-only.  These tests
verify that stripping preserves the role-alternation invariants providers
require, and that the phrase detector fires on the expected error bodies.
"""

from run_agent import _looks_like_image_content_rejection, _strip_images_from_messages


class TestStripImagesPreservesAlternation:
    """_strip_images_from_messages must not break message role alternation."""

    def test_noop_when_no_images(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is False
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]




    def test_tool_message_with_all_images_replaced_not_deleted(self):
        """CRITICAL: tool messages must NEVER be deleted — their tool_call_id
        pairs with an assistant tool_call and providers reject unmatched IDs.
        """
        msgs = [
            {"role": "user", "content": "take a screenshot"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "computer_use", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        # Length preserved — tool message NOT deleted
        assert len(msgs) == 3
        # tool_call_id still present
        assert msgs[2]["tool_call_id"] == "call_abc"
        # Content replaced with text placeholder (now a string, not a list)
        assert isinstance(msgs[2]["content"], str)
        assert "image content removed" in msgs[2]["content"].lower()

    def test_tool_message_with_mixed_content_keeps_text_parts(self):
        msgs = [
            {"role": "user", "content": "screenshot plz"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Captured 1024x768"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        assert len(msgs) == 3
        assert msgs[2]["content"] == [{"type": "text", "text": "Captured 1024x768"}]
        assert msgs[2]["tool_call_id"] == "call_1"


    def test_multiple_tool_messages_all_preserved(self):
        """Parallel tool calls: each tool_call_id must retain a paired message."""
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "image_url", "image_url": {}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c2",
                "content": [{"type": "image_url", "image_url": {}}],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}




class TestImageRejectionPhraseIsolation:
    """The image-rejection phrase list must NOT false-match on other
    image-related error categories (size-too-large, format errors, etc.)
    so they route to the correct recovery handler (e.g. _try_shrink_image_parts).
    """

    def _matches(self, body: str) -> bool:
        return _looks_like_image_content_rejection(body)

    def test_kimi_truncated_image_trips_recovery(self):
        # Kimi/Moonshot reject truncated image bytes with this 400; the
        # bad bytes are in immutable history so stripping must fire.
        body = ("HTTP 400: Invalid request: prepare image failed error, "
                "status code: 400, message: failed to decode image: invalid "
                "or unsupported image format")
        assert self._matches(body) is True

    def test_anthropic_image_too_large_does_not_trip(self):
        # From agent/error_classifier.py _IMAGE_TOO_LARGE_PATTERNS —
        # these must route to image_too_large / _try_shrink_image_parts_in_messages,
        # NOT to our vision-unsupported fallback.
        bodies = [
            "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum",
            "image too large: 6291456 bytes > 5242880 limit",
            "image_too_large",
            "image size exceeds per-request limit",
        ]
        for body in bodies:
            assert self._matches(body) is False, f"false positive on: {body}"



    def test_real_image_rejection_bodies_trip(self):
        """Positive cases — real-world error wordings that should trigger."""
        bodies = [
            "Only 'text' content type is supported.",
            "Bad request: multimodal is not supported by this model",
            "This model does not support images",
            "vision is not supported on this endpoint",
            "model does not support image input",
            # ChatGPT-account Codex backend (issue #23570) — rejects
            # data:image/...base64 URLs in input_image fields. Without this
            # match the agent cascaded into compression / context-too-large
            # recovery instead of just stripping the images.
            "Invalid 'input[56].content[1].image_url'. Expected a valid URL, but got a value with an invalid format.",
            # OpenRouter 404 when no upstream endpoint for the model accepts
            # image input — issue #21160. The exact wording from the report.
            "HTTP 404: No endpoints found that support image input",
            # Alibaba/OpenAI-compatible endpoints can reject image-bearing
            # messages without naming image_url explicitly. The first failed
            # turn should still switch to text-only/aux-vision mode (#57948).
            "The provided messages input is invalid. The error info is [Unexpected item type in content].",
            "The image data you provided does not represent a valid image. Please check your input and try again.",
        ]
        for body in bodies:
            assert self._matches(body) is True, f"false negative on: {body}"


