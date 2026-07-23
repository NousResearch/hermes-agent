"""Tests for the image-rejection fallback in run_agent.

When a server rejects image content (e.g. text-only endpoints), the agent
strips image parts from message history and retries text-only.  These tests
verify that stripping preserves the role-alternation invariants providers
require, and that the phrase detector fires on the expected error bodies.
"""

from run_agent import _strip_images_from_messages


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

    def test_string_content_untouched(self):
        """String content passes through — only list content is inspected."""
        msgs = [{"role": "user", "content": "just text"}]
        changed = _strip_images_from_messages(msgs)
        assert changed is False
        assert msgs[0]["content"] == "just text"

    def test_strips_image_url_part_preserves_text(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        assert msgs[0]["content"] == [{"type": "text", "text": "describe"}]

    def test_strips_all_recognized_image_types(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {}},
                {"type": "image", "source": {}},
                {"type": "input_image", "image_url": "http://x"},
            ],
        }]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        assert msgs[0]["content"] == [{"type": "text", "text": "hi"}]

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

    def test_image_only_user_message_dropped(self):
        """Synthetic image-only user messages (gateway injection pattern) are
        safe to drop — no tool_call_id linkage to preserve."""
        msgs = [
            {"role": "user", "content": "what's in this?"},
            {"role": "assistant", "content": "I'll check."},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        # Synthetic image-only user message dropped
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "assistant"

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

    def test_returns_false_when_nothing_changed(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "hello"},
        ]
        assert _strip_images_from_messages(msgs) is False

    def test_handles_non_dict_entries_gracefully(self):
        msgs = [None, "not a dict", {"role": "user", "content": "ok"}]
        # Must not raise
        changed = _strip_images_from_messages(msgs)
        assert changed is False


class TestImageRejectionPhraseIsolation:
    """The image-rejection phrase list must NOT false-match on other
    image-related error categories (size-too-large, format errors, etc.)
    so they route to the correct recovery handler (e.g. _try_shrink_image_parts).
    """

    # Reproduces the phrase list used in run_agent.py's error-handler block.
    _REJECTION_PHRASES = (
        "only 'text' content type is supported",
        "only text content type is supported",
        "image_url is not supported",
        "image content is not supported",
        "multimodal is not supported",
        "multimodal content is not supported",
        "multimodal input is not supported",
        "vision is not supported",
        "vision input is not supported",
        "does not support images",
        "does not support image input",
        "does not support multimodal",
        "does not support vision",
        "model does not support image",
        "image_url'. expected",
        "no endpoints found that support image input",
    )

    def _matches(self, body: str) -> bool:
        low = body.lower()
        return any(p in low for p in self._REJECTION_PHRASES)

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

    def test_context_overflow_does_not_trip(self):
        bodies = [
            "This model's maximum context length is 200000 tokens.",
            "Request too large: max tokens per request is 200000",
            "The input exceeds the context window.",
        ]
        for body in bodies:
            assert self._matches(body) is False, f"false positive on: {body}"

    def test_rate_limit_does_not_trip(self):
        bodies = [
            "rate limit reached for requests",
            "You exceeded your current quota",
        ]
        for body in bodies:
            assert self._matches(body) is False

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
        ]
        for body in bodies:
            assert self._matches(body) is True, f"false negative on: {body}"

    def test_openrouter_data_policy_no_endpoints_does_not_trip(self):
        """OpenRouter has several 'no endpoints ...' 404 bodies. Only the
        image-input one is an image rejection — the guardrail / data-policy
        variants (agent/error_classifier.py) are about routing restrictions,
        not vision, and must route to their own handler, not get their images
        stripped.
        """
        bodies = [
            "No endpoints available matching your guardrail restrictions",
            "No endpoints available matching your data policy",
            "No endpoints found matching your data policy",
        ]
        for body in bodies:
            assert self._matches(body) is False, f"false positive on: {body}"

    def test_codex_data_url_rejection_does_not_false_match_other_url_errors(self):
        """The narrow 'image_url'. expected' phrase (keyed on the
        field-path apostrophe used in the Codex Responses error format)
        must NOT trip on URL validation errors that aren't about
        image_url specifically. See issue #23570 for the original error.
        """
        bodies = [
            # Generic URL validation errors — should NOT trip
            "Invalid webhook_url. Must be a valid URL.",
            "Expected a valid URL but got an empty string.",
            "redirect_uri does not look like a valid URL.",
            # An image_url error worded differently — also should not trip
            # the narrow phrase (a separate phrase would be needed)
            "image_url field cannot be empty",
        ]
        for body in bodies:
            assert self._matches(body) is False, f"false positive on: {body}"


class TestStripImagesDropsStaleApiContent:
    """The strip runs on the persistent history, not just the per-call copy.

    ``api_content`` is the byte-stability sidecar: it holds the exact bytes
    previously sent for a message, and the next turn substitutes it back into
    ``content``. Leaving it in place on a message this function rewrote would
    replay the images the strip just removed — and the recovery cannot re-fire,
    because it sets ``_vision_supported = False`` and gates itself on that. The
    session would then send rejected images on every subsequent turn.

    Same contract the other content-rewrite paths follow (stale-confirmation
    redaction in ``replay_cleanup``, compression rewrites, merge-into-tail):
    "the cost is one cache boundary miss, never wrong content".
    """

    @staticmethod
    def _wire(msg):
        """What the next turn actually sends for this history message."""
        from agent.turn_context import substitute_api_content

        api_msg = msg.copy()
        substitute_api_content(api_msg)
        return api_msg["content"]

    def _image_msg(self, sidecar="look<IMAGE BYTES SENT LAST TURN>"):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "api_content": sidecar,
        }

    def test_stripped_message_loses_its_sidecar(self):
        msgs = [self._image_msg()]
        assert _strip_images_from_messages(msgs) is True
        assert "api_content" not in msgs[0]

    def test_next_turn_does_not_resend_the_stripped_images(self):
        msgs = [self._image_msg()]
        _strip_images_from_messages(msgs)

        wire = self._wire(msgs[0])
        assert "IMAGE BYTES" not in str(wire), (
            "the stale sidecar replayed the images the strip removed"
        )
        assert wire == [{"type": "text", "text": "look"}]

    def test_tool_placeholder_message_also_loses_its_sidecar(self):
        """An image-only tool result becomes a placeholder — same rewrite."""
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                "api_content": "<SCREENSHOT BYTES>",
            }
        ]
        assert _strip_images_from_messages(msgs) is True
        assert "api_content" not in msgs[0]
        assert "image content removed" in msgs[0]["content"]

    def test_untouched_messages_keep_their_sidecar(self):
        """Only rewritten messages pay the cache boundary — not the whole prefix."""
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "no images here"}],
                "api_content": "no images here<injected ctx>",
            },
            self._image_msg(),
        ]
        _strip_images_from_messages(msgs)

        assert msgs[0]["api_content"] == "no images here<injected ctx>"
        assert "api_content" not in msgs[1]
