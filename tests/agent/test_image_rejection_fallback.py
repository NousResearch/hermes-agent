"""Tests for the image-rejection fallback in run_agent.

When a server rejects image content (e.g. text-only endpoints), the agent
strips image parts from message history and retries text-only.  These tests
verify that stripping preserves the role-alternation invariants providers
require, and that the phrase detector fires on the expected error bodies.
"""

from agent.message_sanitization import _looks_like_image_content_rejection, _strip_images_from_messages


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

    def test_assistant_with_tool_calls_and_image_only_content_preserved(self):
        """Assistant messages carrying tool_calls must NEVER be deleted —
        dropping them would orphan the paired tool responses, which providers
        reject with unmatched tool_call_id errors.
        """
        msgs = [
            {"role": "user", "content": "annotate this screenshot"},
            {
                "role": "assistant",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "annotate", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "done"},
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        # Length preserved — assistant message with tool_calls NOT deleted
        assert len(msgs) == 3
        assert msgs[1]["tool_calls"][0]["id"] == "call_xyz"
        # Content replaced with text placeholder (now a string, not a list)
        assert isinstance(msgs[1]["content"], str)
        assert "image content removed" in msgs[1]["content"].lower()
        # Paired tool response still matches
        assert msgs[2]["tool_call_id"] == "call_xyz"

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


class TestStripImagesDropsStaleApiContent:
    """The strip runs on the persistent history, not just the per-call copy.

    ``api_content`` is the byte-stability sidecar: it holds the exact bytes
    previously sent for a message, and the next turn substitutes it back into
    ``content``. Leaving it in place on a message this function rewrote would
    replay the images the strip just removed — and the recovery cannot re-fire,
    because it records the model in ``_image_rejecting_models`` and gates itself
    on that. The session would then send rejected images on every subsequent turn.

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


class TestRejectionNeverReachesPersistedHistory:
    """A rejection says what the CURRENT model accepts, not what the conversation holds.

    The recovery used to strip images from the canonical ``messages`` and force a full flush,
    which deleted every image — and every image-only message — from state.db for good; a later
    switch to a vision model found them gone. Same failure class as the ASCII strip in #117802.
    The strip now happens on the send path only.
    """

    class _Err(Exception):
        status_code = 400
        body = "This model does not support images."

    @staticmethod
    def _agent(provider="text-only-provider", model="text-model"):
        from types import SimpleNamespace

        return SimpleNamespace(
            provider=provider, model=model, _force_ascii_payload=False,
            _image_rejecting_models=set(), _db_flush_scan_prefix=7, log_prefix="",
            _vprint=lambda *a, **k: None,
        )

    @staticmethod
    def _history():
        return [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ]},
        ]

    def _recover(self, agent, messages, api_messages):
        from agent.turn_recovery import recover_before_classification

        return recover_before_classification(
            agent, self._Err(), messages=messages, api_messages=api_messages,
            api_kwargs={}, active_system_prompt="sys",
        )

    def test_canonical_history_keeps_its_images(self):
        import copy

        agent, history = self._agent(), self._history()
        before = copy.deepcopy(history)
        wire = copy.deepcopy(history)

        retry, _ = self._recover(agent, history, wire)

        assert retry is True
        assert history == before, "the recovery rewrote persisted history"
        assert agent._db_flush_scan_prefix == 7, "the recovery forced a history rewrite"
        # The in-flight request still goes out text-only.
        assert "image_url" not in str(wire)

    def test_later_requests_to_the_same_model_stay_text_only(self):
        """build_api_request strips each attempt's copy — in Hermes's own message format, before
        provider conversion, so every provider shape (incl. Bedrock's type-less blocks) is covered."""
        from agent.message_sanitization import strip_images_for_rejecting_model

        agent = self._agent()
        self._recover(agent, self._history(), [])

        api_messages = self._history()
        assert strip_images_for_rejecting_model(agent, api_messages) is True
        assert "image_url" not in str(api_messages)

    def test_a_model_that_accepts_images_gets_them_again(self):
        from agent.message_sanitization import strip_images_for_rejecting_model

        agent = self._agent()
        self._recover(agent, self._history(), [])

        agent.provider, agent.model = "vision-provider", "vision-model"
        api_messages = self._history()
        assert strip_images_for_rejecting_model(agent, api_messages) is False
        assert str(api_messages).count("data:image/png") == 2

    def test_every_model_in_a_fallback_chain_is_tracked(self):
        """Two models reject images in the same turn (fallback A -> B). A turn-global guard
        skipped B's recovery once A had tripped it, failing the turn; recording only one model
        also forgot A on later turns. Each model is now judged and remembered on its own."""
        from agent.message_sanitization import strip_images_for_rejecting_model

        agent = self._agent(provider="p", model="model-a")
        retry_a, _ = self._recover(agent, self._history(), [])

        # The fallback restart rebuilds api_messages from history, images included, for B.
        agent.model = "model-b"
        rebuilt = self._history()
        assert strip_images_for_rejecting_model(agent, rebuilt) is False
        assert "image_url" in str(rebuilt)

        # B rejects too, in the same turn: its recovery must still run.
        retry_b, _ = self._recover(agent, self._history(), [])

        assert retry_a is True and retry_b is True
        assert agent._image_rejecting_models == {("p", "model-a"), ("p", "model-b")}
        for model in ("model-a", "model-b"):
            agent.model = model
            api_messages = self._history()
            assert strip_images_for_rejecting_model(agent, api_messages) is True, model
            assert "image_url" not in str(api_messages)

        agent.model = "model-c"
        api_messages = self._history()
        assert strip_images_for_rejecting_model(agent, api_messages) is False
        assert str(api_messages).count("data:image/png") == 2

    def test_a_corrupt_image_does_not_mark_the_model_image_rejecting(self):
        """'failed to decode image' says the PAYLOAD is bad, not that the model is text-only.
        The attempt is stripped and retried, but the model stays unmarked so a later request
        with a good image still reaches it — otherwise one bad screenshot blinds the model for
        the rest of the session."""
        import copy

        from agent.message_sanitization import strip_images_for_rejecting_model

        class _CorruptErr(Exception):
            status_code = 400
            body = "Invalid request: prepare image failed: failed to decode image: invalid or unsupported image format"

        from agent.turn_recovery import recover_before_classification

        agent, history = self._agent(), self._history()
        before, wire = copy.deepcopy(history), copy.deepcopy(history)
        retry, _ = recover_before_classification(
            agent, _CorruptErr(), messages=history, api_messages=wire,
            api_kwargs={}, active_system_prompt="sys",
        )

        assert retry is True
        assert "image_url" not in str(wire), "the retry payload should be text-only"
        assert history == before
        assert agent._image_rejecting_models == set()
        api_messages = self._history()
        assert strip_images_for_rejecting_model(agent, api_messages) is False
        assert str(api_messages).count("data:image/png") == 2

    def test_a_repeat_rejection_from_the_same_model_does_not_loop(self):
        """The per-model guard still stops re-entry: a second rejection from a model already
        known to reject images falls through to normal error handling instead of retrying forever."""
        agent = self._agent()
        assert self._recover(agent, self._history(), [])[0] is True
        assert self._recover(agent, self._history(), [])[0] is False
