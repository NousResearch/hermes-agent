"""Tests for reactive multimodal-tool-content recovery.

Covers the full chain for providers that reject list-type content in
``role: "tool"`` messages (Xiaomi MiMo's 400 "text is not set", etc.):

  1. agent/error_classifier.py: 400 with the right wording classifies as
     ``FailoverReason.multimodal_tool_content_unsupported``.
  2. run_agent._try_strip_image_parts_from_tool_messages downgrades tool
     messages whose ``content`` is a list-with-image to a string text
     summary, in-place, and records the active (provider, model) in
     ``self._no_list_tool_content_models`` so future tool results in this
     session preemptively downgrade.
  3. run_agent._tool_result_content_for_active_model short-circuits to a
     text summary when the (provider, model) is in the cache, even though
     ``_model_supports_vision`` returns True — avoiding a wasted round
     trip on every subsequent screenshot in the session.

The end-to-end retry loop wiring (`conversation_loop.py`) is exercised by
the classifier signal + helper-mutation tests; the integration only adds
a trivial flag-and-continue around the existing pattern used for
``image_too_large`` recovery.

See: https://github.com/NousResearch/hermes-agent/issues/27344
"""

from __future__ import annotations


from agent.error_classifier import FailoverReason, classify_api_error


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None


def _make_agent(provider: str = "xiaomi", model: str = "mimo-v2.5"):
    """Build a bare AIAgent for method-level testing, no provider setup."""
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    return agent


# ─── Strip helper ────────────────────────────────────────────────────────────


class TestStripImagePartsHelper:
    def test_no_messages_returns_false(self):
        agent = _make_agent()
        assert agent._try_strip_image_parts_from_tool_messages([]) is False
        assert agent._try_strip_image_parts_from_tool_messages(None) is False

    def test_no_tool_messages_returns_false(self):
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": "plain text"},
            {"role": "assistant", "content": "ack"},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False

    def test_tool_message_with_string_content_unchanged(self):
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": "plain string result"},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False
        assert msgs[0]["content"] == "plain string result"

    def test_tool_message_list_without_image_unchanged(self):
        """List content with only text parts is left alone — caller surfaces
        the original error if this turns out to also be rejected."""
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "hello"},
            ]},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False

    def test_tool_message_list_with_image_downgrades(self):
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "AX summary: 5 buttons visible"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
            ]},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is True
        # Image stripped; text preserved as a string.
        assert isinstance(msgs[0]["content"], str)
        assert "AX summary" in msgs[0]["content"]
        assert "image_url" not in msgs[0]["content"]
        assert "iVBOR" not in msgs[0]["content"]

    def test_tool_message_image_only_gets_placeholder(self):
        """If the list had nothing but image parts, leave a placeholder so
        the assistant message has something to reference."""
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
            ]},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is True
        assert isinstance(msgs[0]["content"], str)
        assert "image content removed" in msgs[0]["content"]

    def test_records_provider_model_in_session_cache(self):
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        assert ("xiaomi", "mimo-v2.5") in agent._no_list_tool_content_models

    def test_only_tool_messages_get_downgraded(self):
        """User / assistant messages with list-type content are out of
        scope — they're handled by the existing image-routing path."""
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,Y"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        # User message untouched.
        assert isinstance(msgs[0]["content"], list)
        assert any(p.get("type") == "image_url" for p in msgs[0]["content"])
        # Tool message downgraded.
        assert isinstance(msgs[1]["content"], str)
        assert "summary" in msgs[1]["content"]

    def test_skips_recording_when_no_model_id(self):
        """Don't poison the cache with empty keys when provider/model is
        unset (e.g. lazy-initialised mid-handshake)."""
        agent = _make_agent(provider="", model="")
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        assert agent._no_list_tool_content_models == set()


# ─── Short-circuit on cached models ──────────────────────────────────────────


class TestToolResultContentShortCircuit:
    """Once the session has learned that (provider, model) rejects list
    content, ``_tool_result_content_for_active_model`` returns a text
    summary even though ``_model_supports_vision`` reports True.
    """

    def _multimodal_result(self, png_b64: str = "iVBORw0KGgoAAAA"):
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "capture mode=som 800x600 app=Safari"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
            ],
            "text_summary": "capture mode=som 800x600 app=Safari",
            "meta": {"mode": "som", "width": 800, "height": 600, "elements": 5,
                     "png_bytes": 1024},
        }

    def test_returns_text_summary_for_xiaomi_proactively(self, monkeypatch):
        """Xiaomi MiMo rejects list-type tool content, so even with an
        empty cache, _tool_result_content_for_active_model should
        proactively downgrade to a text summary."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        agent._no_list_tool_content_models = set()  # explicit empty
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model(
            "computer_use", self._multimodal_result()
        )
        # Proactive downgrade: text summary instead of list with images.
        assert isinstance(out, str)
        assert "data:image" not in out
        assert "image_url" not in out

    def test_returns_text_summary_when_model_in_cache(self, monkeypatch):
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        agent._no_list_tool_content_models = {("xiaomi", "mimo-v2.5")}
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model(
            "computer_use", self._multimodal_result()
        )
        # Short-circuit: a plain string summary, no image_url present.
        assert isinstance(out, str)
        assert "data:image" not in out
        assert "image_url" not in out

    def test_xiaomi_any_model_gets_text_summary(self, monkeypatch):
        """All Xiaomi models reject list-type tool content, so even a
        different model on the same provider gets a text summary."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        agent._no_list_tool_content_models = {("xiaomi", "mimo-v2.5")}
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model(
            "computer_use", self._multimodal_result()
        )
        assert isinstance(out, str)
        assert "data:image" not in out

    def test_missing_cache_attribute_falls_through(self, monkeypatch):
        """Agents built via ``object.__new__`` without calling ``__init__``
        must not crash — the cache attribute may be absent. Xiaomi still
        gets a text summary because the provider profile says so."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        # Deliberately do not assign _no_list_tool_content_models.
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model(
            "computer_use", self._multimodal_result()
        )
        # Xiaomi proactively downgrades regardless of cache state.
        assert isinstance(out, str)
        assert "data:image" not in out


# ─── Classifier ──────────────────────────────────────────────────────────────


class TestRecoveryEndToEndClassification:
    """Lock in that the patterns used by the recovery path classify to
    the right ``FailoverReason``. (The recovery hook in
    ``agent.conversation_loop`` consumes this reason directly.)
    """

    def test_xiaomi_mimo_classifies(self):
        err = _FakeApiError(
            status_code=400,
            message=(
                "Error code: 400 - {'error': {'code': '400', 'message': "
                "'Param Incorrect', 'param': 'text is not set', 'type': ''}}"
            ),
        )
        result = classify_api_error(err, provider="xiaomi", model="mimo-v2.5")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_alibaba_variant_classifies(self):
        err = _FakeApiError(
            status_code=400,
            message="tool_call.content must be string",
        )
        result = classify_api_error(err, provider="alibaba", model="qwen3.5-plus")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_lm_studio_variant_classifies(self):
        """LM Studio / llama.cpp rejects list-type tool content with a
        generic schema-validation error that must still match the multimodal
        tool content recovery path."""
        err = _FakeApiError(
            status_code=400,
            message=(
                "Error code: 400 - {'error': \"Invalid 'messages' in payload. "
                "Please check the structure of your 'messages' and try again.\"}"
            ),
        )
        result = classify_api_error(
            err, provider="custom", model="qwen3.6-27b-mtp"
        )
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True
