"""Tests for the Console Go native-vision deferral (PR #81955).

Providers like opencode-go / Console Go accept multimodal USER messages
but reject ``image_url`` parts inside tool-role messages with HTTP 400
("Upstream request failed: [400] Provider returned error"). When the
provider profile declares ``supports_vision_tool_messages=False`` and the
active model is vision-capable, ``_tool_result_content_for_active_model``
must:

* keep the text parts in the tool message,
* defer the image parts to ``self._pending_tool_image_parts``,
* and ``_append_pending_tool_images_as_user_message`` must append them as
  a user-role message after the whole tool batch (role alternation
  ``tool..tool -> user``).

These tests pin that contract, including the ``input_image`` part type
(Responses-style) which ``_content_has_image_parts`` recognises too.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent(*, vision: bool = True, tool_messages: bool = False, user_messages: bool = True):
    agent = MagicMock(spec=AIAgent)
    agent.provider = "opencode-go"
    agent.model = "mimo-v2.5"
    agent._model_supports_vision = lambda: vision
    agent._provider_supports_vision_tool_messages = lambda: tool_messages
    agent._provider_supports_vision_user_messages = lambda: user_messages
    agent._pending_tool_image_parts = None
    agent._no_list_tool_content_models = None
    # staticmethod — bind it so the code path under test uses the real impl
    agent._content_has_image_parts = lambda content: AIAgent._content_has_image_parts(content)
    return agent


def _tool_result(text="The image shows a red stop sign.", part_type="image_url"):
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text},
            {"type": part_type, "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
        "text_summary": text,
    }


class TestToolResultContentForActiveModel:
    def test_text_kept_and_image_deferred(self):
        agent = _make_agent()
        result = _tool_result()
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert content == [{"type": "text", "text": "The image shows a red stop sign."}]
        assert agent._pending_tool_image_parts == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]

    def test_input_image_parts_are_deferred_too(self):
        """Responses-style ``input_image`` parts must be deferred as well."""
        agent = _make_agent()
        result = _tool_result(part_type="input_image")
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert content == [{"type": "text", "text": "The image shows a red stop sign."}]
        assert len(agent._pending_tool_image_parts) == 1
        assert agent._pending_tool_image_parts[0]["type"] == "input_image"

    def test_image_only_content_falls_back_to_text_summary(self):
        agent = _make_agent()
        result = {
            "_multimodal": True,
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ],
            "text_summary": "summary text",
        }
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert isinstance(content, str)
        assert content == "summary text"
        assert agent._pending_tool_image_parts  # image deferred anyway

    def test_provider_accepting_tool_images_passes_content_through(self):
        agent = _make_agent(tool_messages=True)
        result = _tool_result()
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert content is result["content"]
        assert agent._pending_tool_image_parts is None

    def test_no_user_multimodal_support_keeps_text_summary_downgrade(self):
        """Xiaomi-style providers (no multimodal user messages) keep the
        safe text-summary downgrade and defer nothing."""
        agent = _make_agent(user_messages=False)
        result = _tool_result()
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert isinstance(content, str)
        assert "red stop sign" in content
        assert agent._pending_tool_image_parts is None

    def test_non_vision_model_gets_text_summary_without_deferral(self):
        agent = _make_agent(vision=False)
        result = _tool_result()
        content = AIAgent._tool_result_content_for_active_model(
            agent, "vision_analyze", result
        )
        assert isinstance(content, str)
        assert agent._pending_tool_image_parts is None


class TestAppendPendingToolImagesAsUserMessage:
    def test_appends_user_message_and_clears_pending(self):
        agent = _make_agent()
        image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        agent._pending_tool_image_parts = [image_part]
        messages: list = []
        AIAgent._append_pending_tool_images_as_user_message(agent, messages)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1] is image_part
        assert agent._pending_tool_image_parts is None

    def test_no_pending_is_noop(self):
        agent = _make_agent()
        messages: list = [{"role": "user", "content": "hi"}]
        AIAgent._append_pending_tool_images_as_user_message(agent, messages)
        assert len(messages) == 1
