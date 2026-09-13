"""Tests for tool-image relocation recovery (reactive 400 path).

Covers the multi-step recovery for providers whose 400 rejects images in
tool messages:

  1. relocate: move tool-row images into a following user message and record
     the (provider, model) in ``_relocate_tool_images_models`` so subsequent
     request builds relocate preemptively;
  2. strip tool rows (existing behavior);
  3. strip marker-owned images from user rows if a preemptively-relocated
     request still fails — recording ``_no_list_tool_content_models`` so the
     session converges to text-only tool content.

Also locks the convergence contract: once ``_no_list_tool_content_models``
contains the key, the build-time projection runs in STRIP mode instead of
relocating.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

IMG_URL = "data:image/png;base64,AAAA"
MARKER = "Attached media from tool result:"
GATEWAY_400_STR = (
    "Error code: 400 - {'error': {'message': 'Invalid input', 'type': "
    "'invalid_request_error', 'param': 'messages.2.content'}}"
)


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _img(url: str = IMG_URL) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _assistant(*call_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "vision_analyze", "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def _tool(call_id: str, content) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "vision_analyze",
        "tool_name": "vision_analyze",
        "content": content,
    }


def _has_image_parts(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(p, dict) and p.get("type") in {"image_url", "input_image"}
        for p in content
    )


def _make_agent(provider: str = "custom", model: str = "m"):
    """Bare AIAgent for method-level testing, no provider setup."""
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.api_mode = "chat_completions"
    agent._no_list_tool_content_models = set()
    agent._relocate_tool_images_models = set()
    agent._model_supports_vision = lambda: True
    return agent


class _FakeGateway400(Exception):
    """Stand-in for openai.BadRequestError with the gateway's param body."""

    def __init__(self):
        super().__init__(GATEWAY_400_STR)
        self.status_code = 400
        self.body = {
            "message": "Invalid input",
            "type": "invalid_request_error",
            "param": "messages.2.content",
        }
        self.response = None


# ---------------------------------------------------------------------------
# _try_relocate_image_parts_to_user_message
# ---------------------------------------------------------------------------


class TestRelocateRecoveryHelper:
    def test_in_place_relocation_preserves_list_identity(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_text("loaded"), _img()])]
        original_id = id(api_messages)

        changed = agent._try_relocate_image_parts_to_user_message(api_messages)

        assert changed is True
        assert id(api_messages) == original_id
        assert len(api_messages) == 3
        assert isinstance(api_messages[1]["content"], str)
        assert api_messages[2]["role"] == "user"
        assert api_messages[2]["content"][0]["text"] == MARKER

    def test_returns_false_when_no_tool_images(self):
        agent = _make_agent()
        api_messages = [_tool("c1", [_text("plain")])]
        assert agent._try_relocate_image_parts_to_user_message(api_messages) is False
        assert len(api_messages) == 1

    def test_returns_false_for_none_or_empty(self):
        agent = _make_agent()
        assert agent._try_relocate_image_parts_to_user_message(None) is False
        assert agent._try_relocate_image_parts_to_user_message([]) is False

    def test_records_session_memory(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]

        agent._try_relocate_image_parts_to_user_message(api_messages)

        assert ("custom", "m") in agent._relocate_tool_images_models

    def test_remember_model_false_skips_memory(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]

        agent._try_relocate_image_parts_to_user_message(
            api_messages, remember_model=False
        )

        assert agent._relocate_tool_images_models == set()

    def test_recovery_relocates_even_when_no_list_memory_present(self):
        agent = _make_agent()
        agent._no_list_tool_content_models = {("custom", agent.model)}
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]

        changed = agent._try_relocate_image_parts_to_user_message(api_messages)

        assert changed is True

    def test_allowlisted_provider_not_relocated_in_recovery(self):
        agent = _make_agent("openrouter", "anthropic/claude-opus-4.6")
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]

        changed = agent._try_relocate_image_parts_to_user_message(api_messages)

        assert changed is False

    def test_memory_convergence_makes_next_build_relocate(self):
        agent = _make_agent()
        assert agent._should_relocate_tool_result_images() is False
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]
        agent._try_relocate_image_parts_to_user_message(api_messages)
        assert agent._should_relocate_tool_result_images() is True


# ---------------------------------------------------------------------------
# _try_strip_relocated_images_from_user_messages
# ---------------------------------------------------------------------------


class TestStripRelocatedUserImages:
    def test_strips_images_following_marker(self):
        agent = _make_agent()
        api_messages = [
            {
                "role": "user",
                "content": [_text("before"), _text(MARKER), _img(), _text("after")],
            }
        ]

        changed = agent._try_strip_relocated_images_from_user_messages(api_messages)

        assert changed is True
        assert api_messages[0]["content"] == [_text("before"), _text(MARKER), _text("after")]

    def test_leaves_user_images_not_following_marker(self):
        agent = _make_agent()
        api_messages = [{"role": "user", "content": [_text("q"), _img()]}]

        changed = agent._try_strip_relocated_images_from_user_messages(api_messages)

        assert changed is False
        assert _has_image_parts(api_messages[0]["content"])

    def test_no_marker_noop(self):
        agent = _make_agent()
        api_messages = [{"role": "user", "content": [_text("q"), _img()]}]
        assert agent._try_strip_relocated_images_from_user_messages(api_messages) is False

    def test_records_no_list_memory(self):
        agent = _make_agent()
        api_messages = [{"role": "user", "content": [_text(MARKER), _img()]}]

        agent._try_strip_relocated_images_from_user_messages(api_messages)

        assert ("custom", "m") in agent._no_list_tool_content_models

    def test_stop_at_first_non_image_part(self):
        agent = _make_agent()
        api_messages = [
            {
                "role": "user",
                "content": [_text(MARKER), _img(), _text("then"), _img("data:image/png;base64,USER")],
            }
        ]

        changed = agent._try_strip_relocated_images_from_user_messages(api_messages)

        assert changed is True
        assert api_messages[0]["content"] == [
            _text(MARKER),
            _text("then"),
            _img("data:image/png;base64,USER"),
        ]

    def test_non_list_content_noop(self):
        agent = _make_agent()
        api_messages = [{"role": "user", "content": f"just text {MARKER}"}]
        assert agent._try_strip_relocated_images_from_user_messages(api_messages) is False


# ---------------------------------------------------------------------------
# Full-loop recovery convergence (mock provider)
# ---------------------------------------------------------------------------


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


class TestLoopRecoveryConvergence:
    @pytest.fixture()
    def agent(self):
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.commandcode.ai/provider/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            a._cached_system_prompt = "You are helpful."
            a._use_prompt_caching = False
            a.save_trajectories = False
            a.compression_enabled = False
            a.provider = "custom"
            a.model = "m"
            a.api_mode = "chat_completions"
            a._no_list_tool_content_models = set()
            a._relocate_tool_images_models = set()
            a._model_supports_vision = lambda: True
            return a

    def _history(self):
        return [
            {"role": "user", "content": "describe the image"},
            _assistant("call_1"),
            _tool("call_1", [_text("Image loaded natively."), _img()]),
        ]

    def test_reactive_400_relocates_and_retries(self, agent):
        """Unknown provider + 400 -> relocate -> retry succeeds; memory is
        recorded so future builds relocate preemptively."""
        agent.client.chat.completions.create.side_effect = [
            _FakeGateway400(),
            _mock_response(content="It is a blue circle with ZQ7K-42."),
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("describe it", conversation_history=self._history())

        assert result.get("completed") is True
        assert agent.client.chat.completions.create.call_count == 2

        first = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
        second = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]

        assert any(
            m.get("role") == "tool" and _has_image_parts(m.get("content")) for m in first
        )
        assert not any(
            m.get("role") == "tool" and _has_image_parts(m.get("content")) for m in second
        )
        marker_rows = [
            m
            for m in second
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(p, dict) and p.get("text") == MARKER for p in m["content"]
            )
        ]
        assert len(marker_rows) == 1
        assert json.dumps(second).count("data:image/png;base64") == 1
        assert ("custom", "m") in agent._relocate_tool_images_models

    def test_double_reject_strips_to_text_and_converges(self, agent):
        """400 even after relocation -> strip -> retry succeeds without images;
        _no_list is recorded so the session converges to text-only tool data."""
        agent.client.chat.completions.create.side_effect = [
            _FakeGateway400(),
            _FakeGateway400(),
            _mock_response(content="described from text summary"),
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("describe it", conversation_history=self._history())

        assert result.get("completed") is True
        assert agent.client.chat.completions.create.call_count == 3

        third = agent.client.chat.completions.create.call_args_list[2].kwargs["messages"]
        assert json.dumps(third).count("data:image/png;base64") == 0
        assert ("custom", "m") in agent._no_list_tool_content_models
