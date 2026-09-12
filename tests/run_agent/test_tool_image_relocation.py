"""Tests for the tool-image relocation projection (request-build time).

Opt-in capability: a ``ProviderProfile`` that declares
``relocate_tool_result_images=True`` accepts images in user messages but
rejects them inside ``role:"tool"`` messages (e.g. CommandCode:
400 ``{'message': 'Invalid input', 'param': 'messages.N.content'}``). For
those providers the agent relocates image parts out of the tool rows into a
following user message **at request-build time** — a deterministic
projection over the per-request ``api_messages`` copy. Persisted history is
never touched and no synthetic user row is appended mid-loop (AGENTS.md:
strict role alternation, no mid-loop synthetic user messages).

Shape produced (pre-validated against the live gateway):

    assistant(tool_calls) -> tool(text) -> user(["Attached media from tool
    result:", <image parts...>])

When a real user message already follows the tool run, the synthetic content
is merged INTO that message (marker + images prepended) so two consecutive
user messages can never appear.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

IMG_URL = "data:image/png;base64,AAAA"
IMG_URL_2 = "data:image/png;base64,BBBB"

_OPTED_IN_PROFILE = SimpleNamespace(
    relocate_tool_result_images=True,
    supports_vision_tool_messages=True,
    supports_vision=False,
)
_DEFAULT_PROFILE = SimpleNamespace(
    relocate_tool_result_images=False,
    supports_vision_tool_messages=True,
    supports_vision=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_agent(
    provider: str = "commandcode",
    model: str = "deepseek/deepseek-v4-pro",
    api_mode: str = "chat_completions",
    vision: bool = True,
):
    """Minimal AIAgent mock with the relocation methods bound for real."""
    agent = MagicMock(spec=AIAgent)
    agent.provider = provider
    agent.model = model
    agent.api_mode = api_mode
    agent._no_list_tool_content_models = set()
    agent._relocate_tool_images_models = set()
    agent._model_supports_vision = lambda: vision
    agent._content_has_image_parts = lambda content: _has_image_parts(content)
    agent._provider_supports_vision_tool_messages = lambda: (
        AIAgent._provider_supports_vision_tool_messages(agent)
    )
    agent._provider_relocates_tool_images = lambda: (
        AIAgent._provider_relocates_tool_images(agent)
    )
    agent._should_relocate_tool_result_images = lambda: (
        AIAgent._should_relocate_tool_result_images(agent)
    )
    agent._project_tool_images_for_build = lambda msgs: (
        AIAgent._project_tool_images_for_build(agent, msgs)
    )
    agent._split_tool_content_images = lambda content: (
        AIAgent._split_tool_content_images(content)
    )
    agent._merge_relocated_into_user = lambda msg, imgs, marker: (
        AIAgent._merge_relocated_into_user(msg, imgs, marker)
    )
    agent._relocate_tool_result_images_for_api = lambda msgs: (
        AIAgent._relocate_tool_result_images_for_api(agent, msgs)
    )
    agent._try_relocate_image_parts_to_user_message = lambda msgs, **kw: (
        AIAgent._try_relocate_image_parts_to_user_message(agent, msgs, **kw)
    )
    agent._try_strip_relocated_images_from_user_messages = lambda msgs, **kw: (
        AIAgent._try_strip_relocated_images_from_user_messages(agent, msgs, **kw)
    )
    agent._try_strip_image_parts_from_tool_messages = lambda msgs, **kw: (
        AIAgent._try_strip_image_parts_from_tool_messages(agent, msgs, **kw)
    )
    return agent


# ---------------------------------------------------------------------------
# Trigger predicate
# ---------------------------------------------------------------------------


class TestShouldRelocatePredicate:
    def test_opted_in_profile_enables_relocation(self):
        agent = _make_agent()
        with patch("providers.get_provider_profile", return_value=_OPTED_IN_PROFILE):
            assert agent._should_relocate_tool_result_images() is True

    def test_default_profile_does_not_relocate(self):
        agent = _make_agent()
        with patch("providers.get_provider_profile", return_value=_DEFAULT_PROFILE):
            assert agent._should_relocate_tool_result_images() is False

    def test_real_commandcode_profile_opted_in(self):
        agent = _make_agent()
        assert agent._should_relocate_tool_result_images() is True

    def test_session_memory_enables_relocation(self):
        agent = _make_agent(provider="custom", model="m")
        agent._relocate_tool_images_models = {("custom", "m")}
        assert agent._should_relocate_tool_result_images() is True

    def test_non_vision_model_no_relocation(self):
        agent = _make_agent(vision=False)
        assert agent._should_relocate_tool_result_images() is False

    def test_non_chat_api_mode_no_relocation(self):
        agent = _make_agent(api_mode="anthropic_messages")
        assert agent._should_relocate_tool_result_images() is False

    def test_no_list_memory_blocks_relocation(self):
        agent = _make_agent()
        agent._no_list_tool_content_models = {("commandcode", agent.model)}
        assert agent._should_relocate_tool_result_images() is False

    def test_bare_agent_does_not_crash(self):
        agent = object.__new__(AIAgent)
        assert agent._should_relocate_tool_result_images() is False


# ---------------------------------------------------------------------------
# Build-time projection
# ---------------------------------------------------------------------------


class TestBuildTimeProjection:
    @pytest.fixture(autouse=True)
    def _opt_in(self):
        with patch("providers.get_provider_profile", return_value=_OPTED_IN_PROFILE):
            yield

    def test_tail_append_relocates_to_new_user_row(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_text("loaded"), _img()])]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out is not api_messages
        assert len(out) == 3
        assert isinstance(out[1]["content"], str)
        assert "loaded" in out[1]["content"]
        assert not _has_image_parts(out[1]["content"])
        assert out[2]["role"] == "user"
        content = out[2]["content"]
        assert content[0] == {"type": "text", "text": "Attached media from tool result:"}
        assert content[1] == _img()

    def test_merge_into_following_user_string_content(self):
        agent = _make_agent()
        api_messages = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
            {"role": "user", "content": "next question"},
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert len(out) == 3
        merged = out[2]
        assert merged["role"] == "user"
        assert merged["content"] == [
            {"type": "text", "text": "Attached media from tool result:"},
            _img(),
            {"type": "text", "text": "next question"},
        ]

    def test_merge_into_following_user_list_content_preserves_order(self):
        agent = _make_agent()
        api_messages = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
            {"role": "user", "content": [_text("q"), _img(IMG_URL_2)]},
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out[2]["content"] == [
            {"type": "text", "text": "Attached media from tool result:"},
            _img(),
            _text("q"),
            _img(IMG_URL_2),
        ]

    def test_multi_image_aggregation_single_user_row(self):
        agent = _make_agent()
        api_messages = [
            _assistant("c1", "c2"),
            _tool("c1", [_text("one"), _img()]),
            _tool("c2", [_text("two"), _img(IMG_URL_2)]),
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        user_rows = [m for m in out if m.get("role") == "user"]
        assert len(user_rows) == 1
        content = user_rows[0]["content"]
        assert content[0]["text"] == "Attached media from tool result:"
        assert [_img(), _img(IMG_URL_2)] == content[1:3]

    def test_multi_run_creates_one_user_row_per_run(self):
        agent = _make_agent()
        api_messages = [
            _assistant("c1"),
            _tool("c1", [_text("one"), _img()]),
            {"role": "user", "content": "thanks"},
            _assistant("c2"),
            _tool("c2", [_text("two"), _img(IMG_URL_2)]),
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        marker_rows = [
            m
            for m in out
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and m["content"]
            and m["content"][0].get("text") == "Attached media from tool result:"
        ]
        assert len(marker_rows) == 2

    def test_empty_text_placeholder(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_img()])]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert isinstance(out[1]["content"], str)
        assert "[image" in out[1]["content"]
        assert out[2]["role"] == "user"
        assert out[2]["content"][1] == _img()

    def test_text_only_tool_rows_untouched_no_new_user(self):
        agent = _make_agent()
        api_messages = [_assistant("c1"), _tool("c1", [_text("plain")])]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out is api_messages
        assert out[1]["content"] == [_text("plain")]

    def test_orphan_tool_row_not_relocated(self):
        agent = _make_agent()
        api_messages = [_tool("orphan", [_text("x"), _img()])]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out is api_messages
        assert _has_image_parts(out[0]["content"])

    def test_returns_same_list_object_when_noop(self):
        agent = _make_agent()
        with patch("providers.get_provider_profile", return_value=_DEFAULT_PROFILE):
            api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]
            out = agent._relocate_tool_result_images_for_api(api_messages)
        assert out is api_messages

    def test_returns_new_list_when_changed_input_unchanged(self):
        agent = _make_agent()
        tool_row = _tool("c1", [_text("loaded"), _img()])
        api_messages = [_assistant("c1"), tool_row]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out is not api_messages
        assert api_messages[1] is tool_row
        assert _has_image_parts(tool_row["content"])

    def test_no_mutation_of_persisted_messages_with_shared_content_list(self):
        agent = _make_agent()
        persisted = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
        ]
        snapshot = json.loads(json.dumps(persisted))
        shared_content = persisted[1]["content"]
        api_messages = [m.copy() for m in persisted]

        agent._relocate_tool_result_images_for_api(api_messages)

        assert persisted == snapshot
        assert persisted[1]["content"] is shared_content
        assert _has_image_parts(shared_content)

    def test_deterministic_two_builds_equal(self):
        agent = _make_agent()
        base = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
            _assistant("c2"),
            _tool("c2", [_text("two"), _img(IMG_URL_2)]),
        ]
        first = agent._relocate_tool_result_images_for_api([m.copy() for m in base])
        second = agent._relocate_tool_result_images_for_api([m.copy() for m in base])

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        third = agent._relocate_tool_result_images_for_api(first)
        assert third is first

    def test_alternation_invariant(self):
        agent = _make_agent()
        api_messages = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
            _assistant("c2"),
            _tool("c2", [_text("two"), _img(IMG_URL_2)]),
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        roles = [m.get("role") for m in out]
        assert not any(
            roles[i] == roles[i + 1] == "user" for i in range(len(roles) - 1)
        )
        for i, m in enumerate(out):
            if (
                m.get("role") == "user"
                and isinstance(m.get("content"), list)
                and m["content"]
                and m["content"][0].get("text")
                == "Attached media from tool result:"
            ):
                assert out[i - 1]["role"] == "tool"

    def test_strip_mode_when_model_in_no_list(self):
        agent = _make_agent()
        agent._no_list_tool_content_models = {("commandcode", agent.model)}
        api_messages = [
            _assistant("c1"),
            _tool("c1", [_text("loaded"), _img()]),
        ]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert isinstance(out[1]["content"], str)
        assert not _has_image_parts(out[1]["content"])
        assert not any(m.get("role") == "user" for m in out if m.get("content"))

    def test_strip_mode_noop_without_tool_images(self):
        agent = _make_agent()
        agent._no_list_tool_content_models = {("commandcode", agent.model)}
        api_messages = [_assistant("c1"), _tool("c1", [_text("plain")])]

        out = agent._relocate_tool_result_images_for_api(api_messages)

        assert out is api_messages


class TestNoOpGates:
    def test_non_chat_api_mode_noop(self):
        agent = _make_agent(api_mode="anthropic_messages")
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]
        assert agent._relocate_tool_result_images_for_api(api_messages) is api_messages

    def test_non_vision_noop(self):
        agent = _make_agent(vision=False)
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]
        assert agent._relocate_tool_result_images_for_api(api_messages) is api_messages

    def test_allowlisted_provider_noop(self):
        agent = _make_agent("openai", "gpt-4.1")
        api_messages = [_assistant("c1"), _tool("c1", [_text("x"), _img()])]
        assert agent._relocate_tool_result_images_for_api(api_messages) is api_messages


# ---------------------------------------------------------------------------
# Full-loop request-shape integration (mock provider)
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


class TestConversationLoopWiring:
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
            a.provider = "commandcode"
            a.model = "deepseek/deepseek-v4-pro"
            a.api_mode = "chat_completions"
            a._no_list_tool_content_models = set()
            a._relocate_tool_images_models = set()
            a._model_supports_vision = lambda: True
            return a

    def test_run_conversation_sends_relocated_shape(self, agent):
        history = [
            {"role": "user", "content": "describe the image"},
            _assistant("call_1"),
            _tool("call_1", [_text("Image loaded natively."), _img()]),
        ]
        agent.client.chat.completions.create.return_value = _mock_response(
            content="It is a blue circle with the text ZQ7K-42."
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "please describe it", conversation_history=history
            )

        assert result.get("completed") is True
        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]

        for msg in sent:
            if msg.get("role") == "tool":
                assert not _has_image_parts(msg.get("content")), msg

        marker_rows = [
            m
            for m in sent
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(p, dict) and p.get("text") == "Attached media from tool result:"
                for p in m["content"]
            )
        ]
        assert len(marker_rows) == 1
        assert json.dumps(sent).count("data:image/png;base64") == 1

        roles = [m.get("role") for m in sent]
        assert not any(
            roles[i] == roles[i + 1] == "user" for i in range(len(roles) - 1)
        )
