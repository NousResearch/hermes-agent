"""Regression tests for runtime metadata injection path.

These tests pin the exact behavior for this PR:

* runtime metadata is not stored in the cached system prompt
* runtime metadata is included in the API request user message
* persisted transcript rows keep the original user message only
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent():
    key_name = "api" + "_key"
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            **{
                key_name: "test-key-1234567890",
                "base_url": "https://openrouter.ai/api/v1",
                "quiet_mode": True,
                "skip_context_files": True,
                "skip_memory": True,
            }
        )
    a.client = MagicMock()
    a.session_id = "session-1234"
    a.pass_session_id = True
    a.model = "test-model"
    a.provider = "openai"
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.tool_delay = 0
    a.save_trajectories = False
    return a


def test_runtime_metadata_note_is_api_only_and_not_persisted():
    agent = _make_agent()
    user_message = "What did we discuss yesterday?"
    agent.client.chat.completions.create.return_value = _mock_response("Done")

    with (
        patch.object(agent, "_persist_session") as persist_session,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_message)

    assert result["completed"] is True
    assert "Conversation started:" not in agent._cached_system_prompt
    assert "Session ID:" not in agent._cached_system_prompt
    assert "Model:" not in agent._cached_system_prompt
    assert "Provider:" not in agent._cached_system_prompt

    api_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert api_messages[0]["role"] == "system"
    assert api_messages[0]["content"] == agent._cached_system_prompt

    sent_user_message = api_messages[1]["content"]
    assert user_message in sent_user_message
    assert "Conversation started:" in sent_user_message
    assert "Session ID: session-1234" in sent_user_message
    assert "Model: test-model" in sent_user_message
    assert "Provider: openai" in sent_user_message

    persisted_messages = persist_session.call_args.args[0]
    assert persisted_messages[0]["role"] == "user"
    assert persisted_messages[0]["content"] == user_message


def test_runtime_metadata_note_is_injected_for_multimodal_user_turns():
    agent = _make_agent()
    user_message = [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]
    agent.client.chat.completions.create.return_value = _mock_response("Done")
    captured = {}

    def _fake_build_api_kwargs(api_messages):
        captured["messages"] = api_messages
        return {"model": agent.model, "messages": api_messages, "timeout": 1800.0}

    with (
        patch.object(agent, "_build_api_kwargs", side_effect=_fake_build_api_kwargs),
        patch.object(agent, "_persist_session") as persist_session,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_message)

    assert result["completed"] is True

    api_messages = captured["messages"]
    sent_user_message = next(m["content"] for m in api_messages if m["role"] == "user")
    assert isinstance(sent_user_message, list)
    assert sent_user_message[:2] == user_message
    assert sent_user_message[2]["type"] == "text"
    assert "Conversation started:" in sent_user_message[2]["text"]
    assert "Session ID: session-1234" in sent_user_message[2]["text"]
    assert "Model: test-model" in sent_user_message[2]["text"]
    assert "Provider: openai" in sent_user_message[2]["text"]

    persisted_messages = persist_session.call_args.args[0]
    assert persisted_messages[0]["role"] == "user"
    assert persisted_messages[0]["content"] == user_message


def test_runtime_metadata_note_survives_preflight_compression_rewrite():
    agent = _make_agent()
    user_message = "What changed after compression?"
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
    ]
    agent.client.chat.completions.create.return_value = _mock_response("Done")
    agent.compression_enabled = True
    agent.context_compressor = SimpleNamespace(
        protect_first_n=1,
        protect_last_n=1,
        threshold_tokens=1,
        context_length=1000,
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        should_compress=lambda _tokens: True,
    )
    captured = {}

    def _fake_compress(messages, system_message, approx_tokens=None, task_id=None):
        return (
            [
                {"role": "user", "content": "[compressed summary]"},
                {"role": "assistant", "content": "[summary ack]"},
                {"role": "user", "content": user_message},
            ],
            agent._cached_system_prompt,
        )

    def _fake_build_api_kwargs(api_messages):
        captured["messages"] = api_messages
        return {"model": agent.model, "messages": api_messages, "timeout": 1800.0}

    with (
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_build_api_kwargs", side_effect=_fake_build_api_kwargs),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_message, conversation_history=history)

    assert result["completed"] is True

    api_messages = captured["messages"]
    summary_message = next(
        m["content"] for m in api_messages if m["role"] == "user" and m["content"] == "[compressed summary]"
    )
    assert "Conversation started:" not in summary_message

    current_user_message = next(
        m["content"]
        for m in api_messages
        if m["role"] == "user"
        and isinstance(m["content"], str)
        and user_message in m["content"]
    )
    assert current_user_message.startswith(user_message)
    assert "Conversation started:" in current_user_message
    assert "Session ID: session-1234" in current_user_message


def test_runtime_metadata_note_survives_user_message_repair_merge():
    agent = _make_agent()
    user_message = "Continue with the fix"
    history = [{"role": "user", "content": "Earlier redirect"}]
    agent.client.chat.completions.create.return_value = _mock_response("Done")
    captured = {}

    def _fake_build_api_kwargs(api_messages):
        captured["messages"] = api_messages
        return {"model": agent.model, "messages": api_messages, "timeout": 1800.0}

    with (
        patch.object(agent, "_build_api_kwargs", side_effect=_fake_build_api_kwargs),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_message, conversation_history=history)

    assert result["completed"] is True

    current_user_message = next(
        m["content"]
        for m in captured["messages"]
        if m["role"] == "user"
        and isinstance(m["content"], str)
        and user_message in m["content"]
    )
    assert current_user_message.startswith("Earlier redirect\n\nContinue with the fix")
    assert "Conversation started:" in current_user_message
    assert "Session ID: session-1234" in current_user_message
