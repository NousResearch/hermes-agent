"""Behavior regressions for the session-search output boundary.

``session_search`` returns structured conversation history for the model.  That
tool payload is input to the next model call, not assistant prose for the user.
These tests keep the boundary intact across empty-response recovery, streaming,
the returned result, and durable session history.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.turn_finalizer import (
    _looks_like_session_search_output,
    _sanitize_session_search_reasoning_fields,
)
from agent.transports.types import NormalizedResponse, ToolCall
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from run_agent import AIAgent


RAW_SESSION_SEARCH_RESULT = json.dumps(
    {
        "success": True,
        "mode": "discover",
        "query": "deployment notes",
        "results": [
            {
                "session_id": "prior-session",
                "match_message_id": 42,
                "messages": [
                    {
                        "id": 42,
                        "role": "assistant",
                        "content": "private historical answer",
                    }
                ],
                "messages_before": 3,
                "messages_after": 2,
            }
        ],
        "count": 1,
        "sessions_searched": 4,
    }
)
RAW_SESSION_SEARCH_ERROR = json.dumps(
    {
        "error": "Session database is temporarily unavailable",
        "success": False,
    }
)
SAFE_SUMMARY = "The earlier decision was to pin the release commit."
SAFE_TRAILING_COMPARISON = "Safe comparison: 3 <"
UNRELATED_ERROR = json.dumps(
    {
        "error": "Comparison service is unavailable",
        "success": False,
    }
)


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "session_search",
                "description": "search prior sessions",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _tool_call(*, name="session_search", call_id="search-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=(
                '{"query":"deployment notes"}'
                if name == "session_search"
                else '{"query":"current release"}'
            ),
        ),
    )


def _response(
    *,
    content,
    finish_reason="stop",
    tool_calls=None,
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*, stream_delta_callback=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=stream_delta_callback,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"session_search", "web_search"}
    agent._fallback_chain = []
    return agent


def _append_tool_results(
    assistant_message,
    messages,
    *_args,
    session_search_result=RAW_SESSION_SEARCH_RESULT,
):
    for tool_call in assistant_message.tool_calls:
        name = tool_call.function.name
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call.id,
                "content": (
                    session_search_result
                    if name == "session_search"
                    else "Current release details"
                ),
            }
        )


def _run(
    agent,
    *,
    persist_side_effect=None,
    flush_side_effect=None,
    stream_callback=None,
    session_search_result=RAW_SESSION_SEARCH_RESULT,
):
    def _execute_tools(assistant_message, messages, *args):
        _append_tool_results(
            assistant_message,
            messages,
            *args,
            session_search_result=session_search_result,
        )

    with (
        patch.object(
            agent,
            "_execute_tool_calls",
            side_effect=_execute_tools,
        ),
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=flush_side_effect,
        ),
        patch.object(
            agent,
            "_persist_session",
            side_effect=persist_side_effect,
        ),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        return agent.run_conversation(
            "What did we decide about deployment?",
            stream_callback=stream_callback,
        )


def test_session_search_empty_followup_does_not_reuse_pre_tool_assistant_text():
    """A search is substantive: an empty follow-up must be retried, not replayed."""
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(
            content="I'll search the earlier conversation.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call()],
        ),
        _response(content=""),
        _response(content=SAFE_SUMMARY),
    ]

    result = _run(agent)

    assert result["final_response"] == SAFE_SUMMARY
    assert result["api_calls"] == 3
    assert result["turn_exit_reason"].startswith("text_response")


def test_session_search_raw_json_is_absent_from_return_and_persistence():
    """Raw tool history must not become the final or durable assistant message."""
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(content=RAW_SESSION_SEARCH_RESULT),
    ]
    persisted = []

    def _capture_persist(messages, _conversation_history):
        persisted[:] = [dict(message) for message in messages]

    result = _run(agent, persist_side_effect=_capture_persist)

    assert result["final_response"] != RAW_SESSION_SEARCH_RESULT
    assert "sessions_searched" not in result["final_response"]
    assert persisted
    assert persisted[-1]["role"] == "assistant"
    assert "sessions_searched" not in persisted[-1]["content"]
    assert "private historical answer" not in persisted[-1]["content"]


def test_session_search_followup_bypasses_visible_stream_before_sanitizing():
    """The post-search model reply must be checked before any visible delta fires."""
    streamed = []
    agent = _make_agent(stream_delta_callback=streamed.append)
    streaming_calls = 0

    def _streaming_response(_api_kwargs, **_kwargs):
        nonlocal streaming_calls
        streaming_calls += 1
        if streaming_calls == 1:
            return _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call()],
            )
        agent.stream_delta_callback(RAW_SESSION_SEARCH_RESULT)
        return _response(content=RAW_SESSION_SEARCH_RESULT)

    with (
        patch.object(
            agent,
            "_interruptible_streaming_api_call",
            side_effect=_streaming_response,
        ) as stream_call,
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=_response(content=RAW_SESSION_SEARCH_RESULT),
        ) as complete_call,
    ):
        result = _run(agent)

    visible_text = "".join(delta for delta in streamed if isinstance(delta, str))
    assert stream_call.call_count == 1
    assert complete_call.call_count == 1
    assert "sessions_searched" not in visible_text
    assert "private historical answer" not in visible_text
    assert "sessions_searched" not in result["final_response"]


def test_session_search_raw_intermediate_tool_content_never_reaches_boundaries():
    """Raw history plus a later tool call is scrubbed before every boundary."""
    interim = []
    flush_snapshots = []
    agent = _make_agent()
    agent.interim_assistant_callback = (
        lambda content, **_kwargs: interim.append(content)
    )
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(
            content=RAW_SESSION_SEARCH_RESULT,
            finish_reason="tool_calls",
            tool_calls=[_tool_call(name="web_search", call_id="web-1")],
        ),
        _response(content=SAFE_SUMMARY),
    ]

    result = _run(
        agent,
        flush_side_effect=lambda messages, _history: flush_snapshots.append(
            copy.deepcopy(messages)
        ),
    )

    assert result["final_response"] == SAFE_SUMMARY
    assert len(flush_snapshots) >= 2
    assert any(
        message.get("tool_calls", [{}])[0].get("id") == "web-1"
        for snapshot in flush_snapshots
        for message in snapshot
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assistant_texts = [
        message.get("content", "")
        for message in result["messages"]
        if message.get("role") == "assistant"
    ]
    flushed_assistant_texts = [
        message.get("content", "")
        for snapshot in flush_snapshots
        for message in snapshot
        if message.get("role") == "assistant"
    ]
    for boundary_text in interim + assistant_texts + flushed_assistant_texts:
        assert "sessions_searched" not in boundary_text
        assert "private historical answer" not in boundary_text


def test_session_search_safe_summary_reaches_display_and_tts_callbacks():
    """A checked full response must preserve normal display and TTS delivery."""
    display_deltas = []
    tts_deltas = []
    agent = _make_agent(stream_delta_callback=display_deltas.append)

    with (
        patch.object(
            agent,
            "_interruptible_streaming_api_call",
            return_value=_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call()],
            ),
        ) as stream_call,
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=_response(content=SAFE_SUMMARY),
        ) as complete_call,
    ):
        result = _run(agent, stream_callback=tts_deltas.append)

    display_text = "".join(
        delta for delta in display_deltas if isinstance(delta, str)
    )
    tts_text = "".join(delta for delta in tts_deltas if isinstance(delta, str))
    assert stream_call.call_count == 1
    assert complete_call.call_count == 1
    assert result["final_response"] == SAFE_SUMMARY
    assert SAFE_SUMMARY in display_text
    assert SAFE_SUMMARY in tts_text
    assert result["response_previewed"] is False


def test_session_search_raw_reasoning_is_absent_from_all_boundaries():
    """Raw history in every reasoning field is scrubbed before callbacks/store."""
    reasoning_deltas = []
    persisted = []
    agent = _make_agent()

    def _capture_reasoning(text):
        reasoning_deltas.append(text)

    agent.reasoning_callback = _capture_reasoning
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(
            content=SAFE_SUMMARY,
            reasoning=RAW_SESSION_SEARCH_RESULT,
            reasoning_content=RAW_SESSION_SEARCH_RESULT,
            reasoning_details=[
                {
                    "type": "reasoning.text",
                    "text": RAW_SESSION_SEARCH_RESULT,
                }
            ],
        ),
    ]

    def _capture_persist(messages, _conversation_history):
        persisted[:] = copy.deepcopy(messages)

    result = _run(agent, persist_side_effect=_capture_persist)

    def _assistant_reasoning(messages):
        return [
            {
                key: message.get(key)
                for key in ("reasoning", "reasoning_content", "reasoning_details")
                if key in message
            }
            for message in messages
            if message.get("role") == "assistant"
        ]

    boundary_values = {
        "reasoning_callback": reasoning_deltas,
        "persisted": _assistant_reasoning(persisted),
        "result_messages": _assistant_reasoning(result["messages"]),
        "result_last_reasoning": result["last_reasoning"],
    }
    serialized = json.dumps(boundary_values, default=str)
    assert result["final_response"] == SAFE_SUMMARY
    assert not reasoning_deltas
    assert "sessions_searched" not in serialized
    assert "private historical answer" not in serialized


def test_provider_native_reasoning_is_sanitized_before_final_persistence():
    """Real normalized provider containers cannot persist raw search JSON."""
    persisted = []
    agent = _make_agent()
    transport = agent._get_transport()
    safe_codex_item = {
        "id": "reasoning-safe",
        "type": "reasoning",
        "encrypted_content": "opaque-safe-reasoning",
    }
    safe_anthropic_block = {
        "type": "tool_use",
        "id": "tool-safe",
        "name": "web_search",
        "input": {"query": "current release"},
    }
    unrelated_message_item = {
        "type": "message",
        "id": "message-safe",
        "content": [{"type": "output_text", "text": SAFE_SUMMARY}],
    }
    tool_response = NormalizedResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="search-1",
                name="session_search",
                arguments='{"query":"deployment notes"}',
            )
        ],
        finish_reason="tool_calls",
    )
    final_response = NormalizedResponse(
        content=SAFE_SUMMARY,
        tool_calls=None,
        finish_reason="stop",
        provider_data={
            "codex_reasoning_items": [
                {
                    "id": "reasoning-raw",
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": RAW_SESSION_SEARCH_RESULT}
                    ],
                },
                safe_codex_item,
            ],
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": RAW_SESSION_SEARCH_RESULT},
                safe_anthropic_block,
            ],
            "codex_message_items": [unrelated_message_item],
            "refusal": "unrelated metadata stays intact",
        },
    )
    # The chat-completions loop normalizes once to inspect finish_reason and
    # once at the shared response boundary.  Return the same real normalized
    # object for both reads of each provider response.
    normalized_responses = [
        tool_response,
        tool_response,
        final_response,
        final_response,
    ]
    agent.client.chat.completions.create.side_effect = [
        _response(content=""),
        _response(content=""),
    ]

    def _capture_persist(messages, _conversation_history):
        persisted[:] = copy.deepcopy(messages)

    with patch.object(
        transport,
        "normalize_response",
        side_effect=normalized_responses,
    ):
        result = _run(agent, persist_side_effect=_capture_persist)

    final_assistant = persisted[-1]
    serialized = json.dumps(final_assistant, default=str)
    assert result["final_response"] == SAFE_SUMMARY
    assert "sessions_searched" not in serialized
    assert "private historical answer" not in serialized
    assert final_assistant["codex_reasoning_items"] == [safe_codex_item]
    assert final_assistant["anthropic_content_blocks"] == [safe_anthropic_block]
    assert final_assistant["codex_message_items"] == [unrelated_message_item]
    assert final_response.provider_data["refusal"] == (
        "unrelated metadata stays intact"
    )


def test_normalized_codex_reasoning_sanitizer_reports_change():
    """The exact provider_data-only regression returns True and stays narrow."""
    unrelated_message_items = [{"type": "message", "id": "message-safe"}]
    response = NormalizedResponse(
        content=SAFE_SUMMARY,
        tool_calls=None,
        finish_reason="stop",
        provider_data={
            "codex_reasoning_items": [RAW_SESSION_SEARCH_RESULT],
            "codex_message_items": unrelated_message_items,
        },
    )

    assert _sanitize_session_search_reasoning_fields(response) is True
    assert response.provider_data["codex_reasoning_items"] == []
    assert response.provider_data["codex_message_items"] == unrelated_message_items


def test_unrelated_error_json_after_session_search_is_preserved():
    """Only the observed session_search error may be withheld."""
    persisted = []
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(content=UNRELATED_ERROR),
    ]

    def _capture_persist(messages, _conversation_history):
        persisted[:] = copy.deepcopy(messages)

    result = _run(agent, persist_side_effect=_capture_persist)

    assert result["final_response"] == UNRELATED_ERROR
    assert persisted[-1]["content"] == UNRELATED_ERROR


@pytest.mark.parametrize(
    "echo",
    [
        f"Tool result: {RAW_SESSION_SEARCH_ERROR}",
        f"Tool result:\n```json\n{RAW_SESSION_SEARCH_ERROR}\n```",
    ],
)
def test_observed_session_search_error_echo_with_wrapper_is_withheld(echo):
    """Prefixes and JSON fences do not defeat observed-error correlation."""
    persisted = []
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(content=echo),
    ]

    def _capture_persist(messages, _conversation_history):
        persisted[:] = copy.deepcopy(messages)

    result = _run(
        agent,
        persist_side_effect=_capture_persist,
        session_search_result=RAW_SESSION_SEARCH_ERROR,
    )

    assert "Session database is temporarily unavailable" not in result["final_response"]
    assert "Session database is temporarily unavailable" not in persisted[-1]["content"]


def test_session_search_safe_summary_flushes_trailing_tag_prefix_for_gateway():
    """The complete checked text reaches display/TTS and is gateway-final."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="message-1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="message-1")
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
    )
    tts_deltas = []
    agent = _make_agent(stream_delta_callback=consumer.on_delta)

    with (
        patch.object(
            agent,
            "_interruptible_streaming_api_call",
            return_value=_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call()],
            ),
        ),
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=_response(content=SAFE_TRAILING_COMPARISON),
        ),
    ):
        result = _run(agent, stream_callback=tts_deltas.append)

    consumer.finish()
    asyncio.run(consumer.run())

    platform_text = "".join(
        call.kwargs.get("content", "") for call in adapter.send.call_args_list
    )
    platform_text += "".join(
        call.kwargs.get("content", "") for call in adapter.edit_message.call_args_list
    )
    assert result["final_response"] == SAFE_TRAILING_COMPARISON
    assert SAFE_TRAILING_COMPARISON in platform_text
    assert SAFE_TRAILING_COMPARISON in "".join(tts_deltas)
    assert consumer.final_response_sent is True
    assert adapter.send.call_count == 1


def test_session_search_error_json_is_absent_from_return_and_persistence():
    """The real tool_error shape must not become assistant output or history."""
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
        _response(content=RAW_SESSION_SEARCH_ERROR),
    ]
    persisted = []

    def _capture_persist(messages, _conversation_history):
        persisted[:] = copy.deepcopy(messages)

    result = _run(
        agent,
        persist_side_effect=_capture_persist,
        session_search_result=RAW_SESSION_SEARCH_ERROR,
    )

    assert result["final_response"] != RAW_SESSION_SEARCH_ERROR
    assert "Session database is temporarily unavailable" not in result["final_response"]
    assert persisted[-1]["role"] == "assistant"
    assert '"success": false' not in persisted[-1]["content"]
    assert "Session database is temporarily unavailable" not in persisted[-1]["content"]


def test_session_search_output_recognizer_covers_all_tool_shapes_and_fences():
    for mode, container in (
        ("browse", "results"),
        ("discover", "results"),
        ("read", "messages"),
        ("scroll", "messages"),
    ):
        payload = json.dumps({"success": True, "mode": mode, container: []})
        assert _looks_like_session_search_output(payload)
        assert _looks_like_session_search_output(f"```json\n{payload}\n```")

    wrapped_compact = (
        'Tool result: {"success":true,"mode":"discover",'
        '"results":[{"session_id":"s1"}]}'
    )
    assert _looks_like_session_search_output(wrapped_compact)


def test_session_search_output_recognizer_keeps_unrelated_json():
    ordinary_result = json.dumps(
        {
            "success": True,
            "results": [{"name": "release", "status": "ready"}],
        }
    )
    assert not _looks_like_session_search_output(ordinary_result)
