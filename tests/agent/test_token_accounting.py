import pytest
from agent.token_accounting import (
    estimate_text_tokens,
    estimate_message_tokens,
    classify_message,
    estimate_request_breakdown
)

def test_estimate_text_tokens():
    text = "hello world this is a test string"
    assert estimate_text_tokens(text) > 0
    assert estimate_text_tokens("") == 0

def test_estimate_message_tokens():
    # String content
    msg1 = {"role": "user", "content": "hello"}
    assert estimate_message_tokens(msg1) > 0

    # List content
    msg2 = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert estimate_message_tokens(msg2) > 0

    # Tool call
    msg3 = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "my_tool",
                    "arguments": "{\"arg1\": \"val1\"}"
                }
            }
        ]
    }
    assert estimate_message_tokens(msg3) > 0

    # Reasoning content
    msg4 = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Thinking about it..."
    }
    assert estimate_message_tokens(msg4) > 0

def test_classify_message():
    sys_msg = {"role": "system", "content": "You are a bot"}
    assert classify_message(sys_msg, 0, None) == "system_prompt"

    user_msg = {"role": "user", "content": "hi"}
    assert classify_message(user_msg, 1, 1) == "current_user_turn"
    assert classify_message(user_msg, 1, 5) == "conversation_history_user"

    ast_msg = {"role": "assistant", "content": "ok"}
    assert classify_message(ast_msg, 2, 1) == "conversation_history_assistant"

    tool_call_msg = {"role": "assistant", "tool_calls": [{"id": "1"}]}
    assert classify_message(tool_call_msg, 3, 1) == "tool_call_arguments"

    tool_res_msg = {"role": "tool", "content": "result"}
    assert classify_message(tool_res_msg, 4, 1) == "conversation_history_tool_result"

def test_estimate_request_breakdown():
    api_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"}
    ]
    tools = [
        {"type": "function", "function": {"name": "test_tool", "description": "a test tool"}}
    ]

    breakdown = estimate_request_breakdown(
        api_messages=api_messages,
        tools=tools,
        current_turn_user_idx=3,
        injected_context_chars=100
    )

    assert breakdown.total_estimated_tokens > 0
    assert "system_prompt" in breakdown.buckets
    assert "tool_schemas" in breakdown.buckets
    assert "conversation_history_user" in breakdown.buckets
    assert "conversation_history_assistant" in breakdown.buckets
    assert "current_user_turn" in breakdown.buckets
    assert "injected_context" in breakdown.buckets

    # Check counts
    assert breakdown.buckets["system_prompt"].count == 1
    assert breakdown.buckets["conversation_history_user"].count == 1
    assert breakdown.buckets["conversation_history_assistant"].count == 1
    assert breakdown.buckets["current_user_turn"].count == 1
