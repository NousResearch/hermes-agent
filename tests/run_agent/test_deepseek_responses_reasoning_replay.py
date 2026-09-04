"""DeepSeek Responses plaintext reasoning continuity."""

from types import SimpleNamespace

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _classify_responses_issuer,
    _normalize_codex_response,
)


def _deepseek_tool_response():
    reasoning = SimpleNamespace(
        type="reasoning",
        id="rs_deepseek_1",
        status="completed",
        summary=[],
        encrypted_content=None,
        content=[
            SimpleNamespace(
                type="reasoning_text",
                text="I need to inspect TASK.md before calling the file tool.",
            )
        ],
    )
    function_call = SimpleNamespace(
        type="function_call",
        id="fc_deepseek_1",
        status="completed",
        name="read_file",
        call_id="call_deepseek_1",
        arguments='{"path":"TASK.md"}',
    )
    return SimpleNamespace(
        status="completed",
        output=[reasoning, function_call],
    )


def test_deepseek_responses_issuer_is_stable_across_base_url_variants():
    assert (
        _classify_responses_issuer(base_url="https://api.deepseek.com")
        == "deepseek_responses"
    )
    assert (
        _classify_responses_issuer(base_url="https://api.deepseek.com/v1/")
        == "deepseek_responses"
    )


def test_deepseek_plaintext_reasoning_round_trips_after_tool_call():
    assistant, finish_reason = _normalize_codex_response(
        _deepseek_tool_response(),
        issuer_kind="deepseek_responses",
    )

    assert finish_reason == "tool_calls"
    assert assistant.codex_reasoning_items == [
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": "I need to inspect TASK.md before calling the file tool.",
                }
            ],
            "_issuer_kind": "deepseek_responses",
        }
    ]

    messages = [
        {"role": "user", "content": "Inspect TASK.md now."},
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": assistant.codex_reasoning_items,
            "tool_calls": [
                {
                    "id": "call_deepseek_1",
                    "call_id": "call_deepseek_1",
                    "response_item_id": "fc_deepseek_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"TASK.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_deepseek_1",
            "content": "TASK contents",
        },
    ]
    wire = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="deepseek_responses",
    )

    reasoning_index = next(
        index for index, item in enumerate(wire) if item.get("type") == "reasoning"
    )
    function_index = next(
        index for index, item in enumerate(wire) if item.get("type") == "function_call"
    )
    output_index = next(
        index
        for index, item in enumerate(wire)
        if item.get("type") == "function_call_output"
    )
    assert wire[reasoning_index] == {
        "type": "reasoning",
        "content": [
            {
                "type": "reasoning_text",
                "text": "I need to inspect TASK.md before calling the file tool.",
            }
        ],
    }
    assert reasoning_index < function_index < output_index


def test_deepseek_plaintext_reasoning_is_not_replayed_cross_issuer():
    assistant, _ = _normalize_codex_response(
        _deepseek_tool_response(),
        issuer_kind="deepseek_responses",
    )
    wire = _chat_messages_to_responses_input(
        [
            {"role": "user", "content": "Inspect TASK.md now."},
            {
                "role": "assistant",
                "content": "done",
                "codex_reasoning_items": assistant.codex_reasoning_items,
            },
        ],
        current_issuer_kind="other:https://example.test",
    )

    assert all(item.get("type") != "reasoning" for item in wire)
