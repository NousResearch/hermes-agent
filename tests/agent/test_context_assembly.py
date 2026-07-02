import json

from agent.context_assembly import (
    compact_stale_payloads_for_prompt,
    context_assembly_config_from_mapping,
)


def test_compacts_stale_tool_result_without_mutating_history():
    old_payload = "build log line\n" * 200
    recent_payload = "recent error\n" * 200
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-old",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-old",
            "tool_name": "terminal",
            "content": old_payload,
        },
        {"role": "user", "content": "what failed?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-new",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-new",
            "tool_name": "terminal",
            "content": recent_payload,
        },
    ]

    prompt_view, stats = compact_stale_payloads_for_prompt(
        messages,
        protect_last_n=3,
        min_chars=100,
        preview_chars=80,
    )

    assert prompt_view is not messages
    assert messages[2]["content"] == old_payload
    assert prompt_view[2]["content"].startswith(
        "[stale tool result compacted before model call]"
    )
    assert "tool: terminal" in prompt_view[2]["content"]
    assert "original_chars:" in prompt_view[2]["content"]
    assert prompt_view[5]["content"] == recent_payload
    assert stats.tool_results_compacted == 1
    assert stats.messages_compacted == 1
    assert stats.estimated_tokens_evicted > 0


def test_compacts_old_tool_call_arguments_as_valid_json():
    large_args = json.dumps({"path": "/tmp/out.txt", "content": "x" * 500})
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": large_args},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]

    prompt_view, stats = compact_stale_payloads_for_prompt(
        messages,
        protect_last_n=1,
        min_chars=100,
        preview_chars=40,
    )

    compacted_args = prompt_view[0]["tool_calls"][0]["function"]["arguments"]
    parsed = json.loads(compacted_args)
    assert parsed["_hermes_compacted_stale_tool_arguments"] is True
    assert parsed["tool"] == "write_file"
    assert parsed["original_chars"] == len(large_args)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == large_args
    assert stats.tool_call_args_compacted == 1


def test_preserves_configured_tool_and_recent_media():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "old screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,old"}},
            ],
        },
        {
            "role": "tool",
            "tool_name": "debug_dump",
            "content": "x" * 200,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
    ]

    prompt_view, stats = compact_stale_payloads_for_prompt(
        messages,
        protect_last_n=1,
        min_chars=100,
        preserve_tools=["debug_dump"],
    )

    assert prompt_view is not messages
    assert prompt_view[0]["content"][1] == {
        "type": "text",
        "text": "[stale media payload compacted before model call]",
    }
    assert prompt_view[1]["content"] == "x" * 200
    assert prompt_view[2]["content"][1]["type"] == "image_url"
    assert stats.messages_compacted == 1
    assert stats.media_parts_compacted == 1


def test_context_assembly_config_normalizes_values():
    cfg = context_assembly_config_from_mapping({
        "context_assembly": {
            "enabled": "false",
            "protect_last_n": "-2",
            "min_chars": "bad",
            "preview_chars": 0,
            "preserve_tools": ["terminal", 123],
        }
    })

    assert cfg == {
        "enabled": False,
        "protect_last_n": 0,
        "min_chars": 12_000,
        "preview_chars": 0,
        "preserve_tools": ["terminal", "123"],
    }
