"""Tests for empty / null ``tool_calls`` stripping in ChatCompletionsTransport.

Strict OpenAI-compatible providers (onerouter / Qwen, DeepSeek v4) reject an
assistant message carrying ``tool_calls: []`` (or ``null``) with HTTP 400
"Empty tool_calls is not supported in message."  The pre-API sanitizer in
``agent_runtime_helpers.sanitize_api_messages`` already drops these on the
conversation_loop path, but the transport layer must also normalize them so
auxiliary / custom-provider routes that bypass that sanitizer cannot reach the
wire with an invalid array.  See #58755 (follow-up).
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestEmptyToolCallsStripping:
    """Assistant messages with empty/invalid tool_calls must be normalized."""

    def test_assistant_empty_list_dropped(self, transport):
        msgs = [{"role": "assistant", "content": "ok", "tool_calls": []}]
        out = transport.convert_messages(msgs)
        assert "tool_calls" not in out[0]
        assert out[0]["content"] == "ok"

    def test_assistant_null_dropped(self, transport):
        msgs = [{"role": "assistant", "content": "ok", "tool_calls": None}]
        out = transport.convert_messages(msgs)
        assert "tool_calls" not in out[0]

    def test_assistant_real_calls_preserved(self, transport):
        real_tc = [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]
        msgs = [{"role": "assistant", "content": "c", "tool_calls": real_tc}]
        out = transport.convert_messages(msgs)
        assert out[0]["tool_calls"] == real_tc

    def test_only_empty_assistant_stripped_in_mixed_batch(self, transport):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "thinking", "tool_calls": []},
            {
                "role": "assistant",
                "content": "acting",
                "tool_calls": [{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }],
            },
        ]
        out = transport.convert_messages(msgs)
        assert "tool_calls" not in out[1]
        assert out[1]["content"] == "thinking"
        assert out[2]["tool_calls"] and out[2]["tool_calls"][0]["id"] == "call_x"

    def test_user_role_empty_tool_calls_untouched(self, transport):
        # User messages should not carry tool_calls at all, but if a stray
        # empty array is present we must NOT strip it (it's not the invalid
        # assistant shape, and mutating unrelated roles risks breaking the
        # schema assumptions elsewhere). The transport only normalizes
        # assistant messages.
        msgs = [{"role": "user", "content": "hi", "tool_calls": []}]
        out = transport.convert_messages(msgs)
        assert "tool_calls" in out[0]
        assert out[0]["tool_calls"] == []

    def test_nonempty_array_codex_fields_stripped(self, transport):
        # A non-empty tool_calls array carrying codex scaffolding markers
        # (call_id, response_item_id) must have those fields stripped while
        # the call itself is preserved.
        msgs = [{
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{
                "id": "fc_1",
                "call_id": "call_1",
                "response_item_id": "fc_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }],
        }]
        out = transport.convert_messages(msgs, model="gpt-4o")
        # Non-empty array: codex fields stripped, call preserved.
        assert out[0]["tool_calls"]
        tc = out[0]["tool_calls"][0]
        assert "call_id" not in tc
        assert "response_item_id" not in tc

    def test_clean_list_is_identity(self, transport):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "c",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }],
            },
        ]
        assert transport.convert_messages(msgs) is msgs

    def test_empty_array_triggers_copy_on_write(self, transport):
        msgs = [{"role": "assistant", "content": "ok", "tool_calls": []}]
        out = transport.convert_messages(msgs)
        # Original list/message must not be mutated in place.
        assert msgs[0]["tool_calls"] == []
        assert "tool_calls" not in out[0]
        assert out is not msgs

    @pytest.mark.parametrize(
        "model",
        ["qwen/qwen3.8-max-preview:free", "deepseek/deepseek-v4-flash", "gpt-4o"],
    )
    def test_empty_array_stripped_across_providers(self, transport, model):
        msgs = [{"role": "assistant", "content": "ok", "tool_calls": []}]
        out = transport.convert_messages(msgs, model=model)
        assert "tool_calls" not in out[0]
