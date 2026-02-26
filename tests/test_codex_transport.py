"""Tests for agent/codex_transport.py.

Run with: python -m pytest tests/test_codex_transport.py -v
"""


class TestResponsesToCompletion:
    """Tests for _responses_to_chat_completion SSE reconstruction."""

    def test_multi_tool_calls_correct_arguments(self):
        """Ensure multiple parallel tool calls get the right arguments assigned."""
        from agent.codex_transport import _responses_to_chat_completion

        events = [
            {"type": "response.created", "response": {"id": "resp_1"}},
            # First tool call
            {"type": "response.output_item.added", "item": {
                "id": "item_aaa", "type": "function_call",
                "call_id": "call_111", "name": "web_search",
            }},
            {"type": "response.function_call_arguments.delta",
             "item_id": "item_aaa", "delta": '{"query":'},
            {"type": "response.function_call_arguments.delta",
             "item_id": "item_aaa", "delta": '"first"}'},
            # Second tool call (interleaved)
            {"type": "response.output_item.added", "item": {
                "id": "item_bbb", "type": "function_call",
                "call_id": "call_222", "name": "terminal",
            }},
            {"type": "response.function_call_arguments.delta",
             "item_id": "item_bbb", "delta": '{"command":'},
            {"type": "response.function_call_arguments.delta",
             "item_id": "item_bbb", "delta": '"ls"}'},
            # Done events
            {"type": "response.function_call_arguments.done",
             "item_id": "item_aaa", "arguments": '{"query":"first"}'},
            {"type": "response.function_call_arguments.done",
             "item_id": "item_bbb", "arguments": '{"command":"ls"}'},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
            }}},
        ]

        result = _responses_to_chat_completion(events, "gpt-5.3-codex")
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["id"] == "call_111"
        assert tc[0]["function"]["name"] == "web_search"
        assert tc[0]["function"]["arguments"] == '{"query":"first"}'
        assert tc[1]["id"] == "call_222"
        assert tc[1]["function"]["name"] == "terminal"
        assert tc[1]["function"]["arguments"] == '{"command":"ls"}'
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_text_only_response(self):
        from agent.codex_transport import _responses_to_chat_completion

        events = [
            {"type": "response.created", "response": {"id": "resp_2"}},
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world!"},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
            }}},
        ]

        result = _responses_to_chat_completion(events, "gpt-5.3-codex")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Hello world!"
        assert msg.get("tool_calls") is None
        assert result["choices"][0]["finish_reason"] == "stop"


class TestMessageConversion:
    """Tests for _convert_messages_to_input chat→responses format translation."""

    def test_strips_extra_fields(self):
        from agent.codex_transport import _convert_messages_to_input

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!", "finish_reason": "stop", "reasoning": None},
            {"role": "user", "content": "bye"},
        ]
        instructions, items = _convert_messages_to_input(messages)
        assert instructions == "Be helpful."
        assert len(items) == 3
        # assistant message should not contain finish_reason or reasoning
        assert items[1] == {"role": "assistant", "content": "hello!"}

    def test_tool_calls_to_function_call_items(self):
        from agent.codex_transport import _convert_messages_to_input

        messages = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": '{"q":"test"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "results here"},
        ]
        instructions, items = _convert_messages_to_input(messages)
        assert instructions == "Use tools."
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_1"
        assert items[1]["name"] == "web_search"
        assert items[2]["type"] == "function_call_output"
        assert items[2]["call_id"] == "call_1"
        assert items[2]["output"] == "results here"
