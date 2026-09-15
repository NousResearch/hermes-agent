"""Tests for the opencode-go tool-result ``name`` strip (#112135).

The Go relay's upstream (Console Go) rejects the OpenAI-spec ``name`` field on
role-"tool" messages with HTTP 400 ``messages.N: "name" is not supported by this
endpoint``. ``make_tool_result_message`` stamps ``name`` (wire format) next to the
internal ``tool_name``; the shared sanitizer strips only the internal key, so the
profile's ``prepare_messages`` drops ``name`` on the wire copy — copy-on-write,
same pattern as the NVIDIA NIM strip.
"""

import copy


def _profile():
    from providers import get_provider_profile
    return get_provider_profile("opencode-go")


def _tool_result(name="test_echo"):
    return {
        "role": "tool",
        "name": name,
        "tool_name": name,
        "content": "echoed: hello",
        "tool_call_id": "call_test1",
    }


class TestOpenCodeGoToolResultNameStrip:
    def test_strips_name_from_tool_messages(self):
        prepared = _profile().prepare_messages([_tool_result()])
        assert len(prepared) == 1
        assert "name" not in prepared[0]
        assert prepared[0]["tool_call_id"] == "call_test1"
        assert prepared[0]["content"] == "echoed: hello"

    def test_original_trajectory_message_not_mutated(self):
        original = _tool_result()
        snapshot = copy.deepcopy(original)
        _profile().prepare_messages([original])
        assert original == snapshot

    def test_no_tool_name_anywhere_returns_input_list_as_is(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert _profile().prepare_messages(messages) is messages

    def test_tool_message_without_name_is_untouched(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
        assert _profile().prepare_messages([msg])[0] is msg

    def test_only_tool_role_loses_name(self):
        # ``name`` on other roles (e.g. legacy function-call history) is not the
        # rejected shape — only role-"tool" rows are copied.
        assistant = {"role": "assistant", "name": "test_echo", "content": "calling"}
        prepared = _profile().prepare_messages([assistant, _tool_result()])
        assert prepared[0]["name"] == "test_echo"
        assert "name" not in prepared[1]

    def test_full_conversation_only_tool_row_copied(self):
        conversation = [
            {"role": "user", "content": "Use the test_echo tool with text=hello."},
            {"role": "assistant", "tool_calls": [{"id": "call_test1", "type": "function",
                "function": {"name": "test_echo", "arguments": "{\"text\": \"hello\"}"}}]},
            _tool_result(),
            {"role": "user", "content": "Now tell me what the tool returned."},
        ]
        prepared = _profile().prepare_messages(conversation)
        assert prepared[0] is conversation[0]
        assert prepared[1] is conversation[1]
        assert prepared[1]["tool_calls"][0]["function"]["name"] == "test_echo"
        assert "name" not in prepared[2]
        assert prepared[2]["tool_call_id"] == "call_test1"
        assert prepared[3] is conversation[3]
