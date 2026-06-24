"""Tests for _sanitize_tool_name and _sanitize_messages_tool_names.

Validates that tool/function ``name`` fields violating OpenAI's
``^[a-zA-Z0-9_-]+$`` constraint are coerced before the API call.
"""

from agent.message_sanitization import (
    _VALID_TOOL_NAME_RE,
    _sanitize_messages_tool_names,
    _sanitize_tool_name,
)


class TestSanitizeToolName:
    """Unit tests for _sanitize_tool_name()."""

    def test_valid_name_unchanged(self):
        assert _sanitize_tool_name("web_search") == "web_search"
        assert _sanitize_tool_name("my-tool") == "my-tool"
        assert _sanitize_tool_name("Tool123") == "Tool123"

    def test_dot_replaced_with_underscore(self):
        """multi_tool_use.parallel → multi_tool_use_parallel"""
        assert _sanitize_tool_name("multi_tool_use.parallel") == "multi_tool_use_parallel"

    def test_space_replaced_with_underscore(self):
        assert _sanitize_tool_name("Jane Doe") == "Jane_Doe"

    def test_multiple_dots(self):
        assert _sanitize_tool_name("a.b.c") == "a_b_c"

    def test_mixed_invalid_chars(self):
        assert _sanitize_tool_name("foo.bar baz!") == "foo_bar_baz_"

    def test_empty_string(self):
        # Empty string matches ^[a-zA-Z0-9_-]+$ vacuously
        assert _sanitize_tool_name("") == ""


class TestValidToolNameRe:
    """Regex validation tests."""

    def test_valid_patterns(self):
        assert _VALID_TOOL_NAME_RE.match("web_search")
        assert _VALID_TOOL_NAME_RE.match("multi-tool")
        assert _VALID_TOOL_NAME_RE.match("Tool123")
        assert _VALID_TOOL_NAME_RE.match("_private")

    def test_invalid_patterns(self):
        assert not _VALID_TOOL_NAME_RE.match("multi_tool_use.parallel")
        assert not _VALID_TOOL_NAME_RE.match("Jane Doe")
        assert not _VALID_TOOL_NAME_RE.match("foo@bar")
        assert not _VALID_TOOL_NAME_RE.match("name.with.dots")


class TestSanitizeMessagesToolNames:
    """Integration tests for _sanitize_messages_tool_names()."""

    def test_sanitizes_function_name_in_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "multi_tool_use.parallel",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ]
        assert _sanitize_messages_tool_names(messages) is True
        assert messages[0]["tool_calls"][0]["function"]["name"] == "multi_tool_use_parallel"

    def test_sanitizes_message_name_field(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "multi_tool_use.parallel",
                "content": "result",
            }
        ]
        assert _sanitize_messages_tool_names(messages) is True
        assert messages[0]["name"] == "multi_tool_use_parallel"

    def test_no_change_when_names_valid(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "web_search",
                "content": "result",
            },
        ]
        assert _sanitize_messages_tool_names(messages) is False

    def test_handles_non_dict_messages(self):
        messages = ["not a dict", None, 42]
        assert _sanitize_messages_tool_names(messages) is False

    def test_handles_missing_function_key(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function"}]}
        ]
        assert _sanitize_messages_tool_names(messages) is False

    def test_sanitizes_display_name_with_space(self):
        """Display names like 'Jane Doe' passed as message name."""
        messages = [
            {"role": "user", "name": "Jane Doe", "content": "hello"},
        ]
        assert _sanitize_messages_tool_names(messages) is True
        assert messages[0]["name"] == "Jane_Doe"

    def test_preserves_valid_names(self):
        """Valid names should pass through unchanged."""
        messages = [
            {"role": "tool", "name": "read_file", "content": "data"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": "{}"}}
            ]},
        ]
        assert _sanitize_messages_tool_names(messages) is False
        assert messages[0]["name"] == "read_file"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "write_file"

    def test_multiple_invalid_names_in_one_pass(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "a.b", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "c d", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "a.b", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "name": "c d", "content": "r2"},
        ]
        assert _sanitize_messages_tool_names(messages) is True
        assert messages[0]["tool_calls"][0]["function"]["name"] == "a_b"
        assert messages[0]["tool_calls"][1]["function"]["name"] == "c_d"
        assert messages[1]["name"] == "a_b"
        assert messages[2]["name"] == "c_d"
