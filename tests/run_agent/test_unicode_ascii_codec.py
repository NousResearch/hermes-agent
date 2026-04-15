"""Tests for UnicodeEncodeError recovery with ASCII codec.

Covers the fix for issue #6843 — systems with ASCII locale (LANG=C)
that can't encode non-ASCII characters in API request payloads.
"""

import pytest

from run_agent import (
    _strip_non_ascii,
    _sanitize_messages_non_ascii,
    _sanitize_structure_non_ascii,
    _sanitize_tools_non_ascii,
    _sanitize_messages_surrogates,
)


class TestStripNonAscii:
    """Tests for _strip_non_ascii helper."""

    def test_ascii_only(self):
        assert _strip_non_ascii("hello world") == "hello world"

    def test_removes_non_ascii(self):
        assert _strip_non_ascii("hello ⚕ world") == "hello  world"

    def test_removes_emoji(self):
        assert _strip_non_ascii("test 🤖 done") == "test  done"

    def test_chinese_chars(self):
        assert _strip_non_ascii("你好world") == "world"

    def test_empty_string(self):
        assert _strip_non_ascii("") == ""

    def test_only_non_ascii(self):
        assert _strip_non_ascii("⚕🤖") == ""


class TestSanitizeMessagesNonAscii:
    """Tests for _sanitize_messages_non_ascii."""

    def test_no_change_ascii_only(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _sanitize_messages_non_ascii(messages) is False
        assert messages[0]["content"] == "hello"

    def test_sanitizes_content_string(self):
        messages = [{"role": "user", "content": "hello ⚕ world"}]
        assert _sanitize_messages_non_ascii(messages) is True
        assert messages[0]["content"] == "hello  world"

    def test_sanitizes_content_list(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello 🤖"}]
        }]
        assert _sanitize_messages_non_ascii(messages) is True
        assert messages[0]["content"][0]["text"] == "hello "

    def test_sanitizes_name_field(self):
        messages = [{"role": "tool", "name": "⚕tool", "content": "ok"}]
        assert _sanitize_messages_non_ascii(messages) is True
        assert messages[0]["name"] == "tool"

    def test_sanitizes_tool_calls(self):
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "⚕test.txt"}'
                }
            }]
        }]
        assert _sanitize_messages_non_ascii(messages) is True
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "test.txt"}'

    def test_handles_non_dict_messages(self):
        messages = ["not a dict", {"role": "user", "content": "hello"}]
        assert _sanitize_messages_non_ascii(messages) is False

    def test_empty_messages(self):
        assert _sanitize_messages_non_ascii([]) is False

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "⚕ System prompt"},
            {"role": "user", "content": "Hello 你好"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert _sanitize_messages_non_ascii(messages) is True
        assert messages[0]["content"] == " System prompt"
        assert messages[1]["content"] == "Hello "
        assert messages[2]["content"] == "Hi there!"


class TestSurrogateVsAsciiSanitization:
    """Test that surrogate and ASCII sanitization work independently."""

    def test_surrogates_still_handled(self):
        """Surrogates are caught by _sanitize_messages_surrogates, not _non_ascii."""
        msg_with_surrogate = "test \ud800 end"
        messages = [{"role": "user", "content": msg_with_surrogate}]
        assert _sanitize_messages_surrogates(messages) is True
        assert "\ud800" not in messages[0]["content"]
        assert "\ufffd" in messages[0]["content"]

    def test_surrogates_in_name_and_tool_calls_are_sanitized(self):
        messages = [{
            "role": "assistant",
            "name": "bad\ud800name",
            "content": None,
            "tool_calls": [{
                "id": "call_\ud800",
                "type": "function",
                "function": {
                    "name": "read\ud800_file",
                    "arguments": '{"path": "bad\ud800.txt"}'
                }
            }],
        }]
        assert _sanitize_messages_surrogates(messages) is True
        assert "\ud800" not in messages[0]["name"]
        assert "\ud800" not in messages[0]["tool_calls"][0]["id"]
        assert "\ud800" not in messages[0]["tool_calls"][0]["function"]["name"]
        assert "\ud800" not in messages[0]["tool_calls"][0]["function"]["arguments"]

    def test_ascii_codec_strips_all_non_ascii(self):
        """ASCII codec case: all non-ASCII is stripped, not replaced."""
        messages = [{"role": "user", "content": "test ⚕🤖你好 end"}]
        assert _sanitize_messages_non_ascii(messages) is True
        # All non-ASCII chars removed; spaces around them collapse
        assert messages[0]["content"] == "test  end"

    def test_no_surrogates_returns_false(self):
        """When no surrogates present, _sanitize_messages_surrogates returns False."""
        messages = [{"role": "user", "content": "hello ⚕ world"}]
        assert _sanitize_messages_surrogates(messages) is False


class TestApiKeyNonAsciiSanitization:
    """Tests for API key sanitization in the UnicodeEncodeError recovery.

    Covers the root cause of issue #6843: a non-ASCII character (ʋ U+028B)
    in the API key causes httpx to fail when encoding the Authorization
    header as ASCII.  The recovery block must strip non-ASCII from the key.
    """

    def test_strip_non_ascii_from_api_key(self):
        """_strip_non_ascii removes ʋ from an API key string."""
        key = "sk-proj-abc" + "ʋ" + "def"
        assert _strip_non_ascii(key) == "sk-proj-abcdef"

    def test_api_key_at_position_153(self):
        """Reproduce the exact error: ʋ at position 153 in 'Bearer <key>'."""
        key = "sk-proj-" + "a" * 138 + "ʋ" + "bcd"
        auth_value = f"Bearer {key}"
        # This is what httpx does — and it fails:
        with pytest.raises(UnicodeEncodeError) as exc_info:
            auth_value.encode("ascii")
        assert exc_info.value.start == 153
        # After sanitization, it should work:
        sanitized_key = _strip_non_ascii(key)
        sanitized_auth = f"Bearer {sanitized_key}"
        sanitized_auth.encode("ascii")  # should not raise


class TestSanitizeToolsNonAscii:
    """Tests for _sanitize_tools_non_ascii."""

    def test_sanitizes_tool_description_and_parameter_descriptions(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Print structured output │ with emoji 🤖",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path │ with unicode",
                            }
                        },
                    },
                },
            }
        ]

        assert _sanitize_tools_non_ascii(tools) is True
        assert tools[0]["function"]["description"] == "Print structured output  with emoji "
        assert tools[0]["function"]["parameters"]["properties"]["path"]["description"] == "File path  with unicode"

    def test_no_change_for_ascii_only_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path",
                            }
                        },
                    },
                },
            }
        ]

        assert _sanitize_tools_non_ascii(tools) is False


class TestSanitizeStructureNonAscii:
    def test_sanitizes_nested_dict_structure(self):
        payload = {
            "default_headers": {
                "X-Title": "Hermes │ Agent",
                "User-Agent": "Hermes/1.0 🤖",
            }
        }
        assert _sanitize_structure_non_ascii(payload) is True
        assert payload["default_headers"]["X-Title"] == "Hermes  Agent"
        assert payload["default_headers"]["User-Agent"] == "Hermes/1.0 "
