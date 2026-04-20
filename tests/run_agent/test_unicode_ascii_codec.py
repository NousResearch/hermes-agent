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


class TestApiKeyClientSync:
    """Verify that ASCII recovery updates the live OpenAI client's api_key.

    The OpenAI SDK stores its own copy of api_key which auth_headers reads
    dynamically.  If only self.api_key is updated but self.client.api_key
    is not, the next request still sends the corrupted key in the
    Authorization header.
    """

    def test_client_api_key_updated_on_sanitize(self):
        """Simulate the recovery path and verify client.api_key is synced."""
        from unittest.mock import MagicMock
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        bad_key = "sk-proj-abc\u028bdef"  # ʋ lookalike at position 11
        agent.api_key = bad_key
        agent._client_kwargs = {"api_key": bad_key}
        agent.quiet_mode = True

        # Mock client with its own api_key attribute (like the real OpenAI client)
        mock_client = MagicMock()
        mock_client.api_key = bad_key
        agent.client = mock_client

        # --- replicate the recovery logic from run_agent.py ---
        _raw_key = agent.api_key
        _clean_key = _strip_non_ascii(_raw_key)
        assert _clean_key != _raw_key, "test precondition: key should have non-ASCII"

        agent.api_key = _clean_key
        agent._client_kwargs["api_key"] = _clean_key
        if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
            agent.client.api_key = _clean_key

        # All three locations should now hold the clean key
        assert agent.api_key == "sk-proj-abcdef"
        assert agent._client_kwargs["api_key"] == "sk-proj-abcdef"
        assert agent.client.api_key == "sk-proj-abcdef"
        # The bad char should be gone from all of them
        assert "\u028b" not in agent.api_key
        assert "\u028b" not in agent._client_kwargs["api_key"]
        assert "\u028b" not in agent.client.api_key

    def test_client_none_does_not_crash(self):
        """Recovery should not crash when client is None (pre-init)."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        bad_key = "sk-proj-\u028b"
        agent.api_key = bad_key
        agent._client_kwargs = {"api_key": bad_key}
        agent.client = None

        _clean_key = _strip_non_ascii(bad_key)
        agent.api_key = _clean_key
        agent._client_kwargs["api_key"] = _clean_key
        if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
            agent.client.api_key = _clean_key

        assert agent.api_key == "sk-proj-"
        assert agent.client is None  # should not have been touched


class TestApiMessagesAndApiKwargsSanitized:
    """Regression tests for #6843 follow-up: api_messages and api_kwargs must
    be sanitized alongside messages during ASCII-codec recovery.

    The original fix only sanitized the canonical `messages` list.
    api_messages is a separate API-copy built before the retry loop; it may
    carry extra fields (reasoning_content, extra_body) with non-ASCII chars
    that are not present in `messages`.  Without sanitizing api_messages and
    api_kwargs, the retry still raises UnicodeEncodeError even after the
    'System encoding is ASCII — stripped...' log line appears.
    """

    def test_api_messages_with_reasoning_content_is_sanitized(self):
        """api_messages may contain reasoning_content not in messages."""
        api_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Sure!",
                # reasoning_content is injected by the API-copy builder and
                # is NOT present in the canonical messages list
                "reasoning_content": "Let me think \xab step by step \xbb",
            },
        ]
        found = _sanitize_messages_non_ascii(api_messages)
        assert found is True
        assert "\xab" not in api_messages[2]["reasoning_content"]
        assert "\xbb" not in api_messages[2]["reasoning_content"]

    def test_api_kwargs_with_non_ascii_extra_body_is_sanitized(self):
        """api_kwargs may contain non-ASCII in extra_body or other fields."""
        api_kwargs = {
            "model": "glm-5.1",
            "messages": [{"role": "user", "content": "ok"}],
            "extra_body": {
                "system": "Think carefully \u2192 answer",
            },
        }
        found = _sanitize_structure_non_ascii(api_kwargs)
        assert found is True
        assert "\u2192" not in api_kwargs["extra_body"]["system"]

    def test_messages_clean_but_api_messages_dirty_both_get_sanitized(self):
        """Even when canonical messages are clean, api_messages may be dirty."""
        messages = [{"role": "user", "content": "hello"}]
        api_messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_content": "step \xab done",
            },
        ]
        # messages sanitize returns False (nothing to clean)
        assert _sanitize_messages_non_ascii(messages) is False
        # api_messages sanitize must catch the dirty reasoning_content
        assert _sanitize_messages_non_ascii(api_messages) is True
        assert "\xab" not in api_messages[1]["reasoning_content"]

    def test_reasoning_field_in_canonical_messages_is_sanitized(self):
        """The canonical messages list stores reasoning as 'reasoning', not
        'reasoning_content'.  The extra-fields loop must catch it."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "ok",
                "reasoning": "Let me think \xab carefully \xbb",
            },
        ]
        assert _sanitize_messages_non_ascii(messages) is True
        assert "\xab" not in messages[1]["reasoning"]
        assert "\xbb" not in messages[1]["reasoning"]
