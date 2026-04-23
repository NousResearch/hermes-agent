"""Tests for surrogate character sanitization in user input.

Surrogates (U+D800..U+DFFF) are invalid in UTF-8 and crash json.dumps()
inside the OpenAI SDK. They can appear via clipboard paste from rich-text
editors like Google Docs, OR from byte-level reasoning models (xiaomi/mimo,
kimi, glm) emitting lone halves in reasoning output.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from run_agent import (
    _sanitize_surrogates,
    _sanitize_messages_surrogates,
    _sanitize_structure_surrogates,
    _SURROGATE_RE,
)


class TestSanitizeSurrogates:
    """Test the _sanitize_surrogates() helper."""

    def test_normal_text_unchanged(self):
        text = "Hello, this is normal text with unicode: café ñ 日本語 🎉"
        assert _sanitize_surrogates(text) == text

    def test_empty_string(self):
        assert _sanitize_surrogates("") == ""

    def test_single_surrogate_replaced(self):
        result = _sanitize_surrogates("Hello \udce2 world")
        assert result == "Hello \ufffd world"

    def test_multiple_surrogates_replaced(self):
        result = _sanitize_surrogates("a\ud800b\udc00c\udfff")
        assert result == "a\ufffdb\ufffdc\ufffd"

    def test_all_surrogate_range(self):
        """Verify the regex catches the full surrogate range."""
        for cp in [0xD800, 0xD900, 0xDA00, 0xDB00, 0xDC00, 0xDD00, 0xDE00, 0xDF00, 0xDFFF]:
            text = f"test{chr(cp)}end"
            result = _sanitize_surrogates(text)
            assert '\ufffd' in result, f"Surrogate U+{cp:04X} not caught"

    def test_result_is_json_serializable(self):
        """Sanitized text must survive json.dumps + utf-8 encoding."""
        dirty = "data \udce2\udcb0 from clipboard"
        clean = _sanitize_surrogates(dirty)
        serialized = json.dumps({"content": clean}, ensure_ascii=False)
        # Must not raise UnicodeEncodeError
        serialized.encode("utf-8")

    def test_original_surrogates_fail_encoding(self):
        """Confirm the original bug: surrogates crash utf-8 encoding."""
        dirty = "data \udce2 from clipboard"
        serialized = json.dumps({"content": dirty}, ensure_ascii=False)
        with pytest.raises(UnicodeEncodeError):
            serialized.encode("utf-8")


class TestSanitizeMessagesSurrogates:
    """Test the _sanitize_messages_surrogates() helper for message lists."""

    def test_clean_messages_returns_false(self):
        msgs = [
            {"role": "user", "content": "all clean"},
            {"role": "assistant", "content": "me too"},
        ]
        assert _sanitize_messages_surrogates(msgs) is False

    def test_dirty_string_content_sanitized(self):
        msgs = [
            {"role": "user", "content": "text with \udce2 surrogate"},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\ufffd" in msgs[0]["content"]
        assert "\udce2" not in msgs[0]["content"]

    def test_dirty_multimodal_content_sanitized(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "multimodal \udce2 content"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ]},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\ufffd" in msgs[0]["content"][0]["text"]
        assert "\udce2" not in msgs[0]["content"][0]["text"]

    def test_mixed_clean_and_dirty(self):
        msgs = [
            {"role": "user", "content": "clean text"},
            {"role": "user", "content": "dirty \udce2 text"},
            {"role": "assistant", "content": "clean response"},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert msgs[0]["content"] == "clean text"
        assert "\ufffd" in msgs[1]["content"]
        assert msgs[2]["content"] == "clean response"

    def test_non_dict_items_skipped(self):
        msgs = ["not a dict", {"role": "user", "content": "ok"}]
        assert _sanitize_messages_surrogates(msgs) is False

    def test_tool_messages_sanitized(self):
        """Tool results could also contain surrogates from file reads etc."""
        msgs = [
            {"role": "tool", "content": "result with \udce2 data", "tool_call_id": "x"},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\ufffd" in msgs[0]["content"]


class TestReasoningFieldSurrogates:
    """Surrogates in reasoning fields (byte-level reasoning models).

    xiaomi/mimo, kimi, glm and similar byte-level tokenizers can emit lone
    surrogates in reasoning output. These fields are carried through to the
    API as `reasoning_content` on assistant messages, and must be sanitized
    or json.dumps() crashes with 'utf-8' codec can't encode surrogates.
    """

    def test_reasoning_field_sanitized(self):
        msgs = [
            {"role": "assistant", "content": "ok", "reasoning": "thought \udce2 here"},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\udce2" not in msgs[0]["reasoning"]
        assert "\ufffd" in msgs[0]["reasoning"]

    def test_reasoning_content_field_sanitized(self):
        """api_messages carry `reasoning_content` built from `reasoning`."""
        msgs = [
            {"role": "assistant", "content": "ok", "reasoning_content": "thought \udce2 here"},
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\udce2" not in msgs[0]["reasoning_content"]
        assert "\ufffd" in msgs[0]["reasoning_content"]

    def test_reasoning_details_nested_sanitized(self):
        """reasoning_details is a list of dicts with nested string fields."""
        msgs = [
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_details": [
                    {"type": "reasoning.summary", "summary": "summary \udce2 text"},
                    {"type": "reasoning.text", "text": "chain \udc00 of thought"},
                ],
            },
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert "\udce2" not in msgs[0]["reasoning_details"][0]["summary"]
        assert "\ufffd" in msgs[0]["reasoning_details"][0]["summary"]
        assert "\udc00" not in msgs[0]["reasoning_details"][1]["text"]
        assert "\ufffd" in msgs[0]["reasoning_details"][1]["text"]

    def test_deeply_nested_reasoning_sanitized(self):
        """Nested dicts / lists inside extra fields are recursed into."""
        msgs = [
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_details": [
                    {
                        "type": "reasoning.encrypted",
                        "content": {
                            "encrypted_content": "opaque",
                            "text_parts": ["part1", "part2 \udce2 part"],
                        },
                    },
                ],
            },
        ]
        assert _sanitize_messages_surrogates(msgs) is True
        assert (
            msgs[0]["reasoning_details"][0]["content"]["text_parts"][1]
            == "part2 \ufffd part"
        )

    def test_reasoning_end_to_end_json_serialization(self):
        """After sanitization, the full message dict must serialize clean."""
        msgs = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reasoning with \udce2 surrogate",
                "reasoning_details": [
                    {"summary": "nested \udcb0 surrogate"},
                ],
            },
        ]
        _sanitize_messages_surrogates(msgs)
        # Must round-trip through json + utf-8 encoding without error
        payload = json.dumps(msgs, ensure_ascii=False).encode("utf-8")
        assert b"\\" not in payload[:0]  # sanity — just ensure we got bytes
        assert len(payload) > 0

    def test_no_surrogates_returns_false(self):
        """Clean reasoning fields don't trigger a modification."""
        msgs = [
            {
                "role": "assistant",
                "content": "ok",
                "reasoning": "clean thought",
                "reasoning_content": "also clean",
                "reasoning_details": [{"summary": "clean summary"}],
            },
        ]
        assert _sanitize_messages_surrogates(msgs) is False


class TestSanitizeStructureSurrogates:
    """Test the _sanitize_structure_surrogates() helper for nested payloads."""

    def test_empty_payload(self):
        assert _sanitize_structure_surrogates({}) is False
        assert _sanitize_structure_surrogates([]) is False

    def test_flat_dict(self):
        payload = {"a": "clean", "b": "dirty \udce2 text"}
        assert _sanitize_structure_surrogates(payload) is True
        assert payload["a"] == "clean"
        assert "\ufffd" in payload["b"]

    def test_flat_list(self):
        payload = ["clean", "dirty \udce2"]
        assert _sanitize_structure_surrogates(payload) is True
        assert payload[0] == "clean"
        assert "\ufffd" in payload[1]

    def test_nested_dict_in_list(self):
        payload = [{"x": "dirty \udce2"}, {"x": "clean"}]
        assert _sanitize_structure_surrogates(payload) is True
        assert "\ufffd" in payload[0]["x"]
        assert payload[1]["x"] == "clean"

    def test_deeply_nested(self):
        payload = {
            "level1": {
                "level2": [
                    {"level3": "deep \udce2 surrogate"},
                ],
            },
        }
        assert _sanitize_structure_surrogates(payload) is True
        assert "\ufffd" in payload["level1"]["level2"][0]["level3"]

    def test_clean_payload_returns_false(self):
        payload = {"a": "clean", "b": [{"c": "also clean"}]}
        assert _sanitize_structure_surrogates(payload) is False

    def test_non_string_values_ignored(self):
        payload = {"int": 42, "list": [1, 2, 3], "dict": {"none": None}, "bool": True}
        assert _sanitize_structure_surrogates(payload) is False
        # Non-string values survive unchanged
        assert payload["int"] == 42
        assert payload["list"] == [1, 2, 3]


class TestApiMessagesSurrogateRecovery:
    """Integration: verify the recovery block sanitizes api_messages.

    The bug this guards against: a surrogate in `reasoning_content` on
    api_messages (transformed from `reasoning` during build) crashes the
    OpenAI SDK's json.dumps(), and the recovery block previously only
    sanitized the canonical `messages` list — not `api_messages` — so the
    next retry would send the same broken payload and fail 3 times.
    """

    def test_api_messages_reasoning_content_sanitized(self):
        """The extended sanitizer catches reasoning_content in api_messages."""
        api_messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "response",
                "reasoning_content": "thought \udce2 trail",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        assert _sanitize_messages_surrogates(api_messages) is True
        assert "\udce2" not in api_messages[1]["reasoning_content"]
        # Full payload must now serialize clean
        json.dumps(api_messages, ensure_ascii=False).encode("utf-8")


class TestRunConversationSurrogateSanitization:
    """Integration: verify run_conversation sanitizes user_message."""

    @patch("run_agent.AIAgent._build_system_prompt")
    @patch("run_agent.AIAgent._interruptible_streaming_api_call")
    @patch("run_agent.AIAgent._interruptible_api_call")
    def test_user_message_surrogates_sanitized(self, mock_api, mock_stream, mock_sys):
        """Surrogates in user_message are stripped before API call."""
        from run_agent import AIAgent

        mock_sys.return_value = "system prompt"

        # Mock streaming to return a simple response
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.message.tool_calls = None
        mock_choice.message.refusal = None
        mock_choice.finish_reason = "stop"
        mock_choice.message.reasoning_content = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "test-model"
        mock_response.id = "test-id"

        mock_stream.return_value = mock_response
        mock_api.return_value = mock_response

        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://localhost:1234/v1", quiet_mode=True, skip_memory=True, skip_context_files=True)
        agent.client = MagicMock()

        # Pass a message with surrogates
        result = agent.run_conversation(
            user_message="test \udce2 message",
            conversation_history=[],
        )

        # The message stored in history should have surrogates replaced
        for msg in result.get("messages", []):
            if msg.get("role") == "user":
                assert "\udce2" not in msg["content"], "Surrogate leaked into stored message"
                assert "\ufffd" in msg["content"], "Replacement char not in stored message"
