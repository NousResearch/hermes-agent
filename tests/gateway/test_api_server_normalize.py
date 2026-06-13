"""Tests for API server request/content normalization helpers."""
from gateway.ingress import (
    _normalize_chat_completions_request,
    _normalize_chat_content,
    _normalize_responses_request,
    _normalize_run_request,
    _normalize_session_chat_request,
)


class TestNormalizeSessionChatRequest:
    """Session chat request normalization shared by sync + streaming endpoints."""

    def test_uses_system_message_when_present(self):
        normalized, err = _normalize_session_chat_request(
            {"message": "hello", "system_message": "stay focused"}
        )
        assert err is None
        assert normalized.user_message == "hello"
        assert normalized.system_prompt == "stay focused"

    def test_falls_back_to_instructions(self):
        normalized, err = _normalize_session_chat_request(
            {"message": "hello", "instructions": "be concise"}
        )
        assert err is None
        assert normalized.user_message == "hello"
        assert normalized.system_prompt == "be concise"

    def test_accepts_multimodal_message(self):
        normalized, err = _normalize_session_chat_request(
            {
                "message": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ]
            }
        )
        assert err is None
        assert normalized.user_message == [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        assert normalized.system_prompt is None

    def test_rejects_non_string_system_message(self):
        normalized, err = _normalize_session_chat_request(
            {"message": "hello", "system_message": ["not", "a", "string"]}
        )
        assert normalized is None
        assert err is not None
        assert err.status == 400


class TestNormalizeRunRequest:
    """/v1/runs request normalization shared by the runs handler."""

    def test_accepts_string_input(self):
        normalized, err = _normalize_run_request({"input": "hello"})
        assert err is None
        assert normalized.user_message == "hello"
        assert normalized.instructions is None
        assert normalized.previous_response_id is None
        assert normalized.conversation_history == []
        assert normalized.raw_input == "hello"

    def test_extracts_history_from_multi_message_input(self):
        normalized, err = _normalize_run_request(
            {
                "input": [
                    {"role": "system", "content": "ignore me"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                    {"role": "user", "content": "final"},
                ]
            }
        )
        assert err is None
        assert normalized.user_message == "final"
        assert normalized.conversation_history == [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]

    def test_preserves_explicit_conversation_history(self):
        normalized, err = _normalize_run_request(
            {
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "final"},
                ],
                "conversation_history": [{"role": "assistant", "content": "from body"}],
                "previous_response_id": "resp_123",
                "instructions": "be concise",
                "session_id": "sess_1",
            }
        )
        assert err is None
        assert normalized.user_message == "final"
        assert normalized.conversation_history == [{"role": "assistant", "content": "from body"}]
        assert normalized.previous_response_id == "resp_123"
        assert normalized.instructions == "be concise"
        assert normalized.session_id == "sess_1"

    def test_rejects_missing_input(self):
        normalized, err = _normalize_run_request({"model": "test"})
        assert normalized is None
        assert err is not None
        assert err.status == 400

    def test_rejects_invalid_conversation_history_shape(self):
        normalized, err = _normalize_run_request(
            {"input": "hello", "conversation_history": {"role": "user"}}
        )
        assert normalized is None
        assert err is not None
        assert err.status == 400


class TestNormalizeResponsesRequest:
    """/v1/responses request normalization prior to response-store chaining."""

    def test_wraps_string_input_as_single_user_message(self):
        normalized, err = _normalize_responses_request({"input": "hello"})
        assert err is None
        assert normalized.input_messages == [{"role": "user", "content": "hello"}]
        assert normalized.user_message == "hello"
        assert normalized.conversation_history == []
        assert normalized.instructions is None
        assert normalized.previous_response_id is None
        assert normalized.conversation is None
        assert normalized.store is True

    def test_extracts_history_from_array_input_and_preserves_metadata(self):
        normalized, err = _normalize_responses_request(
            {
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "final"},
                            {"type": "input_image", "image_url": "https://example.com/cat.png"},
                        ],
                    },
                ],
                "instructions": "be concise",
                "previous_response_id": "resp_1",
                "conversation": None,
                "store": "false",
            }
        )
        assert err is None
        assert normalized.input_messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "final"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            },
        ]
        assert normalized.conversation_history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        assert normalized.user_message == [
            {"type": "text", "text": "final"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]
        assert normalized.instructions == "be concise"
        assert normalized.previous_response_id == "resp_1"
        assert normalized.store is False

    def test_explicit_conversation_history_takes_precedence(self):
        normalized, err = _normalize_responses_request(
            {
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "final"},
                ],
                "conversation_history": [{"role": "assistant", "content": [{"type": "text", "text": "from body"}]}],
                "previous_response_id": "resp_123",
            }
        )
        assert err is None
        assert normalized.conversation_history == [{"role": "assistant", "content": "from body"}]
        assert normalized.previous_response_id == "resp_123"
        assert normalized.user_message == "final"

    def test_rejects_missing_input(self):
        normalized, err = _normalize_responses_request({"model": "test"})
        assert normalized is None
        assert err is not None
        assert err.status == 400

    def test_rejects_conversation_and_previous_response_id_together(self):
        normalized, err = _normalize_responses_request(
            {"input": "hello", "conversation": "chat-1", "previous_response_id": "resp_1"}
        )
        assert normalized is None
        assert err is not None
        assert err.status == 400


class TestNormalizeChatCompletionsRequest:
    """/v1/chat/completions request normalization."""

    def test_combines_system_messages_and_extracts_last_user_message(self):
        normalized, err = _normalize_chat_completions_request(
            {
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "system", "content": [{"type": "text", "text": "stay concise"}]},
                    {"role": "user", "content": "final"},
                ]
            }
        )
        assert err is None
        assert normalized.system_prompt == "be helpful\nstay concise"
        assert normalized.user_message == "final"
        assert normalized.history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]

    def test_preserves_multimodal_user_content(self):
        normalized, err = _normalize_chat_completions_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                        ],
                    }
                ]
            }
        )
        assert err is None
        assert normalized.user_message == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]
        assert normalized.history == []

    def test_rejects_missing_messages(self):
        normalized, err = _normalize_chat_completions_request({"model": "test"})
        assert normalized is None
        assert err is not None
        assert err.status == 400

    def test_rejects_missing_user_message(self):
        normalized, err = _normalize_chat_completions_request(
            {"messages": [{"role": "system", "content": "only system"}]}
        )
        assert normalized is None
        assert err is not None
        assert err.status == 400


class TestNormalizeChatContent:
    """Content normalization converts array-based content parts to plain text."""

    def test_none_returns_empty_string(self):
        assert _normalize_chat_content(None) == ""

    def test_plain_string_returned_as_is(self):
        assert _normalize_chat_content("hello world") == "hello world"

    def test_empty_string_returned_as_is(self):
        assert _normalize_chat_content("") == ""

    def test_text_content_part(self):
        content = [{"type": "text", "text": "hello"}]
        assert _normalize_chat_content(content) == "hello"

    def test_input_text_content_part(self):
        content = [{"type": "input_text", "text": "user input"}]
        assert _normalize_chat_content(content) == "user input"

    def test_output_text_content_part(self):
        content = [{"type": "output_text", "text": "assistant output"}]
        assert _normalize_chat_content(content) == "assistant output"

    def test_multiple_text_parts_joined_with_newline(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _normalize_chat_content(content) == "first\nsecond"

    def test_mixed_string_and_dict_parts(self):
        content = ["plain string", {"type": "text", "text": "dict part"}]
        assert _normalize_chat_content(content) == "plain string\ndict part"

    def test_image_url_parts_silently_skipped(self):
        content = [
            {"type": "text", "text": "check this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        assert _normalize_chat_content(content) == "check this:"

    def test_integer_content_converted(self):
        assert _normalize_chat_content(42) == "42"

    def test_boolean_content_converted(self):
        assert _normalize_chat_content(True) == "True"

    def test_deeply_nested_list_respects_depth_limit(self):
        """Nesting beyond max_depth returns empty string."""
        content = [[[[[[[[[[[["deep"]]]]]]]]]]]]
        result = _normalize_chat_content(content)
        # The deep nesting should be truncated, not crash
        assert isinstance(result, str)

    def test_large_list_capped(self):
        """Lists beyond MAX_CONTENT_LIST_SIZE are truncated."""
        content = [{"type": "text", "text": f"item{i}"} for i in range(2000)]
        result = _normalize_chat_content(content)
        # Should not contain all 2000 items
        assert result.count("item") <= 1000

    def test_oversized_string_truncated(self):
        """Strings beyond 64KB are truncated."""
        huge = "x" * 100_000
        result = _normalize_chat_content(huge)
        assert len(result) == 65_536

    def test_empty_text_parts_filtered(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "actual"},
            {"type": "text", "text": ""},
        ]
        assert _normalize_chat_content(content) == "actual"

    def test_dict_without_type_skipped(self):
        content = [{"foo": "bar"}, {"type": "text", "text": "real"}]
        assert _normalize_chat_content(content) == "real"

    def test_empty_list_returns_empty(self):
        assert _normalize_chat_content([]) == ""

    def test_many_small_parts_normalize_without_quadratic_rescan(self, monkeypatch):
        """Large content arrays should normalize in linear time."""
        content = [{"type": "text", "text": "x"} for _ in range(1000)]
        sum_calls = 0

        def counting_sum(values):
            nonlocal sum_calls
            sum_calls += 1
            return sum(values)

        monkeypatch.setattr(api_server, "sum", counting_sum, raising=False)
        result = _normalize_chat_content(content)

        assert result.count("x") == 1000
        assert sum_calls == 0
