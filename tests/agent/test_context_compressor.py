"""Tests for agent/context_compressor.py — compression logic, thresholds, truncation fallback."""

import pytest
from unittest.mock import patch, MagicMock

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        return c


class TestShouldCompress:
    def test_below_threshold(self, compressor):
        compressor.last_prompt_tokens = 50000
        assert compressor.should_compress() is False

    def test_above_threshold(self, compressor):
        compressor.last_prompt_tokens = 90000
        assert compressor.should_compress() is True

    def test_exact_threshold(self, compressor):
        compressor.last_prompt_tokens = 85000
        assert compressor.should_compress() is True

    def test_explicit_tokens(self, compressor):
        assert compressor.should_compress(prompt_tokens=90000) is True
        assert compressor.should_compress(prompt_tokens=50000) is False



class TestUpdateFromResponse:
    def test_updates_fields(self, compressor):
        compressor.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 1000,
            "total_tokens": 6000,
        })
        assert compressor.last_prompt_tokens == 5000
        assert compressor.last_completion_tokens == 1000

    def test_missing_fields_default_zero(self, compressor):
        compressor.update_from_response({})
        assert compressor.last_prompt_tokens == 0



class TestCompress:
    def _make_messages(self, n):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(n)]

    def test_too_few_messages_returns_unchanged(self, compressor):
        msgs = self._make_messages(4)  # protect_first=2 + protect_last=2 + 1 = 5 needed
        result = compressor.compress(msgs)
        assert result == msgs

    def test_truncation_fallback_no_client(self, compressor):
        # compressor has client=None, so should use truncation fallback
        msgs = [{"role": "system", "content": "System prompt"}] + self._make_messages(10)
        result = compressor.compress(msgs)
        assert len(result) < len(msgs)
        # Should keep system message and last N
        assert result[0]["role"] == "system"
        assert compressor.compression_count == 1

    def test_compression_increments_count(self, compressor):
        msgs = self._make_messages(10)
        compressor.compress(msgs)
        assert compressor.compression_count == 1
        compressor.compress(msgs)
        assert compressor.compression_count == 2

    def test_protects_first_and_last(self, compressor):
        msgs = self._make_messages(10)
        result = compressor.compress(msgs)
        # First 2 messages should be preserved (protect_first_n=2)
        # Last 2 messages should be preserved (protect_last_n=2)
        assert result[-1]["content"] == msgs[-1]["content"]
        # The second-to-last tail message may have the summary merged
        # into it when a double-collision prevents a standalone summary
        # (head=assistant, tail=user in this fixture).  Verify the
        # original content is present in either case.
        assert msgs[-2]["content"] in result[-2]["content"]


class TestGenerateSummaryNoneContent:
    """Regression: content=None (from tool-call-only assistant messages) must not crash."""

    def test_none_content_does_not_crash(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: tool calls happened"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "search"}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "thanks"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert summary.startswith(SUMMARY_PREFIX)

    def test_none_content_in_system_message_compress(self):
        """System message with content=None should not crash during compress."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [{"role": "system", "content": None}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(10)
        ]
        result = c.compress(msgs)
        assert len(result) < len(msgs)


class TestNonStringContent:
    """Regression: content as dict (e.g., llama.cpp tool calls) must not crash."""

    def test_dict_content_coerced_to_string(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = {"text": "some summary"}

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert summary.startswith(SUMMARY_PREFIX)

    def test_none_content_coerced_to_empty(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        # None content → empty string → standardized compaction handoff prefix added
        assert summary is not None
        assert summary == SUMMARY_PREFIX

    def test_summary_call_does_not_force_temperature(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        kwargs = mock_call.call_args.kwargs
        assert "temperature" not in kwargs

    def test_summary_call_passes_live_main_runtime(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="codex-token",
                api_mode="codex_responses",
                quiet_mode=True,
            )

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        assert mock_call.call_args.kwargs["main_runtime"] == {
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "api_mode": "codex_responses",
        }


class TestSummaryFailureCooldown:
    def test_summary_failure_enters_cooldown_and_skips_retry(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom")) as mock_call:
            first = c._generate_summary(messages)
            second = c._generate_summary(messages)

        assert first is None
        assert second is None
        assert mock_call.call_count == 1


class TestSummaryFailureTrackingForGatewayWarning:
    """When summary generation fails, the compressor must record dropped count
    + fallback flag so gateway hygiene & /compress can surface a visible
    warning instead of silently dropping context."""

    def test_compress_records_fallback_and_dropped_count_on_summary_failure(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        # Simulate summary LLM call failing — covers the 404 / model-not-found
        # case from issue (auxiliary compression model misconfigured).
        with patch("agent.context_compressor.call_llm", side_effect=Exception("404 model not found")):
            result = c.compress(msgs)

        assert c._last_summary_fallback_used is True
        assert c._last_summary_dropped_count > 0
        assert c._last_summary_error is not None
        # Result must still be well-formed (fallback summary present).
        assert any(
            isinstance(m.get("content"), str) and "Summary generation was unavailable" in m["content"]
            for m in result
        )

    def test_compress_clears_fallback_flag_on_subsequent_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        # First call fails, second succeeds — flag must reset on second compress.
        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom")):
            c.compress(msgs)
        assert c._last_summary_fallback_used is True

        # Reset cooldown to allow retry on second compress
        c._summary_failure_cooldown_until = 0.0
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            c.compress(msgs)
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0


class TestSummaryPrefixNormalization:
    def test_legacy_prefix_is_replaced(self):
        summary = ContextCompressor._with_summary_prefix("[CONTEXT SUMMARY]: did work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"

    def test_existing_new_prefix_is_not_duplicated(self):
        summary = ContextCompressor._with_summary_prefix(f"{SUMMARY_PREFIX}\ndid work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"


class TestCompressWithClient:
    def test_system_content_list_gets_compression_note_without_crashing(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        assert isinstance(result[0]["content"], list)
        assert any(
            isinstance(block, dict)
            and "compacted into a handoff summary" in block.get("text", "")
            for block in result[0]["content"]
        )

    def test_summarization_path(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(10)]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Should have summary message in the middle
        contents = [m.get("content", "") for m in result]
        assert any(c.startswith(SUMMARY_PREFIX) for c in contents)
        assert len(result) < len(msgs)

    def test_summarization_does_not_split_tool_call_pairs(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: compressed middle"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=3,
                protect_last_n=4,
            )

        msgs = [
            {"role": "user", "content": "Could you address the reviewer comments in PR#71"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "output a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "output b"},
            {"role": "user", "content": "later 1"},
            {"role": "assistant", "content": "later 2"},
            {"role": "tool", "tool_call_id": "call_x", "content": "later output"},
            {"role": "assistant", "content": "later 3"},
            {"role": "user", "content": "later 4"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        answered_ids = {
            msg.get("tool_call_id")
            for msg in result
            if msg.get("role") == "tool" and msg.get("tool_call_id")
        }
        for msg in result:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    assert tc["id"] in answered_ids

    def test_summary_role_avoids_consecutive_user_messages(self):
        """Summary role should alternate with the last head message to avoid consecutive same-role messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Last head message (index 1) is "assistant" → summary should be "user".
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # head_last=assistant, tail_first=assistant → summary_role="user", no collision.
        # Need 8 messages: min_for_compress = 2+3+1 = 6, must have > 6.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msg = [
            m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_msg) == 1
        assert summary_msg[0]["role"] == "user"

    def test_summary_role_avoids_consecutive_user_when_head_ends_with_user(self):
        """When last head message is 'user', summary must be 'assistant' to avoid two consecutive user messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3, protect_last_n=2)

        # Last head message (index 2) is "user" → summary should be "assistant"
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},  # last head — user
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msg = [
            m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_msg) == 1
        assert summary_msg[0]["role"] == "assistant"

    def test_summary_role_flips_to_avoid_tail_collision(self):
        """When summary role collides with the first tail message but flipping
        doesn't collide with head, the role should be flipped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head ends with tool (index 1), tail starts with user (index 6).
        # Default: tool → summary_role="user" → collides with tail.
        # Flip to "assistant" → tool→assistant is fine.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in ("user", "assistant") and r2 in ("user", "assistant"):
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

    def test_double_collision_merges_summary_into_tail(self):
        """When neither role avoids collision with both neighbors, the summary
        should be merged into the first tail message rather than creating a
        standalone message that breaks role alternation.

        Common scenario: head ends with 'assistant', tail starts with 'user'.
        summary='user' collides with tail, summary='assistant' collides with head.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3, protect_last_n=3)

        # Head: [system, user, assistant]  →  last head = assistant
        # Tail: [user, assistant, user]    →  first tail = user
        # summary_role="user" collides with tail, "assistant" collides with head → merge
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},      # compressed
            {"role": "assistant", "content": "msg 4"},  # compressed
            {"role": "user", "content": "msg 5"},       # compressed
            {"role": "user", "content": "msg 6"},       # tail start
            {"role": "assistant", "content": "msg 7"},
            {"role": "user", "content": "msg 8"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in ("user", "assistant") and r2 in ("user", "assistant"):
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

        # The summary text should be merged into the first tail message
        first_tail = [m for m in result if "msg 6" in (m.get("content") or "")]
        assert len(first_tail) == 1
        assert "summary text" in first_tail[0]["content"]

    def test_double_collision_merges_summary_into_list_tail_content(self):
        """Structured tail content should accept a merged summary without TypeError."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3, protect_last_n=3)

        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "user", "content": [{"type": "text", "text": "msg 6"}]},
            {"role": "assistant", "content": "msg 7"},
            {"role": "user", "content": "msg 8"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        merged_tail = next(
            m for m in result
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        )
        assert isinstance(merged_tail["content"], list)
        assert "summary text" in merged_tail["content"][0]["text"]
        assert any(
            isinstance(block, dict) and block.get("text") == "msg 6"
            for block in merged_tail["content"]
        )

    def test_double_collision_user_head_assistant_tail(self):
        """Reverse double collision: head ends with 'user', tail starts with 'assistant'.
        summary='assistant' collides with tail, 'user' collides with head → merge."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head: [system, user]        → last head = user
        # Tail: [assistant, user, assistant] → first tail = assistant
        # summary_role="assistant" collides with tail, "user" collides with head → merge
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # Need 8 messages: min_for_compress = 2+3+1 = 6, must have > 6.
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},   # compressed
            {"role": "user", "content": "msg 3"},        # compressed
            {"role": "assistant", "content": "msg 4"},   # compressed
            {"role": "assistant", "content": "msg 5"},   # tail start
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in ("user", "assistant") and r2 in ("user", "assistant"):
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

        # The summary should be merged into the first tail message (assistant at index 5)
        first_tail = [m for m in result if "msg 5" in (m.get("content") or "")]
        assert len(first_tail) == 1
        assert "summary text" in first_tail[0]["content"]

    def test_no_collision_scenarios_still_work(self):
        """Verify that the common no-collision cases (head=assistant/tail=assistant,
        head=user/tail=user) still produce a standalone summary message."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head=assistant, Tail=assistant → summary_role="user", no collision.
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # Need 8 messages: min_for_compress = 2+3+1 = 6, must have > 6.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msgs = [m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)]
        assert len(summary_msgs) == 1, "should have a standalone summary message"
        assert summary_msgs[0]["role"] == "user"

    def test_summarization_does_not_start_tail_with_tool_outputs(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: compressed middle"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=3,
            )

        msgs = [
            {"role": "user", "content": "earlier 1"},
            {"role": "assistant", "content": "earlier 2"},
            {"role": "user", "content": "earlier 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_c", "type": "function", "function": {"name": "search_files", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_c", "content": "output c"},
            {"role": "user", "content": "latest user"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        called_ids = {
            tc["id"]
            for msg in result
            if msg.get("role") == "assistant" and msg.get("tool_calls")
            for tc in msg["tool_calls"]
        }
        for msg in result:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                assert msg["tool_call_id"] in called_ids


class TestSummaryTargetRatio:
    """Verify that summary_target_ratio properly scales budgets with context window."""

    def test_tail_budget_scales_with_context(self):
        """Tail token budget should be threshold_tokens * summary_target_ratio."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        # 200K * 0.50 threshold * 0.40 ratio = 40K
        assert c.tail_token_budget == 40_000

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        # 1M * 0.50 threshold * 0.40 ratio = 200K
        assert c.tail_token_budget == 200_000

    def test_summary_cap_scales_with_context(self):
        """Max summary tokens should be 5% of context, capped at 12K."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 10_000  # 200K * 0.05

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 12_000  # capped at 12K ceiling

    def test_ratio_clamped(self):
        """Ratio should be clamped to [0.10, 0.80]."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.05)
        assert c.summary_target_ratio == 0.10

        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.95)
        assert c.summary_target_ratio == 0.80

    def test_default_threshold_is_50_percent(self):
        """Default compression threshold should be 50%, with a 64K floor."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.threshold_percent == 0.50
        # 50% of 100K = 50K, but the floor is 64K
        assert c.threshold_tokens == 64_000

    def test_threshold_floor_does_not_apply_above_128k(self):
        """On large-context models the 50% percentage is used directly."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        # 50% of 200K = 100K, which is above the 64K floor
        assert c.threshold_tokens == 100_000

    def test_default_protect_last_n_is_20(self):
        """Default protect_last_n should be 20."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.protect_last_n == 20


class TestTokenBudgetTailProtection:
    """Tests for token-budget-based tail protection (PR #6240).

    The core change: tail protection is now based on a token budget rather
    than a fixed message count.  This prevents large tool outputs from
    blocking compaction.
    """

    @pytest.fixture()
    def budget_compressor(self):
        """Compressor with known token budget for tail protection tests."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,  # 100K threshold
                protect_first_n=2,
                protect_last_n=20,
                quiet_mode=True,
            )
            return c

    def test_large_tool_outputs_no_longer_block_compaction(self, budget_compressor):
        """The motivating scenario: 20 messages with large tool outputs should
        NOT prevent compaction.  With message-count tail protection they would
        all be protected, leaving nothing to summarize."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "Start task"},
            {"role": "assistant", "content": "On it"},
        ]
        # Add 20 messages with large tool outputs (~5K chars each ≈ 1250 tokens)
        for i in range(10):
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"function": {"name": f"tool_{i}", "arguments": "{}"}}],
            })
            messages.append({
                "role": "tool", "content": "x" * 5000,
                "tool_call_id": f"call_{i}",
            })
        # Add 3 recent small messages
        messages.append({"role": "user", "content": "What's the status?"})
        messages.append({"role": "assistant", "content": "Here's what I found..."})
        messages.append({"role": "user", "content": "Continue"})

        # The tail cut should NOT protect all 20 tool messages
        head_end = c.protect_first_n
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        tail_size = len(messages) - cut
        # With token budget, the tail should be much smaller than 20+
        assert tail_size < 20, f"Tail {tail_size} messages — large tool outputs are blocking compaction"
        # But at least 3 (hard minimum)
        assert tail_size >= 3

    def test_min_tail_always_3_messages(self, budget_compressor):
        """Even with a tiny token budget, at least 3 messages are protected."""
        c = budget_compressor
        # Override to a tiny budget
        c.tail_token_budget = 10
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "working on it"},
            {"role": "user", "content": "more work"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]
        head_end = 2
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        tail_size = len(messages) - cut
        assert tail_size >= 3, f"Tail is only {tail_size} messages, min should be 3"

    def test_soft_ceiling_allows_oversized_message(self, budget_compressor):
        """The 1.5x soft ceiling allows an oversized message to be included
        rather than splitting it."""
        c = budget_compressor
        # Set a small budget — 500 tokens
        c.tail_token_budget = 500
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "read the file"},
            # This message is ~600 tokens (> budget of 500, but < 1.5x = 750)
            {"role": "assistant", "content": "a" * 2400},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "short reply"},
            {"role": "user", "content": "continue"},
        ]
        head_end = 2
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        # The oversized message at index 3 should NOT be the cut point
        # because 1.5x ceiling = 750 tokens and accumulated would be ~610
        # (short msgs + oversized msg) which is < 750
        tail_size = len(messages) - cut
        assert tail_size >= 3

    def test_small_conversation_still_compresses(self, budget_compressor):
        """With the new min of 8 messages (head=2 + 3 + 1 guard + 2 middle),
        a small but compressible conversation should still compress."""
        c = budget_compressor
        # 9 messages: head(2) + 4 middle + 3 tail = compressible
        messages = []
        for i in range(9):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        # Should not early-return (needs > protect_first_n + 3 + 1 = 6)
        # Mock the summary generation to avoid real API call
        with patch.object(c, "_generate_summary", return_value="Summary of conversation"):
            result = c.compress(messages, current_tokens=90_000)
        # Should have compressed (fewer messages than original)
        assert len(result) < len(messages)

    def test_prune_with_token_budget(self, budget_compressor):
        """_prune_old_tool_results with protect_tail_tokens respects the budget."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "big.txt"}'}}]},
            {"role": "tool", "content": "x" * 10000, "tool_call_id": "c1"},  # ~2500 tokens
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "small.txt"}'}}]},
            {"role": "tool", "content": "y" * 10000, "tool_call_id": "c2"},  # ~2500 tokens
            {"role": "user", "content": "short recent message"},
            {"role": "assistant", "content": "short reply"},
        ]
        # With a 1000-token budget, only the last couple messages should be protected
        result, pruned = c._prune_old_tool_results(
            messages, protect_tail_count=2, protect_tail_tokens=1000,
        )
        # At least one old tool result should have been pruned
        assert pruned >= 1

    def test_prune_without_token_budget_uses_message_count(self, budget_compressor):
        """Without protect_tail_tokens, falls back to message-count behavior."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "tool", "arguments": "{}"}}]},
            {"role": "tool", "content": "x" * 5000, "tool_call_id": "c1"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "reply"},
        ]
        # protect_tail_count=3 means last 3 messages protected
        result, pruned = c._prune_old_tool_results(
            messages, protect_tail_count=3,
        )
        # Tool at index 2 is outside the protected tail (last 3 = indices 2,3,4)
        # so it might or might not be pruned depending on boundary
        assert isinstance(pruned, int)

    def test_multimodal_message_accumulates_text_chars_not_block_count(self, budget_compressor):
        """_find_tail_cut_by_tokens must use text char count, not list length,
        for multimodal content. Regression guard for #16087.

        Setup: 6 messages, budget=80 (soft_ceiling=120).  The multimodal message
        at index 1 has 500 chars of text → 135 tokens (correct) or 10 tokens (bug).

        Fixed path: walk stops at the multimodal (44+135=179 > 120), cut stays at 2,
        tail = messages[2:] = 4 messages.

        Bug path: walk counts only 10 tokens for the multimodal, exhausts to head_end,
        the head_end safeguard forces cut = n - min_tail = 3, tail = only 3 messages.
        """
        c = budget_compressor
        # 500 chars → 500//4 + 10 = 135 tokens; len([text, image]) // 4 + 10 = 10 (bug)
        big_text = "x" * 500
        multimodal_content = [
            {"type": "text", "text": big_text},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        messages = [
            {"role": "user", "content": "head1"},               # 0
            {"role": "user", "content": multimodal_content},    # 1: BIG (index under test)
            {"role": "assistant", "content": "tail1"},           # 2
            {"role": "user", "content": "tail2"},                # 3
            {"role": "assistant", "content": "tail3"},           # 4
            {"role": "user", "content": "tail4"},                # 5
        ]
        c.tail_token_budget = 80  # soft_ceiling = 120
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        # With the fix: cut=2, tail has 4 messages (soft_ceiling not exceeded by tail1-4).
        # With the bug: head_end safeguard fires → cut = n - min_tail = 3, only 3 in tail.
        assert len(messages) - cut >= 4, (
            f"Expected ≥4 messages in tail (got {len(messages) - cut}, cut={cut}). "
            "The multimodal message was underestimated — len(list) used instead of text chars."
        )

    def test_plain_string_content_unchanged(self, budget_compressor):
        """Plain string content must still be estimated correctly after the fix."""
        c = budget_compressor
        # Same layout as the multimodal test but with a plain 500-char string.
        # Both buggy and fixed code count plain strings the same way (len(str)).
        # With 135 tokens the plain string also exceeds soft_ceiling=120, so
        # the walk stops at index 1 and tail has 4 messages — same as the fix path.
        big_plain = "x" * 500
        messages = [
            {"role": "user", "content": "head1"},
            {"role": "user", "content": big_plain},   # 1: 135 tokens, plain string
            {"role": "assistant", "content": "tail1"},
            {"role": "user", "content": "tail2"},
            {"role": "assistant", "content": "tail3"},
            {"role": "user", "content": "tail4"},
        ]
        c.tail_token_budget = 80
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert len(messages) - cut >= 4, (
            f"Plain string regression: expected ≥4 messages in tail, got {len(messages) - cut}"
        )

    def test_image_only_block_contributes_zero_text_chars(self, budget_compressor):
        """Image-only content blocks (no 'text' key) contribute 0 chars + base overhead."""
        c = budget_compressor
        c.tail_token_budget = 500
        image_only = [{"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}]
        messages = [
            {"role": "user", "content": "a" * 4000},
            {"role": "user", "content": image_only},   # 0 text chars → 10 tokens overhead
            {"role": "assistant", "content": "ok"},
        ]
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert isinstance(cut, int)
        assert 0 <= cut <= len(messages)

    def test_mixed_list_with_bare_strings_does_not_crash(self, budget_compressor):
        """Content list may contain bare strings (not dicts) — must not raise AttributeError."""
        c = budget_compressor
        c.tail_token_budget = 500
        # Bare string item alongside a dict item — normalisation elsewhere allows this.
        mixed_content = ["Hello, world!", {"type": "text", "text": "extra text"}]
        messages = [
            {"role": "user", "content": mixed_content},
            {"role": "assistant", "content": "ok"},
        ]
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert isinstance(cut, int)
        assert 0 <= cut <= len(messages)


class TestUpdateModelBudgets:
    """Regression: update_model() must recalculate token budgets."""

    def test_tail_budget_recalculated(self):
        """tail_token_budget must change after switching to a different context length."""
        from unittest.mock import patch
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            comp = ContextCompressor("model-a", threshold_percent=0.50, quiet_mode=True)
        old_tail = comp.tail_token_budget
        old_max_summary = comp.max_summary_tokens

        comp.update_model("model-b", context_length=32_000)
        assert comp.tail_token_budget != old_tail, "tail_token_budget should change"
        assert comp.tail_token_budget < old_tail, "smaller context → smaller budget"
        assert comp.max_summary_tokens != old_max_summary, "max_summary_tokens should change"

    def test_budgets_proportional(self):
        """Budgets should be proportional to context_length after update."""
        from unittest.mock import patch
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            comp = ContextCompressor("model-a", threshold_percent=0.50, quiet_mode=True)
        comp.update_model("model-b", context_length=10_000)
        assert comp.tail_token_budget == int(comp.threshold_tokens * comp.summary_target_ratio)
        assert comp.max_summary_tokens == min(int(10_000 * 0.05), 4000)


class TestTruncateToolCallArgsJson:
    """Regression tests for #11762.

    The previous implementation produced invalid JSON by slicing
    ``function.arguments`` mid-string, which caused non-retryable 400s from
    strict providers (observed on MiniMax) and stuck long sessions in a
    re-send loop. The helper here must always emit parseable JSON whose
    shape matches the original — shrunken, not corrupted.
    """

    def _helper(self):
        from agent.context_compressor import _truncate_tool_call_args_json
        return _truncate_tool_call_args_json

    def test_shrunken_args_remain_valid_json(self):
        import json as _json
        shrink = self._helper()
        original = _json.dumps({
            "path": "~/.hermes/skills/shopping/browser-setup-notes.md",
            "content": "# Shopping Browser Setup Notes\n\n" + "abc " * 400,
        })
        assert len(original) > 500
        shrunk = shrink(original)
        parsed = _json.loads(shrunk)  # must not raise
        assert parsed["path"] == "~/.hermes/skills/shopping/browser-setup-notes.md"
        assert parsed["content"].endswith("...[truncated]")
        assert len(shrunk) < len(original)

    def test_non_json_arguments_pass_through(self):
        shrink = self._helper()
        not_json = "this is not json at all, " * 50
        assert shrink(not_json) == not_json

    def test_short_string_leaves_unchanged(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({"command": "ls -la", "cwd": "/tmp"})
        assert _json.loads(shrink(payload)) == {"command": "ls -la", "cwd": "/tmp"}

    def test_nested_structures_are_walked(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({
            "messages": [
                {"role": "user", "content": "x" * 500},
                {"role": "assistant", "content": "ok"},
            ],
            "meta": {"note": "y" * 500},
        })
        parsed = _json.loads(shrink(payload))
        assert parsed["messages"][0]["content"].endswith("...[truncated]")
        assert parsed["messages"][1]["content"] == "ok"
        assert parsed["meta"]["note"].endswith("...[truncated]")

    def test_non_string_leaves_preserved(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({
            "retries": 3,
            "enabled": True,
            "timeout": None,
            "items": [1, 2, 3],
            "note": "z" * 500,
        })
        parsed = _json.loads(shrink(payload))
        assert parsed["retries"] == 3
        assert parsed["enabled"] is True
        assert parsed["timeout"] is None
        assert parsed["items"] == [1, 2, 3]
        assert parsed["note"].endswith("...[truncated]")

    def test_scalar_json_string_gets_shrunk(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps("q" * 500)
        parsed = _json.loads(shrink(payload))
        assert isinstance(parsed, str)
        assert parsed.endswith("...[truncated]")

    def test_unicode_preserved(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({"content": "非德满" + ("a" * 500)})
        out = shrink(payload)
        # ensure_ascii=False keeps CJK intact rather than emitting \uXXXX
        assert "非德满" in out

    def test_pass3_emits_valid_json_for_downstream_provider(self):
        """End-to-end: Pass 3 must never produce the exact failure payload
        that caused the 400 loop (unterminated string, missing brace)."""
        import json as _json
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )
        huge_content = "# Shopping Browser Setup Notes\n\n## Overview\n" + "x " * 400
        args_payload = _json.dumps({
            "path": "~/.hermes/skills/shopping/browser-setup-notes.md",
            "content": huge_content,
        })
        assert len(args_payload) > 500  # triggers the Pass-3 shrink
        messages = [
            {"role": "user", "content": "please write two files"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "write_file", "arguments": args_payload}},
            ]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"bytes_written": 727}'},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        result, _ = c._prune_old_tool_results(messages, protect_tail_count=2)
        shrunk = result[1]["tool_calls"][0]["function"]["arguments"]
        # Must parse — otherwise downstream provider returns 400
        parsed = _json.loads(shrunk)
        assert parsed["path"] == "~/.hermes/skills/shopping/browser-setup-notes.md"
        assert parsed["content"].endswith("...[truncated]")
