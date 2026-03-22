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


class TestShouldCompressPreflight:
    def test_short_messages(self, compressor):
        msgs = [{"role": "user", "content": "short"}]
        assert compressor.should_compress_preflight(msgs) is False

    def test_long_messages(self, compressor):
        # Each message ~100k chars / 4 = 25k tokens, need >85k threshold
        msgs = [{"role": "user", "content": "x" * 400000}]
        assert compressor.should_compress_preflight(msgs) is True


class TestUpdateFromResponse:
    def test_updates_fields(self, compressor):
        compressor.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 1000,
            "total_tokens": 6000,
        })
        assert compressor.last_prompt_tokens == 5000
        assert compressor.last_completion_tokens == 1000
        assert compressor.last_total_tokens == 6000

    def test_missing_fields_default_zero(self, compressor):
        compressor.update_from_response({})
        assert compressor.last_prompt_tokens == 0


class TestGetStatus:
    def test_returns_expected_keys(self, compressor):
        status = compressor.get_status()
        assert "last_prompt_tokens" in status
        assert "threshold_tokens" in status
        assert "context_length" in status
        assert "usage_percent" in status
        assert "compression_count" in status

    def test_usage_percent_calculation(self, compressor):
        compressor.last_prompt_tokens = 50000
        status = compressor.get_status()
        assert status["usage_percent"] == 50.0


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


class TestSummaryPrefixNormalization:
    def test_legacy_prefix_is_replaced(self):
        summary = ContextCompressor._with_summary_prefix("[CONTEXT SUMMARY]: did work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"

    def test_existing_new_prefix_is_not_duplicated(self):
        summary = ContextCompressor._with_summary_prefix(f"{SUMMARY_PREFIX}\ndid work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"


class TestCompressWithClient:
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

        # Last head message (index 1) is "assistant" → summary should be "user"
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
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

    def test_double_collision_user_head_assistant_tail(self):
        """Reverse double collision: head ends with 'user', tail starts with 'assistant'.
        summary='assistant' collides with tail, 'user' collides with head → merge."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head: [system, user]        → last head = user
        # Tail: [assistant, user]     → first tail = assistant
        # summary_role="assistant" collides with tail, "user" collides with head → merge
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},   # compressed
            {"role": "user", "content": "msg 3"},        # compressed
            {"role": "assistant", "content": "msg 4"},   # compressed
            {"role": "assistant", "content": "msg 5"},   # tail start
            {"role": "user", "content": "msg 6"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in ("user", "assistant") and r2 in ("user", "assistant"):
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

        # The summary should be merged into the first tail message (assistant)
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

        # Head=assistant, Tail=assistant → summary_role="user", no collision
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
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
        """Tail token budget should be context_length * summary_target_ratio."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        assert c.tail_token_budget == 80_000

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        assert c.tail_token_budget == 400_000

    def test_summary_cap_scales_with_context(self):
        """Max summary tokens should be 5% of context, capped at 32K."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 10_000  # 200K * 0.05

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 32_000  # capped at ceiling

    def test_ratio_clamped(self):
        """Ratio should be clamped to [0.10, 0.80]."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.05)
        assert c.summary_target_ratio == 0.10

        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.95)
        assert c.summary_target_ratio == 0.80

    def test_default_threshold_is_80_percent(self):
        """Default compression threshold should be 80%."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.threshold_percent == 0.80
        assert c.threshold_tokens == 80_000

    def test_default_protect_last_n_is_20(self):
        """Default protect_last_n should be 20."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.protect_last_n == 20
