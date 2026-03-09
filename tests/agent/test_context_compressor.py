"""Tests for agent/context_compressor.py — compression logic, thresholds, truncation fallback."""

import pytest
from unittest.mock import patch, MagicMock

from agent.context_compressor import (
    ContextCompressor,
    DEFAULT_COMPACTION_PROMPT,
    LEGACY_SUMMARY_PREFIX,
    MULTI_COMPACTION_WARNING,
    SUMMARY_PREFIX,
)


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with (
        patch("agent.context_compressor.get_model_context_length", return_value=100000),
        patch(
            "agent.context_compressor.get_text_auxiliary_client",
            return_value=(None, None),
        ),
    ):
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
        compressor.update_from_response(
            {
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
                "total_tokens": 6000,
            }
        )
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
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(n)
        ]

    def test_too_few_messages_returns_unchanged(self, compressor):
        msgs = self._make_messages(4)  # protect_first=2 + protect_last=2 + 1 = 5 needed
        result = compressor.compress(msgs, keep_latest_user_full=True)
        assert result == msgs

    def test_truncation_fallback_no_client(self, compressor):
        # compressor has client=None, so should use truncation fallback
        msgs = [{"role": "system", "content": "System prompt"}] + self._make_messages(
            10
        )
        result = compressor.compress(msgs, keep_latest_user_full=True)
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

    def test_preserves_latest_real_user_message(self, compressor):
        msgs = self._make_messages(10)
        result = compressor.compress(msgs, keep_latest_user_full=True)
        preserved_users = [
            msg["content"]
            for msg in result
            if msg.get("role") == "user"
            and not msg.get("content", "").startswith(SUMMARY_PREFIX)
        ]
        assert msgs[-2]["content"] in preserved_users

    def test_preserves_older_user_messages_within_budget_after_latest_request(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fresh handoff summary"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
                user_message_token_budget=10,
            )

        msgs = [
            {"role": "user", "content": "first preserved user"},
            {"role": "assistant", "content": "assistant 1"},
            {"role": "user", "content": "second preserved user"},
            {"role": "assistant", "content": "assistant 2"},
            {"role": "user", "content": "latest user"},
            {"role": "assistant", "content": "assistant 3"},
        ]

        result = compressor.compress(msgs, keep_latest_user_full=True)

        preserved_users = [
            msg["content"]
            for msg in result
            if msg.get("role") == "user"
            and not msg.get("content", "").startswith(SUMMARY_PREFIX)
        ]
        assert preserved_users == [
            "first pr…1 tokens truncated…ved user",
            "second preserved user",
            "latest user",
        ]

    def test_truncates_only_older_preserved_user_messages_to_fit_budget(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fresh handoff summary"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
                user_message_token_budget=2,
            )

        long_user_message = "x" * 40
        result = compressor.compress(
            [
                {"role": "user", "content": "older user"},
                {"role": "assistant", "content": "assistant 1"},
                {"role": "user", "content": long_user_message},
                {"role": "assistant", "content": "assistant 2"},
            ],
            keep_latest_user_full=True,
        )

        preserved_users = [
            msg["content"]
            for msg in result
            if msg.get("role") == "user"
            and not msg.get("content", "").startswith(SUMMARY_PREFIX)
        ]
        assert preserved_users == ["olde…1 tokens truncated…user", long_user_message]

    def test_default_budget_can_still_truncate_newest_preserved_user_message(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fresh handoff summary"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
                user_message_token_budget=2,
            )

        long_user_message = "x" * 40
        result = compressor.compress(
            [
                {"role": "user", "content": "older user"},
                {"role": "assistant", "content": "assistant 1"},
                {"role": "user", "content": long_user_message},
                {"role": "assistant", "content": "assistant 2"},
            ]
        )

        preserved_users = [
            msg["content"]
            for msg in result
            if msg.get("role") == "user"
            and not msg.get("content", "").startswith(SUMMARY_PREFIX)
        ]
        assert preserved_users == ["xxxx…8 tokens truncated…xxxx"]

    def test_skips_old_compaction_messages_when_building_new_summary(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "replacement summary"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )

        msgs = [
            {"role": "user", "content": f"{LEGACY_SUMMARY_PREFIX} old summary"},
            {"role": "assistant", "content": "assistant 1"},
            {"role": "user", "content": "real user request"},
            {"role": "assistant", "content": "assistant 2"},
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nnew-style summary"},
            {"role": "assistant", "content": "assistant 3"},
        ]

        compressor.compress(msgs)

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert "old summary" not in prompt
        assert "new-style summary" not in prompt

    def test_does_not_mutate_system_message_content(self):
        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(None, None),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )

        msgs = [
            {"role": "system", "content": "Original system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
        ]

        result = compressor.compress(msgs)
        assert result[0]["content"] == "Original system prompt"

    def test_warns_after_multiple_compactions(self, caplog):
        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(None, None),
            ),
        ):
            compressor = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )

        with caplog.at_level("WARNING"):
            compressor.compress(self._make_messages(6))

        assert MULTI_COMPACTION_WARNING in caplog.text
        assert caplog.records[-1].message == MULTI_COMPACTION_WARNING


class TestGenerateSummaryNoneContent:
    """Regression: content=None (from tool-call-only assistant messages) must not crash."""

    def test_none_content_does_not_crash(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "[CONTEXT SUMMARY]: tool calls happened"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "search"}}],
            },
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "thanks"},
        ]

        summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert summary.startswith(SUMMARY_PREFIX)

    def test_none_content_in_system_message_compress(self):
        """System message with content=None should not crash during compress."""
        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(None, None),
            ),
        ):
            c = ContextCompressor(
                model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2
            )

        msgs = [{"role": "system", "content": None}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(10)
        ]
        result = c.compress(msgs)
        assert len(result) < len(msgs)


class TestCompressWithClient:
    def test_summarization_path(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            c = ContextCompressor(model="test", quiet_mode=True)

        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(10)
        ]
        result = c.compress(msgs)

        # Should have summary message in the middle
        contents = [m.get("content", "") for m in result]
        assert any(content.startswith(SUMMARY_PREFIX) for content in contents)
        assert len(result) < len(msgs)

    def test_custom_compaction_prompt_override_used(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "custom summary"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test",
                quiet_mode=True,
                compaction_prompt_override="Use this exact custom prompt.",
            )

        compressor._generate_summary(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        )

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert prompt.startswith("Use this exact custom prompt.")

    def test_default_compaction_prompt_matches_codex_text(self):
        assert DEFAULT_COMPACTION_PROMPT == (
            "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
            "Include:\n"
            "- Current progress and key decisions made\n"
            "- Important context, constraints, or user preferences\n"
            "- What remains to be done (clear next steps)\n"
            "- Any critical data, examples, or references needed to continue\n\n"
            "Be concise, structured, and focused on helping the next LLM seamlessly continue the work.\n"
        )

    def test_preflight_mode_inserts_summary_before_current_user_request(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary content"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=1,
                protect_last_n=0,
                user_message_token_budget=100,
            )

        result = compressor.compress(
            [
                {"role": "user", "content": "older request"},
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": "follow-up request"},
                {"role": "assistant", "content": "follow-up reply"},
                {"role": "user", "content": "current request"},
            ],
            keep_latest_user_full=True,
        )

        user_messages = [msg["content"] for msg in result if msg.get("role") == "user"]
        assert user_messages[-1] == "current request"
        assert any(content.startswith(SUMMARY_PREFIX) for content in user_messages[:-1])

    def test_preflight_mode_exempts_current_user_request_from_budget(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary content"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=0,
                protect_last_n=0,
                user_message_token_budget=0,
            )

        current_request = "current request " * 20
        result = compressor.compress(
            [
                {"role": "user", "content": "older request"},
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": current_request},
            ],
            keep_latest_user_full=True,
        )

        user_messages = [msg["content"] for msg in result if msg.get("role") == "user"]
        assert user_messages[-1] == current_request

    def test_preflight_mode_preserves_multimodal_user_messages(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary content"
        mock_client.chat.completions.create.return_value = mock_response

        with (
            patch(
                "agent.context_compressor.get_model_context_length", return_value=100000
            ),
            patch(
                "agent.context_compressor.get_text_auxiliary_client",
                return_value=(mock_client, "test-model"),
            ),
        ):
            compressor = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=0,
                protect_last_n=0,
                user_message_token_budget=1,
            )

        latest_multimodal = [
            {"type": "text", "text": "compare these screenshots"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        result = compressor.compress(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "initial image request"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,BBBB"},
                        },
                    ],
                },
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": latest_multimodal},
            ],
            keep_latest_user_full=True,
        )

        assert result[-1]["content"] == latest_multimodal
