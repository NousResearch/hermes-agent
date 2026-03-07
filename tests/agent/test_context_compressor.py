"""Tests for agent/context_compressor.py — compression logic, thresholds, truncation fallback."""

import pytest
from unittest.mock import patch, MagicMock

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
         patch("agent.context_compressor.get_text_auxiliary_client", return_value=(None, None)):
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
        assert result[-2]["content"] == msgs[-2]["content"]


class TestGenerateSummaryNoneContent:
    """Regression: content=None (from tool-call-only assistant messages) must not crash."""

    def test_none_content_does_not_crash(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: tool calls happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(mock_client, "test-model")):
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

        summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert "CONTEXT SUMMARY" in summary

    def test_none_content_in_system_message_compress(self):
        """System message with content=None should not crash during compress."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(None, None)):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

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
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(mock_client, "test-model")):
            c = ContextCompressor(model="test", quiet_mode=True)

        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(10)]
        result = c.compress(msgs)

        # Should have summary message in the middle
        contents = [m.get("content", "") for m in result]
        assert any("CONTEXT SUMMARY" in c for c in contents)
        assert len(result) < len(msgs)


class TestPruneToolOutputs:
    def _make_compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(None, None)):
            return ContextCompressor(model="test/model", quiet_mode=True)

    def test_prune_replaces_old_tool_outputs(self):
        c = self._make_compressor()
        big_content = "x" * (c._prune_protect_tokens * 4 * 2)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": big_content, "name": "terminal"},
            {"role": "assistant", "content": "ok2"},
            {"role": "tool", "content": big_content, "name": "terminal"},
            {"role": "assistant", "content": "still going"},
            {"role": "tool", "content": big_content, "name": "terminal"},
            {"role": "assistant", "content": "done"},
        ]
        pruned, chars_saved = c._prune_tool_outputs(messages)
        assert chars_saved > 0
        pruned_tool_contents = [m["content"] for m in pruned if m.get("role") == "tool"]
        assert any("pruned" in c for c in pruned_tool_contents)

    def test_protected_tools_never_pruned(self):
        c = self._make_compressor()
        big_content = "x" * (c._prune_protect_tokens * 4 * 2)
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": big_content, "name": "memory"},
            {"role": "tool", "content": big_content, "name": "terminal"},
            {"role": "assistant", "content": "done"},
        ]
        pruned, chars_saved = c._prune_tool_outputs(messages)
        memory_msg = next(m for m in pruned if m.get("name") == "memory")
        assert memory_msg["content"] == big_content

    def test_no_tool_outputs_no_change(self):
        c = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        pruned, chars_saved = c._prune_tool_outputs(messages)
        assert chars_saved == 0
        assert pruned == messages


class TestAdaptiveThresholds:
    def _make_compressor(self, context_length):
        with patch("agent.context_compressor.get_model_context_length", return_value=context_length), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(None, None)):
            return ContextCompressor(model="test/model", quiet_mode=True)

    def test_large_context_gets_large_protect(self):
        c = self._make_compressor(1_000_000)
        assert c._prune_protect_tokens == 100_000

    def test_128k_context_gets_40k_protect(self):
        c = self._make_compressor(128_000)
        assert c._prune_protect_tokens == 40_000

    def test_small_context_gets_small_protect(self):
        c = self._make_compressor(32_000)
        assert c._prune_protect_tokens == 10_000


class TestCompactionTemplate:
    def test_summary_uses_structured_template(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: ## Goal\ntest goal"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000), \
             patch("agent.context_compressor.get_text_auxiliary_client", return_value=(mock_client, "test-model")):
            c = ContextCompressor(model="test", quiet_mode=True)

        msgs = [{"role": "user", "content": "do something"}]
        c._generate_summary(msgs)

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "## Goal" in prompt
        assert "## Accomplished So Far" in prompt
        assert "## Next Steps" in prompt
