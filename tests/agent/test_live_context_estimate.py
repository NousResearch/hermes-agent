"""Tests for mid-turn live context estimates.

The desktop/TUI context bar reads ``context_used``/``context_percent`` off
tool.start/tool.complete event payloads. Before ``live_context_tokens`` the
value was a frozen snapshot of the previous API call's ``prompt_tokens``, so
the bar appeared stuck through long tool batches. These tests pin the new
behavior: provider-exact base plus a rough estimate of only the messages
appended since that snapshot.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough
from agent.tool_executor import _context_usage_for_tool_events


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


BASE_MESSAGES = [
    {"role": "user", "content": "hello " * 50},
    {"role": "assistant", "content": "hi there " * 30},
]


class TestLiveContextTokens:
    def test_no_tail_returns_exact_base(self, compressor):
        compressor.update_from_response(
            {"prompt_tokens": 5000}, messages_len=len(BASE_MESSAGES)
        )
        assert compressor.live_context_tokens(list(BASE_MESSAGES)) == 5000

    def test_prices_tail_appended_since_snapshot(self, compressor):
        compressor.update_from_response(
            {"prompt_tokens": 5000}, messages_len=len(BASE_MESSAGES)
        )
        tool_result = {"role": "tool", "content": "x" * 4000}
        msgs = list(BASE_MESSAGES) + [tool_result]
        expected_tail = estimate_messages_tokens_rough([tool_result])
        assert expected_tail > 0
        assert compressor.live_context_tokens(msgs) == 5000 + expected_tail

    def test_grows_monotonically_as_results_append(self, compressor):
        compressor.update_from_response(
            {"prompt_tokens": 7000}, messages_len=len(BASE_MESSAGES)
        )
        msgs = list(BASE_MESSAGES)
        seen = [compressor.live_context_tokens(msgs)]
        for _ in range(3):
            msgs.append({"role": "tool", "content": "result " * 500})
            seen.append(compressor.live_context_tokens(msgs))
        assert seen == sorted(seen)
        assert seen[-1] > seen[0]

    def test_fallback_without_snapshot_takes_max(self, compressor):
        msgs = [{"role": "tool", "content": "y" * 8000}]
        rough = estimate_messages_tokens_rough(msgs)

        compressor.last_prompt_tokens = 50  # tiny base, no snapshot recorded
        assert compressor.live_context_tokens(msgs) == rough

        compressor.last_prompt_tokens = 10_000_000
        assert compressor.live_context_tokens(msgs) == 10_000_000

    def test_invalid_snapshot_after_compression_prefers_rough(self, compressor):
        compressor.update_from_response({"prompt_tokens": 90_000}, messages_len=10)
        compressor.awaiting_real_usage_after_compression = True
        small = [{"role": "user", "content": "compressed summary"}]
        # Snapshot (10) no longer maps onto the shrunk list; the stale 90k
        # base must not win over the fresh rough estimate.
        assert compressor.live_context_tokens(small) == estimate_messages_tokens_rough(small)

    def test_update_from_response_records_snapshot(self, compressor):
        compressor.update_from_response({"prompt_tokens": 123}, messages_len=7)
        assert compressor.last_prompt_messages_len == 7
        # Callers that don't know the list length leave the snapshot alone.
        compressor.update_from_response({"prompt_tokens": 456})
        assert compressor.last_prompt_messages_len == 7

    def test_provider_usage_cannot_lower_same_preflight_snapshot(self, compressor):
        compressor.last_prompt_tokens = 67_000
        compressor.last_prompt_messages_len = len(BASE_MESSAGES)

        compressor.update_from_response(
            {"prompt_tokens": 48_000, "completion_tokens": 100, "total_tokens": 48_100},
            messages_len=len(BASE_MESSAGES),
        )

        assert compressor.last_prompt_tokens == 67_000
        assert compressor.last_real_prompt_tokens == 48_000
        assert compressor.last_total_tokens == 48_100
        assert compressor.live_context_tokens(list(BASE_MESSAGES)) == 67_000

    def test_provider_usage_can_lower_after_compression(self, compressor):
        compressor.last_prompt_tokens = 67_000
        compressor.last_prompt_messages_len = len(BASE_MESSAGES)
        compressor.awaiting_real_usage_after_compression = True

        compressor.update_from_response(
            {"prompt_tokens": 22_000, "completion_tokens": 100, "total_tokens": 22_100},
            messages_len=len(BASE_MESSAGES),
        )

        assert compressor.last_prompt_tokens == 22_000
        assert compressor.last_real_prompt_tokens == 22_000
        assert compressor.awaiting_real_usage_after_compression is False


class _FakeAgent:
    model = "test/model"
    session_input_tokens = 10
    session_output_tokens = 20
    session_total_tokens = 30
    session_api_calls = 2
    context_compressor = None


class TestToolEventUsage:
    def test_context_fields_use_live_estimate(self, compressor):
        agent = _FakeAgent()
        compressor.update_from_response({"prompt_tokens": 4000}, messages_len=1)
        agent.context_compressor = compressor

        msgs = [{"role": "user", "content": "q"}]
        first = _context_usage_for_tool_events(agent, msgs)
        assert first["context_used"] == 4000
        assert first["context_max"] == 100000
        assert first["context_percent"] == 4

        msgs.append({"role": "tool", "content": "z" * 40_000})
        second = _context_usage_for_tool_events(agent, msgs)
        assert second["context_used"] > first["context_used"]
        assert second["context_percent"] == round(second["context_used"] / 100000 * 100)

    def test_no_compressor_omits_context_fields(self):
        usage = _context_usage_for_tool_events(_FakeAgent(), [])
        assert "context_used" not in usage
        assert usage["model"] == "test/model"
        assert usage["total"] == 30

    def test_session_totals_passthrough(self, compressor):
        agent = _FakeAgent()
        agent.context_compressor = compressor
        usage = _context_usage_for_tool_events(agent, list(BASE_MESSAGES))
        assert usage["input"] == 10
        assert usage["output"] == 20
        assert usage["calls"] == 2
