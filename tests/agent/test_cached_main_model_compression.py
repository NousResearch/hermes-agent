"""Tests for compression.mode: cached_main_model — KV-prefix-reuse summary mode.

The whole point of this mode is that the summary request reuses the live
conversation's already-cached KV prefix instead of a fresh single-user-message
prompt (which has zero prefix reuse and forces a full cold prefill on local
llama.cpp / LM Studio servers). The request becomes [captured last-sent
api_messages + one appended user instruction], so the server only prefills the
small tail.

These tests assert the BEHAVIOR CONTRACTS, not snapshots:
  * the captured prefix is byte-for-byte preserved in the built request (the
    invariant that makes KV reuse possible),
  * exactly ONE message is appended and it is a role="user" instruction,
  * the stored capture is never mutated by building a request (deep copy),
  * the mode falls back to auxiliary (builder returns None) whenever the
    captured prefix cannot safely take a trailing user instruction.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def cached_cc():
    """A compressor configured for cached_main_model mode."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            compression_mode="cached_main_model",
        )
        _ = c.context_length  # resolve while the mock is active
    return c


def _valid_prefix():
    """A captured prefix ending in a plain assistant turn (the safe case)."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there, how can I help?"},
    ]


class TestBuildCachedPrefixSummaryRequest:
    def test_prefix_is_byte_identical_and_one_user_instruction_appended(self, cached_cc):
        captured = _valid_prefix()
        cached_cc.set_last_sent_api_messages(captured)

        req = cached_cc._build_cached_prefix_summary_request(
            focus_topic=None, memory_context="", summary_budget=1000,
        )

        assert req is not None
        # Every original message must be preserved byte-for-byte — this is the
        # invariant that lets the provider reuse its cached KV prefix.
        for i in range(len(captured)):
            assert req[i] == captured[i], f"prefix mismatch at index {i}"
        # Exactly one message appended, and it is a user instruction.
        assert len(req) == len(captured) + 1
        last = req[-1]
        assert last["role"] == "user"
        assert "context-compaction checkpoint summary" in last["content"]

    def test_appended_instruction_carries_structured_template(self, cached_cc):
        cached_cc.set_last_sent_api_messages(_valid_prefix())
        req = cached_cc._build_cached_prefix_summary_request(None, "", 1000)
        content = req[-1]["content"]
        # The summary shape must match the auxiliary path so downstream parsing
        # and continuity logic work unchanged.
        assert "## Goal" in content
        assert "## Critical Context" in content
        assert "NEVER include API keys" in content

    def test_stored_capture_is_not_mutated_by_builder(self, cached_cc):
        captured = _valid_prefix()
        cached_cc.set_last_sent_api_messages(captured)
        before_len = len(captured)
        before_snapshot = [dict(m) for m in captured]

        req = cached_cc._build_cached_prefix_summary_request(None, "", 1000)
        assert req is not None

        # The stored capture must be untouched: same length, same contents.
        assert len(captured) == before_len
        for i, m in enumerate(captured):
            assert m == before_snapshot[i]
        # And the built request's prefix shares no dict identity with the store
        # (deep copy), so mutating the request can't corrupt the capture.
        assert req[0] is not captured[0]

    def test_focus_topic_is_folded_into_instruction(self, cached_cc):
        cached_cc.set_last_sent_api_messages(_valid_prefix())
        req = cached_cc._build_cached_prefix_summary_request(
            focus_topic="compress mode", memory_context="", summary_budget=1000,
        )
        assert 'FOCUS TOPIC: "compress mode"' in req[-1]["content"]

    def test_memory_context_is_folded_into_instruction(self, cached_cc):
        cached_cc.set_last_sent_api_messages(_valid_prefix())
        req = cached_cc._build_cached_prefix_summary_request(
            focus_topic=None, memory_context='{"note": "x"}', summary_budget=1000,
        )
        assert "MEMORY PROVIDER CONTEXT" in req[-1]["content"]


class TestCachedPrefixFallback:
    """The builder must return None (fall back to auxiliary) whenever the
    captured prefix cannot safely take a trailing user instruction."""

    def test_no_captured_prefix_returns_none(self, cached_cc):
        assert cached_cc._build_cached_prefix_summary_request(None, "", 100) is None

    def test_empty_captured_prefix_returns_none(self, cached_cc):
        cached_cc.set_last_sent_api_messages([])
        assert cached_cc._build_cached_prefix_summary_request(None, "", 100) is None

    def test_prefix_ending_in_user_returns_none(self, cached_cc):
        # A trailing user message cannot take another user instruction without
        # breaking strict role alternation.
        cached_cc.set_last_sent_api_messages([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ])
        assert cached_cc._build_cached_prefix_summary_request(None, "", 100) is None

    def test_prefix_ending_in_tool_returns_none(self, cached_cc):
        cached_cc.set_last_sent_api_messages([
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ])
        assert cached_cc._build_cached_prefix_summary_request(None, "", 100) is None

    def test_prefix_ending_in_assistant_with_tool_calls_returns_none(self, cached_cc):
        # An assistant turn with pending tool_calls cannot take a trailing user
        # instruction (the wire expects the matching tool result next).
        cached_cc.set_last_sent_api_messages([
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ])
        assert cached_cc._build_cached_prefix_summary_request(None, "", 100) is None


class TestSetLastSentApiMessages:
    def test_stores_deep_copy(self, cached_cc):
        captured = _valid_prefix()
        cached_cc.set_last_sent_api_messages(captured)
        # The stored list must be a distinct object (deep copy).
        assert cached_cc._last_sent_api_messages is not captured
        assert cached_cc._last_sent_api_messages[0] is not captured[0]

    def test_none_clears_capture(self, cached_cc):
        cached_cc.set_last_sent_api_messages(_valid_prefix())
        assert cached_cc._last_sent_api_messages is not None
        cached_cc.set_last_sent_api_messages(None)
        assert cached_cc._last_sent_api_messages is None


class TestCompressionModeConfig:
    def test_unknown_mode_falls_back_to_auxiliary(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="m", quiet_mode=True, compression_mode="bogus")
        assert c.compression_mode == "auxiliary"

    def test_valid_modes_are_preserved(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            a = ContextCompressor(model="m", quiet_mode=True, compression_mode="auxiliary")
            b = ContextCompressor(model="m", quiet_mode=True, compression_mode="cached_main_model")
        assert a.compression_mode == "auxiliary"
        assert b.compression_mode == "cached_main_model"

    def test_update_model_clears_capture_on_runtime_change(self, cached_cc):
        captured = _valid_prefix()
        cached_cc.set_last_sent_api_messages(captured)
        assert cached_cc._last_sent_api_messages is not None
        # A model/provider switch invalidates the live KV cache.
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            cached_cc.update_model(
                model="other/model", context_length=100000,
                base_url="", api_key="", provider="", api_mode="",
            )
        assert cached_cc._last_sent_api_messages is None
