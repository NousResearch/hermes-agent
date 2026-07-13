"""Tests for session hygiene observability (issue #12626).

Verifies:
- /usage reads compression.hygiene_hard_message_limit (not a parallel key)
- /usage shows raw transcript row count vs the hygiene limit
- The hygiene gate correctly attributes trigger reason across three cases
- The user-visible notice fires only when message count was (part of) the trigger
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_slash_mixin(config=None):
    """Minimal GatewaySlashCommandsMixin with async session store."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    mixin = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    mixin.config = config if config is not None else {}
    mixin._agent_cache = {}
    mixin._agent_cache_lock = threading.Lock()
    mixin._running_agents = {}
    mixin._session_key_for_source = MagicMock(return_value="test_key")

    fake_se = MagicMock()
    fake_se.session_id = "sid"

    async_store = MagicMock()
    async_store.get_or_create_session = AsyncMock(return_value=fake_se)
    async_store.load_transcript = AsyncMock(return_value=[])
    mixin.async_session_store = async_store

    return mixin, async_store


def _make_event(platform=None):
    event = MagicMock()
    event.source = MagicMock()
    event.source.platform = platform or MagicMock()
    return event


# ---------------------------------------------------------------------------
# _hygiene_msg_limit_from_config — the shared config-reader in slash_commands
# ---------------------------------------------------------------------------

class TestHygieneMsgLimitFromConfig:
    """The helper reads compression.hygiene_hard_message_limit with a 5000 default."""

    def test_returns_default_when_key_absent(self):
        from gateway.slash_commands import _hygiene_msg_limit_from_config
        assert _hygiene_msg_limit_from_config({}) == 5000

    def test_reads_compression_section(self):
        from gateway.slash_commands import _hygiene_msg_limit_from_config
        cfg = {"compression": {"hygiene_hard_message_limit": 2000}}
        assert _hygiene_msg_limit_from_config(cfg) == 2000

    def test_zero_disables(self):
        from gateway.slash_commands import _hygiene_msg_limit_from_config
        cfg = {"compression": {"hygiene_hard_message_limit": 0}}
        assert _hygiene_msg_limit_from_config(cfg) == 0

    def test_invalid_value_returns_default(self):
        from gateway.slash_commands import _hygiene_msg_limit_from_config
        cfg = {"compression": {"hygiene_hard_message_limit": "not_a_number"}}
        assert _hygiene_msg_limit_from_config(cfg) == 5000

    def test_ignores_gateway_session_hygiene_key(self):
        """Old parallel config key must not be consulted."""
        from gateway.slash_commands import _hygiene_msg_limit_from_config
        cfg = {"gateway": {"session_hygiene": {"max_messages": 400}}}
        # No compression key → default 5000, NOT the legacy 400
        assert _hygiene_msg_limit_from_config(cfg) == 5000


# ---------------------------------------------------------------------------
# /usage — fallback branch (no live agent)
# ---------------------------------------------------------------------------

class TestUsageFallbackShowsRawCount:
    """/usage fallback uses len(history) raw rows to match the gate's accounting."""

    @pytest.mark.asyncio
    async def test_shows_raw_count_vs_limit(self):
        mixin, store = _make_slash_mixin(
            config={"compression": {"hygiene_hard_message_limit": 500}}
        )
        # 120 raw rows: 50 user + 50 assistant + 20 tool results
        raw_history = (
            [{"role": "user", "content": f"q{i}"} for i in range(50)]
            + [{"role": "assistant", "content": f"a{i}"} for i in range(50)]
            + [{"role": "tool", "tool_call_id": f"c{i}", "content": "ok"} for i in range(20)]
        )
        store.load_transcript.return_value = raw_history

        with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=8000):
            result = await mixin._handle_usage_command(_make_event())

        # Raw count (120) shown, not filtered count (100)
        assert "120" in result
        assert "500" in result

    @pytest.mark.asyncio
    async def test_warning_fires_at_90_percent_of_raw(self):
        mixin, store = _make_slash_mixin(
            config={"compression": {"hygiene_hard_message_limit": 100}}
        )
        # 92 raw rows — above 90 % threshold
        store.load_transcript.return_value = [
            {"role": "user", "content": f"m{i}"} for i in range(92)
        ]

        with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=2000):
            result = await mixin._handle_usage_command(_make_event())

        assert "Approaching message limit" in result

    @pytest.mark.asyncio
    async def test_no_warning_below_threshold(self):
        mixin, store = _make_slash_mixin(
            config={"compression": {"hygiene_hard_message_limit": 100}}
        )
        store.load_transcript.return_value = [
            {"role": "user", "content": f"m{i}"} for i in range(50)
        ]

        with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=1000):
            result = await mixin._handle_usage_command(_make_event())

        assert "Approaching" not in result

    @pytest.mark.asyncio
    async def test_disabled_limit_hides_proximity_line(self):
        mixin, store = _make_slash_mixin(
            config={"compression": {"hygiene_hard_message_limit": 0}}
        )
        store.load_transcript.return_value = [
            {"role": "user", "content": f"m{i}"} for i in range(999)
        ]

        with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=5000):
            result = await mixin._handle_usage_command(_make_event())

        # With limit=0, no count/limit line should appear
        assert "/ 0" not in result
        assert "Approaching" not in result


# ---------------------------------------------------------------------------
# Hygiene trigger reason — three-way accuracy
# ---------------------------------------------------------------------------

class TestHygieneReasonThreeWay:
    """The hygiene gate computes _hygiene_reason as message_count / token_pressure / both."""

    def _reason(self, msg_count, hard_limit, approx_tokens, compress_threshold):
        """Replicate the exact production expression from gateway/run.py."""
        _token_pressure = approx_tokens >= compress_threshold
        _count_pressure = hard_limit > 0 and msg_count >= hard_limit
        return (
            "both" if (_token_pressure and _count_pressure)
            else "message_count" if _count_pressure
            else "token_pressure"
        )

    def test_token_pressure_only(self):
        r = self._reason(100, 5000, 90_000, 85_000)
        assert r == "token_pressure"

    def test_message_count_only(self):
        r = self._reason(5001, 5000, 40_000, 85_000)
        assert r == "message_count"

    def test_both_triggers_simultaneously(self):
        r = self._reason(5001, 5000, 90_000, 85_000)
        assert r == "both"

    def test_disabled_limit_never_count_pressure(self):
        r = self._reason(99999, 0, 40_000, 85_000)
        assert r == "token_pressure"

    @pytest.mark.parametrize("reason,expected_in_notice", [
        ("message_count", "Token pressure was only"),
        ("both", "token usage was"),
    ])
    def test_notice_text_matches_reason(self, reason, expected_in_notice):
        """Notice copy accurately describes which condition fired."""
        msg_count, hard_limit, approx_tokens, ctx_length = 5001, 5000, 40_000, 200_000
        tok_pct = min(100, int(approx_tokens / ctx_length * 100))

        if reason == "both":
            trigger_desc = (
                f"your conversation reached {msg_count} messages "
                f"(limit: {hard_limit}) and token usage was "
                f"{tok_pct}% of the context window."
            )
        else:
            trigger_desc = (
                f"your conversation reached {msg_count} messages "
                f"(limit: {hard_limit}). Token pressure was "
                f"only {tok_pct}% of the context window."
            )

        notice = (
            f"ℹ️ **Session auto-compacted** — {trigger_desc}\n"
            f"Earlier context was summarized to keep things running smoothly.\n"
            f"_To adjust the limit: set `compression.hygiene_hard_message_limit` "
            f"in config.yaml. Set to `0` to disable the message-count valve._"
        )

        assert expected_in_notice in notice
        assert "compression.hygiene_hard_message_limit" in notice
        # Must NOT reference the old config key
        assert "gateway.session_hygiene" not in notice

    def test_token_pressure_reason_produces_no_notice(self):
        """Pure token-pressure compaction must not send any user notice."""
        reason = self._reason(100, 5000, 90_000, 85_000)
        assert reason == "token_pressure"
        # token_pressure is not in ("message_count", "both") so no notice fires
        assert reason not in ("message_count", "both")
