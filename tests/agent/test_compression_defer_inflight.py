"""Tests for compression.defer_while_aux_inflight — contention deferral.

Covers the in-flight auxiliary call registry (agent/auxiliary_client.py) and
the opt-in deferral gate in ContextCompressor.should_compress()
(agent/context_compressor.py).
"""

import asyncio

import pytest
from unittest.mock import patch

from agent.auxiliary_client import (
    _INFLIGHT_AUX_TASKS,
    _inflight_aux_begin,
    _inflight_aux_end,
    _track_inflight_aux_async,
    _track_inflight_aux_sync,
    inflight_aux_count,
)
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty in-flight registry."""
    _INFLIGHT_AUX_TASKS.clear()
    yield
    _INFLIGHT_AUX_TASKS.clear()


def _make_compressor(**kwargs):
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            quiet_mode=True,
            **kwargs,
        )


class TestInflightRegistry:
    def test_begin_end_roundtrip(self):
        _inflight_aux_begin("web_extract")
        assert inflight_aux_count() == 1
        _inflight_aux_end("web_extract")
        assert inflight_aux_count() == 0
        assert "web_extract" not in _INFLIGHT_AUX_TASKS

    def test_counts_accumulate_per_task(self):
        _inflight_aux_begin("web_extract")
        _inflight_aux_begin("web_extract")
        _inflight_aux_begin("vision")
        assert inflight_aux_count() == 3
        _inflight_aux_end("web_extract")
        assert inflight_aux_count() == 2

    def test_no_task_excluded_by_default(self):
        # Another session's compression is the heaviest contention source —
        # it must count by default.
        _inflight_aux_begin("compression")
        assert inflight_aux_count() == 1
        assert inflight_aux_count(exclude_tasks=("compression",)) == 0

    def test_decorators_applied_to_real_entry_points(self):
        # The registry only works if the real call_llm/async_call_llm are
        # actually wrapped — guard the two @decorator lines.
        from agent.auxiliary_client import async_call_llm, call_llm

        assert hasattr(call_llm, "__wrapped__")
        assert hasattr(async_call_llm, "__wrapped__")

    def test_none_task_not_tracked(self):
        _inflight_aux_begin(None)
        _inflight_aux_begin("")
        assert inflight_aux_count(exclude_tasks=()) == 0
        # end() on an untracked task must not underflow or raise
        _inflight_aux_end(None)
        _inflight_aux_end("never_started")
        assert inflight_aux_count(exclude_tasks=()) == 0

    def test_sync_decorator_tracks_and_releases_on_exception(self):
        observed = {}

        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            observed["during"] = inflight_aux_count()
            raise RuntimeError("provider exploded")

        with pytest.raises(RuntimeError):
            fake_call("web_extract")
        assert observed["during"] == 1
        assert inflight_aux_count() == 0

    def test_sync_decorator_accepts_keyword_task(self):
        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            return inflight_aux_count()

        assert fake_call(task="vision") == 1
        assert inflight_aux_count() == 0

    def test_async_decorator_tracks_and_releases(self):
        observed = {}

        @_track_inflight_aux_async
        async def fake_call(task=None, **kwargs):
            observed["during"] = inflight_aux_count()
            return "ok"

        assert asyncio.run(fake_call("web_extract")) == "ok"
        assert observed["during"] == 1
        assert inflight_aux_count() == 0

    def test_async_decorator_releases_on_exception(self):
        @_track_inflight_aux_async
        async def fake_call(task=None, **kwargs):
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            asyncio.run(fake_call("web_extract"))
        assert inflight_aux_count() == 0


class TestShouldCompressDefer:
    # Fixture geometry: context_length=100000, threshold 0.85 -> 85000
    # tokens, default ceiling 0.95 -> 95000 tokens.

    def test_default_off_ignores_inflight(self):
        compressor = _make_compressor()
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_defers_when_aux_inflight(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is False

    def test_compresses_when_nothing_inflight(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_foreign_compression_inflight_defers(self):
        # A session checking should_compress can never see its own
        # summarizer call (check and call are sequential within a turn), so
        # an in-flight "compression" belongs to another session and must
        # serialize behind it.
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("compression")
        assert compressor.should_compress(prompt_tokens=90000) is False

    def test_max_consecutive_defers_then_fires(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        for _ in range(ContextCompressor.DEFER_MAX_CONSECUTIVE_CHECKS):
            assert compressor.should_compress(prompt_tokens=90000) is False
        # Bound reached: fires even though the call is still in flight.
        assert compressor.should_compress(prompt_tokens=90000) is True
        # Firing reset the counter — deferral works again.
        assert compressor.should_compress(prompt_tokens=90000) is False

    def test_counter_resets_when_compression_fires_normally(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is False
        _inflight_aux_end("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is True
        assert compressor._consecutive_defer_checks == 0

    def test_inert_gate_warns(self, caplog):
        import logging

        # Context window small enough that the MINIMUM_CONTEXT_LENGTH floor
        # pushes the threshold past the ceiling: gate can never defer.
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            with patch(
                "agent.context_compressor.get_model_context_length",
                return_value=65536,
            ):
                ContextCompressor(
                    model="test/model",
                    threshold_percent=0.99,
                    quiet_mode=False,
                    defer_while_aux_inflight=True,
                    defer_hard_ceiling=0.95,
                )
        assert any("inert" in r.message for r in caplog.records)

    def test_hard_ceiling_overrides_deferral(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=96000) is True

    def test_below_threshold_unaffected(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=50000) is False

    def test_deferral_is_re_evaluated(self):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is False
        _inflight_aux_end("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_ceiling_clamped_to_valid_range(self):
        compressor = _make_compressor(
            defer_while_aux_inflight=True, defer_hard_ceiling=2.0,
        )
        assert compressor.defer_hard_ceiling == 1.0
        compressor = _make_compressor(
            defer_while_aux_inflight=True, defer_hard_ceiling=-1.0,
        )
        # Degenerate ceiling 0.0: every over-threshold check is past the
        # ceiling, so compression always fires — deferral safely disabled.
        assert compressor.defer_hard_ceiling == 0.0
        _inflight_aux_begin("web_extract")
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_defer_logs_at_info(self, caplog):
        compressor = _make_compressor(defer_while_aux_inflight=True)
        compressor.quiet_mode = False
        _inflight_aux_begin("web_extract")
        import logging

        with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
            assert compressor.should_compress(prompt_tokens=90000) is False
        assert any("Compression deferred" in r.message for r in caplog.records)


class TestGatewayCacheBusting:
    def test_defer_keys_bust_gateway_agent_cache(self):
        """Hot-reload promise: flipping either defer key must change the
        cached-agent signature so a running gateway picks it up."""
        from gateway.run import GatewayRunner

        sig_off = GatewayRunner._extract_cache_busting_config(
            {"compression": {"defer_while_aux_inflight": False}}
        )
        sig_on = GatewayRunner._extract_cache_busting_config(
            {"compression": {"defer_while_aux_inflight": True}}
        )
        assert sig_off != sig_on

        sig_95 = GatewayRunner._extract_cache_busting_config(
            {"compression": {"defer_hard_ceiling": 0.95}}
        )
        sig_85 = GatewayRunner._extract_cache_busting_config(
            {"compression": {"defer_hard_ceiling": 0.85}}
        )
        assert sig_95 != sig_85
