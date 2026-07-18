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
    from agent.auxiliary_client import _INFLIGHT_PENDING_ENDS

    _INFLIGHT_AUX_TASKS.clear()
    _INFLIGHT_PENDING_ENDS.clear()
    yield
    _INFLIGHT_AUX_TASKS.clear()
    _INFLIGHT_PENDING_ENDS.clear()


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

    def test_taskless_sync_call_counts_under_sentinel(self):
        # plugin_llm helpers call with task=None and trajectory_compressor
        # omits task entirely — both must be visible to the deferral gate,
        # while the sentinel itself must never be forwarded to the wrapped
        # function (it would corrupt task-based provider/model routing).
        observed = {}

        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            observed["during"] = inflight_aux_count()
            observed["task_seen"] = task
            return "ok"

        assert fake_call() == "ok"
        assert observed["during"] == 1
        assert observed["task_seen"] is None
        assert inflight_aux_count() == 0

        assert fake_call(task=None) == "ok"
        assert observed["during"] == 1
        assert inflight_aux_count() == 0

    def test_taskless_async_call_counts_under_sentinel(self):
        observed = {}

        @_track_inflight_aux_async
        async def fake_call(task=None, **kwargs):
            observed["during"] = inflight_aux_count()
            observed["task_seen"] = task
            return "ok"

        assert asyncio.run(fake_call()) == "ok"
        assert observed["during"] == 1
        assert observed["task_seen"] is None
        assert inflight_aux_count() == 0

    def test_taskless_release_on_exception(self):
        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            raise RuntimeError("provider exploded")

        with pytest.raises(RuntimeError):
            fake_call()
        assert inflight_aux_count() == 0


class TestStreamLifetime:
    """call_llm(stream=True) returns a raw iterator immediately; the registry
    entry must survive until that stream is exhausted, closed, or errors —
    the accelerator is busy while chunks are being produced, not just until
    the iterator object is handed back (MoA aggregator path).
    """

    @staticmethod
    def _stream_fn(chunks=("a", "b"), error_after=None):
        def gen():
            for i, chunk in enumerate(chunks):
                if error_after is not None and i == error_after:
                    raise RuntimeError("stream broke")
                yield chunk

        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            return gen()

        return fake_call

    def test_entry_survives_return_until_exhaustion(self):
        stream = self._stream_fn()("moa_aggregator", stream=True)
        # Handed back, not yet consumed: still in flight.
        assert inflight_aux_count() == 1
        assert next(stream) == "a"
        assert inflight_aux_count() == 1
        assert next(stream) == "b"
        with pytest.raises(StopIteration):
            next(stream)
        assert inflight_aux_count() == 0

    def test_close_releases_mid_consumption(self):
        stream = self._stream_fn()("moa_aggregator", stream=True)
        assert next(stream) == "a"
        assert inflight_aux_count() == 1
        stream.close()
        assert inflight_aux_count() == 0

    def test_error_mid_stream_releases(self):
        stream = self._stream_fn(error_after=1)("moa_aggregator", stream=True)
        assert next(stream) == "a"
        with pytest.raises(RuntimeError):
            next(stream)
        assert inflight_aux_count() == 0

    def test_release_fires_exactly_once(self):
        # A double release (close + exhaust + GC) must not decrement a
        # NEIGHBORING entry — e.g. another in-flight call of the same key.
        first = self._stream_fn()("moa_aggregator", stream=True)
        second = self._stream_fn()("moa_aggregator", stream=True)
        assert inflight_aux_count() == 2
        first.close()
        first.close()
        with pytest.raises(StopIteration):
            next(first)
        assert inflight_aux_count() == 1  # `second` must still be counted
        second.close()
        assert inflight_aux_count() == 0

    def test_context_manager_releases(self):
        with self._stream_fn()("moa_aggregator", stream=True) as stream:
            assert next(stream) == "a"
            assert inflight_aux_count() == 1
        assert inflight_aux_count() == 0

    def test_call_error_with_stream_flag_still_releases(self):
        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            raise RuntimeError("refused before any stream existed")

        with pytest.raises(RuntimeError):
            fake_call("moa_aggregator", stream=True)
        assert inflight_aux_count() == 0

    def test_non_stream_call_released_at_return(self):
        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            return "complete response"

        assert fake_call("web_extract") == "complete response"
        assert inflight_aux_count() == 0

    def test_attribute_delegation_to_inner_stream(self):
        class _FakeSDKStream:
            response = "http-response-object"

            def __iter__(self):
                return iter(())

        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            return _FakeSDKStream()

        stream = fake_call("moa_aggregator", stream=True)
        assert stream.response == "http-response-object"
        stream.close()
        assert inflight_aux_count() == 0

    def test_complete_response_under_stream_flag_not_wrapped(self):
        # Several in-tree producers accept stream=True but return a completed
        # response object (MoA openai-codex aggregator, Bedrock Converse
        # shim, copilot-acp).  Those calls are over when the frame returns:
        # the result must come back unwrapped and the entry must be released
        # immediately — wrapping it would pin a phantom in-flight entry for
        # the response object's whole lifetime.
        class _CompletedResponse:
            choices = [object()]

        @_track_inflight_aux_sync
        def fake_call(task=None, **kwargs):
            return _CompletedResponse()

        result = fake_call("moa_aggregator", stream=True)
        assert isinstance(result, _CompletedResponse)  # raw, no proxy
        assert inflight_aux_count() == 0

    def test_abandoned_stream_released_via_finalizer_without_lock(self):
        # The consumer abandons streams without close() on interrupt or
        # supersede.  The finalizer must release the entry WITHOUT touching
        # the registry lock (deadlock hazard if cyclic GC fires inside the
        # locked region): it deposits into _INFLIGHT_PENDING_ENDS, drained by
        # the next locked registry operation.
        import gc

        from agent.auxiliary_client import _INFLIGHT_PENDING_ENDS

        stream = self._stream_fn()("moa_aggregator", stream=True)
        assert next(stream) == "a"
        assert inflight_aux_count() == 1
        del stream
        gc.collect()
        # The finalizer deposits, it does not decrement in place.
        assert len(_INFLIGHT_PENDING_ENDS) == 1
        # Any locked registry operation drains the deposit.
        assert inflight_aux_count() == 0
        assert len(_INFLIGHT_PENDING_ENDS) == 0


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


# The defer keys' cache-busting behavior is covered where the rest of the
# compression subkeys are: tests/gateway/test_agent_cache.py
# (TestExtractCacheBustingConfig.test_reads_compression_subkeys).
