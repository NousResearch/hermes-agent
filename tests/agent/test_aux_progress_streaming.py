"""Tests for the auxiliary forward-progress streaming layer.

Slow summary models must not be punished like hung ones (#see PR): when a
forward-progress hook is installed (context compression), the primary
auxiliary call streams and ticks the hook per chunk, so outer watchdogs
(gateway session hygiene) can extend their deadline on liveness. Without a
hook, behavior is byte-for-byte the old non-streaming call.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import (
    _acreate_with_stream,
    _aggregate_chat_stream,
    _aggregate_chat_stream_async,
    _aux_stream_total_ceiling,
    _create_with_progress,
    _notify_aux_progress,
    _provider_requires_stream,
    aux_progress_hook,
)
from agent.conversation_compression import CompressionCommitFence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content=None, reasoning=None, finish_reason=None, usage=None,
           tool_calls=None, model="m1", chunk_id="c1"):
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id=chunk_id, model=model, choices=[choice], usage=usage,
    )


class _FakeClient:
    """OpenAI-shaped client whose create() returns a canned value or stream."""

    def __init__(self, response=None, stream_chunks=None, stream_error=None):
        self.calls = []
        self._response = response
        self._stream_chunks = stream_chunks
        self._stream_error = stream_error
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            if self._stream_error is not None:
                raise self._stream_error
            return iter(self._stream_chunks or [])
        return self._response


_COMPLETE = SimpleNamespace(
    id="r1", model="m1", object="chat.completion",
    choices=[SimpleNamespace(
        index=0,
        message=SimpleNamespace(role="assistant", content="non-streamed"),
        finish_reason="stop",
    )],
    usage=None,
)


# ---------------------------------------------------------------------------
# aux_progress_hook plumbing
# ---------------------------------------------------------------------------

class TestAuxProgressHook:
    def test_hook_installed_and_restored(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            _notify_aux_progress()
        _notify_aux_progress()  # outside — must not tick
        assert ticks == [1]

    def test_none_hook_is_noop_passthrough(self):
        with aux_progress_hook(None):
            _notify_aux_progress()  # must not raise

    def test_hook_exception_is_swallowed(self):
        def _boom():
            raise RuntimeError("hook blew up")
        with aux_progress_hook(_boom):
            _notify_aux_progress()  # must not raise

    def test_hook_is_thread_local(self):
        ticks = []
        seen_in_thread = []

        def _other_thread():
            # No hook installed on this thread.
            _notify_aux_progress()
            seen_in_thread.append(len(ticks))

        with aux_progress_hook(lambda: ticks.append(1)):
            t = threading.Thread(target=_other_thread)
            t.start()
            t.join()
        assert seen_in_thread == [0]


# ---------------------------------------------------------------------------
# _create_with_progress
# ---------------------------------------------------------------------------

class TestCreateWithProgress:
    def test_no_hook_means_plain_nonstreaming_call(self):
        client = _FakeClient(response=_COMPLETE)
        result = _create_with_progress(client, {"model": "m1", "messages": []})
        assert result is _COMPLETE
        assert len(client.calls) == 1
        assert "stream" not in client.calls[0]

    def test_hook_upgrades_to_streaming_and_ticks_per_chunk(self):
        chunks = [
            _chunk(reasoning="thinking..."),
            _chunk(content="Hello "),
            _chunk(content="world", finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2,
                                         total_tokens=7)),
        ]
        client = _FakeClient(stream_chunks=chunks)
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            result = _create_with_progress(
                client, {"model": "m1", "messages": [], "timeout": 30},
            )
        assert client.calls[0]["stream"] is True
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.reasoning == "thinking..."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 7
        # 1 dispatch tick + 1 per chunk
        assert len(ticks) >= len(chunks) + 1

    def test_streaming_rejected_falls_back_to_plain_call(self):
        client = _FakeClient(
            response=_COMPLETE,
            stream_error=RuntimeError("stream is not supported by this model"),
        )
        with aux_progress_hook(lambda: None):
            result = _create_with_progress(
                client, {"model": "m1", "messages": []},
            )
        assert result is _COMPLETE
        # streamed attempt + non-streaming fallback
        assert len(client.calls) == 2
        assert client.calls[0].get("stream") is True
        assert "stream" not in client.calls[1]

    def test_auth_error_propagates_without_nonstreaming_retry(self):
        class _FakeAuthError(Exception):
            status_code = 401
        client = _FakeClient(stream_error=_FakeAuthError("Error code: 401 - unauthorized"))
        with aux_progress_hook(lambda: None):
            with pytest.raises(_FakeAuthError):
                _create_with_progress(client, {"model": "m1", "messages": []})
        assert len(client.calls) == 1  # no silent non-streaming retry

    def test_shim_returning_complete_response_passes_through(self):
        # Adapters may ignore stream=True and hand back a full response.
        class _ShimClient(_FakeClient):
            def _create(self, **kwargs):
                self.calls.append(kwargs)
                return _COMPLETE
        client = _ShimClient()
        with aux_progress_hook(lambda: None):
            result = _create_with_progress(client, {"model": "m1", "messages": []})
        assert result is _COMPLETE


# ---------------------------------------------------------------------------
# _aggregate_chat_stream
# ---------------------------------------------------------------------------

class TestAggregateChatStream:
    def test_tool_call_deltas_are_reassembled(self):
        tc0 = SimpleNamespace(
            index=0, id="call_1",
            function=SimpleNamespace(name="do_thing", arguments='{"a"'),
        )
        tc1 = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments=': 1}'),
        )
        chunks = [
            _chunk(tool_calls=[tc0]),
            _chunk(tool_calls=[tc1], finish_reason="tool_calls"),
        ]
        result = _aggregate_chat_stream(iter(chunks))
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "do_thing"
        assert tool_calls[0].function.arguments == '{"a": 1}'
        assert result.choices[0].finish_reason == "tool_calls"

    def test_total_ceiling_kills_trickle_stream_as_timeout(self):
        def _trickle():
            while True:
                time.sleep(0.01)
                yield _chunk(content="x")
        with pytest.raises(TimeoutError, match="timed out"):
            _aggregate_chat_stream(_trickle(), total_ceiling=0.05)

    def test_stream_close_is_called(self):
        closed = []

        class _Stream:
            def __iter__(self):
                return iter([_chunk(content="ok", finish_reason="stop")])

            def close(self):
                closed.append(True)

        result = _aggregate_chat_stream(_Stream())
        assert result.choices[0].message.content == "ok"
        assert closed == [True]

    def test_empty_choices_chunks_are_skipped(self):
        empty = SimpleNamespace(id="c", model="m", choices=[], usage=None)
        chunks = [empty, _chunk(content="ok", finish_reason="stop")]
        result = _aggregate_chat_stream(iter(chunks))
        assert result.choices[0].message.content == "ok"


# ---------------------------------------------------------------------------
# Ceiling arithmetic
# ---------------------------------------------------------------------------

class TestStreamCeiling:
    def test_floor_applies_to_small_timeouts(self):
        assert _aux_stream_total_ceiling(30) == 600.0

    def test_multiplier_wins_for_large_timeouts(self):
        assert _aux_stream_total_ceiling(300) == 1200.0

    def test_none_timeout_gets_floor(self):
        assert _aux_stream_total_ceiling(None) == 600.0


# ---------------------------------------------------------------------------
# CompressionCommitFence progress surface
# ---------------------------------------------------------------------------

class TestFenceProgress:
    def test_touch_progress_resets_idle_clock(self):
        fence = CompressionCommitFence()
        time.sleep(0.05)
        assert fence.seconds_since_progress() >= 0.04
        fence.touch_progress()
        assert fence.seconds_since_progress() < 0.05

    def test_fence_hook_wiring_matches_compressor_usage(self):
        # conversation_compression installs fence.touch_progress as the hook;
        # verify the pair works end-to-end through _notify_aux_progress.
        fence = CompressionCommitFence()
        time.sleep(0.05)
        with aux_progress_hook(fence.touch_progress):
            _notify_aux_progress()
        assert fence.seconds_since_progress() < 0.05


# ---------------------------------------------------------------------------
# Stream-only providers (credit @kudi88, PR #60686)
# ---------------------------------------------------------------------------

class TestProviderRequiresStream:
    def test_tencent_copilot_is_stream_only(self):
        assert _provider_requires_stream(
            "custom", "https://copilot.tencent.com/v1"
        ) is True

    def test_normal_endpoints_are_not(self):
        assert _provider_requires_stream(
            "openrouter", "https://openrouter.ai/api/v1"
        ) is False
        assert _provider_requires_stream("auto", None) is False
        assert _provider_requires_stream("auto", "") is False

    def test_config_marker_matches_custom_endpoint(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"stream_only_base_urls": ["my-proxy.example.com"]}},
        ):
            assert _provider_requires_stream(
                "custom", "https://my-proxy.example.com/v1"
            ) is True
            assert _provider_requires_stream(
                "custom", "https://other.example.com/v1"
            ) is False

    def test_config_read_failure_fails_open_to_non_streaming(self):
        with patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("config broken"),
        ):
            assert _provider_requires_stream(
                "custom", "https://other.example.com/v1"
            ) is False


class TestForceStream:
    def test_force_stream_streams_without_a_hook(self):
        chunks = [_chunk(content="hi", finish_reason="stop")]
        client = _FakeClient(stream_chunks=chunks)
        # NO aux_progress_hook installed — force_stream alone must stream.
        result = _create_with_progress(
            client, {"model": "m1", "messages": []}, force_stream=True,
        )
        assert client.calls[0]["stream"] is True
        assert result.choices[0].message.content == "hi"

    def test_force_stream_does_not_retry_nonstreaming_on_failure(self):
        client = _FakeClient(
            response=_COMPLETE,
            stream_error=RuntimeError("HTTP 400 bad request"),
        )
        with pytest.raises(RuntimeError, match="bad request"):
            _create_with_progress(
                client, {"model": "m1", "messages": []}, force_stream=True,
            )
        # No silent non-streaming retry — the provider rejects those anyway.
        assert len(client.calls) == 1


class TestAsyncStreamAggregation:
    @pytest.mark.asyncio
    async def test_async_stream_is_consumed_with_async_for(self):
        # The sweeper review of PR #60686 flagged that awaiting create() and
        # then iterating synchronously raises — the async contract is
        # ``async for``. Verify the async aggregator consumes a real async
        # iterator and preserves tool-call deltas.
        tc0 = SimpleNamespace(
            index=0, id="call_9",
            function=SimpleNamespace(name="lookup", arguments='{"q":'),
        )
        tc1 = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments='"x"}'),
        )
        raw_chunks = [
            _chunk(content="part1 "),
            _chunk(tool_calls=[tc0]),
            _chunk(tool_calls=[tc1], content="part2", finish_reason="tool_calls"),
        ]

        class _AsyncStream:
            def __init__(self, items):
                self._items = list(items)
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

            async def close(self):
                self.closed = True

        stream = _AsyncStream(raw_chunks)
        result = await _aggregate_chat_stream_async(stream)
        msg = result.choices[0].message
        assert msg.content == "part1 part2"
        assert msg.tool_calls[0].function.name == "lookup"
        assert msg.tool_calls[0].function.arguments == '{"q":"x"}'
        assert result.choices[0].finish_reason == "tool_calls"
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_acreate_with_stream_passes_stream_kwargs(self):
        calls = []

        class _AsyncStream:
            def __init__(self, items):
                self._items = list(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        class _AsyncClient:
            def __init__(self):
                completions = SimpleNamespace(create=self._create)
                self.chat = SimpleNamespace(completions=completions)

            async def _create(self, **kwargs):
                calls.append(kwargs)
                return _AsyncStream([_chunk(content="ok", finish_reason="stop")])

        result = await _acreate_with_stream(
            _AsyncClient(), {"model": "m1", "messages": [], "timeout": 30},
        )
        assert calls[0]["stream"] is True
        assert result.choices[0].message.content == "ok"
