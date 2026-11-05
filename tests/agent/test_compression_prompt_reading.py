"""Compression leaves llama-server's prompt reading out of its total time limit.

The limit (600 s by default) stops a summary that trickles out a token at a time. On a laptop
GPU, reading a 158K-token conversation took a press reviewer's Spark about nine minutes before the
first summary token, so every compaction of a long session hit the limit and failed. llama-server
reports its reading as progress chunks; the reading they report pushes the deadline back, and the
limit applies to generation only. A server that reports without reading gets no extra time, and a
fence the host has closed stays closed.
"""

import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client, conversation_compression, local_prompt_progress
from agent.auxiliary_client import _ChatStreamAccumulator, _current_aux_stream_deadline
from agent.conversation_compression import CompressionCommitFence, _await_worker_within_budget
from agent.conversation_compression_deadline import compression_deadline_scope, effective_ceiling
from agent.local_prompt_progress import PromptReadingClock, prompt_reading_hook


@pytest.fixture
def clock(monkeypatch):
    """One fake monotonic clock for the stream, the reading clock and the fence."""
    now = [1000.0]
    fake = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    for module in (auxiliary_client, local_prompt_progress, conversation_compression):
        monkeypatch.setattr(module, "time", fake)
    return now


def _progress(processed, time_ms=None):
    payload = {"total": 150_000, "cache": 0, "processed": processed}
    if time_ms is not None:
        payload["time_ms"] = time_ms
    return SimpleNamespace(model_extra={"prompt_progress": payload},
                           choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)],
                           usage=None)


def _token(text):
    return SimpleNamespace(model_extra={}, choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, reasoning=None, reasoning_content=None, tool_calls=None),
        finish_reason=None)], usage=None)


def test_reading_time_pushes_back_the_shared_deadline(clock):
    fence = CompressionCommitFence(total_ceiling_seconds=60)
    with compression_deadline_scope(fence):
        reading = PromptReadingClock(clock[0])
        assert reading.tick({"processed": 0}) is True
        clock[0] += 45
        assert reading.tick({"processed": 40_000}) is True
        assert fence.deadline_monotonic == pytest.approx(1105.0)
        assert _current_aux_stream_deadline() == pytest.approx(1105.0)
    assert reading.seconds == pytest.approx(45.0)


def test_server_reading_time_bounds_each_stretch(clock):
    reading = PromptReadingClock(clock[0])
    clock[0] += 30  # queued behind another request, then the model loads
    reading.tick({"processed": 0, "time_ms": 0})
    clock[0] += 50
    reading.tick({"processed": 20_000, "time_ms": 40_000})
    assert reading.seconds == pytest.approx(40.0)


def test_progress_without_new_tokens_gets_no_time(clock):
    fence = CompressionCommitFence(total_ceiling_seconds=60)
    with compression_deadline_scope(fence):
        acc = _ChatStreamAccumulator(total_ceiling=60, host_deadline=_current_aux_stream_deadline)
        with patch.object(auxiliary_client, "_notify_aux_progress") as alive:
            clock[0] += 10
            acc.feed(_progress(5_000))
            for _ in range(2):  # a server stuck at the same count
                clock[0] += 20
                acc.feed(_progress(5_000))
            clock[0] += 20
            with pytest.raises(TimeoutError, match="timed out"):  # the limit still runs
                acc.feed(_progress(5_000))
        assert alive.call_count == 1
        assert fence.deadline_monotonic == pytest.approx(1070.0)  # only the first 10 s of reading


def test_a_closed_fence_stays_closed(clock):
    cancelled = CompressionCommitFence(total_ceiling_seconds=60)
    assert cancelled.try_cancel_before_commit()
    assert cancelled.extend_deadline(30) is False
    assert cancelled.is_cancelled

    expired = CompressionCommitFence(total_ceiling_seconds=60)
    clock[0] += 61
    assert expired.is_cancelled
    assert expired.extend_deadline(120) is False  # a chunk landing after the deadline
    assert expired.is_cancelled

    committing = CompressionCommitFence(total_ceiling_seconds=60)
    assert committing.begin_commit()
    assert committing.extend_deadline(30) is False  # the commit owns the fence; nothing to extend
    committing.finish_commit()
    assert committing.deadline_monotonic == pytest.approx(clock[0] + 60)


def test_stream_outlives_a_ceiling_shorter_than_its_prompt_reading(clock):
    fence = CompressionCommitFence(total_ceiling_seconds=60)
    with compression_deadline_scope(fence):
        acc = _ChatStreamAccumulator(total_ceiling=60, host_deadline=_current_aux_stream_deadline)
        for processed in (0, 50_000, 100_000, 150_000):  # 50 s between batches, 200 s of reading
            clock[0] += 50
            acc.feed(_progress(processed))
        clock[0] += 20
        assert acc.feed(_token("summary")) is True
        assert not fence.deadline_exceeded
        clock[0] += 45  # 65 s of generation: over the ceiling
        with pytest.raises(TimeoutError, match="timed out"):
            acc.feed(_token(" more"))


def test_reading_hook_is_scoped_to_its_block(clock):
    heard = []
    with prompt_reading_hook(heard.append):
        reading = PromptReadingClock(clock[0])
        reading.tick({"processed": 0})
        clock[0] += 3
        reading.tick({"processed": 100})
    outside = PromptReadingClock(clock[0])
    outside.tick({"processed": 0})
    clock[0] += 3
    outside.tick({"processed": 100})
    assert heard == [0.0, pytest.approx(3.0)]


def test_hosts_apply_the_moved_deadline(clock):
    fence = CompressionCommitFence(total_ceiling_seconds=60)
    wait_started = clock[0]
    assert effective_ceiling(fence, 60, wait_started) == pytest.approx(60)
    fence.extend_deadline(90)
    assert effective_ceiling(fence, 60, wait_started) == pytest.approx(150)


def test_host_keeps_waiting_while_the_deadline_moves():
    fence = CompressionCommitFence(total_ceiling_seconds=2.0)
    wait_started = time.monotonic()
    future = concurrent.futures.Future()
    # Reading already pushed the deadline to 6 s; the summary lands at 3 s, past the 2 s ceiling.
    fence.extend_deadline(4.0)
    fence.touch_progress()
    threading.Timer(3.0, lambda: future.set_result(("summary", ""))).start()
    settled, result = _await_worker_within_budget(future, fence, idle=10.0, ceiling=2.0, wait_started=wait_started)
    assert settled and result == ("summary", "")
