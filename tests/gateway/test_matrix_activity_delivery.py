"""Delivery regressions for sticky-pane sealing and burst coalescing."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tests.gateway.test_matrix_tool_activity_pane import _PaneAdapter, _pane


@pytest.mark.asyncio
async def test_failed_seal_retries_without_new_root():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    adapter.edit_results = [SendResult(success=False, error="temporary")]
    await pane.close()
    assert pane.closed and pane.footer is None
    assert len(adapter.edits) == 2
    assert len(adapter.sends) == 1
    assert all(e["message_id"] == "$root" for e in adapter.edits)


@pytest.mark.asyncio
async def test_exhausted_seal_retains_acknowledged_footer_and_can_retry():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    adapter.edit_results = [SendResult(success=False, error="offline")] * 10
    await pane.close()
    assert not pane.closed
    assert pane.footer == "Working"
    assert 1 < len(adapter.edits) <= 3
    await pane.append_activity("too late")
    assert pane.activity_lines == []
    adapter.edit_results.clear()
    await pane.close()
    assert pane.closed and pane.footer is None
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_cancelled_failed_seal_finishes_retry_before_propagating():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    adapter.edit_results = [SendResult(success=False, error="temporary")]
    adapter.edit_entered, adapter.edit_release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(pane.close())
    await adapter.edit_entered.wait()
    task.cancel()
    adapter.edit_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pane.closed and pane.footer is None
    assert len(adapter.edits) == 2
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_seal_transport_timeout_is_bounded_and_retryable():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    pane.TRANSPORT_TIMEOUT = 0.01
    pane.SEAL_RETRY_DELAY = 0
    adapter.edit_release = asyncio.Event()
    await asyncio.wait_for(pane.close(), timeout=2.0)
    assert not pane.closed and pane.footer == "Working"
    assert len(adapter.edits) == pane.SEAL_ATTEMPTS
    adapter.edit_release.set()
    await pane.close()
    assert pane.closed
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_heartbeat_and_tool_updates_share_interval_and_idle_flush(monkeypatch):
    import gateway.matrix_activity_pane as module
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    pane.publish_interval = 1.5
    await pane.append_activity("status")
    for i in range(30):
        await pane.set_footer(f"Working {i}")
        await pane.append_activity(f"tool {i}")
    assert not adapter.edits
    clock.now = 1.5
    await pane.flush()
    assert len(adapter.edits) == 1
    html = adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert "Working 29" in html and "tool 29" in html
    await pane.flush()
    assert len(adapter.edits) == 1
    await pane.close()
    assert len(adapter.edits) == 2
    assert "Working" not in adapter.edits[-1]["content"]


@pytest.mark.asyncio
async def test_queue_burst_coalesces_and_final_seal_preserves_all_labels():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    ctx = TurnContext()
    ctx.matrix_activity_pane, ctx.progress_queue = pane, queue.Queue()
    ctx._run_still_current = lambda: True
    runner = TurnRunner(SimpleNamespace(), ctx)
    for i in range(80):
        ctx.progress_queue.put(f"tool {i}")
    task = asyncio.create_task(runner._send_matrix_activity_progress())
    while not ctx.progress_queue.empty():
        await asyncio.sleep(0)
    await asyncio.sleep(0.02)
    assert len(adapter.edits) <= 1
    task.cancel()
    await task
    await pane.close()
    assert pane.activity_lines == [f"tool {i}" for i in range(80)]
    assert len(adapter.sends) == 1
    assert len(adapter.edits) <= 2
    assert "Working" not in adapter.edits[-1]["content"]
    assert all(e["message_id"] == "$root" for e in adapter.edits)


@pytest.mark.asyncio
async def test_cancel_preserves_dequeued_label_waiting_for_heartbeat_lock():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    dequeued = asyncio.Event()

    class SignallingQueue(queue.Queue):
        def get_nowait(self):
            raw = super().get_nowait()
            dequeued.set()
            return raw

    ctx = TurnContext()
    ctx.matrix_activity_pane, ctx.progress_queue = pane, SignallingQueue()
    ctx._run_still_current = lambda: True
    runner = TurnRunner(SimpleNamespace(), ctx)
    ctx.progress_queue.put("last tool")
    # A heartbeat owns the same lock while the consumer dequeues its label.
    async with pane.lock:
        task = asyncio.create_task(runner._send_matrix_activity_progress())
        await asyncio.wait_for(dequeued.wait(), timeout=2.0)
        task.cancel()
        await asyncio.sleep(0)
    await asyncio.wait_for(task, timeout=2.0)
    await pane.close()
    assert pane.activity_lines == ["last tool"]
    assert "last tool" in adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_cancel_drain_coalesces_even_when_transport_exceeds_edit_interval(monkeypatch):
    import gateway.matrix_activity_pane as module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    class SlowAdapter(_PaneAdapter):
        async def edit_message(self, *args, **kwargs):
            clock.now += 2.0
            return await super().edit_message(*args, **kwargs)

    adapter = SlowAdapter()
    pane = _pane(adapter)
    await pane.set_footer("Working")
    ctx = TurnContext()
    ctx.matrix_activity_pane, ctx.progress_queue = pane, queue.Queue()
    ctx._run_still_current = lambda: True
    runner = TurnRunner(SimpleNamespace(), ctx)
    task = asyncio.create_task(runner._send_matrix_activity_progress())
    await asyncio.sleep(0)
    labels = [f"tool {i}" for i in range(30)]
    for label in labels:
        ctx.progress_queue.put(label)
    clock.now = 2.0
    task.cancel()
    await asyncio.wait_for(task, timeout=2.0)
    assert not adapter.edits  # The shutdown drain collects; close publishes once.
    await pane.close()
    assert pane.activity_lines == labels
    assert len(adapter.edits) == 1
    assert len(adapter.sends) == 1
    html = adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert all(label in html for label in labels)
    assert "Working" not in html


@pytest.mark.asyncio
async def test_cancelled_progress_transport_error_still_allows_turn_cleanup():
    entered, release = asyncio.Event(), asyncio.Event()

    class FailingAdapter(_PaneAdapter):
        async def send(self, *args, **kwargs):
            entered.set()
            await release.wait()
            raise TimeoutError("homeserver unavailable")

    pane = _pane(FailingAdapter())
    ctx = TurnContext()
    ctx.matrix_activity_pane, ctx.progress_queue = pane, queue.Queue()
    ctx._run_still_current = lambda: True
    ctx.progress_queue.put("last tool")
    runner = TurnRunner(SimpleNamespace(), ctx)
    task = asyncio.create_task(runner._send_matrix_activity_progress())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        task.cancel()
        release.set()
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done, "transport failure swallowed the cleanup cancellation"
        await task
        assert pane.activity_lines == ["last tool"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await task
