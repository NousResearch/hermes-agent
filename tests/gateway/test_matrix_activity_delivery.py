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
    await asyncio.wait_for(pane.close(), timeout=0.5)
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
