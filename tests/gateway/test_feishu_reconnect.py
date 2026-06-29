"""hc-384 self-reconnect supervisor + hc-385 long-task heartbeat.

The Feishu adapter used to hand websocket reconnection to the lark SDK, which
in prod tried exactly once and gave up (0/188 reconnects), leaving bots dead
for hours. hc-384 makes the adapter own reconnection: disable the SDK's retry,
detect the dead socket, run a backoff ladder, verify liveness, and escalate to
a retryable fatal error so the gateway restarts the adapter. hc-385 adds an
opt-in heartbeat that edits a status message in place during long tasks.

Tests are sync + ``asyncio.run`` to match the existing Feishu suite. The lark
SDK is not installed in CI's dev env, so we exercise the adapter-owned logic
(supervisor / ladder / probe / heartbeat) with fakes rather than a real client.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import gateway.platforms.feishu as feishu
from apex_overlay import feishu_supervisor
from gateway.platforms.base import (
    MessageEvent,
    PlatformConfig,
    ProcessingOutcome,
    SendResult,
)

# hc-384 self-reconnect supervisor lives in the apex_overlay.feishu_supervisor
# seam (zero-in-place; applied by the apex-overlay plugin at gateway boot). In
# prod the plugin runs apply() before any adapter connects; this suite imports
# the adapter cold, so apply it here to install the real supervisor methods onto
# FeishuAdapter (matching the booted gateway). Idempotent + fail-safe; per-file
# process isolation keeps this class patch scoped to this test file.
#
# The fast-ladder helper monkeypatches the reconnect tuning on the *seam* module
# now (the constants moved there from feishu.py), so the ladder still runs fast.
feishu_supervisor.apply()


def _adapter(**extra):
    return feishu.FeishuAdapter(PlatformConfig(extra=extra or {}))


def _event(message_id="om_1", chat_id="oc_1"):
    return MessageEvent(
        text="hi", message_id=message_id, source=SimpleNamespace(chat_id=chat_id)
    )


# ---------------------------------------------------------------------------
# hc-384 — settings + SDK auto-reconnect disable
# ---------------------------------------------------------------------------


def test_ws_self_reconnect_defaults_on():
    assert _adapter()._ws_self_reconnect is True


def test_ws_self_reconnect_revert_via_extra():
    assert _adapter(ws_self_reconnect=False)._ws_self_reconnect is False


def test_runtime_overrides_disable_sdk_reconnect_when_enabled():
    adapter = _adapter()
    ws = SimpleNamespace()
    feishu._apply_feishu_ws_runtime_overrides(ws, adapter)
    # hc-384: we take over reconnection, so the SDK's own retry is off.
    assert ws._auto_reconnect is False
    assert ws._reconnect_nonce == adapter._ws_reconnect_nonce
    assert ws._reconnect_interval == adapter._ws_reconnect_interval


def test_runtime_overrides_leave_sdk_reconnect_when_reverted():
    adapter = _adapter(ws_self_reconnect=False)
    ws = SimpleNamespace()
    feishu._apply_feishu_ws_runtime_overrides(ws, adapter)
    # Revert path: do NOT touch the SDK's auto-reconnect (back to old behaviour).
    assert not hasattr(ws, "_auto_reconnect")
    assert ws._reconnect_nonce == adapter._ws_reconnect_nonce


# ---------------------------------------------------------------------------
# hc-384 — death detection
# ---------------------------------------------------------------------------


def test_websocket_appears_dead_no_client():
    adapter = _adapter()
    adapter._ws_client = None
    assert adapter._websocket_appears_dead() is True


def test_websocket_appears_dead_conn_cleared():
    adapter = _adapter()
    # lark's _disconnect() sets _conn=None when the receive loop exits.
    adapter._ws_client = SimpleNamespace(_conn=None)
    adapter._ws_future = SimpleNamespace(done=lambda: False)
    assert adapter._websocket_appears_dead() is True


def test_websocket_appears_dead_future_done():
    adapter = _adapter()
    adapter._ws_client = SimpleNamespace(_conn=object())
    adapter._ws_future = SimpleNamespace(done=lambda: True)
    assert adapter._websocket_appears_dead() is True


def test_websocket_appears_alive():
    adapter = _adapter()
    adapter._ws_client = SimpleNamespace(_conn=object())
    adapter._ws_future = SimpleNamespace(done=lambda: False)
    assert adapter._websocket_appears_dead() is False


# ---------------------------------------------------------------------------
# hc-384 — reconnect ladder
# ---------------------------------------------------------------------------


def _fast_ladder(monkeypatch, *, max_attempts=None):
    monkeypatch.setattr(feishu_supervisor, "_FEISHU_WS_RECONNECT_BASE_DELAY", 0)
    monkeypatch.setattr(feishu_supervisor, "_FEISHU_WS_RECONNECT_MAX_DELAY", 0)
    monkeypatch.setattr(feishu_supervisor, "_FEISHU_WS_RECONNECT_VERIFY_DELAY", 0)
    if max_attempts is not None:
        monkeypatch.setattr(feishu_supervisor, "_FEISHU_WS_RECONNECT_MAX_ATTEMPTS", max_attempts)


def test_reconnect_ladder_succeeds_first_attempt(monkeypatch):
    _fast_ladder(monkeypatch)
    adapter = _adapter()
    adapter._running = True
    adapter._teardown_ws_thread = AsyncMock()
    adapter._connect_websocket = AsyncMock()
    adapter._verify_ws_alive = AsyncMock(return_value=True)
    notify = AsyncMock()
    adapter._notify_fatal_error = notify

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    adapter._teardown_ws_thread.assert_awaited_once()
    adapter._connect_websocket.assert_awaited_once()
    adapter._verify_ws_alive.assert_awaited_once()
    notify.assert_not_awaited()
    assert adapter.has_fatal_error is False
    assert adapter._ws_reconnecting is False


def test_reconnect_ladder_retries_until_verified(monkeypatch):
    _fast_ladder(monkeypatch, max_attempts=5)
    adapter = _adapter()
    adapter._running = True
    adapter._teardown_ws_thread = AsyncMock()
    adapter._connect_websocket = AsyncMock()
    # Fail the liveness probe twice, then succeed.
    adapter._verify_ws_alive = AsyncMock(side_effect=[False, False, True])
    notify = AsyncMock()
    adapter._notify_fatal_error = notify

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    assert adapter._connect_websocket.await_count == 3
    notify.assert_not_awaited()
    assert adapter.has_fatal_error is False


def test_reconnect_ladder_exhaustion_escalates_to_fatal(monkeypatch):
    _fast_ladder(monkeypatch, max_attempts=3)
    adapter = _adapter()
    adapter._running = True
    adapter._teardown_ws_thread = AsyncMock()
    adapter._connect_websocket = AsyncMock()
    adapter._verify_ws_alive = AsyncMock(return_value=False)
    notify = AsyncMock()
    adapter._notify_fatal_error = notify

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    assert adapter._connect_websocket.await_count == 3
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "feishu_ws_reconnect_exhausted"
    # Retryable so the gateway watcher recreates the adapter (re-acquires lock).
    assert adapter.fatal_error_retryable is True
    notify.assert_awaited_once()
    assert adapter._ws_reconnecting is False


def test_reconnect_ladder_continues_when_relaunch_raises(monkeypatch):
    _fast_ladder(monkeypatch, max_attempts=3)
    adapter = _adapter()
    adapter._running = True
    adapter._teardown_ws_thread = AsyncMock()
    # First relaunch raises, second succeeds.
    adapter._connect_websocket = AsyncMock(side_effect=[RuntimeError("boom"), None])
    adapter._verify_ws_alive = AsyncMock(return_value=True)
    notify = AsyncMock()
    adapter._notify_fatal_error = notify

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    assert adapter._connect_websocket.await_count == 2
    # Probe only runs after the successful relaunch.
    adapter._verify_ws_alive.assert_awaited_once()
    notify.assert_not_awaited()


def test_reconnect_ladder_aborts_on_intentional_disconnect(monkeypatch):
    _fast_ladder(monkeypatch)
    adapter = _adapter()
    adapter._running = True
    adapter._intentional_disconnect = True
    adapter._teardown_ws_thread = AsyncMock()
    adapter._connect_websocket = AsyncMock()
    notify = AsyncMock()
    adapter._notify_fatal_error = notify

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    adapter._connect_websocket.assert_not_awaited()
    notify.assert_not_awaited()
    assert adapter.has_fatal_error is False


def test_reconnect_reentrancy_guard(monkeypatch):
    _fast_ladder(monkeypatch)
    adapter = _adapter()
    adapter._running = True
    adapter._ws_reconnecting = True  # a ladder is already in flight
    adapter._connect_websocket = AsyncMock()

    asyncio.run(adapter._reconnect_websocket_with_backoff())

    adapter._connect_websocket.assert_not_awaited()
    # Guard must not clear another ladder's in-flight flag.
    assert adapter._ws_reconnecting is True


# ---------------------------------------------------------------------------
# hc-384 — liveness probe + teardown + supervisor
# ---------------------------------------------------------------------------


def test_verify_ws_alive_requires_socket_and_probe(monkeypatch):
    monkeypatch.setattr(feishu_supervisor, "_FEISHU_WS_RECONNECT_VERIFY_DELAY", 0)
    adapter = _adapter()
    adapter._running = True

    adapter._ws_client = None
    assert asyncio.run(adapter._verify_ws_alive()) is False

    adapter._ws_client = SimpleNamespace(_conn=object())
    adapter._hydrate_bot_identity = AsyncMock(return_value=True)
    assert asyncio.run(adapter._verify_ws_alive()) is True

    # Socket exists but the bot endpoint is unreachable → not alive.
    adapter._hydrate_bot_identity = AsyncMock(return_value=False)
    assert asyncio.run(adapter._verify_ws_alive()) is False


def test_teardown_ws_thread_clears_state():
    adapter = _adapter()
    adapter._ws_client = SimpleNamespace(_auto_reconnect=True)
    adapter._ws_thread_loop = None
    adapter._ws_future = None

    asyncio.run(adapter._teardown_ws_thread())

    # _disable_websocket_auto_reconnect drops the client reference.
    assert adapter._ws_client is None
    assert adapter._ws_future is None
    assert adapter._ws_thread_loop is None


def test_cancel_ws_supervisor_does_not_await_itself():
    """Fatal escalation drives disconnect() from inside the supervisor; the
    cancel must not deadlock by awaiting the running task itself."""
    adapter = _adapter()

    async def scenario():
        async def fake_supervisor():
            await adapter._cancel_ws_supervisor()
            return "completed"

        task = asyncio.ensure_future(fake_supervisor())
        adapter._ws_supervisor_task = task
        return await task

    assert asyncio.run(scenario()) == "completed"


def test_start_ws_supervisor_skips_when_reverted():
    adapter = _adapter(ws_self_reconnect=False)
    adapter._connection_mode = "websocket"
    loop = asyncio.new_event_loop()
    adapter._loop = loop
    adapter._start_ws_supervisor()
    assert adapter._ws_supervisor_task is None
    loop.close()


def test_start_ws_supervisor_skips_webhook_mode():
    adapter = _adapter()
    adapter._connection_mode = "webhook"
    loop = asyncio.new_event_loop()
    adapter._loop = loop
    adapter._start_ws_supervisor()
    assert adapter._ws_supervisor_task is None
    loop.close()


def test_start_ws_supervisor_creates_task_when_enabled():
    adapter = _adapter()
    adapter._connection_mode = "websocket"
    adapter._running = True
    loop = asyncio.new_event_loop()
    adapter._loop = loop
    adapter._start_ws_supervisor()
    task = adapter._ws_supervisor_task
    assert task is not None
    task.cancel()
    loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    loop.close()


# ---------------------------------------------------------------------------
# hc-385 — heartbeat
# ---------------------------------------------------------------------------


def test_heartbeat_disabled_by_default():
    assert _adapter()._heartbeat_enabled is False


def test_heartbeat_settings_from_extra():
    adapter = _adapter(
        heartbeat_enabled=True,
        heartbeat_interval_seconds=30,
        heartbeat_initial_delay_seconds=15,
    )
    assert adapter._heartbeat_enabled is True
    assert adapter._heartbeat_interval == 30
    assert adapter._heartbeat_initial_delay == 15


def test_maybe_start_heartbeat_noop_when_disabled():
    adapter = _adapter()  # heartbeat off

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        adapter._maybe_start_heartbeat(_event())

    asyncio.run(scenario())
    assert adapter._heartbeat_tasks == {}


def test_heartbeat_sends_then_edits_then_finalizes():
    adapter = _adapter(heartbeat_enabled=True)
    adapter._heartbeat_initial_delay = 0
    adapter._heartbeat_interval = 0.02
    adapter._running = True

    sends: list = []
    edits: list = []

    async def fake_send(chat_id, content, reply_to=None, metadata=None):
        sends.append((chat_id, content, reply_to))
        return SendResult(success=True, message_id="hb_1")

    async def fake_edit(chat_id, message_id, content, **kwargs):
        edits.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)

    adapter.send = fake_send
    adapter.edit_message = fake_edit

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        adapter._maybe_start_heartbeat(_event())
        task = adapter._heartbeat_tasks["om_1"]
        await asyncio.sleep(0.07)  # allow the first send + a couple of edits
        adapter._stop_heartbeat("om_1")
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    # Exactly one message is posted, then edited in place — never spammed.
    assert len(sends) == 1
    assert sends[0][0] == "oc_1"
    assert sends[0][2] == "om_1"  # threaded under the triggering message
    assert "仍在执行" in sends[0][1]
    assert edits, "expected at least one in-place edit"
    assert edits[-1][1] == "hb_1"
    # The finally block leaves an accurate final line.
    assert any("本轮处理结束" in content for _, _, content in edits)
    assert adapter._heartbeat_tasks == {}


def test_heartbeat_silent_for_short_task():
    adapter = _adapter(heartbeat_enabled=True)
    adapter._heartbeat_initial_delay = 10  # task ends well before this
    adapter._running = True

    sends: list = []

    async def fake_send(*args, **kwargs):
        sends.append(args)
        return SendResult(success=True, message_id="hb")

    adapter.send = fake_send
    adapter.edit_message = AsyncMock()

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        adapter._maybe_start_heartbeat(_event())
        task = adapter._heartbeat_tasks["om_1"]
        await asyncio.sleep(0)  # let it reach the initial-delay sleep
        adapter._stop_heartbeat("om_1")
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    # Short tasks must not post or finalize anything.
    assert sends == []
    adapter.edit_message.assert_not_awaited()
    assert adapter._heartbeat_tasks == {}


def test_on_processing_complete_stops_heartbeat():
    adapter = _adapter(heartbeat_enabled=True)
    adapter._running = True

    async def scenario():
        adapter._loop = asyncio.get_running_loop()

        async def hb():
            await asyncio.sleep(100)

        task = asyncio.ensure_future(hb())
        adapter._heartbeat_tasks["om_1"] = task
        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        await asyncio.gather(task, return_exceptions=True)
        return task.cancelled()

    assert asyncio.run(scenario()) is True


def test_cancel_all_heartbeats_on_disconnect():
    adapter = _adapter(heartbeat_enabled=True)

    async def scenario():
        async def hb():
            await asyncio.sleep(100)

        t1 = asyncio.ensure_future(hb())
        t2 = asyncio.ensure_future(hb())
        adapter._heartbeat_tasks = {"a": t1, "b": t2}
        await adapter._cancel_all_heartbeats()
        return t1.cancelled(), t2.cancelled(), dict(adapter._heartbeat_tasks)

    c1, c2, remaining = asyncio.run(scenario())
    assert c1 is True
    assert c2 is True
    assert remaining == {}
