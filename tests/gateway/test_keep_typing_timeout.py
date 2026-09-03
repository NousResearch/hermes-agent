"""Tests for BasePlatformAdapter._keep_typing timeout-per-tick behavior.

When the gateway is waiting on a long upstream provider response (e.g.
Anthropic/opus-4.7 first-token latency climbing during an upstream blip),
the model-call socket is blocked on the worker thread but the asyncio loop
is still running, and ``_keep_typing`` refreshes the platform typing
indicator every 2 seconds.

The bug: each ``send_typing`` call is an HTTP round-trip to the platform API
(Telegram/Discord). If the same network instability that's slowing the model
call also makes ``send_typing`` slow (5-30s response time), the refresh loop
stalls inside the ``await self.send_typing(...)`` call. Platform-side typing
expires at ~5s, so the bubble dies and doesn't come back until that stuck
call returns — exactly when the user most needs the "yes, still working"
signal.

The fix: bound each ``send_typing`` with ``asyncio.wait_for``. If a
send_typing takes longer than the per-tick budget (default 1.5s when
interval=2.0), abandon it and let the next scheduled tick fire a fresh
call. As long as any one of them succeeds within the ~5s platform window,
the bubble stays visible across provider stalls.
"""

import asyncio
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    PlatformConfig,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class TestTypingLeaseGeneration:
    def test_private_lease_metadata_is_not_stamped_for_unfenced_adapter(self):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = False
        adapter._begin_typing_lease("123", "session-1")

        assert adapter._typing_metadata_for_session(
            "session-1", {"thread_id": "topic-1"}
        ) == {"thread_id": "topic-1"}

    @pytest.mark.asyncio
    async def test_lock_registry_does_not_retain_idle_chat(self):
        adapter = _StubAdapter()
        lock = adapter._typing_lease_lock("one-shot-chat")
        lock_ref = weakref.ref(lock)

        del lock
        gc.collect()

        assert lock_ref() is None

    def test_replacement_adapter_recognizes_live_typing_lease(self):
        runner = SimpleNamespace()
        ingress = _StubAdapter()
        ingress.gateway_runner = runner
        lease_id = ingress._begin_typing_lease("123", "session-1")
        replacement = _StubAdapter()
        replacement.gateway_runner = runner

        assert replacement._typing_lease_allows(
            "123", {"_hermes_typing_lease_id": lease_id}
        )

    @pytest.mark.asyncio
    async def test_distinct_gateway_runtimes_isolate_typing_leases(self):
        first = _StubAdapter()
        first.gateway_runner = SimpleNamespace()
        second = _StubAdapter()
        second.gateway_runner = SimpleNamespace()

        first_lease = first._begin_typing_lease("123", "session-1")
        first_metadata = {"_hermes_typing_lease_id": first_lease}
        second._begin_typing_lease("123", "session-1")

        assert first._typing_lease_allows("123", first_metadata)
        await second._fence_all_typing_leases()
        assert first._typing_lease_allows("123", first_metadata)
        first._revoke_typing_lease(first_lease)

    def test_replacement_turn_invalidates_prior_lease(self):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = True
        first = adapter._begin_typing_lease("123", "session-1")
        old_metadata = adapter._typing_metadata_for_session("session-1", {})

        second = adapter._begin_typing_lease("123", "session-1")
        new_metadata = adapter._typing_metadata_for_session("session-1", {})

        assert first != second
        assert not adapter._typing_lease_allows("123", old_metadata)
        assert adapter._typing_lease_allows("123", new_metadata)

    @pytest.mark.asyncio
    async def test_finalization_fence_closes_then_revokes_lease(self):
        adapter = _StubAdapter()
        lease_id = adapter._begin_typing_lease("123", "session-1")
        metadata = {"_hermes_typing_lease_id": lease_id}
        fence = getattr(adapter, "_fence_typing_lease_before_final", None)

        assert callable(fence), "final delivery must fence admitted typing sends"
        lock = adapter._typing_lease_lock("123")
        await lock.acquire()
        try:
            fence_task = asyncio.create_task(
                fence("123", lease_id)
            )
            await asyncio.sleep(0)
            assert not adapter._typing_lease_allows("123", metadata)
            assert not fence_task.done()
        finally:
            lock.release()
        await fence_task
        assert not adapter._typing_lease_allows("123", metadata)

    @pytest.mark.asyncio
    async def test_shutdown_revokes_live_typing_leases(self):
        adapter = _StubAdapter()
        lease_id = adapter._begin_typing_lease("123", "session-1")
        metadata = {"_hermes_typing_lease_id": lease_id}

        await adapter.cancel_background_tasks()

        assert not adapter._typing_lease_allows("123", metadata)


    @pytest.mark.asyncio
    async def test_shutdown_drains_admitted_typing_action_before_returning(self):
        adapter = _StubAdapter()
        lease_id = adapter._begin_typing_lease("123", "session-1")
        metadata = {"_hermes_typing_lease_id": lease_id}
        lock = adapter._typing_lease_lock("123")
        await lock.acquire()
        try:
            shutdown_task = asyncio.create_task(adapter.cancel_background_tasks())
            await asyncio.sleep(0)
            assert not adapter._typing_lease_allows("123", metadata)
            assert not shutdown_task.done()
        finally:
            lock.release()
        await shutdown_task

    @pytest.mark.asyncio
    async def test_outer_finalization_fences_successor_queued_turn_lease(self):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = True
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        )
        event = MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m1",
        )
        session_key = build_session_key(source)
        successor = {}
        successor_allowed_at_send = []

        async def handler(_event):
            successor["id"] = adapter._begin_typing_lease(
                source.chat_id, session_key
            )
            return "queued final"

        async def send(chat_id, content, reply_to=None, metadata=None):
            successor_allowed_at_send.append(
                adapter._typing_lease_allows(
                    chat_id,
                    {"_hermes_typing_lease_id": successor["id"]},
                )
            )
            return SendResult(success=True, message_id="m1")

        adapter._message_handler = handler
        adapter.send = send

        await adapter._process_message_background(event, session_key)

        assert successor_allowed_at_send == [False]
        assert not adapter._typing_lease_allows(
            source.chat_id,
            {"_hermes_typing_lease_id": successor["id"]},
        )

    @pytest.mark.asyncio
    async def test_stale_task_cleanup_preserves_replacement_turn_lease(self):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = True
        adapter.config.typing_indicator = False
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        )
        event = MessageEvent(
            text="stale",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m1",
        )
        session_key = build_session_key(source)
        stale_guard = asyncio.Event()
        adapter._active_sessions[session_key] = stale_guard
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_event):
            started.set()
            await release.wait()
            return None

        adapter._message_handler = handler
        stale_task = asyncio.create_task(
            adapter._process_message_background(event, session_key)
        )
        adapter._session_tasks[session_key] = stale_task
        await started.wait()

        replacement_guard = asyncio.Event()
        replacement_task = asyncio.create_task(asyncio.sleep(0))
        adapter._active_sessions[session_key] = replacement_guard
        adapter._session_tasks[session_key] = replacement_task
        replacement_lease = adapter._begin_typing_lease(
            source.chat_id, session_key
        )
        replacement_metadata = {
            "_hermes_typing_lease_id": replacement_lease
        }
        assert adapter._typing_lease_allows(
            source.chat_id, replacement_metadata
        )

        release.set()
        await stale_task
        await replacement_task

        assert adapter._active_sessions.get(session_key) is replacement_guard
        assert adapter._typing_lease_allows(
            source.chat_id, replacement_metadata
        )
        adapter._revoke_typing_lease(replacement_lease)

    @pytest.mark.asyncio
    async def test_keep_typing_refreshes_successor_generation(self, monkeypatch):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = True
        session_key = "session-1"
        first_lease = adapter._begin_typing_lease("123", session_key)
        first_metadata = adapter._typing_metadata_for_session(session_key, {})
        successor_seen = asyncio.Event()
        calls = []

        async def record_typing(chat_id, metadata=None):
            lease_id = (metadata or {}).get("_hermes_typing_lease_id")
            calls.append(lease_id)
            if lease_id != first_lease:
                successor_seen.set()

        monkeypatch.setattr(adapter, "send_typing", record_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                "123",
                interval=0.05,
                metadata=first_metadata,
                stop_event=stop_event,
                session_key=session_key,
            )
        )
        while not calls:
            await asyncio.sleep(0)

        successor_lease = adapter._begin_typing_lease("123", session_key)
        await asyncio.wait_for(successor_seen.wait(), timeout=1.0)
        stop_event.set()
        await task

        assert calls[0] == first_lease
        assert successor_lease in calls[1:]

    @pytest.mark.asyncio
    async def test_fenced_typing_send_is_not_cancelled_by_heartbeat_timeout(self, monkeypatch):
        adapter = _StubAdapter()
        adapter.typing_send_requires_final_fence = True
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_send_typing(chat_id, metadata=None):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(adapter, "send_typing", blocked_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing("123", interval=0.5, stop_event=stop_event)
        )
        await started.wait()
        await asyncio.sleep(0.4)
        assert not cancelled.is_set()

        release.set()
        stop_event.set()
        await task

    @pytest.mark.asyncio
    async def test_slow_send_typing_does_not_block_cadence(self, monkeypatch):
        """A send_typing that hangs longer than the per-tick budget must be
        abandoned so the next scheduled tick can fire a fresh call."""
        adapter = _StubAdapter()
        call_events = []

        async def slow_send_typing(chat_id, metadata=None):
            # Simulate a stuck HTTP round-trip. If _keep_typing awaits this
            # unconditionally, the loop stalls for the full duration.
            call_events.append("start")
            try:
                await asyncio.sleep(10)
            finally:
                call_events.append("finish-or-cancel")

        monkeypatch.setattr(adapter, "send_typing", slow_send_typing)
        # Avoid stop_typing side-effects in the finally block.
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop_event = asyncio.Event()
        # Start the typing loop, let it run ~3s (should fire 2 ticks) then stop.
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="123",
                interval=1.0,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(3.0)
        stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            pytest.fail(
                "_keep_typing did not exit within 2s of stop_event.set() — "
                "it is blocked on a slow send_typing call"
            )

        # With per-tick timeout, we should see MULTIPLE send_typing starts
        # despite each being slow (abandoned via TimeoutError).  Without the
        # fix there would be exactly 1 start (the one still stuck).
        starts = [e for e in call_events if e == "start"]
        assert len(starts) >= 2, (
            f"expected at least 2 send_typing ticks across 3s of slow "
            f"operation, got {len(starts)} — refresh cadence is stalled "
            f"on a slow send_typing"
        )

    @pytest.mark.asyncio
    async def test_fast_send_typing_still_gets_awaited(self, monkeypatch):
        """When send_typing is fast (normal case), it must still complete
        normally — the timeout is only an upper bound, not a cap on
        successful calls."""
        adapter = _StubAdapter()
        completed = []

        async def fast_send_typing(chat_id, metadata=None):
            await asyncio.sleep(0.01)  # well under the timeout
            completed.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", fast_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="456",
                interval=0.5,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(1.2)  # ~3 ticks
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert len(completed) >= 2, (
            f"expected multiple completed send_typing calls, got "
            f"{len(completed)}"
        )
        assert all(c == "456" for c in completed)

    @pytest.mark.asyncio
    async def test_send_typing_exception_does_not_kill_loop(self, monkeypatch):
        """A send_typing that raises (e.g. transient HTTP 500) must be
        caught so the loop continues refreshing on schedule."""
        adapter = _StubAdapter()
        tick_count = {"n": 0}

        async def flaky_send_typing(chat_id, metadata=None):
            tick_count["n"] += 1
            if tick_count["n"] == 1:
                raise RuntimeError("transient upstream error")
            # Subsequent calls succeed.

        monkeypatch.setattr(adapter, "send_typing", flaky_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="789",
                interval=0.3,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(1.0)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert tick_count["n"] >= 2, (
            f"loop exited after first send_typing exception; expected it to "
            f"keep ticking (got {tick_count['n']} ticks)"
        )

    @pytest.mark.asyncio
    async def test_paused_chat_skips_send_typing(self, monkeypatch):
        """When a chat is in _typing_paused (e.g. awaiting approval), the
        loop must not call send_typing at all. Regression guard — existing
        behavior, preserved through the timeout change."""
        adapter = _StubAdapter()
        calls = []

        async def recording_send_typing(chat_id, metadata=None):
            calls.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))
        adapter._typing_paused.add("paused-chat")

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="paused-chat",
                interval=0.3,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(1.0)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert calls == [], (
            f"send_typing was called on a paused chat: {calls}"
        )

    @pytest.mark.asyncio
    async def test_stop_typing_refresh_blocks_late_cancel_tick(self, monkeypatch):
        """Final cleanup must not let a cancelled refresh loop send typing again."""
        adapter = _StubAdapter()
        late_sends = []
        stop_calls = []

        async def send_typing(chat_id, metadata=None):
            late_sends.append(chat_id)

        async def stop_typing(chat_id):
            stop_calls.append((chat_id, chat_id in adapter._typing_paused))

        monkeypatch.setattr(adapter, "send_typing", send_typing)
        monkeypatch.setattr(adapter, "stop_typing", stop_typing)

        async def late_refresh_after_cancel():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                if "discord-chat" not in adapter._typing_paused:
                    await adapter.send_typing("discord-chat")
                raise

        task = asyncio.create_task(late_refresh_after_cancel())
        await asyncio.sleep(0)

        await adapter._stop_typing_refresh("discord-chat", task, timeout=1.0)

        assert late_sends == []
        assert stop_calls == [
            ("discord-chat", True),
            ("discord-chat", True),
        ]
        assert "discord-chat" not in adapter._typing_paused
