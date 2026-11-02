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
import contextlib
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    Platform,
    PlatformConfig,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
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


class _ForwardingStubAdapter(_StubAdapter):
    """Match adapters such as LINE that forward optional lifecycle kwargs."""

    def __init__(self):
        super().__init__()
        self.forwarded_typing_kwargs = []
        self.typing_exited = asyncio.Event()

    async def _keep_typing(self, chat_id, *args, **kwargs):
        self.forwarded_typing_kwargs.append(kwargs)
        try:
            return await super()._keep_typing(chat_id, *args, **kwargs)
        finally:
            self.typing_exited.set()


class TestKeepTypingTimeoutPerTick:
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
        except TimeoutError:
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


class TestPreDeliveryTypingSuspend:
    """Regression guard for #117300: once delivery begins, the refresh loop
    must be suspended so a stalled final send (platform accepted it, HTTP ack
    lost) cannot keep ``send_typing`` firing while the agent is idle.

    The fix reuses the existing ``_typing_paused`` mechanism: the delivery path
    pauses the chat before the first send attempt, and ``_stop_typing_refresh``'s
    finally discards it so it never leaks into the next turn.
    """

    @pytest.mark.asyncio
    async def test_paused_before_delivery_stall_suppresses_refresh(self, monkeypatch):
        """Pause the chat the way the delivery path does; a refresh loop running
        during the stall must never call send_typing (0 ticks)."""
        adapter = _StubAdapter()
        ticks = []

        async def recording_send_typing(chat_id, metadata=None):
            ticks.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        # The delivery path pauses typing before the first send attempt.
        adapter.pause_typing_for_chat("123")

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="123",
                interval=0.2,
                stop_event=stop_event,
            )
        )
        # Simulate the 9s stall from the issue's repro: the refresh loop is
        # still live but the chat is paused, so no tick should reach send_typing.
        await asyncio.sleep(0.8)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert ticks == [], (
            f"typing kept refreshing across a pre-delivery pause: {ticks}"
        )
        # Pausing the delivery path must not leave the chat permanently paused.
        await adapter._stop_typing_refresh("123", None, stop_attempts=1)
        assert "123" not in adapter._typing_paused

    @pytest.mark.asyncio
    async def test_unpaused_stall_still_ticks(self, monkeypatch):
        """Control case: without the pre-delivery pause, the refresh loop keeps
        ticking (the buggy pre-fix behavior), proving the test discriminates."""
        adapter = _StubAdapter()
        ticks = []

        async def recording_send_typing(chat_id, metadata=None):
            ticks.append(chat_id)

        monkeypatch.setattr(adapter, "send_typing", recording_send_typing)
        adapter.stop_typing = MagicMock(return_value=asyncio.sleep(0))

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing(
                chat_id="456",
                interval=0.2,
                stop_event=stop_event,
            )
        )
        await asyncio.sleep(0.8)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert len(ticks) >= 2, (
            f"expected the unpaused refresh loop to keep ticking, "
            f"got {len(ticks)}"
        )


@pytest.mark.asyncio
async def test_process_message_pauses_typing_before_a_stalled_final_send():
    """Production entry point (#117300): ``_process_message_background`` must pause the chat
    BEFORE awaiting the final send. With the send parked forever the turn's ``finally`` never
    runs, so only a pre-delivery pause keeps ``_keep_typing`` from refreshing an idle agent."""
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    adapter = _StubAdapter()
    send_started = asyncio.Event()
    never = asyncio.Event()

    async def _stalled_final_send(*args, **kwargs):
        send_started.set()
        await never.wait()  # platform accepted the message; the HTTP ack never comes back

    async def _handler(evt):
        return "a long final answer the user can already read"

    adapter.set_message_handler(_handler)
    adapter._send_final_text = _stalled_final_send
    adapter._start_typing_refresh = MagicMock(return_value=None)  # loop shape irrelevant here
    event = MessageEvent(
        text="hi", message_id="msg-1", message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="77", user_id="u-1"),
    )
    turn = asyncio.create_task(adapter._process_message_background(event, "agent:main:telegram:private:77"))
    try:
        await asyncio.wait_for(send_started.wait(), timeout=5)
        assert "77" in adapter._typing_paused, "typing not paused while the final send is stalled"
    finally:
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn


class TestTypingTurnLease:
    """A turn's typing refresh must not outlive the turn (per-turn completion lease)."""

    @pytest.mark.asyncio
    async def test_base_refresh_swallowing_cancel_stops_after_turn(self, monkeypatch):
        """End-to-end with the real base ``_keep_typing``: a send_typing that swallows
        the cleanup cancel (cancellation-resistant client) must not keep refreshing
        typing once the turn's cleanup has unpaused the chat."""
        adapter = _StubAdapter()
        first_tick = asyncio.Event()
        calls = []

        async def send_typing(chat_id, metadata=None):
            calls.append(chat_id)
            first_tick.set()
            if len(calls) == 1:
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    # Swallow the cleanup cancel and keep the task alive briefly.
                    await asyncio.sleep(0.05)

        async def handler(event):
            await first_tick.wait()

        monkeypatch.setattr(adapter, "send_typing", send_typing)
        adapter._message_handler = handler
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="lease-chat", chat_type="dm", user_id="u1",
        )
        event = MessageEvent(
            text="hello", message_type=MessageType.TEXT, source=source, message_id="m-lease",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        typing_tasks = []
        real_start = adapter._start_typing_refresh

        def _capture_start(*args, **kwargs):
            task = real_start(*args, **kwargs)
            typing_tasks.append(task)
            return task

        monkeypatch.setattr(adapter, "_start_typing_refresh", _capture_start)
        try:
            await asyncio.wait_for(
                adapter._process_message_background(event, session_key), timeout=3.0)
            calls_at_cleanup = len(calls)
            await asyncio.sleep(2.5)  # > one refresh interval after cleanup
            assert len(calls) == calls_at_cleanup, (
                f"typing refreshed {len(calls) - calls_at_cleanup}x after the turn finished")
            assert typing_tasks and typing_tasks[0].done()
        finally:
            for task in typing_tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    async def test_completed_turn_signals_cancellation_resistant_typing_loop(
        self, monkeypatch
    ):
        adapter = _StubAdapter()
        started = asyncio.Event()
        stopped = asyncio.Event()
        observed_stop_events = []

        async def handler(event):
            await asyncio.wait_for(started.wait(), timeout=1.0)

        async def cancellation_resistant_keep_typing(
            chat_id, metadata=None, stop_event=None, completion_event=None
        ):
            observed_stop_events.append(completion_event)
            assert completion_event is not None
            started.set()
            while not completion_event.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue
            stopped.set()

        monkeypatch.setattr(adapter, "_keep_typing", cancellation_resistant_keep_typing)
        adapter._message_handler = handler
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="completion-chat",
            chat_type="dm",
            user_id="u1",
        )
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-completion",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()

        try:
            await asyncio.wait_for(
                adapter._process_message_background(event, session_key),
                timeout=2.0,
            )
            assert observed_stop_events[0].is_set()
            await asyncio.wait_for(stopped.wait(), timeout=1.0)
        finally:
            if observed_stop_events:
                observed_stop_events[0].set()
                await asyncio.wait_for(stopped.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_forwarding_override_receives_interrupt_and_completion_events(
        self, monkeypatch
    ):
        adapter = _ForwardingStubAdapter()
        first_tick = asyncio.Event()
        release_handler = asyncio.Event()
        calls = []

        async def send_typing(chat_id, metadata=None):
            calls.append(chat_id)
            first_tick.set()

        async def handler(event):
            await release_handler.wait()

        monkeypatch.setattr(adapter, "send_typing", send_typing)
        adapter._message_handler = handler
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="forwarding-chat",
            chat_type="dm",
            user_id="u1",
        )
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-forward",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        task = asyncio.create_task(
            adapter._process_message_background(event, session_key)
        )

        await asyncio.wait_for(first_tick.wait(), timeout=1.0)
        forwarded = adapter.forwarded_typing_kwargs[0]
        assert forwarded["stop_event"] is adapter._active_sessions[session_key]

        await adapter.interrupt_session_activity(session_key, source.chat_id)
        calls_after_interrupt = len(calls)
        await asyncio.wait_for(adapter.typing_exited.wait(), timeout=1.0)
        assert len(calls) == calls_after_interrupt
        assert forwarded["stop_event"].is_set()
        assert not forwarded["completion_event"].is_set()

        release_handler.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert forwarded["completion_event"].is_set()

    @pytest.mark.asyncio
    async def test_explicit_stop_event_override_keeps_interrupt_contract(
        self, monkeypatch
    ):
        adapter = _StubAdapter()
        started = asyncio.Event()
        exited = asyncio.Event()
        release_handler = asyncio.Event()
        observed_stop_events = []

        async def explicit_stop_only_keep_typing(
            chat_id, metadata=None, stop_event=None
        ):
            observed_stop_events.append(stop_event)
            assert stop_event is not None
            started.set()
            await stop_event.wait()
            exited.set()

        async def handler(event):
            await release_handler.wait()

        monkeypatch.setattr(adapter, "_keep_typing", explicit_stop_only_keep_typing)
        adapter._message_handler = handler
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="legacy-override-chat",
            chat_type="dm",
            user_id="u1",
        )
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-legacy",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        task = asyncio.create_task(
            adapter._process_message_background(event, session_key)
        )

        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert observed_stop_events == [adapter._active_sessions[session_key]]
        await adapter.interrupt_session_activity(session_key, source.chat_id)
        await asyncio.wait_for(exited.wait(), timeout=1.0)

        release_handler.set()
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_opaque_legacy_override_fails_closed_and_releases_session(
        self, monkeypatch
    ):
        adapter = _StubAdapter()
        started = asyncio.Event()
        exited = asyncio.Event()
        release_handler = asyncio.Event()
        observed_stop_events = []

        class OpaqueLegacyKeepTyping:
            @property
            def __signature__(self):
                raise ValueError("opaque callable")

            async def __call__(self, chat_id, metadata=None, stop_event=None):
                observed_stop_events.append(stop_event)
                assert stop_event is not None
                started.set()
                await stop_event.wait()
                exited.set()

        async def handler(event):
            await release_handler.wait()

        monkeypatch.setattr(adapter, "_keep_typing", OpaqueLegacyKeepTyping())
        adapter._message_handler = handler
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="opaque-override-chat",
            chat_type="dm",
            user_id="u1",
        )
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-opaque",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        task = asyncio.create_task(
            adapter._process_message_background(event, session_key)
        )
        adapter._session_tasks[session_key] = task

        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert observed_stop_events == [adapter._active_sessions[session_key]]
        await adapter.interrupt_session_activity(session_key, source.chat_id)
        await asyncio.wait_for(exited.wait(), timeout=1.0)

        release_handler.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert session_key not in adapter._active_sessions
        assert session_key not in adapter._session_tasks

    @pytest.mark.asyncio
    async def test_pending_handoff_gets_fresh_typing_completion_event(
        self, monkeypatch
    ):
        adapter = _ForwardingStubAdapter()
        first_tick = asyncio.Event()
        second_tick = asyncio.Event()
        release_second = asyncio.Event()
        send_count = 0

        async def send_typing(chat_id, metadata=None):
            nonlocal send_count
            send_count += 1
            (first_tick if send_count == 1 else second_tick).set()

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="handoff-chat",
            chat_type="dm",
            user_id="u1",
        )
        first = MessageEvent(
            text="first",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-first",
        )
        second = MessageEvent(
            text="second",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m-second",
        )
        session_key = build_session_key(source)

        async def handler(event):
            if event.text == "first":
                await first_tick.wait()
                adapter._pending_messages[session_key] = second
                return
            await second_tick.wait()
            await release_second.wait()
            return

        monkeypatch.setattr(adapter, "send_typing", send_typing)
        adapter._message_handler = handler
        adapter._active_sessions[session_key] = asyncio.Event()
        first_task = asyncio.create_task(
            adapter._process_message_background(first, session_key)
        )
        adapter._session_tasks[session_key] = first_task

        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(second_tick.wait(), timeout=1.0)
        first_events, second_events = adapter.forwarded_typing_kwargs
        assert first_events["completion_event"].is_set()
        assert not second_events["completion_event"].is_set()
        assert (
            first_events["completion_event"]
            is not second_events["completion_event"]
        )
        assert first_events["stop_event"] is second_events["stop_event"]

        release_second.set()
        await asyncio.wait_for(adapter._session_tasks[session_key], timeout=1.0)
        assert second_events["completion_event"].is_set()
