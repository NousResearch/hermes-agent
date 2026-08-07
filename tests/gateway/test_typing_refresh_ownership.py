"""Tests for BasePlatformAdapter._keep_typing per-chat ownership.

The bug: ``_keep_typing`` trusts that whoever started it will cancel it.
When that cancellation is lost (e.g. a shutdown race between the turn
finishing and the refresh task being awaited), the orphaned loop keeps
refreshing the platform typing indicator on its 2-second cadence with
nothing to stop it. Observed in production: a leaked refresh task hammered
a Matrix homeserver's ``PUT /typing`` every 2 seconds for over three hours
after its turn had ended normally, pinning clients' "working" indicator
the whole time.

The fix encodes the invariant instead of patching one leak path: at most
one live refresh loop per chat. A loop claims its chat in
``_typing_refresh_owners`` on start and re-checks the claim every tick, so
any orphan self-terminates within one tick of being stopped or superseded
— whatever path leaked it. Only the current owner may clear platform
typing state, so a dying stale loop can't stomp a newer turn's indicator.
"""

import asyncio

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    Platform,
    PlatformConfig,
    SendResult,
)


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.stop_typing_calls = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def stop_typing(self, chat_id):
        self.stop_typing_calls.append(chat_id)


class TestTypingRefreshOwnership:
    @pytest.mark.asyncio
    async def test_superseded_loop_self_terminates(self):
        """A leaked refresh loop must exit on its own — without being
        cancelled — as soon as a newer turn's loop claims the same chat."""
        adapter = _StubAdapter()

        leaked = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        assert not leaked.done()

        newer = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.2)

        assert leaked.done(), "orphaned loop should exit once superseded"
        assert not newer.done(), "the newer turn's loop must keep running"

        newer.cancel()
        await asyncio.gather(newer, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_stop_refresh_kills_loop_even_when_cancellation_is_lost(self):
        """_stop_typing_refresh must revoke the claim so a loop whose task
        handle was lost (cancellation never delivered) still self-terminates
        on its next tick instead of typing forever."""
        adapter = _StubAdapter()

        leaked = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        assert not leaked.done()

        # No task handle passed — simulates the caller having lost track of
        # the task, the exact condition of the production leak.
        await adapter._stop_typing_refresh("123", None)
        await asyncio.sleep(0.2)

        assert leaked.done(), "loop must exit after its claim is revoked"

    @pytest.mark.asyncio
    async def test_stale_loop_does_not_clear_newer_turns_typing(self):
        """A superseded loop's cleanup must not call stop_typing — that
        would kill the typing indicator of the turn that superseded it."""
        adapter = _StubAdapter()

        stale = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        newer = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.2)

        assert stale.done()
        assert adapter.stop_typing_calls == [], (
            "superseded loop cleared platform typing state it no longer owns"
        )

        newer.cancel()
        await asyncio.gather(newer, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_stopping_stale_task_leaves_live_claim_alone(self):
        """Stopping an old task must not revoke a claim now held by a
        different live loop (a newer turn already running)."""
        adapter = _StubAdapter()

        stale = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.12)
        newer = asyncio.create_task(adapter._keep_typing("123", interval=0.05))
        await asyncio.sleep(0.2)
        assert stale.done()

        await adapter._stop_typing_refresh("123", stale)
        assert adapter._typing_refresh_owners.get("123") is newer

        newer.cancel()
        await asyncio.gather(newer, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_normal_stop_releases_claim(self):
        """The normal stop_event path must leave no claim behind."""
        adapter = _StubAdapter()
        stop_event = asyncio.Event()

        task = asyncio.create_task(
            adapter._keep_typing("123", interval=0.05, stop_event=stop_event)
        )
        await asyncio.sleep(0.12)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

        assert "123" not in adapter._typing_refresh_owners
