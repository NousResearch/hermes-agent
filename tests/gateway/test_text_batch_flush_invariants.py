"""Invariants for BasePlatformAdapter's inbound text-batch flush.

Every adapter that batches inbound text used to carry its own ``_flush_text_batch``; only Discord
shielded the dispatch (#12444) and only WeCom/Weixin checked task identity before the pop. Both
fixes now live in the base flush; these tests pin them against the base class and sweep every
adapter that still overrides the flush (Telegram keeps a hold-queue variant) for the same contract.
"""

import asyncio
from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self._text_batch_delay_seconds = 0.0
        self._text_batch_split_delay_seconds = 0.0
        self.dispatched: list = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *a: Any, **k: Any) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}

    async def handle_message(self, event: MessageEvent) -> None:
        self.entered.set()
        await self.release.wait()  # a real agent turn awaits here
        self.dispatched.append(event)


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT,
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c", chat_type="dm"))


@pytest.mark.asyncio
async def test_cancel_mid_dispatch_does_not_abort_the_turn():
    """A cancel landing while handle_message is in flight (a late chunk re-arming the timer) must not
    abort the dispatch: the batch is delivered exactly once and the task exits clean."""
    adapter = _Adapter()
    adapter._pending_text_batches["k"] = _event("hello")
    task = asyncio.create_task(adapter._flush_text_batch("k"))
    adapter._pending_text_batch_tasks["k"] = task
    await adapter.entered.wait()
    task.cancel()
    adapter.release.set()
    await task  # no CancelledError escapes
    assert [e.text for e in adapter.dispatched] == ["hello"]
    assert adapter._pending_text_batch_tasks == {}


@pytest.mark.asyncio
async def test_superseded_flush_leaves_batch_for_its_successor():
    """When a newer flush task owns the key by the time the old one wakes, the old one must neither
    pop nor dispatch — otherwise the successor finds an empty batch and the message is lost."""
    adapter = _Adapter()
    adapter.release.set()
    event = _event("hello")
    adapter._pending_text_batches["k"] = event
    stale = asyncio.create_task(adapter._flush_text_batch("k"))
    adapter._pending_text_batch_tasks["k"] = stale
    successor = asyncio.create_task(asyncio.sleep(60))
    adapter._pending_text_batch_tasks["k"] = successor  # re-armed before `stale` ran
    await stale
    successor.cancel()
    assert adapter.dispatched == []
    assert adapter._pending_text_batches.get("k") is event
    assert adapter._pending_text_batch_tasks.get("k") is successor


@pytest.mark.asyncio
async def test_enqueue_then_flush_delivers_merged_text_once():
    adapter = _Adapter()
    adapter.release.set()
    adapter._enqueue_text_event(_event("part one"))
    adapter._enqueue_text_event(_event("part two"))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    assert [e.text for e in adapter.dispatched] == ["part one\npart two"]
    assert adapter._pending_text_batches == {} and adapter._pending_text_batch_tasks == {}
