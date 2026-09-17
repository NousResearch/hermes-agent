"""Notification ownership through actual turn wiring and Telegram transport methods.

Fixture adapted from the isolated incident reproducer; only Bot API I/O is fake.
"""

import asyncio
import queue
from types import SimpleNamespace as NS
import pytest
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.session import SessionSource
from gateway.session_context import (
    set_session_vars,
    clear_session_vars,
    get_session_env,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


class Wire:
    def __init__(self):
        self.messages = {}
        self.events = []
        self.next_id = 90000
        self.accepted = asyncio.Event()
        self.release = asyncio.Event()
        self.block_once = False

    async def chunk(
        self, chat, text, i, reply, metadata, thread, fallback, errors, **kw
    ):
        self.next_id += 1
        mid = str(self.next_id)
        self.messages[mid] = {"thread": str(thread), "text": text}
        self.events.append(["send", mid, str(thread), text])
        if self.block_once:
            self.block_once = False
            self.accepted.set()
            await self.release.wait()  # server accepted; receipt delayed
        return NS(message_id=int(mid)), fallback

    async def edit(self, chat, **kw):
        mid = str(kw["message_id"])
        self.events.append([
            "edit",
            mid,
            get_session_env("HERMES_SESSION_THREAD_ID"),
            kw["text"],
        ])
        if mid not in self.messages:
            raise ValueError("Message to edit not found")
        self.messages[mid]["text"] = kw["text"]

    async def edit_message_text(self, chat_id, **kw):
        return await self.edit(chat_id, **kw)

    async def delete_message(self, **kw):
        mid = str(kw["message_id"])
        self.events.append(["delete", mid])
        if mid not in self.messages:
            raise ValueError("Message to delete not found")
        del self.messages[mid]


async def noop(*a, **kw):
    pass


def setup():
    wire = Wire()
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = wire
    a._should_attempt_rich = lambda *a, **kw: False
    a._rich_eligible = lambda *a, **kw: False
    a._send_chunk_with_retries = wire.chunk
    a._edit_message_text_with_cooldown = wire.edit
    a._retrigger_typing = noop
    a.send_typing = noop
    g = GatewayRunner.__new__(GatewayRunner)
    g._adapter_for_source = lambda source: a
    g.hooks = NS(loaded_hooks=[])
    return wire, a, g


def turn(g, topic):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123456",
        chat_type="dm",
        thread_id=topic,
        user_id="123456",
    )
    ctx = TurnContext(
        source=source,
        session_key="agent:main:telegram:dm:123456:" + topic,
        session_id="test-" + topic,
        run_generation=20,
        _run_still_current=lambda: True,
        _cleanup_progress=True,
        progress_queue=queue.Queue(),
        progress_mode="all",
        tool_progress_enabled=True,
    )
    tr = TurnRunner(g, ctx)
    g._run_agent_bind_turn_wiring(ctx, tr, source, "123", False)
    ctx.progress_callback = tr.progress_callback
    return ctx, tr


async def status(ctx, text):
    tok = set_session_vars(
        platform="telegram",
        chat_id="123456",
        thread_id=ctx.source.thread_id,
        session_key=ctx.session_key,
    )
    try:
        before = len(ctx._cleanup_msg_ids)
        ctx._status_callback_sync("lifecycle", text)
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(ctx._cleanup_msg_ids) > before:
                return ctx._cleanup_msg_ids[-1]
        raise AssertionError("callback did not complete")
    finally:
        clear_session_vars(tok)


@pytest.mark.asyncio
async def test_status_ownership_and_late_cleanup():
    w, a, g = setup()
    c1, t1 = turn(g, "101")
    c2, t2 = turn(g, "202")
    c3, t3 = turn(g, "101")  # even reused session/generation cannot reuse a turn token
    mids = await asyncio.gather(*(status(c, "Working") for c in (c1, c2, c3)))
    assert len(set(mids)) == 3
    assert [w.messages[mid]["thread"] for mid in mids] == ["101", "202", "101"]
    assert await status(c1, "Still working") == mids[0]
    await a.delete_message(c1.source.chat_id, mids[0])
    assert mids[0] not in a._status_message_ids.values()
    assert await status(c1, "After deletion") != mids[0]
    for c in (c1, c2, c3):
        g._run_agent_schedule_bubble_cleanup({"final_response": "done"}, a, c)
        a.pop_post_delivery_callback(c.session_key, generation=c.run_generation)()
        await asyncio.sleep(0.01)
    assert not w.messages
    assert not a._status_message_ids

    # First receipt arrives after callback invocation; a queued old callback must not send.
    c, t = turn(g, "303")
    w.block_once = True
    pending = asyncio.create_task(
        t._status_delivery.send(
            a, c.source.chat_id, "lifecycle", "late", c._status_thread_metadata
        )
    )
    await asyncio.wait_for(w.accepted.wait(), 5)
    queued = asyncio.create_task(
        t._status_delivery.send(
            a, c.source.chat_id, "lifecycle", "stale", c._status_thread_metadata
        )
    )
    g._run_agent_schedule_bubble_cleanup({"final_response": "done"}, a, c)
    a.pop_post_delivery_callback(c.session_key, generation=c.run_generation)()
    await asyncio.sleep(0.01)
    _, replacement, _ = setup()
    g._adapter_for_source = lambda source: replacement
    w.release.set()
    await pending
    await queued
    await asyncio.gather(*t._status_delivery.tasks)
    assert not w.messages
    assert not a._status_message_ids
    assert not replacement._status_message_ids


@pytest.mark.asyncio
async def test_adapter_status_cache_serialization_and_invalidation():
    w, a, g = setup()
    w.block_once = True
    first = asyncio.create_task(
        a.send_or_update_status(
            "123456", "owner", "first", metadata={"thread_id": "101"}
        )
    )
    await asyncio.wait_for(w.accepted.wait(), 5)
    second = asyncio.create_task(
        a.send_or_update_status(123456, "owner", "second", metadata={"thread_id": 101})
    )
    w.release.set()
    one, two = await asyncio.gather(first, second)
    assert one.message_id == two.message_id
    assert len([event for event in w.events if event[0] == "send"]) == 1
    other = await a.send_or_update_status(
        "123456", "owner", "other", metadata={"thread_id": "202"}
    )
    assert other.message_id != one.message_id
    # Already-deleted messages invalidate the local cache too.
    del w.messages[str(one.message_id)]
    await a.delete_message("123456", one.message_id)
    assert str(one.message_id) not in a._status_message_ids.values()
    fresh = await a.send_or_update_status(
        "123456", "owner", "fresh", metadata={"thread_id": "101"}
    )
    assert fresh.message_id != one.message_id
    # Late successful edits must not resurrect an entry deleted while awaiting I/O.
    entered, release = asyncio.Event(), asyncio.Event()
    original_edit = a.edit_message

    async def delayed_edit(*args, **kwargs):
        result = await original_edit(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    a.edit_message = delayed_edit
    pending = asyncio.create_task(
        a.send_or_update_status(
            "123456", "owner", "editing", metadata={"thread_id": "101"}
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    await a.delete_message("123456", fresh.message_id)
    release.set()
    await pending
    assert str(fresh.message_id) not in a._status_message_ids.values()
    a._status_message_ids = {("123456", f"old-{i}", "101"): str(i) for i in range(2000)}
    latest = await a.send_or_update_status(
        "123456", "new-owner", "bounded", metadata={"thread_id": "101"}
    )
    assert len(a._status_message_ids) <= 2000
    assert latest.message_id in a._status_message_ids.values()
