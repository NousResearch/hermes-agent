"""Weixin bursts must reach the gateway with their captions and ordered images."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from gateway.platforms import weixin
from gateway.platforms.weixin import ITEM_IMAGE, ITEM_TEXT, WeixinAdapter


def _adapter(monkeypatch):
    adapter = WeixinAdapter(PlatformConfig(
        enabled=True, token="test-token", extra={
            "account_id": "bot", "dm_policy": "allowlist", "allow_from": ["peer", "alice", "bob"],
        },
    ))
    adapter._poll_session = SimpleNamespace(closed=True)
    adapter._token = ""  # no typing-ticket network requests
    # Flush explicitly: timing/load must not decide which messages belong to the test burst.
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 60
    monkeypatch.setattr(adapter, "_text_batch_delay_for", lambda _: 60)
    return adapter


def _message(number, kind, peer="peer"):
    item = {"type": ITEM_TEXT, "text_item": {"text": f"caption-{number}"}}
    if kind == "photo":
        item = {"type": ITEM_IMAGE, "image_item": {"test_path": f"image-{number}.jpg"}}
    return {"from_user_id": peer, "message_id": str(number), "item_list": [item]}


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "steer"])
@pytest.mark.parametrize("deadline", [False, True], ids=["quiet-flush", "continuous-input-cap"])
@pytest.mark.parametrize("kinds", [
    ("text", "photo", "text"),
    ("photo", "text"),
    ("photo", "text", "photo", "text"),
])
async def test_burst_keeps_captions_and_download_order(monkeypatch, kinds, busy, deadline):
    adapter = _adapter(monkeypatch)
    downloaded, release = asyncio.Event(), asyncio.Event()
    delivered, steered = [], []

    async def collect(item, paths, types):
        if item.get("type") == ITEM_IMAGE:
            if not downloaded.is_set():
                downloaded.set()
                await release.wait()
            paths.append(item["image_item"]["test_path"])
            types.append("image/jpeg")

    monkeypatch.setattr(adapter, "_collect_media", collect)
    adapter.handle_message = AsyncMock(side_effect=lambda event: delivered.append(event))
    if busy:
        # Exercise the real busy router: a PHOTO must queue as one unit, never steer its caption.
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._prepare_busy_steer_text = AsyncMock(side_effect=lambda e: e.text)
        runner._pending_event_audio_paths = lambda e: []
        runner._try_agent_verb = lambda agent, verb, text, *a, **kw: steered.append(text) or True
        agent = SimpleNamespace(steer=lambda text: True)

        async def route(event):
            outcome = await runner._resolve_busy_steer_or_redirect(event, "test-session", "steer", agent)
            if not outcome.steered:
                delivered.append(event)

        adapter.handle_message = AsyncMock(side_effect=route)

    tasks = []
    try:
        for i, kind in enumerate(kinds):
            tasks.append(asyncio.create_task(adapter._process_message(_message(i, kind))))
            if kind == "photo" and not downloaded.is_set():
                await asyncio.wait_for(downloaded.wait(), 5)
        release.set()
        await asyncio.gather(*tasks)
        for key in list(adapter._pending_text_batches):
            if deadline:
                # Advance only the batching clock, not asyncio's scheduling clock. Continuous
                # chunks must eventually dispatch without waiting for another quiet period.
                batch = adapter._pending_text_batches[key]
                started = getattr(batch, "_weixin_batch_started", 0)
                monkeypatch.setattr(weixin, "time", SimpleNamespace(monotonic=lambda: started + 10))
                monkeypatch.delattr(adapter, "_text_batch_delay_for")
                adapter._pending_text_batch_tasks[key].cancel()
                task = asyncio.create_task(adapter._flush_text_batch(key))
                adapter._pending_text_batch_tasks[key] = task
                await asyncio.wait_for(asyncio.shield(task), 5)
            else:
                await adapter._flush_text_batch_now(key)

        assert steered == []
        assert len(delivered) == 1
        batch = delivered[0]
        assert batch.message_type == MessageType.PHOTO
        assert batch.media_urls == [f"image-{i}.jpg" for i, k in enumerate(kinds) if k == "photo"]
        assert batch.media_types == ["image/jpeg"] * kinds.count("photo")
        captions = [f"caption-{i}" for i, k in enumerate(kinds) if k == "text"]
        assert all(batch.text.count(caption) == 1 for caption in captions)
        assert [batch.text.index(caption) for caption in captions] == sorted(batch.text.index(caption) for caption in captions)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/approve"])
@pytest.mark.parametrize("download", ["success", "empty", "cancelled"])
async def test_commands_bypass_collection_and_peers_remain_separate(monkeypatch, command, download):
    adapter = _adapter(monkeypatch)
    delivered = []
    adapter.handle_message = AsyncMock(side_effect=lambda event: delivered.append(event))
    await adapter._process_message(_message(1, "text", "alice"))
    await adapter._process_message(_message(2, "text", "bob"))
    started, release = asyncio.Event(), asyncio.Event()

    async def collect(item, paths, types):
        if item.get("type") == ITEM_IMAGE:
            started.set()
            await release.wait()
            if download == "success":
                paths.append("alice.jpg")
                types.append("image/jpeg")

    monkeypatch.setattr(adapter, "_collect_media", collect)
    photo = asyncio.create_task(adapter._process_message(_message(5, "photo", "alice")))
    await asyncio.wait_for(started.wait(), 5)
    control = _message(3, "text", "alice")
    control["item_list"][0]["text_item"]["text"] = command
    try:
        await adapter._process_message(control)
        assert [e.text for e in delivered] == [command]
        release.set()
        if download == "cancelled":
            photo.cancel()
        await asyncio.gather(photo, return_exceptions=True)
        for key in list(adapter._pending_text_batches):
            await adapter._flush_text_batch_now(key)
        assert [(e.source.user_id, e.text) for e in delivered[1:]] == [
            ("alice", "caption-1"), ("bob", "caption-2"),
        ]
        assert delivered[1].media_urls == (["alice.jpg"] if download == "success" else [])
        # A disconnected adapter must not wake later with a buffered message.
        await adapter._process_message(_message(4, "text", "alice"))
        await adapter.disconnect()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        assert len(delivered) == 3
    finally:
        release.set()
        await asyncio.gather(photo, return_exceptions=True)
        await adapter.disconnect()
