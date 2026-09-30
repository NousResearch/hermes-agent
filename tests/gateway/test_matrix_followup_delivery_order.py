"""Reaction novelty follows final delivery, independently of either clock."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.reaction_followups import ReactionWatchStore


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["plain", "streamed", "failed-final"])
@pytest.mark.parametrize("server_offset", [-120, 0, 120])
async def test_reaction_novelty_uses_final_delivery_order_after_restart(
    tmp_path, delivery, server_offset,
):
    wall = [1_800_000_000.0]
    room = "!room:test"
    timeline = []
    streamed = delivery != "plain"

    def record(sender, content, *, timestamp=None):
        event = SimpleNamespace(
            room_id=room, event_id=f"$event{len(timeline)}", sender=sender,
            content=content,
            timestamp=timestamp if timestamp is not None else wall[0] + server_offset,
        )
        timeline.append(event)
        return event

    async def send(_room, _kind, content):
        if delivery == "failed-final" and content.get("m.relates_to", {}).get("rel_type") == "m.replace":
            raise RuntimeError("final edit failed")
        return record("@hermes:test", content).event_id

    async def request(_method, path, *, query_params):
        if "/context/" in path:
            event_id = unquote(path.rsplit("/", 1)[1])
            index = next(index for index, event in enumerate(timeline) if event.event_id == event_id)
            return {"end": str(index + 1)}
        assert path.endswith("/messages")
        assert query_params["dir"] == "f"
        start = int(query_params["from"])
        chunk = timeline[start:start + 1]
        return {"chunk": [{"event_id": event.event_id} for event in chunk],
                **({"end": str(start + 1)} if chunk else {})}

    client = SimpleNamespace(
        send_message_event=AsyncMock(side_effect=send),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
    )
    path = tmp_path / "watches.sqlite"

    def adapter():
        result = MatrixAdapter(PlatformConfig(enabled=True, extra={"user_id": "@hermes:test"}))
        result._reaction_watch_store = ReactionWatchStore(path, clock=lambda: wall[0])
        result._client = client
        result._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        result._source_session_key = lambda _source: "session"
        result._session_store = SimpleNamespace(peek_session_id=lambda _key: "sid")
        result.build_source = lambda **kwargs: SessionSource(platform=Platform.MATRIX, **kwargs)
        result.set_authorization_check(lambda *_args, **_kwargs: True)
        result._message_handler = AsyncMock()
        result.handle_message = AsyncMock()
        return result

    sender = adapter()
    sender._active_sessions["session"] = asyncio.Event()
    source = SessionSource(platform=Platform.MATRIX, chat_id=room, chat_type="group",
                           user_id="@alice:test", thread_id="$thread")
    assert await sender.configure_reaction_followups(
        "session", True, (), room_id=room, requester=source.user_id,
        thread_id=source.thread_id, profile="", session_id="sid",
    )
    sent = await sender.send(room, "Final answer", metadata={"expect_edits": True} if streamed else None)
    target_id = sent.message_id

    def reaction(*, timestamp=None):
        return record("@alice:test", {"m.relates_to": {
            "rel_type": "m.annotation", "event_id": target_id, "key": "👍",
        }}, timestamp=timestamp)

    old = None
    if streamed:
        old = reaction(timestamp=wall[0] + 1000)
        consumer = GatewayStreamConsumer(sender, room)
        consumer._message_id = target_id
        consumer._last_sent_text = "Final answer"
        consumer._preview_message_ids.add(target_id)
        consumer._segment_preview_message_ids.add(target_id)
        consumer._should_send_fresh_final = lambda: False
        edited = await consumer._edit_existing("Final answer", finalize=True, is_turn_final=True)
        if delivery == "failed-final":
            assert not edited
            await consumer._fallback_when_nothing_unseen("Final answer")
            assert consumer.final_content_delivered
        else:
            assert edited

    fresh = reaction()
    wall[0] += 5
    sender.on_streamed_final_delivery(source, "session", (target_id,), "Final answer")
    restarted = adapter()
    if delivery == "failed-final":
        assert sender._followup_store().candidate(room, target_id) is None
        await restarted._on_reaction(old)
        await restarted._on_reaction(fresh)
        restarted.handle_message.assert_not_awaited()
        return
    if old:
        await restarted._on_reaction(old)
        restarted.handle_message.assert_not_awaited()
    await restarted._on_reaction(fresh)
    restarted.handle_message.assert_awaited_once()
    event = restarted.handle_message.await_args.args[0]
    assert (event.source.user_id, event.source.thread_id, event.message_id,
            event.reply_to_message_id, event.reply_to_text, event.defer_until_idle) == (
        source.user_id, source.thread_id, fresh.event_id, target_id, "Final answer", True,
    )
    await restarted._on_reaction(fresh)
    replayed = adapter()
    await replayed._on_reaction(fresh)
    replayed.handle_message.assert_not_awaited()
    restarted.handle_message.assert_awaited_once()
    assert ReactionWatchStore(path, clock=lambda: wall[0]).candidate(room, target_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {"end": "cycle", "chunk": []}, RuntimeError("offline")])
async def test_unproven_delivery_order_does_not_consume_watch(tmp_path, response):
    adapter = MatrixAdapter(PlatformConfig(enabled=True))
    adapter._reaction_watch_store = ReactionWatchStore(tmp_path / "watches.sqlite")
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._source_session_key = lambda _source: "session"
    adapter._session_store = SimpleNamespace(peek_session_id=lambda _key: "sid")
    adapter.build_source = lambda **kwargs: SessionSource(platform=Platform.MATRIX, **kwargs)
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:test", user_id="@alice:test")
    store = adapter._followup_store()
    store.arm("turn", ("$reply",), profile="", room_id=source.chat_id, thread_id="",
              session_key="session", session_id="sid", requester=source.user_id,
              source=source.to_dict(), emoji_filter=(), delivery_event_id="$final")
    before = store.candidate(source.chat_id, "$reply")
    request = (AsyncMock(side_effect=response) if isinstance(response, Exception)
               else AsyncMock(return_value=response))
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=request))
    await adapter._handle_followup_reaction(source.chat_id, "$reply", "👍", source.user_id, "$reaction")
    adapter.handle_message.assert_not_awaited()
    assert store.candidate(source.chat_id, "$reply") == before
    assert store.claim("", source.chat_id, "$reply", source.user_id, "👍") is None
    adapter._client.api.request = AsyncMock(side_effect=[
        {"end": "after-final"}, {"chunk": [{"event_id": "$reaction"}]},
    ])
    await adapter._handle_followup_reaction(source.chat_id, "$reply", "👍", source.user_id, "$reaction")
    adapter.handle_message.assert_awaited_once()
    assert store.candidate(source.chat_id, "$reply") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
async def test_streamed_reply_sends_a_final_edit_only_for_a_pending_followup(followup):
    room = "!room:test"
    sent = []

    async def send(_room, _kind, content):
        sent.append(content.get("m.relates_to", {}).get("rel_type", "message"))
        return f"$event{len(sent)}"

    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"user_id": "@hermes:test"}))
    adapter._client = SimpleNamespace(send_message_event=AsyncMock(side_effect=send))
    adapter._active_sessions["session"] = asyncio.Event()
    if followup:
        assert await adapter.configure_reaction_followups(
            "session", True, (), room_id=room, requester="@alice:test",
            thread_id="", profile="", session_id="sid",
        )
    consumer = GatewayStreamConsumer(adapter, room)
    consumer.on_delta("Final answer")
    consumer.finish()
    await consumer.run()
    anchor = adapter._followup_delivery_events.latest(room, consumer.final_message_ids)
    assert (sent, anchor) == (
        (["message", "m.replace"], "$event2") if followup else (["message"], "$event1")
    )
