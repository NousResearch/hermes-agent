"""Pending Matrix resolutions retain withdrawal state through cache pressure."""

import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.matrix.read_context import read_matrix_context
from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
)
from plugins.platforms.matrix.room_context import fetch_room_entries
from plugins.platforms.matrix.thread_context import fetch_thread_entries
from tests.gateway.test_matrix import _make_adapter
from tests.gateway.test_matrix_effective_event_state import (
    ROOM,
    SENDER,
    _edit_store,
    _edited,
    _original,
)


async def _resolve(adapter, scope):
    cache, client = adapter._event_context_cache, adapter._client
    if scope == "reply":
        return await cache.resolve(client, ROOM, "$target")
    if scope == "room":
        return await fetch_room_entries(client, cache, ROOM, "$current", limit=5)
    if scope in {"thread-root", "thread-child"}:
        return await fetch_thread_entries(
            client,
            cache,
            ROOM,
            "$target" if scope == "thread-root" else "$root",
            before_event_id="$current",
            limit=5,
        )
    return await read_matrix_context(
        adapter, scope.removesuffix("-read"), ROOM, "$target", 5, requester=SENDER
    )


def _client(adapter, raw, scope, barrier, started, release):
    types = pytest.importorskip("mautrix.types")
    original = _original(
        "$target",
        "WITHDRAWN ORIGINAL",
        root="$root" if scope == "thread-child" else None,
    )
    root = _original("$root", "root")

    async def pause():
        started.set()
        await release.wait()

    async def request(_method, path, **_kwargs):
        if "/event/" in path:
            if path.endswith("%24root"):
                return root
            if barrier == "fetch":
                await pause()
            return raw
        if "/context/" in path:
            return {"start": "boundary"}
        if "/messages" in path or path.endswith("/m.thread"):
            chunk = [raw] if scope not in {"thread-root", "thread-read"} else []
            if barrier == "sibling":
                blocker = _encrypted(
                    _original(
                        "$blocker",
                        "blocker",
                        root="$root" if scope == "thread-child" else None,
                    )
                )
                chunk = [blocker, raw] if scope == "room-read" else [raw, blocker]
            if barrier == "page" and path.endswith("/m.thread"):
                await pause()
            return {"start": "page", "chunk": chunk}
        if path.endswith("/m.annotation"):
            return {"chunk": []}
        raise AssertionError(path)

    async def decrypt(event):
        if barrier == "sibling" and event.event_id == "$blocker":
            await pause()
            return types.Event.deserialize({
                **_original(
                    "$blocker",
                    "blocker",
                    root="$root" if scope == "thread-child" else None,
                ),
                "origin_server_ts": 1,
            })
        if barrier == ("original" if event.event_id == "$target" else "replacement"):
            await pause()
        if event.event_id == "$edit":
            replacement = _edited(original, "WITHDRAWN EDIT")["unsigned"][
                "m.relations"
            ]["m.replace"]
            return types.Event.deserialize({**replacement, "origin_server_ts": 1})
        return types.Event.deserialize({**original, "origin_server_ts": 1})

    store = _edit_store({
        "m.new_content": {"msgtype": "m.text", "body": "WITHDRAWN EDIT"}
    })
    original_get = store.get_group_session

    async def get_session(*args):
        if barrier == "store":
            await pause()
        return await original_get(*args)

    store.get_group_session = get_session
    adapter._client = SimpleNamespace(
        api=SimpleNamespace(request=request),
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt, crypto_store=store),
        sync_store=SimpleNamespace(get_next_batch=AsyncMock(return_value="boundary")),
    )


def _adapter():
    adapter = _make_adapter()
    adapter._event_context_cache = MatrixEventContextCache(max_entries=2)
    adapter._joined_rooms = {ROOM}
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_sender_authorized = lambda *_args, **_kwargs: True
    return adapter


def _encrypted(raw):
    return {
        **raw,
        "origin_server_ts": 1,
        "type": "m.room.encrypted",
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2",
            "ciphertext": "ciphertext",
            "session_id": "session",
            "sender_key": "sender",
            "device_id": "device",
            **(
                {"m.relates_to": raw["content"]["m.relates_to"]}
                if "m.relates_to" in raw["content"]
                else {}
            ),
        },
    }


def _evict(cache):
    for index in range(cache.max_entries):
        cache.store(ROOM, f"$pressure{index}", MatrixEventContext(SENDER, "unrelated"))
    gc.collect()


async def _collect_completed(cache):
    drained = asyncio.Event()
    asyncio.get_running_loop().call_soon(drained.set)
    await drained.wait()
    cache._entries.clear()
    gc.collect()
    assert list(cache._active_states) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,barrier",
    [
        ("event-read", "fetch"),
        ("thread-read", "fetch"),
        ("reply", "fetch"),
        ("thread-root", "fetch"),
        ("event-read", "original"),
        ("room-read", "original"),
        ("thread-read", "original"),
        ("reply", "original"),
        ("room", "original"),
        ("thread-root", "original"),
        ("thread-child", "original"),
        ("room", "sibling"),
        ("room-read", "sibling"),
        ("thread-child", "sibling"),
        ("thread-child", "page"),
    ],
)
async def test_uncached_original_withdrawal_remains_visible_during_resolution(
    scope, barrier
):
    adapter = _adapter()
    cache = adapter._event_context_cache
    raw = _original(
        "$target",
        "WITHDRAWN ORIGINAL",
        root="$root" if scope == "thread-child" else None,
    )
    if barrier == "original":
        raw = _encrypted(raw)
    started, release = asyncio.Event(), asyncio.Event()
    _client(adapter, raw, scope, barrier, started, release)
    independent = MatrixEventContextCache()
    independent.store(ROOM, "$target", MatrixEventContext(SENDER, "other owner"))
    pending = asyncio.create_task(_resolve(adapter, scope))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        cache.redact(ROOM, "$target")
        _evict(cache)
    finally:
        release.set()
        result = await pending

    redacted = MatrixEventContext(SENDER, "", redacted=True, event_id="$target")
    if scope.endswith("-read"):
        blocker = (
            [
                {
                    "event_id": "$blocker",
                    "sender": SENDER,
                    "body": "blocker",
                    "msgtype": "m.text",
                    "thread_id": None,
                    "timestamp": 1,
                    "sender_authorized": True,
                }
            ]
            if barrier == "sibling"
            else []
        )
        assert result == {
            "events": [
                *blocker,
                {
                    "event_id": "$target",
                    "sender": SENDER,
                    "body": "[redacted]",
                    "msgtype": None,
                    "thread_id": None,
                    "timestamp": 1 if barrier == "original" else None,
                    "sender_authorized": True,
                    "redacted": True,
                },
            ],
            "errors": [],
        }
    elif scope == "reply":
        assert result is None
    else:
        root = (
            [MatrixEventContext(SENDER, "root", event_id="$root")]
            if scope == "thread-child"
            else []
        )
        blocker = (
            [MatrixEventContext(SENDER, "blocker", event_id="$blocker")]
            if barrier == "sibling"
            else []
        )
        assert result == [*root, *blocker, redacted]
    assert independent.history_entry(ROOM, "$target") == MatrixEventContext(
        SENDER, "other owner", event_id="$target"
    )
    assert len(cache._entries) <= cache.max_entries
    del result, pending
    await _collect_completed(cache)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope", ["event-read", "reply", "room", "thread-root", "thread-child"]
)
@pytest.mark.parametrize("barrier", ["original", "replacement", "store"])
async def test_bundled_replacement_withdrawal_remains_visible_through_registration(
    scope, barrier
):
    adapter = _adapter()
    cache = adapter._event_context_cache
    raw = _encrypted(
        _edited(
            _original(
                "$target",
                "WITHDRAWN ORIGINAL",
                root="$root" if scope == "thread-child" else None,
            ),
            "WITHDRAWN EDIT",
        )
    )
    raw["unsigned"]["m.relations"]["m.replace"] = _encrypted(
        raw["unsigned"]["m.relations"]["m.replace"]
    )
    started, release = asyncio.Event(), asyncio.Event()
    _client(adapter, raw, scope, barrier, started, release)
    pending = asyncio.create_task(_resolve(adapter, scope))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        cache.redact(ROOM, "$edit")
        _evict(cache)
    finally:
        release.set()
        result = await pending

    unavailable = MatrixEventContext(
        SENDER,
        "[event content unavailable]",
        event_id="$target",
        state_error="replacement was redacted",
        replacement_id="$edit",
    )
    if scope == "event-read":
        assert result == {
            "events": [
                {
                    "event_id": "$target",
                    "sender": SENDER,
                    "body": "[event content unavailable]",
                    "msgtype": None,
                    "thread_id": None,
                    "timestamp": 1,
                    "sender_authorized": True,
                }
            ],
            "errors": [{"event_id": "$target", "error": "replacement was redacted"}],
        }
    elif scope == "reply":
        assert result == unavailable
    else:
        root = (
            [MatrixEventContext(SENDER, "root", event_id="$root")]
            if scope == "thread-child"
            else []
        )
        assert result == [*root, unavailable]
    assert len(cache._entries) <= cache.max_entries
    _evict(cache)
    assert cache.is_redacted(ROOM, "$edit") is (scope != "event-read")
    del result, pending
    await _collect_completed(cache)


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["room-not-allowed", "startup-grace"])
async def test_ignored_replies_do_not_evict_cached_events(gate):
    adapter = _adapter()
    adapter._startup_ts = 1_000.0
    adapter._is_allowed_matrix_room_event = AsyncMock(
        return_value=gate != "room-not-allowed"
    )
    cache = adapter._event_context_cache
    kept = {
        event_id: cache.store(ROOM, event_id, MatrixEventContext(SENDER, event_id))
        for event_id in ("$first", "$second")
    }

    await adapter._on_room_message(
        SimpleNamespace(
            room_id="!other:example.org",
            sender=SENDER,
            event_id="$ignored",
            timestamp=1.0 if gate == "startup-grace" else 1_000.0,
            content={
                "msgtype": "m.text",
                "body": "reply",
                "m.relates_to": {"m.in_reply_to": {"event_id": "$parent"}},
            },
        )
    )

    assert cache.snapshot(ROOM) == kept
