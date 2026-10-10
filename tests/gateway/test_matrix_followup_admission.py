"""Cancelled reaction intake remains retryable until gateway admission."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionStore
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.sync_transport import DurableSyncStore, SyncDispatch
from plugins.platforms.matrix.reaction_followups import ReactionWatchStore


async def _admission(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "@alice:test")
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:test")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._busy_input_mode = "queue"
    runner._draining = False
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"user_id": "@bot:test"}))
    runner.adapters = {Platform.MATRIX: adapter}
    adapter.gateway_runner = runner
    adapter.set_session_store(runner.session_store)
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATRIX))
    adapter.set_message_handler(AsyncMock())
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._store_dir = tmp_path / "matrix"
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(display_name="Room", room_topic=None))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    cursor = DurableSyncStore(tmp_path / "sync", "server", "@bot:test", "device", "token")
    await cursor.load()
    adapter._client = SimpleNamespace(sync_store=cursor, api=SimpleNamespace(request=AsyncMock(return_value={"end": "after-final", "chunk": [{"event_id": "$reaction"}]})))
    dispatch = SyncDispatch(adapter._client)
    dispatch.intake_handlers = {adapter._on_reaction}
    source = adapter.build_source(chat_id="!room:test", chat_type="group", user_id="@alice:test")
    entry = runner.session_store.get_or_create_session(source)
    adapter._followup_store().arm("turn", ("$reply",), profile=source.profile or "", room_id=source.chat_id, thread_id="", session_key=entry.session_key, session_id=entry.session_id, requester=source.user_id, source=source.to_dict(), emoji_filter=(), delivery_event_id="$reply", text_content="The answer")
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    return runner, adapter, cursor, dispatch, source, entry


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_cancel_before_busy_admission_keeps_reaction_retryable(tmp_path, monkeypatch, restart):
    runner, adapter, cursor, dispatch, source, entry = await _admission(tmp_path, monkeypatch)
    reached, release = asyncio.Event(), asyncio.Event()
    facade = runner.async_session_store
    lookup = facade.lookup_by_session_key

    async def blocked_lookup(key):
        result = await lookup(key)
        reached.set()
        await release.wait()
        return result

    facade.lookup_by_session_key = blocked_lookup
    event = SimpleNamespace(room_id=source.chat_id, sender=source.user_id, event_id="$reaction", content={"m.relates_to": {"rel_type": "m.annotation", "event_id": "$reply", "key": "👍"}})
    intake = asyncio.create_task(dispatch._catch_errors(adapter._on_reaction, event))
    try:
        await asyncio.wait_for(reached.wait(), timeout=2)
        intake.cancel()
        with pytest.raises(asyncio.CancelledError):
            await intake
        assert (adapter._pending_messages, await cursor.get_next_batch(), cursor.reserve_intake("$reaction"), adapter._followup_store().candidate(source.chat_id, "$reply") is not None) == ({}, None, True, True)
        cursor.release_intake("$reaction")
        if restart:
            prior_store = adapter._followup_store()
            prior = prior_store.claim(source.profile or "", source.chat_id, "$reply", source.user_id,
                                      "👍", reaction_event_id="$reaction", verified_delivery_event_id="$reply")
            replacement_store = ReactionWatchStore(prior_store.path)
            replacement = replacement_store.claim(source.profile or "", source.chat_id, "$reply", source.user_id,
                                                  "👍", reaction_event_id="$reaction", verified_delivery_event_id="$reply")
            prior_store.finish_claim(prior, consumed=False)
            prior_store.finish_claim(prior, consumed=True)
            assert (
                replacement_store.candidate(source.chat_id, "$reply") is not None,
                replacement_store.claim(source.profile or "", source.chat_id, "$reply", source.user_id,
                                        "👍", reaction_event_id="$other", verified_delivery_event_id="$reply"),
            ) == (True, None)
            replacement_store.finish_claim(replacement, consumed=False)
            adapter._reaction_watch_store = replacement_store
        facade.lookup_by_session_key = lookup
        adapter._forget_processed_event("$reaction")
        await dispatch._catch_errors(adapter._on_reaction, event)
        assert (
            adapter._pending_messages[entry.session_key].message_id,
            cursor.reserve_intake("$reaction"),
            adapter._followup_store().candidate(source.chat_id, "$reply"),
        ) == ("$reaction", False, None)
    finally:
        release.set()
        intake.cancel()
        await asyncio.gather(intake, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_after_queue_admission_consumes_watch_once(tmp_path, monkeypatch):
    runner, adapter, cursor, dispatch, source, entry = await _admission(tmp_path, monkeypatch)
    reached, release = threading.Event(), threading.Event()
    write = cursor._write

    def blocked_write(*args):
        reached.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Admission test did not release the durable write")
        write(*args)

    cursor._write = blocked_write
    event = SimpleNamespace(room_id=source.chat_id, sender=source.user_id, event_id="$reaction",
                            content={"m.relates_to": {"rel_type": "m.annotation", "event_id": "$reply", "key": "👍"}})
    intake = asyncio.create_task(dispatch._catch_errors(adapter._on_reaction, event))
    try:
        assert await asyncio.to_thread(reached.wait, 5)
        intake.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await intake
        assert (
            adapter._pending_messages[entry.session_key].message_id,
            cursor.intake_accepted("$reaction"),
            adapter._followup_store().candidate(source.chat_id, "$reply"),
        ) == ("$reaction", True, None)
        await dispatch._catch_errors(adapter._on_reaction, event)
        assert (adapter._pending_messages[entry.session_key].message_id,
                runner._overflow_queue(entry.session_key)) == ("$reaction", None)
    finally:
        release.set()
        intake.cancel()
        await asyncio.gather(intake, return_exceptions=True)
