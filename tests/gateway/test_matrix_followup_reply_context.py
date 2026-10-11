"""A reaction to a split final uses the logical reply's current excerpt."""

import asyncio
import json
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from plugins.platforms.matrix.followup_mixin import _MatrixFollowupChoice
from plugins.platforms.matrix.followup_context import REPLY_EXCERPT_CHARS
from plugins.platforms.matrix.reply_context import MatrixEventContext
from tests.gateway.test_matrix import (
    _OPS_STATE, _ROOM_ID, _room_context_adapter, _room_context_runner, _room_session,
)


async def _split_followup(tmp_path):
    store, source = _room_session(tmp_path)
    entry = store.get_or_create_session(source)
    adapter = _room_context_adapter(_OPS_STATE)
    adapter._store_dir = tmp_path / "matrix"
    adapter._reaction_followup_actions[entry.session_key] = _MatrixFollowupChoice(
        "turn", (), _ROOM_ID, source.user_id, "", "", entry.session_id,
    )
    parts = ("alpha " * 56, "beta " * 67, "gamma " * 56)
    ids = ("$z-first", "$a-second", "$m-last")
    adapter._client.send_message_event = AsyncMock(side_effect=ids)
    for body in parts:
        await adapter._send_room_message(_ROOM_ID, {"msgtype": "m.text", "body": body})
    final = "\n".join(parts)
    adapter.on_streamed_final_delivery(source, entry.session_key, tuple(sorted(ids)), final)
    adapter._watch_purge_handle.cancel()

    restarted = _room_context_adapter(_OPS_STATE)
    restarted._store_dir = adapter._store_dir
    restarted.set_session_store(store)
    restarted.set_authorization_check(lambda *_args, **_kwargs: True)
    restarted._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    restarted._message_handler = AsyncMock()
    restarted.handle_message = AsyncMock()

    async def request(_method, path, **kwargs):
        if "/context/" in path:
            return {"end": "after-final"}
        if path.endswith("/messages"):
            return {"chunk": [{"event_id": "$reaction"}]}
        event_id = unquote(path.rsplit("/", 1)[1])
        return {"event_id": event_id, "room_id": _ROOM_ID,
                "sender": restarted._user_id, "type": "m.room.message",
                "content": {"msgtype": "m.text", "body": parts[ids.index(event_id)]}}

    restarted._client.api.request = AsyncMock(side_effect=request)
    await restarted._handle_followup_reaction(
        _ROOM_ID, ids[-1], "👍", source.user_id, "$reaction",
    )
    followup = restarted.handle_message.await_args.args[0]
    return restarted, followup, _room_context_runner(store, restarted), ids, parts, final


@pytest.mark.asyncio
async def test_split_logical_reply_survives_restart_with_delivery_order(tmp_path):
    adapter, event, runner, ids, parts, final = await _split_followup(tmp_path)
    try:
        prompt = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])
        excerpt = json.dumps(final[:REPLY_EXCERPT_CHARS], ensure_ascii=False)
        assert prompt == (
            f'[Replying to your previous message: {excerpt}]\n\n'
            f"Matrix reaction by {event.user_id}: 👍 on reply {ids[-1]} (reaction event $reaction)."
        )
        resolved = [unquote(call.args[1].rsplit("/", 1)[1]) for call in adapter._client.api.request.await_args_list
                    if "/event/" in call.args[1]]
        assert resolved == [ids[0], ids[1], ids[2]]
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prefix-edit", "prefix-redaction", "prefix-error", "target-redaction", "target-edit"])
async def test_split_quote_rechecks_current_prefix_and_reacted_target(tmp_path, change):
    adapter, event, runner, ids, parts, final = await _split_followup(tmp_path)
    try:
        snapshot = await adapter.fetch_inbound_context(event)
        await snapshot.refresh()
        cache = adapter._event_context_cache
        if change == "prefix-edit":
            cache.store(_ROOM_ID, ids[0], MatrixEventContext(adapter._user_id, "Updated decision"))
            expected = ("Updated decision\n" + parts[1].strip())[:REPLY_EXCERPT_CHARS]
        elif change == "prefix-redaction":
            cache.redact(_ROOM_ID, ids[0])
            expected = "[redacted]"
        elif change == "prefix-error":
            cache.store(_ROOM_ID, ids[0], MatrixEventContext(adapter._user_id, "", state_error="undecryptable"))
            expected = "[event content unavailable]"
        elif change == "target-edit":
            cache.store(_ROOM_ID, ids[-1], MatrixEventContext(adapter._user_id, "Changed reaction target"))
            expected = "Changed reaction target"
        else:
            cache.redact(_ROOM_ID, ids[-1])
            expected = "[redacted]"
        assert snapshot.reply_event(event).reply_to_text == expected
    finally:
        runner.session_store.close_all_db_handles()
