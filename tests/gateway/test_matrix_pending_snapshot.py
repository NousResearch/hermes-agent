"""Pending Matrix input preserves data without serialising its live context."""

from dataclasses import fields
import json

import pytest

from gateway.shutdown_pending import PendingQueueSnapshot, _write_snapshot
from plugins.platforms.matrix.followup_context import REPLY_EXCERPT_CHARS
from tests.gateway.test_matrix_followup_reply_context import _split_followup


@pytest.mark.asyncio
async def test_pending_logical_reply_is_serialisable_without_losing_live_revalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter, event, runner, ids, _parts, final = await _split_followup(tmp_path)
    dependency = event._inbound_context_dependencies[0]
    source = event.source
    key = runner._session_key_for_source(source)
    public = {item.name: getattr(event, item.name) for item in fields(event)
              if item.init and not item.name.startswith("_") and item.name not in {"raw_message", "source"}}
    public.update(message_type=event.message_type.value, timestamp=event.timestamp.isoformat(),
                  source={item.name: getattr(source, item.name) for item in fields(source)})
    public["source"]["platform"] = source.platform.value

    _write_snapshot(runner, key, [event])

    payload, = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
    snapshot = PendingQueueSnapshot.from_payload(payload)
    record, = snapshot.events
    assert record == {
        "uid": event._pending_snapshot_uid, "event": public, "routing": None,
        "timestamp": event.timestamp.timestamp(),
        "context": {"quoted_media": [], "snapshots": [{
            "room_id": source.chat_id, "reply_event_id": event.reply_to_message_id,
            "logical_reply": {"excerpt": final[:REPLY_EXCERPT_CHARS], "delivery": dependency.logical_reply.delivery.to_json()},
        }]},
    }
    assert event._inbound_context_dependencies == (dependency,)
    from gateway.shutdown_pending_codec import decode_pending_event
    restored = decode_pending_event(record, adapter=adapter)
    assert [snapshot.pending_state() for snapshot in restored._inbound_context_dependencies] == record["context"]["snapshots"]
    adapter._event_context_cache.redact(source.chat_id, ids[0])
    prompt = await runner._prepare_inbound_message_text(event=restored, source=restored.source, history=[])
    assert prompt == (
        '[Replying to your previous message: "[redacted]"]\n\n'
        f"Matrix reaction by {event.user_id}: 👍 on reply {ids[-1]} (reaction event $reaction)."
    )
