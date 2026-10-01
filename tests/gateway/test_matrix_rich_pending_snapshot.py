"""Native Matrix contributions remain data-only in pending snapshots."""

import json

import pytest

from gateway.shutdown_pending import PendingQueueSnapshot, _write_snapshot
from plugins.platforms.matrix.rich_content import MatrixRichContentSnapshot
from tests.gateway.test_matrix_rich_content import _adapter, _body, _event, _runner, _typed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["emote", "sticker"])
async def test_pending_native_contributions_preserve_identity_and_live_revalidation(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter, received = _adapter(monkeypatch)
    await adapter._on_room_message(_typed(_event(kind)))
    event = received.await_args.args[0]
    runner = _runner(adapter, tmp_path)
    dependency, = (
        snapshot for snapshot in event._inbound_context_dependencies
        if isinstance(snapshot, MatrixRichContentSnapshot)
    )
    contribution, = dependency.contributions

    _write_snapshot(runner, runner._session_key_for_source(event.source), [event])

    payload, = [json.loads(path.read_text()) for path in (tmp_path / "pending_messages").glob("*.json")]
    record, = PendingQueueSnapshot.from_payload(payload).events
    assert record["context"] == {
        "quoted_media": [],
        "snapshots": [
            dependency.context.pending_state(),
            {
                "kind": "matrix_rich_content",
                "context": dependency.context.pending_state(),
                "contributions": [{
                    "event_id": event.message_id,
                    "sender": event.source.user_id,
                    "original_text": _body(kind),
                    "original_content_text": contribution.original_content_text,
                    "original_media_identity": contribution.original_media_identity,
                    "media_paths": event.media_urls,
                }],
            },
        ],
    }
    adapter._event_context_cache.redact(event.source.chat_id, event.message_id)
    assert (
        dependency.authored_text(event.text),
        dependency.media_event(event).media_urls,
        event._inbound_context_dependencies[-1] is dependency,
    ) == ("[redacted]", [], True)
