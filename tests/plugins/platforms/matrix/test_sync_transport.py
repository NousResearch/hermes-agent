"""Durable Matrix transport checkpoint decoding."""

import json

import pytest

from plugins.platforms.matrix.sync_transport import DurableSyncStore


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
async def test_checkpoint_encoding_preserves_cursor_and_accepted_intakes(
    tmp_path, encoding
):
    store = DurableSyncStore(
        tmp_path, "https://matrix.test", "@user:matrix.test", "device", "token"
    )
    store.path.write_text(
        json.dumps({"next_batch": "cursor", "accepted_events": ["$accepted"]}),
        encoding=encoding,
    )

    await store.load()

    assert (
        await store.get_next_batch(),
        store.intake_accepted("$accepted"),
        store.reserve_intake("$accepted"),
    ) == ("cursor", True, False)
