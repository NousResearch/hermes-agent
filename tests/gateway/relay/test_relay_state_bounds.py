"""Regression: draft/seal coordination dicts are bounded (PR 85796
review, M1).

_sealed_draft_by_chat's key embeds a per-turn identity, so every
completed turn added a permanent entry — unbounded growth for the life
of a long-running gateway process. _open_draft_by_chat could grow the
same way via abandoned entries. Both now FIFO-evict at the same cap the
sibling bounded caches use (and the connector's own tombstone store).
"""

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class OkTransport:
    async def send_outbound(self, payload, platform=None):
        return {"success": True, "message_id": "ts.1"}


@pytest.mark.asyncio
async def test_sealed_tombstones_are_bounded():
    adapter, _ = _connected_adapter()
    adapter._transport = OkTransport()
    cap = adapter._DRAFT_STATE_CAP
    for n in range(cap + 300):
        md = {"message_id": f"evt.{n}"}
        await adapter.send_draft("C1", 1000 + n, "x", metadata=md)
        await adapter.send("C1", "final", metadata=dict(md))
    assert len(adapter._sealed_draft_by_chat) <= cap
    assert len(adapter._open_draft_by_chat) <= cap
    # Newest tombstone survives (FIFO evicts oldest first).
    newest_key = adapter._draft_key("C1", {"message_id": f"evt.{cap + 299}"})
    assert newest_key in adapter._sealed_draft_by_chat


@pytest.mark.asyncio
async def test_open_drafts_are_bounded():
    adapter, _ = _connected_adapter()
    adapter._transport = OkTransport()
    cap = adapter._DRAFT_STATE_CAP
    # Arm many streams without ever sealing (abandoned-turn shape).
    for n in range(cap + 300):
        await adapter.send_draft(
            "C1", 2000 + n, "x", metadata={"message_id": f"evt.{n}"}
        )
    assert len(adapter._open_draft_by_chat) <= cap
