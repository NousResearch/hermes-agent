"""Registered private inventory → Send → acknowledgement → exact-room detail."""
import re

import pytest

from tests.gateway.test_canonical_group_messaging_send import (
    authorized_send, bound, canonical_rows, canonical_tasks, consumer, send_event,
)
from tests.gateway.test_canonical_group_messaging_list import route


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["idle", "runner_busy", "adapter_busy"])
async def test_registered_list_send_ack_detail_journey(authorized_send, monkeypatch, lane):
    c = authorized_send
    monkeypatch.setattr(c.runner, "_handle_legacy_rooms_command", c.forbidden)
    await route(c, c.event("/group list"), lane)
    assert len(c.adapter.sent) == 1
    listing = c.adapter.sent.pop()[1]
    selected = re.search(r"([0-9]+)\. Send room", listing)
    assert selected is not None
    room_ref = int(selected.group(1))
    assert room_ref == c.room_grant["room_ref"]

    event = send_event(c, "exact list-to-detail message", room_ref=room_ref)
    await route(c, event, lane)
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    acknowledgement = c.adapter.sent.pop()[1]
    command = acknowledgement.split("Check: ", 1)[1]
    assert command == f"/group {room_ref}"

    await route(c, c.event(command), lane)
    assert len(c.adapter.sent) == 1
    target, detail, reply_to, metadata = c.adapter.sent.pop()
    assert target == "private-chat" and reply_to is None
    assert metadata == {"_interim_send": True}
    assert f"Group {room_ref} — Send room" in detail
    assert "exact list-to-detail message" in detail
    assert "send-room" not in detail and "inert-gateway" not in detail
    await route(c, event, lane)
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    assert len(c.adapter.sent) == 1 and not c.adapter.generic_sent
