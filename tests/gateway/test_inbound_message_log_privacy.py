"""The per-message ``inbound message`` INFO line carries metadata only.

Message text and the sender's display name go to a separate DEBUG line, so a
gateway running at the default INFO level does not keep user content in its
rotated logs.
"""

import logging

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource

_TEXT = "my order 4471 for Jane Doe"


class _Runner(GatewayTurnMixin):
    async def _hmwa_resolve_session(self, event, source):
        return None  # stop right after the inbound log lines


def _event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="dm",
        user_id="12345",
        user_name="Jane Doe",
    )
    return MessageEvent(text=_TEXT, source=source, message_id="msg-42")


async def _inbound_records(caplog, level):
    event = _event()
    with caplog.at_level(level, logger="gateway.run"):
        await _Runner()._handle_message_with_agent(event, event.source, "q", 1)
    return [r for r in caplog.records if r.getMessage().startswith("inbound message")]


@pytest.mark.asyncio
async def test_info_line_has_metadata_and_no_text(caplog):
    records = await _inbound_records(caplog, logging.INFO)
    assert [r.levelno for r in records] == [logging.INFO]
    line = records[0].getMessage()
    assert "platform=telegram chat=-1001 message_id=msg-42" in line
    assert f"chars={len(_TEXT)}" in line
    assert "4471" not in line and "Jane" not in line


@pytest.mark.asyncio
async def test_debug_line_keeps_text_and_sender(caplog):
    records = await _inbound_records(caplog, logging.DEBUG)
    debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert "user=Jane Doe" in debug[0] and _TEXT in debug[0]
