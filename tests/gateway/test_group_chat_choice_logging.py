"""Compatibility checks for extracted gateway slash dispatch."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageEvent
from gateway.slash_commands_model import GatewayModelCommandsMixin


class _FailingPicker:
    async def send_choice_picker(self, **kwargs):
        raise RuntimeError("picker unavailable")


class _Runner(GatewayModelCommandsMixin):
    def __init__(self):
        self.adapter = _FailingPicker()

    def _adapter_for_source(self, source):
        return self.adapter

    def _thread_metadata_for_source(self, source, anchor=None):
        return {}

    def _reply_anchor_for_event(self, event):
        return None


@pytest.mark.asyncio
async def test_picker_failure_keeps_gateway_logger_identity(caplog):
    source = SimpleNamespace(chat_id="chat", user_id="user")
    event = MessageEvent(text="/group", source=source)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        sent = await _Runner()._try_send_choice_picker(
            event,
            "session-key",
            "Choose",
            [],
            AsyncMock(),
        )

    assert sent is False
    assert any(
        record.name == "gateway.run" and "falling back to text" in record.message
        for record in caplog.records
    )
