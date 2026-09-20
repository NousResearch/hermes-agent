"""Adapter warnings are separate from captions, attachment receipts and failed state."""
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.matrix.adapter import MatrixAdapter


class LegacyAdapter(BasePlatformAdapter):
    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "fixture", "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((content, reply_to, metadata))
        return SendResult(success=True, message_id="actual-text-send")


