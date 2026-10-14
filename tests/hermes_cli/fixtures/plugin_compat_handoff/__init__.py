"""Frozen pre-metadata adapter; keep the two-argument handoff override."""

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult


class LegacyThreadAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config, Platform.DISCORD)
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "group"}

    async def create_handoff_thread(self, parent_chat_id, name):
        self.calls.append((parent_chat_id, name))
        return "777000"


def register(ctx):
    ctx.register_platform("discord", "Frozen handoff adapter", LegacyThreadAdapter, lambda: True)
