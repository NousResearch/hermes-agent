"""Regression: Discord exec approvals are mirrored to operator DMs.

Dangerous-command approvals posted only in a noisy channel/thread are easy to
miss. The adapter keeps the original channel prompt as the canonical visible
gate, but also duplicates the same approval request to configured Discord
operator DMs (DISCORD_ALLOWED_USERS / adapter._allowed_user_ids). A DM button
resolves the same pending approval because it carries the same session_key.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.discord.adapter import DiscordAdapter


class _SentMessage:
    def __init__(self, message_id):
        self.id = message_id


class _FakeChannel:
    def __init__(self, message_id="channel-msg"):
        self.sent = []
        self._message_id = message_id

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return _SentMessage(self._message_id)


class _FakeUser:
    def __init__(self, user_id, *, fail=False):
        self.id = int(user_id)
        self.fail = fail
        self.sent = []

    async def send(self, **kwargs):
        if self.fail:
            raise RuntimeError("DM closed")
        self.sent.append(kwargs)
        return _SentMessage(f"dm-{self.id}")


class _FakeClient:
    def __init__(self, *, channel, users):
        self._channel = channel
        self._users = {int(u.id): u for u in users}
        self.fetch_user = AsyncMock(side_effect=self._fetch_user)

    def get_channel(self, channel_id):
        return self._channel if int(channel_id) == 123 else None

    async def fetch_channel(self, channel_id):
        if int(channel_id) == 123:
            return self._channel
        raise RuntimeError("missing channel")

    def get_user(self, user_id):
        return self._users.get(int(user_id))

    async def _fetch_user(self, user_id):
        user = self._users.get(int(user_id))
        if user is None:
            raise RuntimeError("missing user")
        return user


@pytest.fixture
def adapter():
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._allowed_user_ids = {"111", "222"}
    a._allowed_role_ids = set()
    return a


@pytest.mark.asyncio
async def test_send_exec_approval_mirrors_prompt_to_allowed_user_dms(adapter):
    channel = _FakeChannel()
    user_111 = _FakeUser(111)
    user_222 = _FakeUser(222)
    adapter._client = _FakeClient(channel=channel, users=[user_111, user_222])

    result = await adapter.send_exec_approval(
        chat_id="123",
        command="rm -rf /important",
        session_key="agent:main:discord:thread:123:456",
        description="recursive delete",
        metadata={"thread_id": "123"},
    )

    assert isinstance(result, SendResult)
    assert result.success is True
    assert result.message_id == "channel-msg"
    assert len(channel.sent) == 1
    assert len(user_111.sent) == 1
    assert len(user_222.sent) == 1

    channel_view = channel.sent[0]["view"]
    dm_view = user_111.sent[0]["view"]
    assert channel_view.session_key == "agent:main:discord:thread:123:456"
    assert dm_view.session_key == channel_view.session_key
    assert user_111.sent[0]["embed"].title == channel.sent[0]["embed"].title


@pytest.mark.asyncio
async def test_send_exec_approval_keeps_channel_prompt_when_operator_dm_fails(adapter):
    channel = _FakeChannel()
    user_111 = _FakeUser(111, fail=True)
    user_222 = _FakeUser(222)
    adapter._client = _FakeClient(channel=channel, users=[user_111, user_222])

    result = await adapter.send_exec_approval(
        chat_id="123",
        command="rm -rf /important",
        session_key="agent:main:discord:thread:123:456",
        description="recursive delete",
    )

    assert result.success is True
    assert result.message_id == "channel-msg"
    assert len(channel.sent) == 1
    assert len(user_111.sent) == 0
    assert len(user_222.sent) == 1
