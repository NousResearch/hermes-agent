import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def _adapter() -> RubikaAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"token": "TESTTOKEN"}
    return RubikaAdapter(cfg)


@pytest.mark.asyncio
async def test_get_chat_info_dm():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={
        "chat": {"chat_id": "c1", "chat_type": "User", "first_name": "Ali"}})
    info = await adapter.get_chat_info("c1")
    assert info == {"name": "Ali", "type": "dm"}


@pytest.mark.asyncio
async def test_get_chat_info_group():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={
        "chat": {"chat_id": "g1", "chat_type": "Group", "title": "Team Chat"}})
    info = await adapter.get_chat_info("g1")
    assert info == {"name": "Team Chat", "type": "group"}


@pytest.mark.asyncio
async def test_get_chat_info_group_missing_title():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={
        "chat": {"chat_id": "g2", "chat_type": "Group"}})
    info = await adapter.get_chat_info("g2")
    assert info == {"name": "g2", "type": "group"}
