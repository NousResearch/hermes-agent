"""Group help describes commands without looking up private room names."""

import pytest

from gateway.config import Platform
from tests.gateway.test_hosted_room_messaging import _event, _runner


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", ["help", "HELP", "?"])
@pytest.mark.parametrize("authorized", [False, True])
async def test_group_help_never_queries_room_names(monkeypatch, keyword, authorized):
    runner = _runner()
    runner._can_control_group_chats = lambda event, **kwargs: authorized

    def no_private_lookup():
        raise AssertionError("Help must not resolve a room named help")

    monkeypatch.setattr("gateway.hosted_room_messaging.current_room_backend", no_private_lookup)
    result = await runner._handle_rooms_command(_event(f"/group {keyword}"))
    assert "**Group Chats**" in result
    assert "`/group 7 send <message>`" in result
    assert "`/group 7 approvals`" in result
    assert "No group chat matches" not in result
    assert " files" not in result


@pytest.mark.asyncio
async def test_group_help_uses_the_transport_command_prefix():
    runner = _runner(platform=Platform.MATRIX)
    result = await runner._handle_rooms_command(_event("!group help", platform=Platform.MATRIX))
    assert "`!group 7 stop`" in result
    assert "/group" not in result


@pytest.mark.asyncio
async def test_requesting_help_does_not_unlock_group_history():
    runner = _runner(extra={})
    assert "**Group Chats**" in await runner._handle_rooms_command(_event("/group help"))
    denial = await runner._handle_rooms_command(_event("/group 7"))
    assert "can’t control Group Chats" in denial
