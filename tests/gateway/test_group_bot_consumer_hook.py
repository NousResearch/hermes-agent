"""Core Bot commands remain usable with absent, older or optional consumers."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from tests.gateway.test_hosted_room_messaging import (
    _FakeService, _PickerAdapter, _event, _runner, _seed_rooms,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["bots", "bot 1"])
@pytest.mark.parametrize("consumer", ["absent", "older", "declines", "handles"])
async def test_optional_bot_consumer_preserves_core_fallback(tmp_path, monkeypatch, query, consumer):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr("gateway.hosted_room_messaging.current_room_backend", lambda: service)
    adapter = _PickerAdapter()
    runner = _runner(platform=Platform.TELEGRAM)
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None

    module = ModuleType("gateway.hosted_room_messaging_files")
    old_entry = AsyncMock(side_effect=AssertionError("old room entry is not a Bot entry"))
    module.try_room_menu = old_entry
    hook = AsyncMock(return_value=consumer == "handles")
    if consumer in {"declines", "handles"}:
        module.try_bot_menu = hook
    monkeypatch.setitem(sys.modules, module.__name__, None if consumer == "absent" else module)

    event = _event(f"/group 1 {query}", platform=Platform.TELEGRAM)
    result = await runner._handle_rooms_command(event)
    old_entry.assert_not_called()
    if consumer in {"declines", "handles"}:
        hook.assert_awaited_once()
        args = hook.await_args.args
        assert args[:3] == (runner, event, service)
        assert args[3]["room_id"] == "release-room"
        assert args[4] == "/group"
        assert hook.await_args.kwargs == {"bot_query": "1" if query == "bot 1" else None}
    else:
        hook.assert_not_called()
    if consumer == "handles":
        assert result is None and adapter.calls == []
    elif query == "bots":
        assert result is None
        assert adapter.calls[0]["title"].startswith("🤖 Bots\n")
        assert all(choice["value"].startswith("p=") for choice in adapter.calls[0]["choices"])
    else:
        assert isinstance(result, str) and "Group Chat:" in result
