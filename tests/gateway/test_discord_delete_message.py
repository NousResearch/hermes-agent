import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.NotFound = type("NotFound", (Exception,), {})
    discord_mod.Forbidden = type("Forbidden", (Exception,), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

_discord_mod = sys.modules["discord"]

# tests/gateway/conftest.py installs its own comprehensive discord mock
# (overwrite) before this file collects, so the local _ensure_discord_mock
# above short-circuits and the shared mock has no real exception classes for
# NotFound/Forbidden.  Make sure they are real Exception subclasses on the
# module object the adapter sees, so `except (discord.NotFound,
# discord.Forbidden)` matches what the tests raise.
if not isinstance(_discord_mod.NotFound, type) or not issubclass(_discord_mod.NotFound, Exception):
    _discord_mod.NotFound = type("NotFound", (Exception,), {})
    _discord_mod.Forbidden = type("Forbidden", (Exception,), {})


_UNSET = object()


def _make_adapter(channel, *, get_channel_return=_UNSET, fetch_channel_return=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel if get_channel_return is _UNSET else get_channel_return),
        fetch_channel=AsyncMock(return_value=fetch_channel_return),
    )
    return adapter


@pytest.mark.asyncio
async def test_delete_message_success():
    message = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=message))
    adapter = _make_adapter(channel)

    result = await adapter.delete_message("555", "888")

    assert result is True
    adapter._client.get_channel.assert_called_once_with(555)
    channel.fetch_message.assert_awaited_once_with(888)
    message.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_message_fetch_fallback():
    message = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=message))
    adapter = _make_adapter(channel, get_channel_return=None, fetch_channel_return=channel)

    result = await adapter.delete_message("555", "888")

    assert result is True
    adapter._client.fetch_channel.assert_awaited_once_with(555)
    channel.fetch_message.assert_awaited_once_with(888)
    message.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_message_client_unavailable():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = None

    result = await adapter.delete_message("555", "888")

    assert result is False


@pytest.mark.asyncio
async def test_delete_message_discord_unavailable(monkeypatch):
    channel = SimpleNamespace(id=555, fetch_message=AsyncMock())
    adapter = _make_adapter(channel)
    monkeypatch.setattr("plugins.platforms.discord.adapter.DISCORD_AVAILABLE", False)

    result = await adapter.delete_message("555", "888")

    assert result is False
    adapter._client.get_channel.assert_not_called()


@pytest.mark.asyncio
async def test_delete_message_channel_missing_after_fetch():
    adapter = _make_adapter(None, get_channel_return=None, fetch_channel_return=None)

    result = await adapter.delete_message("555", "888")

    assert result is False
    adapter._client.fetch_channel.assert_awaited_once_with(555)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_cls", [_discord_mod.NotFound, _discord_mod.Forbidden])
@pytest.mark.parametrize("where", ["fetch_message", "delete"])
async def test_delete_message_not_found_forbidden(exc_cls, where):
    if where == "fetch_message":
        channel = SimpleNamespace(id=555, fetch_message=AsyncMock(side_effect=exc_cls()))
    else:
        message = SimpleNamespace(delete=AsyncMock(side_effect=exc_cls()))
        channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=message))
    adapter = _make_adapter(channel)

    result = await adapter.delete_message("555", "888")

    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["fetch_message", "delete"])
async def test_delete_message_unexpected_exception(where):
    if where == "fetch_message":
        channel = SimpleNamespace(id=555, fetch_message=AsyncMock(side_effect=RuntimeError("boom")))
    else:
        message = SimpleNamespace(delete=AsyncMock(side_effect=RuntimeError("boom")))
        channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=message))
    adapter = _make_adapter(channel)

    result = await adapter.delete_message("555", "888")

    assert result is False
