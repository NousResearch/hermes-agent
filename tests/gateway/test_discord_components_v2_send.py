"""Components V2 send path: flag gating and fallback to chunked text."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.gateway.conftest import _ensure_discord_mock

# Do NOT call _ensure_discord_mock() here. tests/gateway/conftest.py already installs the
# mock at import time, and calling it again mints a *fresh* MagicMock (it only short-circuits
# for the real library, which has __file__). That new object replaces the one sibling modules
# captured at collection, breaking isinstance checks against discord.Thread /
# discord.ForumChannel and assertions on discord.MessageReference: 15 unrelated gateway tests
# fail in the same session, while every file still passes in isolation.
assert "discord" in sys.modules, "tests/gateway/conftest.py should have installed the mock"

from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(components_v2: bool):
    a = DiscordAdapter.__new__(DiscordAdapter)
    # `name` is a read-only property on the base adapter; back it with the config it reads.
    a.config = SimpleNamespace(name="discord", extra={})
    a.platform = SimpleNamespace(value="discord")
    a._components_v2 = components_v2
    a._last_self_message_id = {}
    a._nonconversational_messages = SimpleNamespace(mark_many=AsyncMock())
    a.format_message = lambda c: c
    return a


@pytest.mark.asyncio
class TestComponentsV2SendGating:
    async def test_disabled_flag_never_builds_a_view(self):
        channel = MagicMock()
        channel.send = AsyncMock()
        out = await _adapter(False)._try_send_components_v2(
            channel, "hello", None, "c1", None, False)
        assert out is None
        channel.send.assert_not_called()

    async def test_enabled_flag_sends_a_view_and_no_content(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id=777))
        out = await _adapter(True)._try_send_components_v2(
            channel, "hello there", None, "c1", None, False)
        assert out is not None
        assert out.success is True
        assert out.message_id == "777"
        kwargs = channel.send.await_args.kwargs
        assert kwargs.get("view") is not None
        # A v2 message rejects the legacy content field outright.
        assert "content" not in kwargs or kwargs["content"] is None

    async def test_oversized_reply_falls_back_to_chunked_path(self):
        """Past v2's 4000-char whole-tree budget the legacy chunker must still deliver."""
        channel = MagicMock()
        channel.send = AsyncMock()
        out = await _adapter(True)._try_send_components_v2(
            channel, "x" * 9000, None, "c1", None, False)
        assert out is None
        channel.send.assert_not_called()

    async def test_send_failure_degrades_instead_of_losing_the_reply(self):
        channel = MagicMock()
        channel.send = AsyncMock(side_effect=RuntimeError("50035 invalid components"))
        out = await _adapter(True)._try_send_components_v2(
            channel, "hello", None, "c1", None, False)
        assert out is None

    async def test_tracks_last_message_id_for_history_backfill(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id=42))
        a = _adapter(True)
        await a._try_send_components_v2(channel, "hello", None, "c1", None, False)
        assert a._last_self_message_id["c1"] == "42"

    async def test_thread_id_wins_over_chat_id_for_tracking(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id=43))
        a = _adapter(True)
        await a._try_send_components_v2(channel, "hello", None, "c1", "t9", False)
        assert a._last_self_message_id["t9"] == "43"
        assert "c1" not in a._last_self_message_id

    async def test_nonconversational_marks_instead_of_tracking(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id=44))
        a = _adapter(True)
        await a._try_send_components_v2(channel, "hello", None, "c1", None, True)
        a._nonconversational_messages.mark_many.assert_awaited_once_with(["44"])
        assert a._last_self_message_id == {}

    async def test_reply_reference_is_forwarded(self):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id=45))
        ref = object()
        await _adapter(True)._try_send_components_v2(
            channel, "hello", ref, "c1", None, False)
        assert channel.send.await_args.kwargs.get("reference") is ref


def test_components_v2_yaml_key_bridges_to_env():
    """discord.components_v2 in config.yaml must reach the adapter.

    The adapter reads DISCORD_COMPONENTS_V2 via os.getenv, and top-level `discord:` keys only
    reach it through _apply_yaml_config's YAML->env bridge. Without an entry in
    _YAML_BOOL_ENV_KEYS the key is written to config.yaml, passes validation, and is then
    silently dropped -- the flag reads False forever with no error anywhere.
    """
    from plugins.platforms.discord.adapter import _YAML_BOOL_ENV_KEYS

    assert ("components_v2", "DISCORD_COMPONENTS_V2") in _YAML_BOOL_ENV_KEYS
