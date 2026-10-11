"""The agent's ``discord_voice`` tool drives the same join/leave paths as /voice."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_voice_command import _make_runner  # installs the discord mock
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars

GUILD, TEXT, USER = 1488044804787933264, "1553941325869748224", "942640304123674735"


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    th.join(5)
    loop.close()


def _adapter(connected=False):
    adapter = AsyncMock()
    adapter.join_voice_channel = AsyncMock(return_value=True)
    channel = MagicMock()
    channel.name = "Lounge"
    adapter.get_user_voice_channel = AsyncMock(return_value=channel)
    state = {"in": connected}
    adapter.is_in_voice_channel = MagicMock(side_effect=lambda gid: state["in"])

    async def _join(ch):
        state["in"] = True
        return True

    async def _leave(gid):
        state["in"] = False

    adapter.join_voice_channel = AsyncMock(side_effect=_join)
    adapter.leave_voice_channel = AsyncMock(side_effect=_leave)
    adapter._voice_text_channels, adapter._voice_sources = {}, {}
    adapter._voice_clients = {}
    adapter._voice_input_callback = "sentinel" if connected else None
    adapter.get_voice_channel_info = MagicMock(return_value=None)
    return adapter


@pytest.fixture
def runner(tmp_path, gateway_loop):
    r = _make_runner(tmp_path)
    r._gateway_loop = gateway_loop
    return r


def _bind(user=USER, platform="discord", chat=TEXT, scope=str(GUILD), cron=""):
    return set_session_vars(platform=platform, chat_id=chat, chat_type="group", user_id=user,
                            user_name="alice", scope_id=scope, cron_session=cron)


def _call(runner, action):
    import tools.discord_voice_tool as tool
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        return json.loads(tool.discord_voice(action))


def test_leave_uses_shared_path_and_clears_state(runner):
    adapter = _adapter(connected=True)
    runner.adapters[Platform.DISCORD] = adapter
    runner._voice_mode[f"discord:{TEXT}"] = "all"
    tokens = _bind()
    try:
        with patch.object(type(runner), "_voice_channel_leave_for_source",
                          wraps=runner._voice_channel_leave_for_source, autospec=False) as spy:
            out = _call(runner, "leave")
        assert spy.called
    finally:
        clear_session_vars(tokens)
    assert out["success"] is True
    adapter.leave_voice_channel.assert_awaited_once_with(GUILD)
    assert runner._voice_mode[f"discord:{TEXT}"] == "off"
    assert adapter._voice_input_callback is None


def test_join_binds_requesters_voice_channel(runner):
    adapter = _adapter(connected=False)
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        out = _call(runner, "join")
    finally:
        clear_session_vars(tokens)
    assert out["success"] is True and "Lounge" in out["message"]
    adapter.get_user_voice_channel.assert_awaited_once_with(GUILD, USER)
    assert adapter._voice_text_channels[GUILD] == int(TEXT)
    assert runner._voice_mode[f"discord:{TEXT}"] == "all"
    assert adapter._voice_input_callback is not None


def test_unauthorized_user_refused(runner):
    adapter = _adapter(connected=True)
    runner.adapters[Platform.DISCORD] = adapter
    runner._is_user_authorized = lambda source: False
    tokens = _bind(user="123")
    try:
        out = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    assert "error" in out
    adapter.leave_voice_channel.assert_not_awaited()


@pytest.mark.parametrize("kw", [{"platform": "telegram"}, {"cron": "1"}, {"user": ""}])
def test_non_discord_or_cron_refused(runner, kw):
    runner.adapters[Platform.DISCORD] = _adapter(connected=True)
    tokens = _bind(**kw)
    try:
        out = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    assert "error" in out


def test_no_gateway_refused():
    import tools.discord_voice_tool as tool
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        out = json.loads(tool.discord_voice("leave"))
    assert "error" in out
    assert tool.check_discord_voice_requirements() is False


def test_status_reports_connection(runner):
    adapter = _adapter(connected=True)
    adapter.get_voice_channel_info = MagicMock(return_value={
        "channel_name": "Lounge", "member_count": 1, "members": [{"display_name": "alice"}]})
    adapter._voice_text_channels[GUILD] = int(TEXT)
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        out = _call(runner, "status")
    finally:
        clear_session_vars(tokens)
    assert out["connected"] is True and out["channel"] == "Lounge"
    assert out["bound_text_channel_id"] == TEXT


def test_leave_from_dm_finds_connected_guild(runner):
    adapter = _adapter(connected=True)
    vc = MagicMock()
    vc.is_connected.return_value = True
    adapter._voice_clients = {GUILD: vc}
    adapter._voice_sources[GUILD] = SessionSource(
        platform=Platform.DISCORD, chat_id=TEXT, chat_type="group", user_id=USER).to_dict()
    runner.adapters[Platform.DISCORD] = adapter
    runner._voice_mode[f"discord:{TEXT}"] = "all"
    tokens = _bind(chat="999", scope="")
    try:
        out = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    assert out["success"] is True
    adapter.leave_voice_channel.assert_awaited_once_with(GUILD)
    assert runner._voice_mode[f"discord:{TEXT}"] == "off"


def test_unknown_action():
    import tools.discord_voice_tool as tool
    assert "error" in json.loads(tool.discord_voice("dance"))


def test_tool_registered_in_discord_toolset():
    import tools.discord_voice_tool  # noqa: F401
    from toolsets import resolve_toolset
    assert "discord_voice" in resolve_toolset("discord")
