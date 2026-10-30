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

GUILD, TEXT, USER = 1001, "2002", "3003"


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


def _bind(user=USER, platform="discord", chat=TEXT, scope=str(GUILD), cron="", chat_type=None):
    chat_type = chat_type or ("group" if scope else "dm")
    return set_session_vars(platform=platform, chat_id=chat, chat_type=chat_type, user_id=user,
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


@pytest.mark.parametrize("kw", [{"platform": "telegram"}, {"cron": "1"}, {"cron": "true"}, {"user": ""}])
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


def test_status_connected_follows_connection_not_channel_info(runner):
    # In voice, but channel info is unavailable: still connected.
    adapter = _adapter(connected=True)
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        out = _call(runner, "status")
    finally:
        clear_session_vars(tokens)
    assert out["connected"] is True and out["channel"] is None

    # Channel info present, but the voice client is not connected: not connected.
    adapter = _adapter(connected=False)
    adapter.get_voice_channel_info = MagicMock(return_value={
        "channel_name": "Lounge", "member_count": 0, "members": []})
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        out = _call(runner, "status")
    finally:
        clear_session_vars(tokens)
    assert out["connected"] is False


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


def test_join_from_dm_uses_profile_scoped_home_channel(runner, monkeypatch):
    """The DM fallback reads DISCORD_HOME_CHANNEL from the turn's profile secret scope, not the
    launch process env, and binds the voice channel to that home channel's guild."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    home = "4004"
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    adapter = _adapter(connected=False)
    adapter._client = MagicMock()
    adapter._client.get_channel = MagicMock(return_value=SimpleNamespace(guild=SimpleNamespace(id=GUILD)))
    runner.adapters[Platform.DISCORD] = adapter
    scope = set_secret_scope({"DISCORD_HOME_CHANNEL": home})
    tokens = _bind(chat="999", scope="")
    try:
        out = _call(runner, "join")
    finally:
        clear_session_vars(tokens)
        reset_secret_scope(scope)
    assert out["success"] is True and out["text_channel_id"] == home
    adapter._client.get_channel.assert_called_once_with(int(home))
    assert adapter._voice_text_channels[GUILD] == int(home)


def _vc(connected=True):
    vc = MagicMock()
    vc.is_connected.return_value = connected
    return vc


def test_server_channel_never_falls_back_to_another_server(runner):
    """A leave/status from server A must not act on the bot's connection in server B."""
    other = 5005
    adapter = _adapter(connected=False)
    adapter.is_in_voice_channel = MagicMock(side_effect=lambda gid: gid == other)
    adapter._voice_clients = {other: _vc()}
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        left = _call(runner, "leave")
        status = _call(runner, "status")
    finally:
        clear_session_vars(tokens)
    adapter.leave_voice_channel.assert_not_awaited()
    assert left["success"] is False
    assert status["connected"] is False and status["guild_id"] == str(GUILD)

    # A server-channel session whose guild id is missing must not take the DM fallback either.
    tokens = _bind(scope="", chat_type="group")
    try:
        left = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    adapter.leave_voice_channel.assert_not_awaited()
    assert left["success"] is False


def test_server_channel_without_guild_id_never_joins_the_home_server(runner, monkeypatch):
    """The DM home-channel fallback is for DMs only; a group session with no guild id is refused."""
    import tools.discord_voice_tool as tool
    adapter = _adapter(connected=False)
    runner.adapters[Platform.DISCORD] = adapter
    fallback = MagicMock()
    monkeypatch.setattr(tool, "_home_channel_source", fallback)
    tokens = _bind(scope="", chat_type="group")
    try:
        out = _call(runner, "join")
    finally:
        clear_session_vars(tokens)
    assert "error" in out
    fallback.assert_not_called()
    adapter.join_voice_channel.assert_not_awaited()


def test_dm_with_two_voice_connections_is_refused(runner):
    adapter = _adapter(connected=True)
    adapter._voice_clients = {GUILD: _vc(), 5005: _vc()}
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind(chat="999", scope="")
    try:
        out = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    assert "error" in out and "more than one server" in out["error"]


def test_dm_with_numeric_scope_still_uses_dm_rules(runner):
    """A DM session carrying a numeric scope id must not act on that id as a guild."""
    adapter = _adapter(connected=True)
    adapter._voice_clients = {GUILD: _vc(), 5005: _vc()}
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind(chat="999", scope=str(GUILD), chat_type="dm")
    try:
        out = _call(runner, "leave")
    finally:
        clear_session_vars(tokens)
    assert "error" in out and "more than one server" in out["error"]
    adapter.leave_voice_channel.assert_not_awaited()


def test_failed_join_with_older_connection_is_not_success(runner):
    """Already connected, but the requester is not in voice: the join path returns an error message
    and writes no binding, so the tool must not report success from the old connection."""
    adapter = _adapter(connected=True)
    old = SessionSource(platform=Platform.DISCORD, chat_id=TEXT, chat_type="group", user_id=USER).to_dict()
    adapter._voice_sources[GUILD] = old
    adapter.get_user_voice_channel = AsyncMock(return_value=None)
    runner.adapters[Platform.DISCORD] = adapter
    tokens = _bind()
    try:
        out = _call(runner, "join")
    finally:
        clear_session_vars(tokens)
    assert out["success"] is False
    assert adapter._voice_sources[GUILD] is old


def test_timeout_cancels_the_pending_action(runner, monkeypatch):
    import tools.discord_voice_tool as tool
    started, cancelled = threading.Event(), threading.Event()

    async def _hang(source, guild_id):
        started.set()
        try:
            await asyncio.Event().wait()  # never set: hangs until the tool cancels it
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "late"

    adapter = _adapter(connected=False)
    runner.adapters[Platform.DISCORD] = adapter
    runner._voice_channel_join_for_source = _hang
    monkeypatch.setattr(tool, "_TIMEOUT_S", 0.5)
    tokens = _bind()
    try:
        out = _call(runner, "join")
    finally:
        clear_session_vars(tokens)
    assert started.is_set() and "timed out" in out["error"]
    assert cancelled.wait(5)


def test_unknown_action():
    import tools.discord_voice_tool as tool
    assert "error" in json.loads(tool.discord_voice("dance"))


def test_tool_registered_in_discord_toolset():
    import tools.discord_voice_tool
    from toolsets import resolve_toolset
    assert "discord_voice" in resolve_toolset("discord")
