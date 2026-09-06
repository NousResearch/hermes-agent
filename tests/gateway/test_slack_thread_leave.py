"""Behavior contract for Slack thread-scoped ``@bot !leave``."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


TEAM = "T_WORK"
CHANNEL = "C_SHARED"
THREAD = "1700000000.000001"
USER = "U_HUMAN"


def _adapter(tmp_path, monkeypatch, bot_id):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1700000001.000001"}
    )
    adapter._bot_user_id = bot_id
    adapter._team_bot_user_ids[TEAM] = bot_id
    adapter._team_clients[TEAM] = adapter._app.client
    adapter._authorization_check = lambda *_args: True
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="")
    adapter.handle_message = AsyncMock()
    return adapter


def _event(text, ts):
    return {
        "type": "message",
        "text": text,
        "user": USER,
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "thread_ts": THREAD,
        "ts": ts,
        "client_msg_id": f"client-{ts}",
    }


@pytest.mark.asyncio
async def test_leave_targets_only_named_bot_and_mutes_until_rementioned(tmp_path, monkeypatch):
    luna = _adapter(tmp_path / "luna", monkeypatch, "U111LUNA")
    sol = _adapter(tmp_path / "sol", monkeypatch, "U222SOL")
    leave = _event("<@U111LUNA> !leave", "1700000000.000010")

    await luna._handle_slack_message(leave)
    await sol._handle_slack_message(leave)

    luna.handle_message.assert_not_awaited()
    sol.handle_message.assert_not_awaited()
    luna._app.client.chat_postMessage.assert_awaited_once()
    sol._app.client.chat_postMessage.assert_not_awaited()

    await luna._handle_slack_message(_event("ordinary follow-up", "1700000000.000011"))
    luna.handle_message.assert_not_awaited()

    # The mute is profile-scoped durable state, not an in-memory accident.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "luna"))
    restarted_luna = _adapter(tmp_path / "luna", monkeypatch, "U111LUNA")
    await restarted_luna._handle_slack_message(
        _event("follow-up after restart", "1700000000.000012")
    )
    restarted_luna.handle_message.assert_not_awaited()

    await restarted_luna._handle_slack_message(
        _event("<@U111LUNA> come back", "1700000000.000013")
    )
    restarted_luna.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_leave_accepts_labeled_slack_mention(tmp_path, monkeypatch):
    luna = _adapter(tmp_path / "luna", monkeypatch, "U111LUNA")
    await luna._handle_slack_message(
        _event("<@U111LUNA|luna> !leave", "1700000000.000020")
    )
    luna.handle_message.assert_not_awaited()
    luna._app.client.chat_postMessage.assert_awaited_once()
    await luna._handle_slack_message(_event("ordinary follow-up", "1700000000.000021"))
    luna.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_leave_accepts_display_name_mention_pattern(tmp_path, monkeypatch):
    luna = _adapter(tmp_path / "luna", monkeypatch, "U111LUNA")
    luna.config.extra = {"mention_patterns": [r"@MacBook-Pro-Work-Luna"]}
    await luna._handle_slack_message(
        _event("@MacBook-Pro-Work-Luna !leave", "1700000000.000030")
    )
    luna.handle_message.assert_not_awaited()
    luna._app.client.chat_postMessage.assert_awaited_once()
    await luna._handle_slack_message(_event("ordinary follow-up", "1700000000.000031"))
    luna.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_bare_leave_mutes_participating_bot_only(tmp_path, monkeypatch):
    luna = _adapter(tmp_path / "luna", monkeypatch, "U111LUNA")
    sol = _adapter(tmp_path / "sol", monkeypatch, "U222SOL")
    sol._has_active_session_for_thread = MagicMock(return_value=False)
    leave = _event("!leave", "1700000000.000040")

    await luna._handle_slack_message(leave)
    await sol._handle_slack_message(leave)

    luna.handle_message.assert_not_awaited()
    sol.handle_message.assert_not_awaited()
    luna._app.client.chat_postMessage.assert_awaited_once()
    sol._app.client.chat_postMessage.assert_not_awaited()
    await luna._handle_slack_message(_event("ordinary follow-up", "1700000000.000041"))
    luna.handle_message.assert_not_awaited()
