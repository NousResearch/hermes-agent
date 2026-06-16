"""Tests for mirroring goal-completion verdicts to the Discord home channel.

Issue #47191: ``/goal`` completion only echoed the "Goal achieved" line
back to the chat the goal was set from. Cron jobs already have a home
channel convention (``DISCORD_HOME_CHANNEL`` / ``_home_target_env_var``)
that delivers regardless of where a job was triggered from; this extends
the same convention to goal completion.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_source(platform=Platform.TELEGRAM, chat_id="c1") -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


def _make_runner_with_adapters(session_id: str = None, *, with_discord_adapter: bool = True):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    src = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    telegram_adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = telegram_adapter

    discord_adapter = None
    if with_discord_adapter:
        discord_adapter = _RecordingAdapter()
        runner.adapters[Platform.DISCORD] = discord_adapter

    return runner, telegram_adapter, discord_adapter, session_entry, src


@pytest.mark.asyncio
async def test_goal_done_mirrors_to_discord_home_channel(hermes_home, monkeypatch):
    """A goal completed from a non-Discord chat must also notify the
    configured Discord home channel."""
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan-1")

    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters()

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1
    assert len(discord_adapter.sends) == 1
    notice = discord_adapter.sends[0]
    assert notice["chat_id"] == "home-chan-1"
    assert "Goal achieved" in notice["content"]


@pytest.mark.asyncio
async def test_goal_done_skips_home_channel_when_unset(hermes_home, monkeypatch):
    """No DISCORD_HOME_CHANNEL configured -> no extra send, no crash."""
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters()

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1
    assert discord_adapter.sends == []


@pytest.mark.asyncio
async def test_goal_done_in_discord_home_channel_is_not_duplicated(hermes_home, monkeypatch):
    """Goal already running in the Discord home channel itself must not
    receive a second, duplicate notice."""
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan-1")

    runner, _telegram_adapter, discord_adapter, session_entry, _src = _make_runner_with_adapters()

    discord_source = _make_source(platform=Platform.DISCORD, chat_id="home-chan-1")

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=discord_source,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(discord_adapter.sends) == 1


@pytest.mark.asyncio
async def test_goal_done_home_channel_survives_missing_discord_adapter(hermes_home, monkeypatch):
    """DISCORD_HOME_CHANNEL set but no Discord gateway connected must not
    crash the judge hook."""
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan-1")

    runner, telegram_adapter, _discord_adapter, session_entry, src = _make_runner_with_adapters(
        with_discord_adapter=False
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1


@pytest.mark.asyncio
async def test_goal_continue_does_not_notify_home_channel(hermes_home, monkeypatch):
    """A non-terminal verdict ("continue") must not trigger a home-channel
    notice - only a "done" verdict should."""
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan-1")

    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters()

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="partial edit",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1
    assert discord_adapter.sends == []