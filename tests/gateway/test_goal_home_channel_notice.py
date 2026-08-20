"""Tests for mirroring goal-completion verdicts to the Discord home channel.

Issue #47191: ``/goal`` completion only echoed the "Goal achieved" line
back to the chat the goal was set from. Cron jobs already have a home
channel convention (``GatewayConfig.get_home_channel()``) that delivers
regardless of where a job was triggered from; this extends the same
convention to goal completion.

The home channel is resolved from the running gateway's config
(``self.config.get_home_channel(Platform.DISCORD)``) rather than only an
env var, so config-only targets (no matching env var set) are covered.
Dedup against the source chat compares platform, chat_id, AND thread_id,
so a different thread inside the same Discord channel as the home
channel still receives the notice.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
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


def _make_source(platform=Platform.TELEGRAM, chat_id="c1", thread_id=None) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
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


def _make_runner_with_adapters(
    session_id: str = None,
    *,
    with_discord_adapter: bool = True,
    home_channel: HomeChannel = None,
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***", home_channel=home_channel),
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
async def test_goal_done_mirrors_to_discord_home_channel(hermes_home):
    """A goal completed from a non-Discord chat must also notify the
    configured Discord home channel. No env var is set - the home channel
    comes entirely from ``GatewayConfig``, matching how a real running
    gateway resolves it."""
    home_channel = HomeChannel(platform=Platform.DISCORD, chat_id="home-chan-1", name="Home")

    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters(
        home_channel=home_channel
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None)):
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
async def test_goal_done_skips_home_channel_when_unset(hermes_home):
    """No Discord home channel configured -> no extra send, no crash."""
    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters(
        home_channel=None
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False, None)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1
    assert discord_adapter.sends == []


@pytest.mark.asyncio
async def test_goal_done_in_discord_home_channel_is_not_duplicated(hermes_home):
    """Goal already running in the Discord home channel itself (same
    platform, chat_id, AND thread_id) must not receive a second, duplicate
    notice."""
    home_channel = HomeChannel(platform=Platform.DISCORD, chat_id="home-chan-1", name="Home")

    runner, _telegram_adapter, discord_adapter, session_entry, _src = _make_runner_with_adapters(
        home_channel=home_channel
    )

    discord_source = _make_source(platform=Platform.DISCORD, chat_id="home-chan-1", thread_id=None)

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False, None)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=discord_source,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(discord_adapter.sends) == 1


@pytest.mark.asyncio
async def test_goal_done_distinct_thread_same_channel_is_not_deduplicated(hermes_home):
    """A goal completed in a DIFFERENT thread of the same Discord channel
    as the configured home thread must still receive the home-channel
    notice - matching only on chat_id would incorrectly suppress it."""
    home_channel = HomeChannel(
        platform=Platform.DISCORD, chat_id="shared-chan-1", name="Home", thread_id="home-thread-1"
    )

    runner, _telegram_adapter, discord_adapter, session_entry, _src = _make_runner_with_adapters(
        home_channel=home_channel
    )

    other_thread_source = _make_source(
        platform=Platform.DISCORD, chat_id="shared-chan-1", thread_id="other-thread-2"
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False, None)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=other_thread_source,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    # The source's own send plus the mirrored home-channel notice.
    assert len(discord_adapter.sends) == 2
    home_sends = [s for s in discord_adapter.sends if s["chat_id"] == "shared-chan-1"]
    assert len(home_sends) == 2


@pytest.mark.asyncio
async def test_goal_done_home_channel_survives_missing_discord_adapter(hermes_home):
    """Home channel configured but no Discord gateway connected must not
    crash the judge hook."""
    home_channel = HomeChannel(platform=Platform.DISCORD, chat_id="home-chan-1", name="Home")

    runner, telegram_adapter, _discord_adapter, session_entry, src = _make_runner_with_adapters(
        with_discord_adapter=False, home_channel=home_channel
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "shipped", False, None)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="done.",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1


@pytest.mark.asyncio
async def test_goal_continue_does_not_notify_home_channel(hermes_home):
    """A non-terminal verdict ("continue") must not trigger a home-channel
    notice - only a "done" verdict should."""
    home_channel = HomeChannel(platform=Platform.DISCORD, chat_id="home-chan-1", name="Home")

    runner, telegram_adapter, discord_adapter, session_entry, src = _make_runner_with_adapters(
        home_channel=home_channel
    )

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False, None)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="partial edit",
        )
        await asyncio.sleep(0.05)

    assert len(telegram_adapter.sends) == 1
    assert discord_adapter.sends == []
