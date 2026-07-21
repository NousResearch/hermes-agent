from unittest.mock import MagicMock
from typing import Any

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    session_id = "sid-gateway-goal-config"


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-config"


@pytest.mark.asyncio
async def test_gateway_goal_uses_goals_max_turns_from_full_config(tmp_path, monkeypatch):
    """Gateway /goal should honor top-level goals.max_turns from config.yaml."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal ship the benchmark",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-config",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_resume_enqueues_the_next_turn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    adapter = object()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._queued_events = {}
    runner._session_key_for_source = lambda _source: "quick-resume"
    runner._enqueue_fifo = MagicMock()

    manager = goals.GoalManager(_FakeSessionEntry.session_id)
    manager.set("resume this standing goal")
    manager.pause("test")
    event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-resume",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "Goal resumed" in response
        runner._enqueue_fifo.assert_called_once()
        assert runner._enqueue_fifo.call_args is not None
        queued_event = runner._enqueue_fifo.call_args.args[1]
        assert "Continue working toward this goal" in queued_event.text
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_resume_releases_claim_when_enqueue_fails(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {Platform.DISCORD: object()}
    runner._queued_events = {}
    runner._session_key_for_source = lambda _source: "quick-resume"
    runner._enqueue_fifo = MagicMock(side_effect=RuntimeError("fifo unavailable"))

    manager = goals.GoalManager(_FakeSessionEntry.session_id)
    manager.set("reconcile this work")
    manager.pause("test")
    monkeypatch.setattr(
        goals,
        "prepare_goal_resume",
        lambda *_a, **_k: {
            "prompt": "internal reconciliation",
            "reconciliation_claim": "recon-gateway",
            "reconciliation_attempt": 3,
            "goal_id": "goal-id",
            "goal_session_id": _FakeSessionEntry.session_id,
            "delegation_ids": ["deleg-a"],
        },
    )
    release = MagicMock(return_value={"released": True})
    monkeypatch.setattr(goals, "release_goal_reconciliation_turn", release)
    event = MessageEvent(
        text="/goal resume",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-resume-failure",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "Goal resumed" in response
        release.assert_called_once_with(
            "recon-gateway",
            session_id=_FakeSessionEntry.session_id,
            goal_id="goal-id",
            attempt=3,
            turn_started=False,
        )
    finally:
        goals._DB_CACHE.clear()
