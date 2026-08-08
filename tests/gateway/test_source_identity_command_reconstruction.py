from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SOURCE_IDENTITY_AMBIGUOUS,
    trusted_source_message_id,
)
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource
from hermes_cli import goals


def _source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


def _ambiguous_event(text: str, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(platform),
        message_id="first-message-id",
        metadata={SOURCE_IDENTITY_AMBIGUOUS: True, "keep": "value"},
    )


def _busy_runner(state=None):
    runner = object.__new__(GatewayRunner)
    adapter = object()
    captured = []
    runner._adapter_for_source = lambda _source: adapter
    runner._enqueue_fifo = lambda _key, event, _adapter: captured.append(event)
    runner._queue_depth = lambda _key, adapter=None: len(captured)
    runner._peek_session_state = lambda _key: state
    return runner, captured


@pytest.mark.asyncio
async def test_queue_reconstruction_preserves_ambiguous_source_metadata():
    runner, captured = _busy_runner()
    event = _ambiguous_event("/queue do the work")

    await runner._busy_queue_command(event, "session-key", event.source)

    assert len(captured) == 1
    assert captured[0].metadata == event.metadata
    assert captured[0].metadata is not event.metadata
    assert trusted_source_message_id(captured[0]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        None,
        SimpleNamespace(turn=SimpleNamespace(agent=_AGENT_PENDING_SENTINEL)),
    ],
)
async def test_steer_reconstruction_preserves_ambiguous_source_metadata(state):
    runner, captured = _busy_runner(state)
    event = _ambiguous_event("/steer do the work")

    await runner._busy_steer_command(event, "session-key", event.source)

    assert len(captured) == 1
    assert captured[0].metadata == event.metadata
    assert captured[0].metadata is not event.metadata
    assert trusted_source_message_id(captured[0]) is None


class _GoalSessionStore:
    entry = SimpleNamespace(session_id="source-identity-goal-session")

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:dm:chat-1"


@pytest.mark.asyncio
async def test_goal_kickoff_preserves_ambiguous_source_metadata(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="token"),
        }
    )
    runner.session_store = _GoalSessionStore()
    runner.adapters = {Platform.DISCORD: object()}
    captured = []
    runner._enqueue_fifo = lambda _key, event, _adapter: captured.append(event)

    event = _ambiguous_event("/goal ship safely", Platform.DISCORD)
    try:
        await GatewayRunner._handle_goal_command(runner, event)

        assert len(captured) == 1
        assert captured[0].metadata == event.metadata
        assert captured[0].metadata is not event.metadata
        assert trusted_source_message_id(captured[0]) is None
    finally:
        goals._DB_CACHE.clear()
