"""User-originated turn clones retain authenticated completion-gate disposition."""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_STATE = {
    "completion_gate_source_state": {
        "request_id": "1789000000.123456",
        "status": "deferred_quiet_hours",
        "card_id": "t_ab12cd34",
        "destination": "telegram:123456789",
    }
}


def _event(text):
    source = SessionSource(platform=Platform.SLACK, chat_id="C123", user_id="U123")
    return MessageEvent(
        text=text, message_type=MessageType.COMMAND, source=source,
        message_id="1789000000.123456", metadata=dict(_STATE),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["queue", "steer"])
async def test_busy_followup_turns_preserve_source_state(command, monkeypatch):
    runner = GatewayRunner(config=GatewayConfig())
    event = _event(f"/{command} continue")
    queued = []
    adapter = object()
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_enqueue_fifo",
                        lambda key, turn, selected: queued.append((key, turn, selected)))
    monkeypatch.setattr(runner, "_queue_depth", lambda *args, **kwargs: 1)
    if command == "queue":
        await runner._busy_queue_command(event, "key", event.source)
    else:
        monkeypatch.setattr(runner, "_peek_session_state", lambda key: None)
        await runner._busy_steer_command(event, "key", event.source)
    assert len(queued) == 1
    assert queued[0][1].metadata == _STATE
    assert queued[0][1].metadata is not event.metadata


def test_goal_kickoff_preserves_source_state_but_resume_does_not(monkeypatch):
    runner = GatewayRunner(config=GatewayConfig())
    event = _event("/goal build it")
    queued = []
    adapter = object()
    monkeypatch.setattr(runner, "_adapter_and_key_for", lambda incoming: (adapter, "key"))
    monkeypatch.setattr(runner, "_enqueue_fifo", lambda key, turn, selected: queued.append(turn))

    runner._enqueue_goal_turn(event, "build it", label="kickoff", kickoff=True)
    runner._enqueue_goal_turn(event, "continue", label="resume", kickoff=False)

    assert queued[0].metadata == _STATE
    assert queued[0].metadata is not event.metadata
    assert queued[1].metadata == {}
