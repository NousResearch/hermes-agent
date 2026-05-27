"""Regression tests for durable gateway inbound turn persistence."""

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        user_id="user-1",
    )


def _make_runner(tmp_path):
    config = GatewayConfig(sessions_dir=tmp_path)
    store = SessionStore(sessions_dir=tmp_path, config=config)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    return runner, store


def test_pre_persist_inbound_turn_survives_before_agent_finishes(tmp_path):
    runner, store = _make_runner(tmp_path)
    source = _make_source()
    entry = store.get_or_create_session(source)
    event = MessageEvent(
        text="write the incident report",
        message_type=MessageType.TEXT,
        source=source,
        message_id="discord-msg-42",
    )

    persisted = runner._pre_persist_inbound_turn(
        session_id=entry.session_id,
        message_text=event.text,
        event=event,
        history=[],
        model="test-model",
    )

    assert persisted.user_message_id is not None
    messages = store.load_transcript(entry.session_id)
    assert [m["role"] for m in messages] == ["session_meta", "user"]
    assert messages[-1]["content"] == "write the incident report"
    assert messages[-1]["message_id"] == "discord-msg-42"


def test_pre_persisted_inbound_turn_can_be_removed_after_success(tmp_path):
    runner, store = _make_runner(tmp_path)
    source = _make_source()
    entry = store.get_or_create_session(source)
    event = MessageEvent(text="hello", source=source, message_id="m1")

    persisted = runner._pre_persist_inbound_turn(
        session_id=entry.session_id,
        message_text=event.text,
        event=event,
        history=[{"role": "user", "content": "prior"}],
        model="test-model",
    )
    assert len(store.load_transcript(entry.session_id)) == 1

    runner._remove_pre_persisted_inbound_turn(persisted)

    assert store.load_transcript(entry.session_id) == []
