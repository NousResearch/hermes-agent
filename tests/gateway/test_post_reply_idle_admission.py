"""Accepted inbound must fence a pending summary before the next turn begins."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn import GatewayTurnMixin
from gateway.session_identity import RoutingIdentity
from gateway.session import SessionSource
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_ended_route_is_allowed_to_auto_reset(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("old", source="gateway", session_key="chat")
    db.end_session("old", end_reason="reset")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "compression": {"post_reply_idle": {"channels": [
            {"platform": "signal", "chat_id": "chat-id", "after_seconds": 300}
        ]}}
    })
    source = SessionSource(platform=Platform.SIGNAL, chat_id="chat-id", chat_type="group")
    source._identity = RoutingIdentity("default", "default", tmp_path, tmp_path, multiplexed=False)
    event = MessageEvent(text="new turn", message_type=MessageType.TEXT, source=source)
    runner = object.__new__(GatewayTurnMixin)
    runner._session_db = db
    runner._profile_scope_for_source = lambda _: nullcontext()
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda _: SimpleNamespace(session_id="old"))
    try:
        await runner._invalidate_post_reply_idle_for_turn(event, source, "chat")
        assert not hasattr(event, "_post_reply_idle_generation")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_invalid_idle_policy_does_not_block_inbound(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "compression": {"post_reply_idle": {"channels": [{"platform": "signal"}]}}
    })
    source = SessionSource(platform=Platform.SIGNAL, chat_id="chat-id", chat_type="group")
    source._identity = RoutingIdentity("default", "default", tmp_path, tmp_path, multiplexed=False)
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)
    runner = object.__new__(GatewayTurnMixin)
    runner._profile_scope_for_source = lambda _: nullcontext()
    await runner._invalidate_post_reply_idle_for_turn(event, source, "chat")
    assert not hasattr(event, "_post_reply_idle_generation")


@pytest.mark.asyncio
async def test_stale_reply_does_not_arm_after_followup(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="gateway", session_key="chat")
    config = {"compression": {"post_reply_idle": {"channels": [
        {"platform": "signal", "chat_id": "chat-id", "after_seconds": 300}
    ]}}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    source = SessionSource(platform=Platform.SIGNAL, chat_id="chat-id", chat_type="group")
    source._identity = RoutingIdentity("default", "default", tmp_path, tmp_path, multiplexed=False)
    event = MessageEvent(text="reply", message_type=MessageType.TEXT, source=source)
    runner = object.__new__(GatewayTurnMixin)
    runner._session_db = db
    runner._profile_scope_for_source = lambda _: nullcontext()
    event._post_reply_session_id = "s"
    try:
        event._post_reply_idle_generation = db.invalidate_post_reply_idle("s")
        db.invalidate_post_reply_idle("s")  # the next message was admitted before delivery completed
        assert not await runner._arm_post_reply_idle(event, "chat")
        assert db.claim_due_post_reply_idle("idle", now=10**11) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_inbound_invalidates_claimed_idle_summary_before_turn(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="gateway", session_key="chat")
    db.append_message("s", role="user", content="hi")
    db.append_message("s", role="assistant", content="hello")
    config = {"compression": {"post_reply_idle": {"channels": [
        {"platform": "signal", "chat_id": "chat-id", "after_seconds": 300}
    ]}}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    source = SessionSource(platform=Platform.SIGNAL, chat_id="chat-id", chat_type="group")
    source._identity = RoutingIdentity("default", "default", tmp_path, tmp_path, multiplexed=False)
    event = MessageEvent(text="follow up", message_type=MessageType.TEXT, source=source)
    runner = object.__new__(GatewayTurnMixin)
    runner._session_db = db
    runner._profile_scope_for_source = lambda _: nullcontext()
    runner._session_key_for_source = lambda _: "chat"
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda _: SimpleNamespace(session_id="s"))
    try:
        gen = db.invalidate_post_reply_idle("s")
        assert db.arm_post_reply_idle("s", "chat", 1, expected_generation=gen)
        assert db.claim_due_post_reply_idle("idle", now=2) is not None
        await runner._invalidate_post_reply_idle_for_turn(event, source, "chat")
        assert event._post_reply_idle_generation > gen
        with pytest.raises(ValueError, match="idle compaction fence"):
            db.archive_and_compact("s", [{"role": "assistant", "content": "stale"}],
                                   idle_claim=(gen, 2, "idle"))
    finally:
        db.close()
