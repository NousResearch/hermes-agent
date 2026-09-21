"""A delivered webhook answer is written into the chat that received it.

A webhook event runs in its own conversation (``history=0``), so without a mirror
the receiving-side agent has no record of what it sent and a follow-up like "make
that shorter" has no referent. ``send_message`` and cron already mirror; this
covers cross-platform webhook delivery.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


class _Runner:
    def __init__(self, adapter):
        self.adapters = {Platform.TELEGRAM: adapter}
        self._profile_adapters = {}
        self._primary_profile_name = "default"
        self.config = GatewayConfig()

    def _authorization_adapter(self, platform, profile):
        return self.adapters.get(platform)


def _webhook(send_result):
    target = MagicMock()
    target.send = AsyncMock(return_value=send_result)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": {}}))
    adapter.gateway_runner = _Runner(target)
    return adapter, target


@pytest.fixture
def mirrored(monkeypatch):
    calls = []
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: calls.append((a, kw)) or True)
    return calls


@pytest.mark.asyncio
async def test_successful_delivery_is_mirrored_as_user_role_with_a_label(mirrored):
    adapter, target = _webhook(SendResult(success=True))
    result = await adapter._deliver_cross_platform(
        "telegram", "the draft", {"deliver_extra": {"chat_id": "chat-1"}})

    assert result.success
    assert target.send.await_count == 1
    (args, kwargs) = mirrored[0]
    assert args[0] == "telegram"
    assert args[1] == "chat-1"
    assert args[2].endswith("the draft")
    assert "not an instruction" in args[2]
    # assistant-role mirrors replay as real turns and break strict alternation (#2221)
    assert kwargs["role"] == "user"
    # the turn's user is webhook:<route>, which matches no session in the target chat
    assert kwargs["user_id"] is None
    assert kwargs["source_label"] == "webhook"
    # "" selects the chat's main conversation; None would match any reply thread
    assert kwargs["thread_id"] == ""


@pytest.mark.asyncio
async def test_a_failed_send_is_not_mirrored(mirrored):
    adapter, _ = _webhook(SendResult(success=False, error="nope"))
    result = await adapter._deliver_cross_platform(
        "telegram", "never arrived", {"deliver_extra": {"chat_id": "chat-1"}})

    assert not result.success
    assert mirrored == []


@pytest.mark.asyncio
async def test_thread_id_is_carried_so_the_mirror_lands_in_the_same_thread(mirrored):
    adapter, _ = _webhook(SendResult(success=True))
    await adapter._deliver_cross_platform(
        "telegram", "in a topic", {"deliver_extra": {"chat_id": "chat-1", "message_thread_id": "77"}})

    assert mirrored[0][1]["thread_id"] == "77"


@pytest.mark.asyncio
async def test_long_content_is_capped(mirrored):
    adapter, _ = _webhook(SendResult(success=True))
    await adapter._deliver_cross_platform(
        "telegram", "x" * 9000, {"deliver_extra": {"chat_id": "chat-1"}})

    assert "[...truncated]" in mirrored[0][0][2]
    assert len(mirrored[0][0][2]) < 4300


@pytest.mark.asyncio
async def test_a_mirror_that_fails_never_fails_a_delivered_send(monkeypatch):
    """The message is already out; failing the send afterwards helps nobody."""
    def boom(*a, **kw):
        raise RuntimeError("state.db is locked")
    monkeypatch.setattr("gateway.mirror.mirror_to_session", boom)
    adapter, _ = _webhook(SendResult(success=True))

    result = await adapter._deliver_cross_platform(
        "telegram", "delivered anyway", {"deliver_extra": {"chat_id": "chat-1"}})

    assert result.success


def test_empty_thread_id_selects_the_main_conversation_over_a_newer_thread(tmp_path):
    """The bug this guards: a DM that has ever had a reply thread.

    With thread_id=None the lookup applies no thread filter and returns the most
    recently started live session, which here is a stale reply thread. The mirror
    then lands where nobody is talking and the next message in the main
    conversation cannot see it.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("main", "buzz", user_id="u1", session_key="agent:main:buzz:dm:c1",
                      chat_id="c1", chat_type="dm")
    db.create_session("thread", "buzz", user_id="u1", session_key="agent:main:buzz:dm:c1:t9",
                      chat_id="c1", chat_type="dm", thread_id="t9")  # started later

    assert db.find_session_by_origin(platform="buzz", chat_id="c1") == "thread"  # the old trap
    assert db.find_session_by_origin(platform="buzz", chat_id="c1", thread_id="") == "main"
    assert db.find_session_by_origin(platform="buzz", chat_id="c1", thread_id="t9") == "thread"
