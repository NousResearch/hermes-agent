"""A recipient-scoped handoff cannot return an inaccessible Discord thread."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from plugins.platforms.discord import adapter as discord_adapter


@pytest.fixture
def handoff(monkeypatch):
    monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(discord_adapter, "discord", SimpleNamespace(DMChannel=type("DMChannel", (), {})))
    thread = SimpleNamespace(id=300, add_user=AsyncMock(), fetch_member=AsyncMock(return_value=SimpleNamespace(id=200)))
    parent = SimpleNamespace(create_thread=AsyncMock(return_value=thread), send=AsyncMock())
    adapter = object.__new__(discord_adapter.DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._client = SimpleNamespace(get_channel=Mock(return_value=parent))
    return adapter, parent, thread


def test_handoff_adds_recipient_and_reads_membership_before_returning(handoff):
    adapter, parent, thread = handoff
    events = []

    async def add(user):
        assert user.id == 200
        events.append("add")

    async def fetch(user_id):
        assert user_id == 200
        events.append("read")
        return SimpleNamespace(id=user_id)

    thread.add_user.side_effect = add
    thread.fetch_member.side_effect = fetch
    result = asyncio.run(adapter.create_handoff_thread("100", "Lesson", recipient_user_id="200"))
    assert result == "300"
    assert events == ["add", "read"]
    parent.send.assert_not_called()


@pytest.mark.parametrize("failure", ["invite", "read", "missing", "wrong_member"])
def test_handoff_membership_failure_never_returns_thread_or_creates_public_fallback(handoff, failure, caplog):
    adapter, parent, thread = handoff
    if failure == "invite":
        thread.add_user.side_effect = PermissionError("cannot invite")
    elif failure == "read":
        thread.fetch_member.side_effect = PermissionError("cannot read")
    elif failure == "missing":
        thread.fetch_member.return_value = None
    else:
        thread.fetch_member.return_value = SimpleNamespace(id=201)
    assert asyncio.run(adapter.create_handoff_thread("100", "Lesson", recipient_user_id="200")) is None
    assert "membership not verified" in caplog.text
    parent.send.assert_not_called()


def test_seed_message_fallback_also_requires_verified_membership(handoff):
    adapter, parent, thread = handoff
    parent.create_thread.side_effect = PermissionError("no direct threads")
    seed = SimpleNamespace(create_thread=AsyncMock(return_value=thread))
    parent.send.return_value = seed
    thread.fetch_member.side_effect = PermissionError("not a member")

    assert asyncio.run(adapter.create_handoff_thread("100", "Lesson", recipient_user_id="200")) is None
    thread.add_user.assert_awaited_once()
    thread.fetch_member.assert_awaited_once_with(200)


@pytest.mark.parametrize("outcome", ["verified", "invite_failure", "read_failure", "missing_origin", "fallback_failed"])
def test_cron_delivery_uses_thread_only_after_membership_readback(handoff, monkeypatch, outcome):
    from concurrent.futures import Future
    from cron.scheduler import _deliver_result
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.base import SendResult

    adapter, parent, thread = handoff
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True)})
    adapter.config = config.platforms[Platform.DISCORD]
    adapter._session_store = None
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    events = []

    async def add(user):
        events.append("add")
        if outcome == "invite_failure":
            raise PermissionError("cannot invite")

    async def read(user_id):
        events.append("read")
        if outcome in {"read_failure", "fallback_failed"}:
            raise PermissionError("cannot verify membership")
        return SimpleNamespace(id=user_id)

    async def send(chat_id, content, **kwargs):
        events.append("send")
        assert content == "Lesson payload"
        metadata = kwargs.get("metadata") or {}
        assert metadata.get("thread_id") == ("300" if outcome == "verified" else None)
        if outcome == "fallback_failed":
            return SendResult(success=False, error="channel unavailable")
        return SendResult(success=True, message_id="400")

    def schedule(coro, loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    thread.add_user.side_effect = add
    thread.fetch_member.side_effect = read
    adapter.send = AsyncMock(side_effect=send)
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", schedule)
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: True)
    monkeypatch.setattr("cron.scheduler_delivery._standalone_send",
                        lambda *a: (None, "channel unavailable"))
    job = {"id": "lesson", "name": "Lesson", "deliver": "discord:101", "attach_to_session": True,
           "origin": {"platform": "discord", "chat_id": "100"}}
    if outcome != "missing_origin":
        job["origin"]["user_id"] = "200"
    error = _deliver_result(job, "Lesson payload", adapters={Platform.DISCORD: adapter},
                            loop=SimpleNamespace(is_running=lambda: True))
    if outcome == "fallback_failed":
        assert error is not None
        assert "channel unavailable" in error
    else:
        assert error is None
    assert events == {"verified": ["add", "read", "send"], "invite_failure": ["add", "send"],
                      "read_failure": ["add", "read", "send"],
                      "fallback_failed": ["add", "read", "send"],
                      "missing_origin": ["send"]}[outcome]
    if outcome == "missing_origin":
        parent.create_thread.assert_not_called()


def test_legacy_handoff_call_remains_two_argument_compatible(handoff):
    adapter, parent, thread = handoff
    assert asyncio.run(adapter.create_handoff_thread("100", "Branch")) == "300"
    thread.add_user.assert_not_called()
    thread.fetch_member.assert_not_called()
