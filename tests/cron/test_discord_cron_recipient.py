"""Continuable Discord cron threads carry explicit scheduling-user provenance."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cron.scheduler_delivery import _open_continuable_cron_thread


def _schedule(coro, loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except Exception as exc:
        future.set_exception(exc)
    return future


def test_discord_thread_receives_origin_user_for_explicit_destination(monkeypatch):
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", _schedule)
    adapter = SimpleNamespace(name="discord", create_handoff_thread=AsyncMock(return_value="300"))
    job = {"id": "lesson", "origin": {"platform": "discord", "chat_id": "100", "user_id": "200"}}

    assert _open_continuable_cron_thread(job, adapter, "101", object()) == "300"
    assert adapter.create_handoff_thread.call_args.kwargs == {"recipient_user_id": "200"}


@pytest.mark.parametrize("origin", [
    None, "legacy", {}, {"platform": "slack", "chat_id": "100", "user_id": "200"},
    {"platform": "discord", "chat_id": "100"},
    *({"platform": "discord", "chat_id": "100", "user_id": value}
      for value in ["system:cron", True, 0, -1, " 200", "٢٠٠", {}, [200]]),
])
def test_discord_without_explicit_recipient_falls_back_before_thread_creation(monkeypatch, origin, caplog):
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", _schedule)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "200,201")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "202")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "202")
    adapter = SimpleNamespace(name="discord", create_handoff_thread=AsyncMock(return_value="300"))

    assert _open_continuable_cron_thread({"id": "lesson", "origin": origin}, adapter, "101", object()) is None
    adapter.create_handoff_thread.assert_not_called()
    assert "recipient" in caplog.text.lower()


def test_discord_destination_cannot_bypass_recipient_guard_with_transport_name(monkeypatch):
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", _schedule)
    adapter = SimpleNamespace(name="relay", create_handoff_thread=AsyncMock(return_value="300"))
    assert _open_continuable_cron_thread({"id": "lesson"}, adapter, "101", object(), platform_name="discord") is None
    adapter.create_handoff_thread.assert_not_called()


@pytest.mark.parametrize("platform", ["slack", "matrix", "telegram"])
def test_other_platform_handoff_contract_remains_two_arguments(monkeypatch, platform):
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", _schedule)

    async def legacy_create(chat_id, name):
        assert chat_id == "101"
        return "300"

    adapter = SimpleNamespace(name=platform, create_handoff_thread=legacy_create)
    assert _open_continuable_cron_thread({"id": "lesson"}, adapter, "101", object(), platform_name=platform) == "300"
