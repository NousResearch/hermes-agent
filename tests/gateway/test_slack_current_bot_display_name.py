"""Opt-in Slack sender presentation is separate from installed routing identity."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter, _slack_mod


class AuthResponse(dict):
    def __init__(self, *, scope=True):
        super().__init__(team_id="T1", user_id="U1", user="oldhandle", bot_id="B1")
        self.headers = {"X-OAuth-Scopes": "chat:write,chat:write.customize" if scope else "chat:write"}


async def make_adapter(monkeypatch, *, enabled=True, scope=True, lookup_error=None):
    config = PlatformConfig(enabled=True, extra={"use_current_bot_display_name": enabled})
    adapter = SlackAdapter(config)
    client = SimpleNamespace(
        auth_test=AsyncMock(return_value=AuthResponse(scope=scope)),
        bots_info=AsyncMock(return_value={"bot": {"name": "Current Name"}}, side_effect=lookup_error),
        chat_postMessage=AsyncMock(return_value={"ts": "2.0"}),
        chat_startStream=AsyncMock(return_value={"ts": "2.0"}),
        assistant_threads_setStatus=AsyncMock(),
        agents_sessions_setStatus=AsyncMock(),
    )
    adapter._app = SimpleNamespace(client=MagicMock())
    monkeypatch.setattr(adapter, "_new_web_client", lambda *_: client)
    await adapter._authenticate_workspace("unused", None)
    return adapter, client


@pytest.mark.asyncio
async def test_reply_uses_current_name_without_changing_routing_handle(monkeypatch):
    adapter, client = await make_adapter(monkeypatch)
    await adapter._post_chunks("D1", "T1", "Test", "Test", "1.0")
    client.bots_info.assert_awaited_once_with(bot="B1")
    assert client.chat_postMessage.await_args.kwargs["username"] == "Current Name"
    assert adapter._team_bot_names["T1"] == "oldhandle"
    assert adapter._team_bot_user_ids["T1"] == "U1"


@pytest.mark.asyncio
async def test_stream_uses_current_name(monkeypatch):
    adapter, client = await make_adapter(monkeypatch)
    await adapter._start_stream(client, ("T1", "D1", "1.0"), 1, "Test", {})
    assert client.chat_startStream.await_args.kwargs["username"] == "Current Name"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sessions", [False, True])
async def test_status_uses_current_name_only_when_setting(monkeypatch, use_sessions):
    adapter, client = await make_adapter(monkeypatch)
    monkeypatch.setattr(_slack_mod, "_AGENT_SESSIONS_SUPPORTED", use_sessions)
    setter = client.agents_sessions_setStatus if use_sessions else client.assistant_threads_setStatus
    await adapter._set_thread_status("D1", "T1", "1.0", "is working", "set")
    assert setter.await_args.kwargs["username"] == "Current Name"
    await adapter._set_thread_status("D1", "T1", "1.0", "", "clear")
    assert "username" not in setter.await_args.kwargs


@pytest.mark.asyncio
async def test_primary_fallback_client_uses_current_name(monkeypatch):
    adapter, _ = await make_adapter(monkeypatch)
    app = adapter._app
    assert app is not None
    post = AsyncMock(return_value={"ts": "2.0"})
    app.client.chat_postMessage = post
    await adapter._post_chunks("D1", "", "Test", "Test", None)
    post.assert_awaited_once_with(channel="D1", text="Test", mrkdwn=True, username="Current Name")


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,scope", [(False, True), (True, False)])
async def test_disabled_or_missing_scope_preserves_delivery(monkeypatch, enabled, scope):
    adapter, client = await make_adapter(monkeypatch, enabled=enabled, scope=scope)
    await adapter._post_chunks("D1", "T1", "Test", "Test", None)
    client.bots_info.assert_not_awaited()
    assert "username" not in client.chat_postMessage.await_args.kwargs


@pytest.mark.asyncio
async def test_block_retry_uses_identity_of_the_client_that_sends(monkeypatch):
    adapter, primary = await make_adapter(monkeypatch)
    primary.chat_postMessage.side_effect = ValueError("invalid_blocks")
    secondary = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ts": "2.0"}))
    clients = iter((primary, secondary))
    monkeypatch.setattr(adapter, "_is_block_payload_rejection", lambda _: True)
    await adapter._call_with_block_fallback(
        lambda: next(clients), "chat_postMessage", {"channel": "D1", "blocks": [object()]}, "send"
    )
    assert primary.chat_postMessage.await_args.kwargs["username"] == "Current Name"
    secondary.chat_postMessage.assert_awaited_once_with(channel="D1")


@pytest.mark.asyncio
async def test_task_card_stream_uses_current_name(monkeypatch):
    adapter, client = await make_adapter(monkeypatch)
    api_call = AsyncMock(return_value={"ts": "2.0"})
    client.api_call = api_call
    stream = _slack_mod._NativeTaskCardStream(team_id="T1", channel="D1", thread_ts="1.0")
    await adapter._start_native_task_card_stream(client, stream, {})
    api_call.assert_awaited_once_with(
        "chat.startStream",
        json={"channel": "D1", "thread_ts": "1.0", "task_display_mode": "plan", "username": "Current Name"},
    )
    assert stream.stream_ts == "2.0"


@pytest.mark.asyncio
async def test_lookup_error_does_not_block_delivery(monkeypatch):
    adapter, client = await make_adapter(monkeypatch, lookup_error=TimeoutError())
    await adapter._post_chunks("D1", "T1", "Test", "Test", None)
    client.bots_info.assert_awaited_once()
    assert "username" not in client.chat_postMessage.await_args.kwargs
