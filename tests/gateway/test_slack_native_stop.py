"""Slack's native Stop button (``agent_session_stopped``) acts as ``/stop`` in its thread."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from hermes_cli.slack_cli import _build_full_manifest
from plugins.platforms.slack.adapter_native_stop import handle_agent_session_stopped
from tests.gateway.test_slack import SlackAdapter


class _AgentSessionsClient:
    agents_sessions_setStatus = AsyncMock()
    assistant_threads_setStatus = AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
async def test_native_stop_acts_as_stop_in_its_thread_for_authorized_users_only(authorized):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    client = _AgentSessionsClient()
    client.agents_sessions_setStatus.reset_mock()
    adapter._app = object()
    adapter._get_client = lambda *a, **kw: client
    adapter._authorization_check = lambda *a, **kw: authorized
    adapter.handle_message = AsyncMock()
    event = {"type": "agent_session_stopped", "channel": "C1", "user": "U1",
             "thread_ts": "1700.0001", "streaming_message_ts": [], "event_ts": "1700.0009"}

    await handle_agent_session_stopped(adapter, event, {"team_id": "T1", "event": event})

    if not authorized:
        adapter.handle_message.assert_not_awaited()
        client.agents_sessions_setStatus.assert_not_awaited()
        return
    stop = adapter.handle_message.await_args.args[0]
    assert stop.text == "/stop" and stop.get_command() == "stop"
    assert (stop.source.chat_id, stop.source.thread_id, stop.source.user_id, stop.source.scope_id) == (
        "C1", "1700.0001", "U1", "T1")
    client.agents_sessions_setStatus.assert_any_await(channel_id="C1", thread_ts="1700.0001", status="active")


def test_agent_manifest_subscribes_to_native_stop():
    manifest = _build_full_manifest("Hermes", "x", messaging_experience="agent")
    assert "agent_session_stopped" in manifest["settings"]["event_subscriptions"]["bot_events"]
