"""Regression tests for Slack Socket Mode history recovery and ordering."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock() -> None:
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, module in (
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ):
        sys.modules.setdefault(name, module)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as slack_module  # noqa: E402

slack_module.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


class _SlackResponseLike:
    """Minimal Slack SDK response stand-in: mapping access without dict inheritance."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: str, default=None):
        return self._data.get(key, default)


@pytest.fixture
def adapter() -> SlackAdapter:
    instance = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    instance._team_bot_user_ids["T1"] = "BOT"
    return instance


@pytest.mark.asyncio
async def test_normal_message_without_thread_uses_existing_handler(adapter):
    adapter._handle_slack_message_inner = AsyncMock()

    event = {"channel": "D1", "team": "T1", "ts": "1.000", "text": "hello"}
    await adapter._handle_slack_message(event, {"team_id": "T1"})

    adapter._handle_slack_message_inner.assert_awaited_once_with(
        event, {"team_id": "T1"}
    )


@pytest.mark.asyncio
async def test_duplicate_event_is_dispatched_only_once(adapter):
    adapter.handle_message = AsyncMock()
    event = {
        "channel": "D1",
        "channel_type": "im",
        "team": "T1",
        "ts": "1.000",
        "user": "U1",
        "client_msg_id": "client-1",
        "text": "hello",
    }

    await adapter._handle_slack_message_inner(event, {"team_id": "T1"})
    await adapter._handle_slack_message_inner(event, {"team_id": "T1"})

    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_same_thread_handlers_are_serialized_in_arrival_order(adapter):
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    handled: list[str] = []

    adapter._recover_thread_messages_before_dispatch = AsyncMock()

    async def inner(event, payload):
        handled.append(event["ts"])
        if event["ts"] == "2.000":
            first_entered.set()
            await release_first.wait()

    adapter._handle_slack_message_inner = inner
    first = {"channel": "D1", "thread_ts": "1.000", "team": "T1", "ts": "2.000"}
    second = {"channel": "D1", "thread_ts": "1.000", "team": "T1", "ts": "3.000"}

    first_task = asyncio.create_task(adapter._handle_slack_message(first, {"team_id": "T1"}))
    await first_entered.wait()
    second_task = asyncio.create_task(adapter._handle_slack_message(second, {"team_id": "T1"}))
    await asyncio.sleep(0)
    assert handled == ["2.000"]

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert handled == ["2.000", "3.000"]


@pytest.mark.asyncio
async def test_thread_history_is_replayed_oldest_first_and_cursor_advances(adapter):
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {"ts": "3.000", "user": "U3"},
                {"ts": "1.000", "user": "U1"},
                {"ts": "2.000", "user": "U2"},
            ]
        }
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._handle_slack_message_inner = AsyncMock()

    event = {"channel": "D1", "thread_ts": "1.000", "team": "T1", "ts": "3.000"}
    await adapter._recover_thread_messages_before_dispatch(event, {"team_id": "T1"})

    replayed = [
        call.args[0]["ts"]
        for call in adapter._handle_slack_message_inner.await_args_list
    ]
    assert replayed == ["1.000", "2.000", "3.000"]
    assert adapter._socket_thread_recovery_cursors[("T1", "D1", "1.000")] == "3.000"
    assert all(
        call.args[0]["_hermes_socket_recovery_replay"] is True
        for call in adapter._handle_slack_message_inner.await_args_list
    )


@pytest.mark.asyncio
async def test_thread_history_recovery_drains_all_cursor_pages(adapter):
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        side_effect=[
            {
                "messages": [{"ts": "4.000", "user": "U4"}],
                "response_metadata": {"next_cursor": "thread-page-2"},
            },
            {
                "messages": [
                    {"ts": "3.000", "user": "U3"},
                    {"ts": "1.000", "user": "U1"},
                    {"ts": "2.000", "user": "U2"},
                ]
            },
        ]
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._handle_slack_message_inner = AsyncMock()

    event = {"channel": "D1", "thread_ts": "1.000", "team": "T1", "ts": "4.000"}
    await adapter._recover_thread_messages_before_dispatch(event, {"team_id": "T1"})

    assert [
        call.args[0]["ts"]
        for call in adapter._handle_slack_message_inner.await_args_list
    ] == ["1.000", "2.000", "3.000", "4.000"]
    assert client.conversations_replies.await_count == 2
    assert client.conversations_replies.await_args_list[1].kwargs["cursor"] == "thread-page-2"
    assert adapter._socket_thread_recovery_cursors[("T1", "D1", "1.000")] == "4.000"


@pytest.mark.asyncio
async def test_thread_recovery_retains_cursor_when_slack_omits_next_cursor(adapter):
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={"messages": [{"ts": "2.000", "user": "U2"}], "has_more": True}
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._handle_slack_message_inner = AsyncMock()

    event = {"channel": "D1", "thread_ts": "1.000", "team": "T1", "ts": "2.000"}
    await adapter._recover_thread_messages_before_dispatch(event, {"team_id": "T1"})

    assert ("T1", "D1", "1.000") not in adapter._socket_thread_recovery_cursors


@pytest.mark.asyncio
async def test_dm_history_recovery_preserves_timestamp_order(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(return_value={"channels": [{"id": "D1"}]})
    client.conversations_history = AsyncMock(
        return_value={
            "messages": [
                {"ts": "3.000", "user": "U3"},
                {"ts": "1.000", "user": "U1"},
                {"ts": "2.000", "user": "U2"},
            ]
        }
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._socket_recovery_interval_s = 0
    adapter._handle_slack_message = AsyncMock()

    recovered = await adapter._recover_missed_socket_messages()

    assert recovered == 3
    replayed = [call.args[0]["ts"] for call in adapter._handle_slack_message.await_args_list]
    assert replayed == ["1.000", "2.000", "3.000"]
    assert adapter._socket_recovery_cursors[("T1", "D1")] == "3.000"


@pytest.mark.asyncio
async def test_dm_history_recovery_drains_all_cursor_pages(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(return_value={"channels": [{"id": "D1"}]})
    client.conversations_history = AsyncMock(
        side_effect=[
            {
                "messages": [{"ts": "4.000", "user": "U4"}],
                "response_metadata": {"next_cursor": "history-page-2"},
            },
            {
                "messages": [
                    {"ts": "3.000", "user": "U3"},
                    {"ts": "1.000", "user": "U1"},
                    {"ts": "2.000", "user": "U2"},
                ]
            },
        ]
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._socket_recovery_interval_s = 0
    adapter._handle_slack_message = AsyncMock()

    recovered = await adapter._recover_missed_socket_messages()

    assert recovered == 4
    assert [call.args[0]["ts"] for call in adapter._handle_slack_message.await_args_list] == [
        "1.000",
        "2.000",
        "3.000",
        "4.000",
    ]
    assert client.conversations_history.await_count == 2
    assert client.conversations_history.await_args_list[1].kwargs["cursor"] == "history-page-2"
    assert adapter._socket_recovery_cursors[("T1", "D1")] == "4.000"


@pytest.mark.asyncio
async def test_dm_history_recovery_accepts_slack_sdk_response_like(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(
        return_value=_SlackResponseLike({"channels": [{"id": "D1"}]})
    )
    client.conversations_history = AsyncMock(
        return_value=_SlackResponseLike({"messages": [{"ts": "1.000", "user": "U1"}]})
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._socket_recovery_interval_s = 0
    adapter._handle_slack_message = AsyncMock()

    assert await adapter._recover_missed_socket_messages() == 1
    assert adapter._socket_recovery_cursors[("T1", "D1")] == "1.000"


@pytest.mark.asyncio
async def test_dm_thread_recovery_drains_all_cursor_pages_before_advancing(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(return_value={"channels": [{"id": "D1"}]})
    client.conversations_history = AsyncMock(
        return_value={
            "messages": [
                {"ts": "1.000", "user": "U1", "latest_reply": "4.000"}
            ]
        }
    )
    client.conversations_replies = AsyncMock(
        side_effect=[
            {
                "messages": [{"ts": "4.000", "user": "U4"}],
                "response_metadata": {"next_cursor": "reply-page-2"},
            },
            {
                "messages": [
                    {"ts": "3.000", "user": "U3"},
                    {"ts": "1.000", "user": "U1"},
                    {"ts": "2.000", "user": "U2"},
                ]
            },
        ]
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._socket_recovery_interval_s = 0
    adapter._handle_slack_message = AsyncMock()

    recovered = await adapter._recover_missed_socket_messages()

    assert recovered == 4
    assert [call.args[0]["ts"] for call in adapter._handle_slack_message.await_args_list] == [
        "1.000",
        "2.000",
        "3.000",
        "4.000",
    ]
    assert client.conversations_replies.await_count == 2
    assert client.conversations_replies.await_args_list[1].kwargs["cursor"] == "reply-page-2"
    assert adapter._socket_recovery_cursors[("T1", "D1")] == "4.000"


@pytest.mark.asyncio
async def test_dm_recovery_retains_cursor_when_slack_omits_next_cursor(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(return_value={"channels": [{"id": "D1"}]})
    client.conversations_history = AsyncMock(
        return_value={"messages": [{"ts": "2.000", "user": "U2"}], "has_more": True}
    )
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_since = "0.000"
    adapter._socket_recovery_interval_s = 0
    adapter._handle_slack_message = AsyncMock()

    await adapter._recover_missed_socket_messages()

    assert ("T1", "D1") not in adapter._socket_recovery_cursors


@pytest.mark.asyncio
async def test_slack_api_failure_returns_without_retry_loop(adapter):
    client = MagicMock()
    client.conversations_list = AsyncMock(side_effect=RuntimeError("rate limited"))
    adapter._team_clients["T1"] = client
    adapter._socket_recovery_interval_s = 0

    assert await asyncio.wait_for(adapter._recover_missed_socket_messages(), timeout=1) == 0
    client.conversations_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_message_dispatch_still_works_after_socket_restart(adapter):
    adapter._running = True
    adapter._app = MagicMock()
    adapter._app_token = "xapp-test"
    adapter._stop_socket_mode_handler = AsyncMock()
    adapter._start_socket_mode_handler = MagicMock()

    await adapter._restart_socket_mode("test reconnect")

    adapter._handle_slack_message_inner = AsyncMock()
    event = {"channel": "D1", "team": "T1", "ts": "2.000", "text": "after"}
    await adapter._handle_slack_message(event, {"team_id": "T1"})

    adapter._stop_socket_mode_handler.assert_awaited_once()
    adapter._start_socket_mode_handler.assert_called_once()
    adapter._handle_slack_message_inner.assert_awaited_once_with(
        event, {"team_id": "T1"}
    )
