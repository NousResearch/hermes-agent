"""Slack Code channels: one agent session per channel (https://docs.slack.dev/ai/slack-code/).

Behaviour contract:
  * Recognition uses ``properties.record_channel.record_type == "agent_channel"`` from
    conversations.info; ordinary channels are unchanged and a failed lookup is retried.
  * In a code channel an unmentioned top-level message reaches the agent and the whole channel is
    one session (no synthetic thread); a thread the user started keeps its own session.
  * Replies to top-level code-channel messages go top-level, including the streaming path, where
    ``metadata.thread_id`` is the message's own ts and no ``reply_to`` is passed.
  * Status is channel-level: ``agents.sessions.setStatus`` without ``thread_ts`` -- ``processing``
    once per turn, ``active`` when the turn ends.
  * ``agent_session_stopped`` is registered before the catch-all ack and runs ``/stop``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

CODE, PLAIN = "C0CODE", "C0PLAIN"


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    client = AsyncMock()
    client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})

    async def conversations_info(channel):
        props = {"record_channel": {"record_type": "agent_channel"}} if channel == CODE else {}
        return {"channel": {"id": channel, "properties": props}}
    client.conversations_info = AsyncMock(side_effect=conversations_info)
    client.conversations_replies = AsyncMock(return_value={"messages": []})
    client.chat_postMessage = AsyncMock(return_value={"ts": "99.0"})
    client.api_call = AsyncMock(return_value={"ok": True})
    a._app.client = client
    a._get_client = MagicMock(return_value=client)
    a._client_for = MagicMock(return_value=client)
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _msg(channel, ts, text="add a risks section", **extra):
    return {"text": text, "user": "U_USER", "channel": channel, "channel_type": "group", "ts": ts, **extra}


class TestRecognition:
    @pytest.mark.asyncio
    async def test_record_channel_marks_a_code_channel(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message(_msg(PLAIN, "11.0", text="hi all"))
        assert adapter._is_known_code_channel(CODE)
        assert not adapter._is_known_code_channel(PLAIN)

    @pytest.mark.asyncio
    async def test_failed_lookup_is_retried(self, adapter):
        adapter._get_client.return_value.conversations_info = AsyncMock(side_effect=RuntimeError("boom"))
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        assert CODE not in adapter._code_channels


class TestSession:
    @pytest.mark.asyncio
    async def test_unmentioned_message_reaches_the_agent_as_one_channel_session(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args[0][0].source.thread_id is None

    @pytest.mark.asyncio
    async def test_plain_channel_still_needs_a_mention(self, adapter):
        await adapter._handle_slack_message(_msg(PLAIN, "11.0", text="hi all"))
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_started_thread_keeps_its_own_session(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message(_msg(CODE, "12.0", text="in a thread", thread_ts="10.0"))
        assert adapter.handle_message.await_args[0][0].source.thread_id == "10.0"


class TestReplies:
    @pytest.mark.asyncio
    async def test_replies_go_top_level_including_the_streaming_path(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        assert adapter._resolve_thread_ts("10.0", {"thread_id": None}) is None
        assert adapter._resolve_thread_ts(None, {"thread_id": "10.0"}) is None  # stream frames

    @pytest.mark.asyncio
    async def test_send_posts_top_level(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter.send(CODE, "Report ready", metadata={"thread_id": "10.0", "user_id": "U_USER"})
        kwargs = adapter._client_for.return_value.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == CODE and kwargs.get("thread_ts") is None

    @pytest.mark.asyncio
    async def test_replies_in_a_user_thread_stay_threaded(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message(_msg(CODE, "12.0", text="in a thread", thread_ts="10.0"))
        assert adapter._resolve_thread_ts(None, {"thread_id": "10.0"}) == "10.0"
        assert adapter._resolve_thread_ts("12.0", {"thread_id": "10.0"}) == "10.0"

    @pytest.mark.asyncio
    async def test_plain_channel_replies_unchanged(self, adapter):
        await adapter._handle_slack_message(_msg(PLAIN, "11.0", text="<@U_BOT> hi"))
        assert adapter._resolve_thread_ts("11.0", {}) == "11.0"


class TestStatus:
    @pytest.mark.asyncio
    async def test_channel_status_processing_once_then_active(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        api_call = adapter._get_client.return_value.api_call
        api_call.reset_mock()
        await adapter.send_typing(CODE, {"message_id": "10.0", "thread_id": "10.0"})
        await adapter.send_typing(CODE, {"message_id": "10.0", "thread_id": "10.0"})
        await adapter.stop_typing(CODE, {"thread_id": "10.0"})
        calls = [c.kwargs["json"] for c in api_call.await_args_list if c.args[0] == "agents.sessions.setStatus"]
        assert calls == [{"channel_id": CODE, "status": "processing"}, {"channel_id": CODE, "status": "active"}]


class TestStop:
    def test_stop_handler_registered_before_the_catch_all(self, adapter):
        order = []
        adapter._app.event = lambda t, *a, **k: (order.append(t), (lambda fn: fn))[1]
        adapter._app.command = adapter._app.action = lambda *a, **k: (lambda fn: fn)
        adapter._register_plugin_action_handlers = lambda *a, **k: None
        adapter._wire_plugin_handlers = lambda *a, **k: None
        adapter._register_bolt_handlers()
        assert "agent_session_stopped" in order
        catch_all = next(i for i, t in enumerate(order) if not isinstance(t, str))
        assert order.index("agent_session_stopped") < catch_all

    @pytest.mark.asyncio
    async def test_stop_runs_the_session_stop_command(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        adapter.handle_message.reset_mock()
        await adapter._handle_agent_session_stopped(
            {"type": "agent_session_stopped", "channel": CODE, "user": "U_USER", "event_ts": "13.0"}, {})
        event = adapter.handle_message.await_args[0][0]
        assert event.text == "/stop" and event.source.thread_id is None
