"""Slack Code channels: one agent session per channel (https://docs.slack.dev/ai/slack-code/).

Behaviour contract (opt-in, ``platforms.slack.extra.code_channels: true``):
  * Off (the default), nothing changes: no conversations.info lookup, code channels act like any
    channel.
  * Recognition uses ``properties.record_channel.record_type == "agent_channel"`` from
    conversations.info; ordinary channels are unchanged and a failed lookup is retried.
  * In a code channel an unmentioned top-level message reaches the agent and the whole channel is
    one session (no synthetic thread); a thread the user started keeps its own session.
  * Replies to top-level code-channel messages go top-level, including the streaming path, where
    ``metadata.thread_id`` is the message's own ts and no ``reply_to`` is passed.
  * Status is channel-level: ``agents.sessions.setStatus`` without ``thread_ts`` -- ``processing``
    once per turn, ``active`` when the turn ends.
  * Everyone in a code channel shares the one session, even with ``group_sessions_per_user``.
  * ``agent_session_stopped`` is registered before the catch-all ack, runs ``/stop`` for the
    channel's session (authorized users only) and moves the status to ``active``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

CODE, PLAIN = "C0CODE", "C0PLAIN"


def _make_adapter(**extra):
    a = SlackAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
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


@pytest.fixture()
def adapter():
    return _make_adapter(code_channels=True)


def _msg(channel, ts, text="add a risks section", **extra):
    return {"text": text, "user": "U_USER", "channel": channel, "channel_type": "group", "ts": ts, **extra}


class TestOptIn:
    @pytest.mark.asyncio
    async def test_off_by_default_no_lookup_and_mention_still_required(self):
        a = _make_adapter()
        await a._handle_slack_message(_msg(CODE, "10.0"))
        a._get_client.return_value.conversations_info.assert_not_awaited()
        a.handle_message.assert_not_awaited()


class TestRecognition:
    @pytest.mark.asyncio
    async def test_record_channel_marks_a_code_channel(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message(_msg(PLAIN, "11.0", text="hi all"))
        assert adapter._code_channels.is_code_channel(CODE)
        assert not adapter._code_channels.is_code_channel(PLAIN)

    @pytest.mark.asyncio
    async def test_failed_lookup_is_retried(self, adapter):
        adapter._get_client.return_value.conversations_info = AsyncMock(side_effect=RuntimeError("boom"))
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        assert not adapter._code_channels.is_code_channel(CODE)
        adapter._get_client.return_value.conversations_info = AsyncMock(
            return_value={"channel": {"properties": {"record_channel": {"record_type": "agent_channel"}}}})
        await adapter._handle_slack_message(_msg(CODE, "10.1"))
        assert adapter._code_channels.is_code_channel(CODE)


class TestSession:
    @pytest.mark.asyncio
    async def test_unmentioned_message_reaches_the_agent_as_one_channel_session(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args[0][0].source.thread_id is None

    @pytest.mark.asyncio
    async def test_participants_share_the_channel_session(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message({**_msg(CODE, "10.5", text="and a summary"), "user": "U_OTHER"})
        first, second = (c.args[0] for c in adapter.handle_message.await_args_list)
        assert adapter._event_session_key(first) == adapter._event_session_key(second)

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
        assert adapter._resolve_thread_ts("10.0", {"slack_team_id": "T1"}) is None
        assert adapter._resolve_thread_ts("10.0", {"thread_id": "10.0"}) is None  # synthetic root

    @pytest.mark.asyncio
    async def test_send_posts_top_level(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter.send(CODE, "Report ready", reply_to="10.0", metadata={"user_id": "U_USER"})
        kwargs = adapter._client_for.return_value.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == CODE and kwargs.get("thread_ts") is None

    @pytest.mark.asyncio
    async def test_a_side_thread_opened_mid_turn_does_not_move_the_channel_answer(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        await adapter._handle_slack_message(_msg(CODE, "12.0", text="in a thread", thread_ts="10.0"))
        assert adapter._resolve_thread_ts("10.0", {}) is None

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
        await adapter.send_typing(CODE, {"message_id": "10.0"})
        await adapter.send_typing(CODE, {"message_id": "10.0"})
        await adapter.stop_typing(CODE, {"thread_id": "10.0"})  # a side thread ends: not ours
        await adapter.stop_typing(CODE, {"slack_team_id": "T1"})
        calls = [c.kwargs["json"] for c in api_call.await_args_list if c.args[0] == "agents.sessions.setStatus"]
        assert calls == [{"channel_id": CODE, "status": "processing"}, {"channel_id": CODE, "status": "active"}]


class TestSlash:
    @pytest.mark.asyncio
    async def test_slash_before_any_message_uses_the_shared_channel_session(self, adapter):
        await adapter._handle_slash_command(
            {"command": "/new", "text": "", "user_id": "U_USER", "channel_id": CODE, "team_id": ""})
        await adapter._handle_slack_message({**_msg(CODE, "10.0"), "user": "U_OTHER"})
        slash, message = (c.args[0] for c in adapter.handle_message.await_args_list)
        assert adapter._event_session_key(slash) == adapter._event_session_key(message)


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
    async def test_stop_runs_the_session_stop_command_and_clears_status(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        message = adapter.handle_message.await_args[0][0]
        await adapter.send_typing(CODE, {"message_id": "10.0", "thread_id": None})
        adapter.handle_message.reset_mock()
        api_call = adapter._get_client.return_value.api_call
        api_call.reset_mock()
        await adapter._code_channels.on_session_stopped(
            {"type": "agent_session_stopped", "channel": CODE, "user": "U_USER", "event_ts": "13.0",
             "streaming_message_ts": []}, {})
        stop = adapter.handle_message.await_args[0][0]
        assert stop.text == "/stop" and stop.get_command() == "stop"
        assert adapter._event_session_key(stop) == adapter._event_session_key(message)
        calls = [c.kwargs["json"] for c in api_call.await_args_list if c.args[0] == "agents.sessions.setStatus"]
        assert calls == [{"channel_id": CODE, "status": "active"}]

    @pytest.mark.asyncio
    async def test_stop_from_an_unauthorized_user_changes_nothing(self, adapter):
        await adapter._handle_slack_message(_msg(CODE, "10.0"))
        adapter.handle_message.reset_mock()
        adapter._authorization_check = lambda *a, **kw: False
        await adapter._code_channels.on_session_stopped(
            {"type": "agent_session_stopped", "channel": CODE, "user": "U_EVIL", "event_ts": "13.0"}, {})
        adapter.handle_message.assert_not_awaited()
