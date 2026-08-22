from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import MessageType
from plugins.platforms.slack import native_stop
from plugins.platforms.slack.native_stop import SlackAgentStopAdapter


class _FakeApp:
    def __init__(self):
        self.registrations = []

    def event(self, event_name):
        def decorator(callback):
            self.registrations.append((event_name, callback))
            return callback

        return decorator


def _bare_adapter():
    adapter = SlackAgentStopAdapter.__new__(SlackAgentStopAdapter)
    adapter._app = object()
    adapter._channel_team = {}
    adapter._channel_teams = {}
    adapter._CHANNEL_TEAM_MAX = 10000
    adapter._channel_name_cache = {}
    adapter._active_streams = {}
    adapter._event_team_id = MagicMock(return_value="")
    adapter._resolve_user_name = AsyncMock(return_value="Ada")
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    adapter.handle_message = AsyncMock()
    adapter.stop_typing = AsyncMock()
    client = SimpleNamespace(chat_stopStream=AsyncMock())
    adapter._get_client = MagicMock(return_value=client)
    return adapter, client


@pytest.mark.asyncio
async def test_agent_session_stopped_routes_through_canonical_stop_and_clears_ui():
    adapter, client = _bare_adapter()
    adapter._active_streams["C123ABC456"] = {
        "ts": "1782234987.693923",
        "draft_id": 7,
        "sent": "working",
    }
    event = {
        "type": "agent_session_stopped",
        "channel": "C123ABC456",
        "thread_ts": "1782234671.392669",
        "message_ts": "1782234987.693923",
        "user": "U123ABC456",
        "team_id": "T0123ABC456",
        "event_ts": "1783536983.783769",
    }

    await adapter._handle_agent_session_stopped(event, {})

    assert adapter._channel_team == {"C123ABC456": "T0123ABC456"}
    assert adapter._channel_teams == {"C123ABC456": {"T0123ABC456"}}
    adapter._resolve_user_name.assert_awaited_once_with(
        "U123ABC456",
        chat_id="C123ABC456",
        team_id="T0123ABC456",
    )
    adapter.build_source.assert_called_once_with(
        chat_id="C123ABC456",
        chat_name="C123ABC456",
        chat_type="group",
        user_id="U123ABC456",
        user_name="Ada",
        thread_id="1782234671.392669",
        scope_id="T0123ABC456",
    )

    stop_event = adapter.handle_message.await_args.args[0]
    assert stop_event.text == "/stop"
    assert stop_event.message_type is MessageType.COMMAND
    assert stop_event.raw_message is event
    assert stop_event.message_id == "1783536983.783769"
    assert stop_event.metadata["native_agent_session_stop"] is True
    assert stop_event.metadata["slack_team_id"] == "T0123ABC456"
    assert stop_event.metadata["thread_id"] == "1782234671.392669"

    adapter.stop_typing.assert_awaited_once_with(
        "C123ABC456",
        metadata={
            "thread_id": "1782234671.392669",
            "thread_ts": "1782234671.392669",
            "slack_team_id": "T0123ABC456",
        },
    )
    adapter._get_client.assert_called_once_with(
        "C123ABC456", team_id="T0123ABC456"
    )
    client.chat_stopStream.assert_awaited_once_with(
        channel="C123ABC456",
        ts="1782234987.693923",
    )
    assert "C123ABC456" not in adapter._active_streams


@pytest.mark.asyncio
async def test_agent_session_stopped_fails_closed_without_exact_lane_identity():
    adapter, client = _bare_adapter()
    event = {
        "type": "agent_session_stopped",
        "channel": "C123ABC456",
        "user": "U123ABC456",
        "team_id": "T0123ABC456",
    }

    await adapter._handle_agent_session_stopped(event, {})

    adapter.handle_message.assert_not_awaited()
    adapter.stop_typing.assert_not_awaited()
    client.chat_stopStream.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_session_stopped_does_not_drop_unrelated_channel_stream():
    adapter, client = _bare_adapter()
    adapter._active_streams["C123ABC456"] = {
        "ts": "1782234000.000000",
        "draft_id": 3,
        "sent": "other thread",
    }
    event = {
        "type": "agent_session_stopped",
        "channel": "C123ABC456",
        "thread_ts": "1782234671.392669",
        "message_ts": "1782234987.693923",
        "user": "U123ABC456",
        "team_id": "T0123ABC456",
    }

    await adapter._handle_agent_session_stopped(event, {})

    client.chat_stopStream.assert_awaited_once_with(
        channel="C123ABC456",
        ts="1782234987.693923",
    )
    assert adapter._active_streams["C123ABC456"]["ts"] == "1782234000.000000"


@pytest.mark.asyncio
async def test_ui_stream_stop_still_runs_when_typing_cleanup_fails():
    adapter, client = _bare_adapter()
    adapter.stop_typing = AsyncMock(side_effect=RuntimeError("status unavailable"))

    await adapter._settle_agent_session_stopped_ui(
        channel_id="C123ABC456",
        thread_ts="1782234671.392669",
        team_id="T0123ABC456",
        message_ts="1782234987.693923",
    )

    client.chat_stopStream.assert_awaited_once_with(
        channel="C123ABC456",
        ts="1782234987.693923",
    )


def test_native_stop_listener_is_registered_before_socket_mode_and_only_once():
    adapter, _ = _bare_adapter()
    app = _FakeApp()
    adapter._app = app

    with patch.object(
        native_stop._adapter.SlackAdapter,
        "_start_socket_mode_handler",
    ) as parent_start:
        adapter._start_socket_mode_handler()
        adapter._start_socket_mode_handler()

    assert [name for name, _ in app.registrations] == ["agent_session_stopped"]
    assert parent_start.call_count == 2


@pytest.mark.asyncio
async def test_native_stop_listener_rebinds_shared_app_to_replacement_adapter():
    first_adapter, _ = _bare_adapter()
    replacement_adapter, _ = _bare_adapter()
    first_adapter._handle_agent_session_stopped = AsyncMock()
    replacement_adapter._handle_agent_session_stopped = AsyncMock()
    app = _FakeApp()
    first_adapter._app = app
    replacement_adapter._app = app

    first_adapter._register_agent_session_stopped_listener()
    replacement_adapter._register_agent_session_stopped_listener()

    assert [name for name, _ in app.registrations] == ["agent_session_stopped"]
    callback = app.registrations[0][1]
    event = {"type": "agent_session_stopped"}
    body = {"team_id": "T0123ABC456"}
    await callback(event, body)

    first_adapter._handle_agent_session_stopped.assert_not_awaited()
    replacement_adapter._handle_agent_session_stopped.assert_awaited_once_with(
        event,
        body,
    )


def test_plugin_registration_preserves_metadata_and_replaces_only_factory():
    captured = {}

    class Context:
        def register_platform(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    native_stop.register(Context())

    kwargs = captured["kwargs"]
    assert kwargs["name"] == "slack"
    assert kwargs["label"] == "Slack"
    assert kwargs["adapter_factory"] is native_stop._build_adapter
    assert kwargs["check_fn"] is native_stop._adapter.slack_deps_present
    assert kwargs["ensure_deps_fn"] is native_stop._adapter.check_slack_requirements


def test_registration_proxy_refuses_unknown_base_factory():
    proxy = native_stop._PlatformRegistrationProxy(SimpleNamespace())

    with pytest.raises(RuntimeError, match="canonical Slack adapter factory"):
        proxy.register_platform(adapter_factory=lambda config: config)
