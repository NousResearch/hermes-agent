from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, SUBCOMMANDS, gateway_help_lines, resolve_command

from types import SimpleNamespace
import sys

import pytest

import gateway.run as gateway_run
from gateway.calls.native.ports import NativeCallInvitation
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from hermes_cli.config import DEFAULT_CONFIG


def test_call_command_is_gateway_visible():
    cmd = resolve_command("call")

    assert cmd is not None
    assert cmd.name == "call"
    assert "call" in GATEWAY_KNOWN_COMMANDS
    assert "/call" in "\n".join(gateway_help_lines())


def test_call_command_has_expected_subcommands():
    assert SUBCOMMANDS["/call"] == ["browser", "native", "status", "end"]


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(extra={})
    runner._call_manager = None
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True
    return runner


def _event(text="/call", chat_type="dm", platform="telegram", wrap_platform=True):
    source_platform = SimpleNamespace(value=platform) if wrap_platform else platform
    source = SimpleNamespace(
        platform=source_platform,
        chat_id="123",
        user_id="456",
        chat_type=chat_type,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_handle_call_rejects_group_chat():
    result = await _runner()._handle_call_command(_event(chat_type="group"))

    assert "private-only" in result


@pytest.mark.asyncio
async def test_handle_call_native_is_loudly_unavailable_for_telegram():
    result = await _runner()._handle_call_command(_event("/call native"))

    assert "SimpleX-native" in result
    assert "SimpleX" in result


@pytest.mark.asyncio
async def test_handle_call_native_reports_missing_simplex_adapter():
    result = await _runner()._handle_call_command(_event("/call native", platform="simplex"))

    assert "SimpleX adapter not connected" in result


@pytest.mark.asyncio
async def test_handle_call_native_reports_native_disabled():
    runner = _runner()
    runner.adapters["simplex"] = SimpleNamespace(native_calls_enabled=False)

    result = await runner._handle_call_command(_event("/call native", platform="simplex"))

    assert "native WebRTC bridge is not enabled" in result


@pytest.mark.asyncio
async def test_handle_call_native_reports_handler_not_ready_with_platform_key():
    runner = _runner()
    platform = Platform("simplex")
    runner.adapters[platform] = SimpleNamespace(native_calls_enabled=True)

    result = await runner._handle_call_command(
        _event("/call native", platform=platform, wrap_platform=False)
    )

    assert "native WebRTC bridge" in result
    assert "not ready" in result


@pytest.mark.asyncio
async def test_handle_call_native_reports_sidecar_not_configured():
    runner = _runner()
    runner.adapters["simplex"] = SimpleNamespace(
        native_calls_enabled=True,
        native_call_handler=lambda source, invitation: None,
        config=SimpleNamespace(extra={"native_calls": {"enabled": True}}),
    )

    result = await runner._handle_call_command(_event("/call native", platform="simplex"))

    assert "SimpleX-native calls are unavailable" in result
    assert "sidecar" in result
    assert "not configured" in result


@pytest.mark.asyncio
async def test_handle_call_native_requires_simplex_dm():
    runner = _runner()
    runner.adapters["simplex"] = SimpleNamespace(
        native_calls_enabled=True,
        native_call_handler=lambda source, invitation: None,
        config=SimpleNamespace(
            extra={
                "native_calls": {
                    "enabled": True,
                    "sidecar_command": [sys.executable, "-c", "print('ok')"],
                }
            }
        ),
    )

    result = await runner._handle_call_command(
        _event("/call native", chat_type="group", platform="simplex")
    )

    assert "private-only" in result


@pytest.mark.asyncio
async def test_handle_call_native_reports_native_enabled():
    runner = _runner()
    runner.adapters["simplex"] = SimpleNamespace(
        native_calls_enabled=True,
        native_call_handler=lambda call: call,
        config=SimpleNamespace(
            extra={
                "native_calls": {
                    "sidecar_command": [sys.executable, "-c", "print('ok')"],
                }
            }
        ),
    )

    result = await runner._handle_call_command(_event("/call native", platform="simplex"))

    assert "SimpleX-native calls are enabled" in result
    assert "incoming SimpleX app calls" in result


@pytest.mark.asyncio
async def test_handle_call_native_finds_platform_key_from_value_wrapper():
    runner = _runner()
    runner.adapters[Platform("simplex")] = SimpleNamespace(
        platform=Platform("simplex"),
        native_calls_enabled=True,
        native_call_handler=lambda source, invitation: None,
        config=SimpleNamespace(
            extra={
                "native_calls": {
                    "sidecar_command": [sys.executable, "-c", "print('ok')"],
                }
            }
        ),
    )

    result = await runner._handle_call_command(_event("/call native", platform="simplex"))

    assert "SimpleX-native calls are enabled" in result


def test_configure_native_call_handlers_installs_simplex_handler():
    runner = _runner()
    adapter = SimpleNamespace(
        platform=Platform("simplex"),
        native_calls_enabled=True,
        native_call_handler=None,
    )
    runner.adapters[Platform("simplex")] = adapter

    runner._configure_native_call_handlers()

    assert callable(adapter.native_call_handler)


@pytest.mark.asyncio
async def test_configure_native_call_handler_uses_pending_adapter_before_registry_insert():
    runner = _runner()
    adapter = SimpleNamespace(
        platform=Platform("simplex"),
        native_calls_enabled=True,
        native_call_handler=None,
    )
    observed = {}

    async def handle_with_adapter(pending_adapter, source, invitation):
        observed["adapter"] = pending_adapter
        observed["source"] = source
        observed["invitation"] = invitation
        return SimpleNamespace(ok=True, code="accepted", message="", call_id="call-1")

    runner._handle_simplex_native_call_with_adapter = handle_with_adapter
    runner._configure_native_call_handler(adapter)

    source = SimpleNamespace(platform=Platform("simplex"), chat_id="123")
    invitation = NativeCallInvitation(contact_id="contact-1")
    result = await adapter.native_call_handler(source, invitation)

    assert result.ok is True
    assert observed == {
        "adapter": adapter,
        "source": source,
        "invitation": invitation,
    }


@pytest.mark.asyncio
async def test_start_installs_simplex_native_handler_before_connect(monkeypatch):
    runner = _runner()
    platform = Platform("simplex")
    observed = {}
    platform_config = SimpleNamespace(enabled=True)
    adapter = SimpleNamespace(
        platform=platform,
        native_calls_enabled=True,
        native_call_handler=None,
        has_fatal_error=False,
        set_message_handler=lambda _handler: None,
        set_fatal_error_handler=lambda _handler: None,
        set_session_store=lambda _store: None,
        set_busy_session_handler=lambda _handler: None,
    )

    async def connect_adapter(connecting_adapter, connecting_platform):
        observed["platform"] = connecting_platform
        observed["handler_ready"] = callable(connecting_adapter.native_call_handler)
        return False

    async def noop_async(*_args, **_kwargs):
        return None

    def close_task(coro):
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    runner.config = SimpleNamespace(
        sessions_dir="/tmp/hermes-test-sessions",
        platforms={platform: platform_config},
    )
    runner.session_store = SimpleNamespace(suspend_recently_active=lambda: 0)
    runner.hooks = SimpleNamespace(
        loaded_hooks=[],
        discover_and_load=lambda: None,
        emit=noop_async,
    )
    runner.delivery_router = SimpleNamespace(adapters={})
    runner._restart_drain_timeout = 1
    runner._failed_platforms = {}
    runner._create_adapter = lambda _platform, _config: adapter
    runner._connect_adapter_with_timeout = connect_adapter
    runner._safe_adapter_disconnect = noop_async
    runner._sync_voice_mode_state_to_adapter = lambda _adapter: None
    runner._update_platform_runtime_status = lambda *_args, **_kwargs: None
    runner._update_runtime_status = lambda *_args, **_kwargs: None
    runner._wire_teams_pipeline_runtime = lambda: None
    async def send_update_notification():
        return False

    runner._send_update_notification = send_update_notification
    runner._send_restart_notification = noop_async
    runner._send_home_channel_startup_notifications = noop_async
    runner._schedule_resume_pending_sessions = lambda: None
    runner._session_expiry_watcher = noop_async
    runner._kanban_notifier_watcher = noop_async
    runner._kanban_dispatcher_watcher = noop_async
    runner._platform_reconnect_watcher = noop_async
    runner._handoff_watcher = noop_async

    monkeypatch.setattr(gateway_run.asyncio, "create_task", close_task)

    await runner.start()

    assert observed == {"platform": platform, "handler_ready": True}


@pytest.mark.asyncio
async def test_reconnect_installs_simplex_native_handler_before_connect(monkeypatch):
    runner = _runner()
    platform = Platform("simplex")
    observed = {}
    platform_config = SimpleNamespace(enabled=True)
    adapter = SimpleNamespace(
        platform=platform,
        native_calls_enabled=True,
        native_call_handler=None,
        has_fatal_error=False,
        fatal_error_code=None,
        fatal_error_message=None,
        set_message_handler=lambda _handler: None,
        set_fatal_error_handler=lambda _handler: None,
        set_session_store=lambda _store: None,
        set_busy_session_handler=lambda _handler: None,
    )

    async def connect_adapter(connecting_adapter, connecting_platform):
        observed["platform"] = connecting_platform
        observed["handler_ready"] = callable(connecting_adapter.native_call_handler)
        runner._running = False
        return False

    async def sleep_without_waiting(_seconds):
        return None

    runner.session_store = SimpleNamespace()
    runner._running = True
    runner._failed_platforms = {
        platform: {"config": platform_config, "attempts": 0, "next_retry": 0}
    }
    runner._create_adapter = lambda _platform, _config: adapter
    runner._connect_adapter_with_timeout = connect_adapter
    runner._update_platform_runtime_status = lambda *_args, **_kwargs: None
    runner._pause_failed_platform = lambda *_args, **_kwargs: None

    monkeypatch.setattr(gateway_run.asyncio, "sleep", sleep_without_waiting)

    await runner._platform_reconnect_watcher()

    assert observed == {"platform": platform, "handler_ready": True}


@pytest.mark.asyncio
async def test_handle_simplex_native_call_reports_disconnected_without_adapter():
    runner = _runner()
    source = SimpleNamespace(
        platform=Platform("simplex"),
        chat_id="123",
        user_id="456",
        chat_type="dm",
    )

    result = await runner._handle_simplex_native_call(
        source,
        NativeCallInvitation(contact_id="contact-1"),
    )

    assert result.ok is False
    assert result.code == "call_simplex_ws_disconnected"
    assert result.message == (
        "SimpleX-native call setup failed: SimpleX adapter is not connected."
    )


@pytest.mark.asyncio
async def test_handle_simplex_native_call_records_connecting_native_call():
    runner = _runner()
    source = SimpleNamespace(
        platform=Platform("simplex"),
        chat_id="123",
        user_id="456",
        chat_type="dm",
    )
    adapter = SimpleNamespace(
        platform=Platform("simplex"),
        native_calls_enabled=True,
        config=SimpleNamespace(
            extra={
                "native_calls": {
                    "sidecar_command": [
                        sys.executable,
                        "-c",
                        (
                            "import json; "
                            "print(json.dumps({'ok': True, 'offer': "
                            "{'rtcSession': 'v=0', 'rtcIceCandidates': 'candidate:1'}}))"
                        ),
                    ]
                }
            }
        ),
        offers=[],
        statuses=[],
        rejected=[],
        ended=[],
    )

    async def send_offer(contact_id, offer):
        adapter.offers.append((contact_id, offer))

    async def send_status(contact_id, status):
        adapter.statuses.append((contact_id, status))

    async def reject(contact_id, reason_code):
        adapter.rejected.append((contact_id, reason_code))

    async def end(contact_id):
        adapter.ended.append(contact_id)

    adapter.send_offer = send_offer
    adapter.send_status = send_status
    adapter.reject = reject
    adapter.end = end
    runner.adapters[Platform("simplex")] = adapter

    result = await runner._handle_simplex_native_call(
        source,
        NativeCallInvitation(contact_id="contact-1"),
    )
    status = await runner._get_call_manager().status(source)

    assert result.ok is True
    assert result.call_id
    assert adapter.offers[0][0] == "contact-1"
    assert adapter.statuses == [("contact-1", "connecting")]
    assert status.session.call_id == result.call_id
    assert status.session.mode == "simplex_native"
    assert status.session.state.value == "connecting"


@pytest.mark.asyncio
async def test_handle_call_status_reports_idle():
    result = await _runner()._handle_call_command(_event("/call status"))

    assert "No active call" in result


def test_default_config_has_tailnet_call_settings():
    calls = DEFAULT_CONFIG["calls"]["browser"]

    assert calls["base_url"] == ""
    assert calls["public_exposure_enabled"] is False
    assert calls["ttl_seconds"] == 600


def test_gateway_config_preserves_calls_extra():
    cfg = GatewayConfig.from_dict(
        {
            "calls": {
                "browser": {
                    "base_url": "https://host.ts.net/call",
                    "public_exposure_enabled": False,
                    "ttl_seconds": 300,
                }
            }
        }
    )

    assert cfg.extra["calls"]["browser"]["base_url"] == "https://host.ts.net/call"


def test_load_gateway_config_preserves_calls_from_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """
calls:
  browser:
    base_url: https://host.ts.net/call
    public_exposure_enabled: false
    ttl_seconds: 300
""",
        encoding="utf-8",
    )
    from gateway.config import load_gateway_config

    cfg = load_gateway_config()

    assert cfg.extra["calls"]["browser"]["base_url"] == "https://host.ts.net/call"
    assert cfg.extra["calls"]["browser"]["ttl_seconds"] == 300
