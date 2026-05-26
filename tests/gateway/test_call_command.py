from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, SUBCOMMANDS, gateway_help_lines, resolve_command

from types import SimpleNamespace

import pytest

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
async def test_handle_call_native_reports_native_enabled():
    runner = _runner()
    runner.adapters["simplex"] = SimpleNamespace(
        native_calls_enabled=True,
        native_call_handler=lambda call: call,
    )

    result = await runner._handle_call_command(_event("/call native", platform="simplex"))

    assert "SimpleX-native calls are enabled" in result
    assert "incoming SimpleX app calls" in result


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
