from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    return runner


def test_image_model_command_is_registered_for_cli_and_gateway():
    from hermes_cli.commands import is_gateway_known_command, resolve_command

    cmd = resolve_command("image_model")

    assert cmd is not None
    assert cmd.name == "image_model"
    assert is_gateway_known_command("image_model") is True


@pytest.mark.asyncio
async def test_gateway_dispatch_routes_image_model(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    sentinel = "image model handler reached"
    runner._handle_image_model_command = AsyncMock(return_value=sentinel)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run,
        "_run_from_gateway_async",
        AsyncMock(return_value={"final": "agent should not run"}),
        raising=False,
    )

    out = await runner._handle_message(_make_event("/image_model openai/gpt-image-2-medium"))

    assert out == sentinel
    runner._handle_image_model_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_image_model_handler_returns_switch_result(monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli.image_model_switch import ImageModelSwitchResult

    runner = _make_runner()
    event = _make_event("/image_model openai/gpt-image-2-medium")

    monkeypatch.setattr(
        "hermes_cli.image_model_switch.apply_image_model_switch",
        lambda raw_args: ImageModelSwitchResult(
            success=True,
            message="✓ Image model switched: openai / gpt-image-2-medium",
            provider="openai",
            model="gpt-image-2-medium",
        ),
    )

    out = await GatewayRunner._handle_image_model_command(runner, event)

    assert "openai" in out
    assert "gpt-image-2-medium" in out
