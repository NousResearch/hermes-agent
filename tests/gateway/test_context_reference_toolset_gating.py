from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WEBHOOK: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="route",
        chat_name="Webhook",
        chat_type="webhook",
        user_name="external",
    )


@pytest.mark.asyncio
async def test_gateway_does_not_expand_file_reference_without_file_toolset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = tmp_path / "secret.txt"
    secret.write_text("LOCAL_SECRET_SHOULD_NOT_LEAK\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    runner = _make_runner()
    event = MessageEvent(text="inspect @file:secret.txt", source=_source())

    with (
        patch("gateway.run._load_gateway_config") as mock_config,
        patch("gateway.run._resolve_runtime_agent_kwargs") as mock_runtime,
        patch(
            "agent.model_metadata.get_model_context_length"
        ) as mock_context_length,
    ):
        mock_config.return_value = {"platform_toolsets": {"webhook": ["web"]}}
        mock_runtime.return_value = {"api_key": "test", "base_url": None}
        mock_context_length.return_value = 100_000
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=_source(),
            history=[],
        )

    assert result is not None
    assert "LOCAL_SECRET_SHOULD_NOT_LEAK" not in result
    assert "@file:secret.txt" in result
    assert "context reference expansion is disabled" in result


@pytest.mark.asyncio
async def test_gateway_expands_file_reference_with_file_toolset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    note = tmp_path / "note.txt"
    note.write_text("allowed context\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    runner = _make_runner()
    event = MessageEvent(text="inspect @file:note.txt", source=_source())

    with (
        patch("gateway.run._load_gateway_config") as mock_config,
        patch("gateway.run._resolve_runtime_agent_kwargs") as mock_runtime,
        patch(
            "agent.model_metadata.get_model_context_length"
        ) as mock_context_length,
    ):
        mock_config.return_value = {"platform_toolsets": {"webhook": ["file"]}}
        mock_runtime.return_value = {"api_key": "test", "base_url": None}
        mock_context_length.return_value = 100_000
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=_source(),
            history=[],
        )

    assert result is not None
    assert "allowed context" in result
    assert "@file:note.txt" not in result.split("--- Attached Context ---", 1)[0]
