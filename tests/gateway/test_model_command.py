"""Tests for gateway /model and /models behavior."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: SimpleNamespace(_component_runtime=object())}
    runner._effective_model = None
    runner._effective_provider = None
    return runner


def _make_event(text: str, *, native_slash: bool = False):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="c1",
        chat_name="Hermes / #general",
        chat_type="group",
        user_id="u1",
        user_name="alan",
    )
    metadata = {"is_native_slash": True} if native_slash else {}
    interaction = None
    if native_slash:
        interaction = SimpleNamespace(
            response=SimpleNamespace(is_done=lambda: True, send_message=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
    return MessageEvent(
        text=text,
        source=source,
        raw_message=interaction,
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_model_without_args_returns_numbered_catalog(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.models.curated_models_for_provider",
        lambda _provider: [
            ("anthropic/claude-opus-4.6", "recommended"),
            ("openai/gpt-5.4", ""),
        ],
    )

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/model"))

    assert "Model Catalog" in result
    assert "1. `anthropic/claude-opus-4.6`" in result
    assert "2. `openai/gpt-5.4`" in result
    assert "/model <number>" in result


@pytest.mark.asyncio
async def test_model_numeric_selection_persists_and_records_recent(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.models.curated_models_for_provider",
        lambda _provider: [
            ("anthropic/claude-opus-4.6", "recommended"),
            ("openai/gpt-5.4", ""),
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        },
    )
    recent_calls = []
    monkeypatch.setattr(
        "gateway.platforms.discord_impl.model_picker.record_recent_model",
        lambda user_id, provider, model: recent_calls.append((user_id, provider, model)),
    )

    runner = _make_runner()
    runner._resolve_model_runtime_details = lambda _provider: (
        {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"},
        None,
    )

    result = await runner._handle_model_command(_make_event("/model 2"))

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert result.startswith("🤖 Model changed to `openai/gpt-5.4`")
    assert saved["model"]["default"] == "openai/gpt-5.4"
    assert saved["model"]["provider"] == "openrouter"
    assert recent_calls == [("u1", "openrouter", "openai/gpt-5.4")]


@pytest.mark.asyncio
async def test_model_status_reports_runtime_details(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    runner._resolve_model_runtime_details = lambda _provider: (
        {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "env/config",
        },
        None,
    )

    result = await runner._handle_model_command(_make_event("/model status"))

    assert "Model Status" in result
    assert "**Runtime provider:** OpenRouter (`openrouter`)" in result
    assert "**API mode:** `chat_completions`" in result
    assert "**Base URL:** `https://openrouter.ai/api/v1`" in result
    assert "**Credentials:** configured" in result


@pytest.mark.asyncio
async def test_native_slash_model_without_args_opens_picker(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    open_picker = AsyncMock()
    monkeypatch.setattr(
        "gateway.platforms.discord_impl.model_picker.open_model_picker",
        open_picker,
    )

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/model", native_slash=True))

    assert result is None
    open_picker.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_slash_models_without_args_opens_picker(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    open_picker = AsyncMock()
    monkeypatch.setattr(
        "gateway.platforms.discord_impl.model_picker.open_model_picker",
        open_picker,
    )

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/models", native_slash=True))

    assert result is None
    open_picker.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_slash_model_status_uses_ephemeral_followup(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    runner._resolve_model_runtime_details = lambda _provider: (
        {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
        },
        None,
    )
    event = _make_event("/model status", native_slash=True)

    result = await runner._handle_model_command(event)

    assert result is None
    event.raw_message.followup.send.assert_awaited_once()
    sent = event.raw_message.followup.send.await_args.args[0]
    assert "Model Status" in sent
