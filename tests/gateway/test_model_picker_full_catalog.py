"""Chat /model picker: every model of a provider reaches the adapter; each adapter bounds its own
rendering (Telegram pages, Discord/Slack cap and say "N more"), so the shared listing never cuts
a row below what the adapter can show (#129260)."""

import json
import types

import pytest

import hermes_agent.gateway.run as gateway_run
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.platforms.base import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource

_MODELS = [f"bigcat-model-{i:02d}" for i in range(1, 81)]


class _CapturingPickerAdapter:
    def __init__(self):
        self.providers = None

    async def send_model_picker(self, *, providers, **kwargs):
        self.providers = providers
        return types.SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_model_picker_receives_a_providers_whole_catalog(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    # JSON is valid YAML: a built-in provider whose catalog the user extended past 50 models.
    (home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "deepseek", "default": "bigcat-model-01"},
        "providers": {"deepseek": {"models": _MODELS}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-key")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr("hermes_agent.agent.models_dev.fetch_models_dev", dict)

    adapter = _CapturingPickerAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode, runner._session_model_overrides, runner._running_agents = {}, {}, {}
    event = MessageEvent(text="/model", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"))

    assert await runner._handle_model_command(event) is None  # the picker was sent
    row = next(p for p in adapter.providers if p["slug"] == "deepseek")
    assert row["total_models"] >= len(_MODELS)
    assert len(row["models"]) == row["total_models"]
    assert set(_MODELS) <= set(row["models"])
