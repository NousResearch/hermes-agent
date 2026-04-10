"""Tests for session-scoped gateway model overrides and picker persistence."""

import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


class _PickerCaptureAdapter:
    """Fake interactive picker adapter."""

    def __init__(self):
        self.providers = None
        self.callback = None

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata=None,
    ) -> SendResult:
        self.providers = providers
        self.callback = on_model_selected
        return SendResult(success=True, message_id="picker-1")


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner_for_model_command(adapter):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    return runner, session_key


def _make_runner_for_run_agent():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


@pytest.mark.asyncio
async def test_run_agent_uses_session_model_override(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: default-model\n"
        "  provider: openrouter\n",
        encoding="utf-8",
    )

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "default-key",
        },
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    _CapturingAgent.last_init = None
    runner = _make_runner_for_run_agent()
    session_key = "agent:main:telegram:dm:c1"
    runner._session_model_overrides[session_key] = {
        "model": "ag-gemini-3.1-pro-high",
        "provider": "custom",
        "api_key": "override-key",
        "base_url": "https://relay.example/v1",
        "api_mode": "responses",
    }

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c1",
        chat_type="dm",
        user_id="u1",
    )

    result = await runner._run_agent(
        message="ping",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=session_key,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "ag-gemini-3.1-pro-high"
    assert _CapturingAgent.last_init["provider"] == "custom"
    assert _CapturingAgent.last_init["api_key"] == "override-key"
    assert _CapturingAgent.last_init["base_url"] == "https://relay.example/v1"
    assert _CapturingAgent.last_init["api_mode"] == "responses"


@pytest.mark.asyncio
async def test_model_picker_persists_default_and_uses_short_custom_list(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: ag-claude-opus-4-6-thinking\n"
        "  provider: custom\n"
        "  base_url: https://relay.old/v1\n"
        "  api_key: local-key\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda current_provider="", user_providers=None, max_models=8: [],
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key=None, base_url=None, timeout=8.0: [f"model-{i}" for i in range(1, 11)],
    )

    switch_result = SimpleNamespace(
        success=True,
        new_model="ag-gemini-3.1-pro-high",
        target_provider="custom",
        provider_changed=False,
        api_key="local-key",
        base_url="https://relay.new/v1",
        api_mode="responses",
        error_message="",
        warning_message="",
        provider_label="Custom endpoint",
        resolved_via_alias="",
        capabilities=None,
        model_info=None,
        is_global=True,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: switch_result,
    )

    adapter = _PickerCaptureAdapter()
    runner, _session_key = _make_runner_for_model_command(adapter)
    event = MessageEvent(text="/model", source=_make_source(), message_id="m1")

    result = await runner._handle_model_command(event)

    assert result is None
    assert adapter.providers is not None
    assert len(adapter.providers) == 1
    assert adapter.providers[0]["slug"] == "custom"
    assert len(adapter.providers[0]["models"]) == 5
    assert adapter.providers[0]["models"][0] == "ag-claude-opus-4-6-thinking"
    assert adapter.providers[0]["total_models"] == 11

    confirmation = await adapter.callback("c1", "ag-gemini-3.1-pro-high", "custom")
    saved = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))

    assert saved["model"]["default"] == "ag-gemini-3.1-pro-high"
    assert saved["model"]["provider"] == "custom"
    assert saved["model"]["base_url"] == "https://relay.new/v1"
    assert "Saved to config.yaml" in confirmation
    assert "session only" not in confirmation
