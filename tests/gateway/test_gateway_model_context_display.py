"""Regression tests for gateway model context display helpers.

Covers three user-visible paths that must agree:
- direct /model <name> --provider <slug>
- interactive /model picker callback
- /new session reset banner via _format_session_info()
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key


AI_API_CONFIG = """
model:
  default: MiniMax-M2.7
  provider: ai-api-chat
providers:
  ai-api-codex:
    name: AI API Codex
    base_url: https://ai-api.home.sadlay.cn:22843/v1
    api_key: sk-test
    api_mode: codex_responses
    default_model: gpt-5.4
    models:
      gpt-5.4:
        context_length: 400000
  ai-api-chat:
    name: AI API Chat
    base_url: https://ai-api.home.sadlay.cn:22843/v1
    api_key: sk-test
    api_mode: chat_completions
    default_model: MiniMax-M2.7
    models:
      MiniMax-M2.7:
        context_length: 204800
"""


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


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._clear_session_boundary_security_state = lambda _session_key: None
    runner._invalidate_session_run_generation = lambda *_args, **_kwargs: None
    runner._cleanup_agent_resources = lambda *_args, **_kwargs: None
    runner._evict_cached_agent = lambda *_args, **_kwargs: None
    runner._is_user_authorized = lambda _source: True

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
    return runner


class _PickerAdapter:
    def __init__(self):
        self.last_result = None

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata=None,
    ):
        self.last_result = await on_model_selected(chat_id, "gpt-5.4", "ai-api-codex")
        return SimpleNamespace(success=True, message_id="picker-msg-1")


@pytest.mark.asyncio
async def test_interactive_model_picker_uses_provider_model_context_length(tmp_path):
    runner = _make_runner()
    adapter = _PickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}

    (tmp_path / "config.yaml").write_text(AI_API_CONFIG)

    fake_result = SimpleNamespace(
        success=True,
        new_model="gpt-5.4",
        target_provider="ai-api-codex",
        provider_label="ai-api-codex",
        api_key="sk-test",
        base_url="https://ai-api.home.sadlay.cn:22843/v1",
        api_mode="codex_responses",
        error_message="",
        model_info=None,
    )
    fake_providers = [{"slug": "ai-api-codex", "name": "AI API Codex", "models": ["gpt-5.4"], "is_current": False, "total_models": 1}]

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "hermes_cli.model_switch.list_authenticated_providers", return_value=fake_providers
    ), patch("hermes_cli.model_switch.switch_model", return_value=fake_result):
        response = await runner._handle_model_command(_make_event("/model"))

    assert response is None
    assert adapter.last_result is not None
    assert "Model switched to `gpt-5.4`" in adapter.last_result
    assert "Provider: ai-api-codex" in adapter.last_result
    assert "Context: 400,000 tokens" in adapter.last_result


@pytest.mark.asyncio
async def test_new_banner_uses_provider_model_context_length_from_config(tmp_path):
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: MagicMock(send=AsyncMock())}

    (tmp_path / "config.yaml").write_text(AI_API_CONFIG)

    with patch("gateway.run._hermes_home", tmp_path):
        response = await runner._handle_reset_command(_make_event("/new"))

    assert "Session reset! Starting fresh." in response
    assert "◆ Model: `MiniMax-M2.7`" in response
    assert "◆ Provider: ai-api-chat" in response
    assert "◆ Context: 204K tokens" in response


def test_format_session_info_prefers_session_override_context_length(tmp_path):
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "provider": "ai-api-codex",
        "api_key": "sk-test",
        "base_url": "https://ai-api.home.sadlay.cn:22843/v1",
        "api_mode": "codex_responses",
    }

    (tmp_path / "config.yaml").write_text(AI_API_CONFIG)

    runtime = {
        "provider": "ai-api-chat",
        "base_url": "https://ai-api.home.sadlay.cn:22843/v1",
        "api_key": "sk-test",
    }

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "gateway.run._resolve_gateway_model", return_value="MiniMax-M2.7"
    ), patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime):
        info = runner._format_session_info(_make_source(), session_key)

    assert "◆ Model: `gpt-5.4`" in info
    assert "◆ Provider: ai-api-codex" in info
    assert "◆ Context: 400K tokens" in info
