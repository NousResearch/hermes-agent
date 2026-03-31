"""Tests for the gateway /model command."""

import os
import sys
import types


def _install_fake_prompt_toolkit():
    if 'prompt_toolkit' in sys.modules:
        return
    prompt_toolkit = types.ModuleType('prompt_toolkit')
    auto_suggest = types.ModuleType('prompt_toolkit.auto_suggest')
    completion = types.ModuleType('prompt_toolkit.completion')

    class AutoSuggest:  # pragma: no cover - simple test shim
        pass

    class Suggestion:
        def __init__(self, text=''):
            self.text = text

    class Completer:  # pragma: no cover - simple test shim
        pass

    class Completion:
        def __init__(self, text='', start_position=0, display=None, display_meta=None):
            self.text = text
            self.start_position = start_position
            self.display = display
            self.display_meta = display_meta
            self.display_text = display or text
            self.display_meta_text = display_meta or ''

    auto_suggest.AutoSuggest = AutoSuggest
    auto_suggest.Suggestion = Suggestion
    completion.Completer = Completer
    completion.Completion = Completion
    sys.modules['prompt_toolkit'] = prompt_toolkit
    sys.modules['prompt_toolkit.auto_suggest'] = auto_suggest
    sys.modules['prompt_toolkit.completion'] = completion

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.model_switch import ModelSwitchResult


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
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._effective_model = None
    runner._effective_provider = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._evict_cached_agent = MagicMock()
    runner._session_key_for_source = lambda source: f"key:{source.chat_id}"
    return runner


@pytest.mark.asyncio
async def test_model_command_appears_in_gateway_help():
    _install_fake_prompt_toolkit()
    runner = _make_runner()

    result = await runner._handle_help_command(_make_event('/help'))

    assert '/model [name]' in result


@pytest.mark.asyncio
async def test_handle_model_command_shows_current_model(tmp_path, monkeypatch):
    hermes_home = tmp_path / 'hermes'
    hermes_home.mkdir()
    (hermes_home / 'config.yaml').write_text(
        """model:
  default: openai/gpt-5.4
  provider: openrouter
""",
        encoding='utf-8',
    )
    monkeypatch.setattr(gateway_run, '_hermes_home', hermes_home)

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event('/model'))

    assert 'Current model' in result
    assert 'openai/gpt-5.4' in result
    assert 'OpenRouter' in result


@pytest.mark.asyncio
async def test_handle_model_command_switches_and_persists(tmp_path, monkeypatch):
    hermes_home = tmp_path / 'hermes'
    hermes_home.mkdir()
    config_path = hermes_home / 'config.yaml'
    config_path.write_text(
        """model:
  default: anthropic/claude-sonnet-4
  provider: openrouter
""",
        encoding='utf-8',
    )
    monkeypatch.setattr(gateway_run, '_hermes_home', hermes_home)
    monkeypatch.delenv('HERMES_MODEL', raising=False)
    monkeypatch.delenv('HERMES_INFERENCE_PROVIDER', raising=False)

    runner = _make_runner()

    monkeypatch.setattr(
        'hermes_cli.model_switch.switch_model',
        lambda *args, **kwargs: ModelSwitchResult(
            success=True,
            new_model='openai/gpt-5.5',
            target_provider='openrouter',
            provider_changed=False,
            persist=True,
            provider_label='OpenRouter',
        ),
    )
    monkeypatch.setattr(
        'hermes_cli.runtime_provider.resolve_runtime_provider',
        lambda requested=None: {'base_url': 'https://openrouter.ai/api/v1', 'api_key': 'k'},
    )

    result = await runner._handle_model_command(_make_event('/model openai/gpt-5.5'))

    saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved['model']['default'] == 'openai/gpt-5.5'
    assert os.environ['HERMES_MODEL'] == 'openai/gpt-5.5'
    runner._evict_cached_agent.assert_called_once_with('key:c1')
    assert 'saved to config' in result
    assert 'takes effect on next message' in result


@pytest.mark.asyncio
async def test_handle_message_dispatches_model_command():
    _install_fake_prompt_toolkit()
    runner = _make_runner()
    runner._handle_model_command = AsyncMock(return_value='ok-model')

    result = await runner._handle_message(_make_event('/model openai/gpt-5.5'))

    assert result == 'ok-model'
    runner._handle_model_command.assert_awaited_once()
