"""Regression tests for gateway /model support of config.yaml custom_providers."""

from types import SimpleNamespace

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._evict_cached_agent = lambda *_a, **_kw: None
    runner._thread_metadata_for_source = lambda *_a, **_kw: {}
    runner._reply_anchor_for_event = lambda *_a, **_kw: None
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_handle_model_command_lists_saved_custom_provider(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                },
                "providers": {},
                "custom_providers": [
                    {
                        "name": "Local (127.0.0.1:4141)",
                        "base_url": "http://127.0.0.1:4141/v1",
                        "model": "rotator-openrouter-coding",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    result = await _make_runner()._handle_model_command(_make_event())

    assert result is not None
    assert "Local (127.0.0.1:4141)" in result
    assert "custom:local-(127.0.0.1:4141)" in result
    assert "rotator-openrouter-coding" in result


class _PickerAdapter:
    def __init__(self):
        self.providers = None
        self.callback_result = None

    async def send_model_picker(
        self,
        chat_id,
        providers,
        current_model,
        current_provider,
        session_key,
        on_model_selected,
        metadata=None,
    ):
        self.providers = providers
        chosen = providers[0]
        self.callback_result = await on_model_selected(
            chat_id, "gpt-5-mini", chosen["slug"]
        )
        return SimpleNamespace(success=True, message_id="101")


@pytest.mark.asyncio
async def test_handle_model_command_picker_prefers_copilot_acp_and_shows_slug(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {
                "default": "gpt-5.5",
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            "providers": {},
        },
    )
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: [
            {
                "slug": "copilot",
                "name": "GitHub Copilot",
                "is_current": False,
                "models": ["gpt-5-mini", "claude-sonnet-4.6"],
                "total_models": 2,
                "source": "hermes",
            },
            {
                "slug": "copilot-acp",
                "name": "GitHub Copilot ACP",
                "is_current": False,
                "models": ["gpt-5-mini", "claude-sonnet-4.6"],
                "total_models": 2,
                "source": "hermes",
            },
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: ModelSwitchResult(
            success=True,
            new_model="gpt-5-mini",
            target_provider="copilot-acp",
            provider_changed=True,
            api_key="copilot-acp",
            base_url="",
            api_mode="responses",
            provider_label="GitHub Copilot ACP",
            model_info=None,
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **kw: None,
    )

    adapter = _PickerAdapter()
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: adapter}

    result = await runner._handle_model_command(_make_event())

    assert result is None
    assert [p["slug"] for p in adapter.providers] == ["copilot-acp", "copilot"]
    assert "GitHub Copilot ACP (copilot-acp)" in adapter.callback_result
    assert any(
        "GitHub Copilot ACP (copilot-acp)" in note
        for note in runner._pending_model_notes.values()
    )
