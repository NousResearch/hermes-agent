"""Regression coverage for profile-scoped gateway ``/model`` reads."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _CapturingPickerAdapter:
    def __init__(self):
        self.kwargs = None

    async def send_model_picker(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(success=True)


def _make_event():
    return MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="secondary-chat",
            chat_type="group",
        ),
    )


@pytest.mark.asyncio
async def test_model_picker_reads_routed_profile_config(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    default_home = tmp_path / "default"
    secondary_home = tmp_path / "profiles" / "secondary"
    default_home.mkdir()
    secondary_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: default-provider\n",
        encoding="utf-8",
    )
    (secondary_home / "config.yaml").write_text(
        "model:\n  default: secondary-model\n  provider: secondary-provider\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **_kwargs: [
            {
                "slug": "secondary-provider",
                "name": "Secondary Provider",
                "is_current": True,
                "models": ["secondary-model"],
                "total_models": 1,
            }
        ],
    )

    runner = object.__new__(GatewayRunner)
    adapter = _CapturingPickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._resolve_profile_home_for_source = lambda _source: secondary_home
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._reply_anchor_for_event = lambda *_args, **_kwargs: None

    result = await runner._handle_model_command(_make_event())

    assert result is None
    assert adapter.kwargs is not None
    assert adapter.kwargs["current_model"] == "secondary-model"
    assert adapter.kwargs["current_provider"] == "secondary-provider"


@pytest.mark.asyncio
async def test_model_picker_rehydrates_persisted_session_override_after_restart(tmp_path, monkeypatch):
    """Bare /model must report the persisted session choice after process-local state is lost."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: global-model\n  provider: global-provider\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_key": "test-key",
            "base_url": "https://api.venice.test/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **_kwargs: [],
    )

    runner = object.__new__(GatewayRunner)
    adapter = _CapturingPickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._voice_mode = {}
    runner._session_model_overrides = {}  # fresh process: no in-memory override
    runner._running_agents = {}
    runner.session_store = MagicMock()
    runner.session_store.get_model_override.return_value = {
        "model": "persisted-session-model",
        "provider": "venice",
        "base_url": "https://api.venice.test/v1",
    }
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._reply_anchor_for_event = lambda *_args, **_kwargs: None

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "Current: `persisted-session-model` on venice" in result
    runner.session_store.get_model_override.assert_called_once()
