"""Regression coverage for profile-scoped gateway ``/model`` reads."""

from types import SimpleNamespace

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


def _make_minimal_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _write_null_provider_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: configured-model\n"
        "  provider: null\n"
        "  base_url: null\n",
        encoding="utf-8",
    )


def _model_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="null-provider-chat",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_typed_model_command_treats_null_config_route_as_absent(
    tmp_path, monkeypatch
):
    """YAML null route fields must not reach switch_model as None."""
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    _write_null_provider_config(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    captured = {}

    def _fake_switch(**kwargs):
        captured.update(kwargs)
        return ModelSwitchResult(success=False, error_message="stop after capture")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)

    reply = await _make_minimal_runner()._handle_model_command(
        _model_event("/model gpt-6-astra")
    )

    assert reply is not None and "stop after capture" in reply
    assert captured["current_model"] == "configured-model"
    assert captured["current_provider"] == "openrouter"
    assert captured["current_base_url"] == ""


@pytest.mark.asyncio
async def test_bare_model_command_treats_null_config_route_as_absent(
    tmp_path, monkeypatch
):
    """The text picker fallback must not call get_label with None."""
    import gateway.run as gateway_run

    _write_null_provider_config(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", lambda **_kwargs: []
    )

    reply = await _make_minimal_runner()._handle_model_command(_model_event("/model"))

    assert reply is not None
    assert "configured-model" in reply
    assert "OpenRouter" in reply
