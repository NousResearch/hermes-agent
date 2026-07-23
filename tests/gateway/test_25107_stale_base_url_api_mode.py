"""Regression tests for #25107: gateway /model switch left a stale
``base_url``/never persisted ``api_mode`` in config.yaml when switching to a
custom provider whose resolver returned an empty ``base_url``.

Root cause: both the picker-tap path (``_on_model_selected`` in
``gateway/slash_commands.py``) and the typed ``/model X --global`` path
(``_finish_switch``) guarded the persist block with two *independent*
``if``s:

    if result.base_url:
        model_cfg["base_url"] = result.base_url
    if target_provider != "custom":
        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)

For a NAMED provider the second ``if`` always clears any stale value, so the
bug was invisible there. But for a ``custom`` provider with an empty
resolved ``base_url``, NEITHER branch fires: the old base_url survives
untouched, and ``api_mode`` was never written to ``model_cfg`` at all in
either branch (only present in the in-memory session override).

Fix: an explicit set-or-clear for both fields when the target provider is
custom, so a genuine switch always leaves config.yaml coherent with the
resolved result.
"""

import types

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakePickerAdapter:
    def __init__(self):
        self.captured_callback = None

    async def send_model_picker(self, *, on_model_selected, **kwargs):
        self.captured_callback = on_model_selected
        return types.SimpleNamespace(success=True)


def _make_runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result(*, base_url="", api_mode=""):
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="local-llama",
        target_provider="custom",
        provider_changed=True,
        api_key="sk-local",
        base_url=base_url,
        api_mode=api_mode,
        provider_label="Custom",
        is_global=True,
    )


def _setup_isolated_home(tmp_path, monkeypatch, model_yaml_value, *, base_url="", api_mode=""):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": model_yaml_value, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kw: [{"slug": "custom", "name": "Custom", "models": ["local-llama"]}],
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(base_url=base_url, api_mode=api_mode),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: 8192,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


_STALE_MODEL_CFG = {
    "default": "old-custom-model",
    "provider": "custom",
    "base_url": "https://old-stale-endpoint.example/v1",
    "api_key": "sk-stale",
    "api_mode": "anthropic_messages",
}


# ---------------------------------------------------------------------------
# Typed `/model X --global` path (_finish_switch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typed_switch_to_custom_clears_stale_base_url_and_api_mode(tmp_path, monkeypatch):
    """Switching to a custom provider whose resolver returns no base_url/
    api_mode must clear the previous custom endpoint's leftovers, not keep
    routing at the old host/protocol (#25107)."""
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, dict(_STALE_MODEL_CFG))

    result = await _make_runner()._handle_model_command(
        _make_event("/model local-llama --provider custom --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "local-llama"
    assert "base_url" not in written["model"], (
        "stale base_url from the old custom endpoint must be cleared"
    )
    assert "api_mode" not in written["model"], (
        "stale api_mode from the old custom endpoint must be cleared"
    )


@pytest.mark.asyncio
async def test_typed_switch_to_custom_persists_resolved_base_url_and_api_mode(tmp_path, monkeypatch):
    """The normal case: a custom-provider switch that DOES resolve a fresh
    base_url/api_mode must persist both (api_mode was never written here
    before the fix)."""
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        dict(_STALE_MODEL_CFG),
        base_url="https://new-endpoint.example/v1",
        api_mode="anthropic_messages",
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model local-llama --provider custom --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["base_url"] == "https://new-endpoint.example/v1"
    assert written["model"]["api_mode"] == "anthropic_messages"


# ---------------------------------------------------------------------------
# Picker-tap path (_on_model_selected)
# ---------------------------------------------------------------------------


async def _drive_picker(runner, event):
    sent = await runner._handle_model_command(event)
    assert sent is None
    adapter = runner.adapters[Platform.TELEGRAM]
    assert adapter.captured_callback is not None, "picker callback was not wired"
    return await adapter.captured_callback("12345", "local-llama", "custom")


@pytest.mark.asyncio
async def test_picker_tap_to_custom_clears_stale_base_url_and_api_mode(tmp_path, monkeypatch):
    """Same bug, picker-tap path: tapping a custom-provider model with an
    empty resolved base_url must clear the previous custom endpoint (#25107).
    """
    adapter = _FakePickerAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, dict(_STALE_MODEL_CFG))

    confirmation = await _drive_picker(_make_runner(adapter), _make_event("/model --global"))

    assert confirmation is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert "base_url" not in written["model"]
    assert "api_mode" not in written["model"]
