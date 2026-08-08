"""Regression tests for the bare ``/model`` picker when config.yaml has a
flat-string ``model:`` value instead of a nested dict.

``_handle_model_command`` reads the current model out of ``cfg["model"]``
behind an ``isinstance(model_cfg, dict)`` guard, so a scalar
``model: deepseek-v4-flash`` was dropped on the floor and the picker opened
with ``current_model=""`` — no "current" marker, and the switch callback
resolved against an empty current model. ``gateway/run.py`` already accepts
the scalar form (``_get_model_config``/``_resolve_gateway_model``), so this
is a picker-path-only gap.

The sibling file ``test_model_command_flat_string_config.py`` covers the
*persistence* path (``/model X --global``), not the display path.
"""

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _PickerAdapter:
    """Minimal adapter that advertises picker support."""

    async def send_model_picker(self, *args, **kwargs):  # pragma: no cover - not reached
        return True


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _setup(tmp_path, monkeypatch, model_yaml_value):
    """Write config.yaml with the given ``model:`` value, capture picker args."""
    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": model_yaml_value, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured = {}

    def _fake_list_picker_providers(**kwargs):
        captured.update(kwargs)
        return []  # empty -> handler falls through to the text list, fine for us

    monkeypatch.setattr(
        model_switch, "list_picker_providers", _fake_list_picker_providers
    )
    return captured


@pytest.mark.asyncio
async def test_picker_shows_flat_string_model_as_current(tmp_path, monkeypatch):
    """Regression: ``model: deepseek-v4-flash`` (flat string) must reach the
    picker as ``current_model``, not be dropped to an empty string."""
    captured = _setup(tmp_path, monkeypatch, "deepseek-v4-flash")

    runner = _make_runner()
    monkeypatch.setattr(
        type(runner), "_adapter_for_source", lambda self, src: _PickerAdapter(), raising=False
    )

    await runner._handle_model_command(_make_event())

    assert captured.get("current_model") == "deepseek-v4-flash", (
        "scalar model: should be passed to the picker, got %r"
        % (captured.get("current_model"),)
    )


@pytest.mark.asyncio
async def test_picker_shows_nested_dict_model_as_current(tmp_path, monkeypatch):
    """Companion: the nested-dict form keeps working (default + provider)."""
    captured = _setup(
        tmp_path,
        monkeypatch,
        {"default": "gpt-5.5", "provider": "openai", "base_url": "https://api.openai.com/v1"},
    )

    runner = _make_runner()
    monkeypatch.setattr(
        type(runner), "_adapter_for_source", lambda self, src: _PickerAdapter(), raising=False
    )

    await runner._handle_model_command(_make_event())

    assert captured.get("current_model") == "gpt-5.5"
    assert captured.get("current_provider") == "openai"
    assert captured.get("current_base_url") == "https://api.openai.com/v1"
