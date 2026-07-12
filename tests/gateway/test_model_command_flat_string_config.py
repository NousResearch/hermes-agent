"""Regression tests for gateway /model --global persistence when config.yaml
has a flat-string ``model:`` value instead of a nested dict.

Before fix: ``cfg.setdefault("model", {})`` returned the existing string and
the next assignment raised ``TypeError: 'str' object does not support item
assignment``, so every ``/model X --global`` from Telegram/Discord crashed
silently and the user-visible result was "switch failed" with no persist.

After fix: the persist block coerces a scalar ``model:`` into a nested dict
before mutation, so ``--global`` succeeds and the config is rewritten in
the proper ``model: {default: ..., provider: ...}`` form.
"""

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
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    """Build a successful ModelSwitchResult that bypasses real provider resolution."""
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openrouter",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        is_global=True,
    )


def _setup_isolated_home(tmp_path, monkeypatch, model_yaml_value):
    """Write a config.yaml with the given ``model:`` value and stub the heavy bits."""
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
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    # save_config writes to ``get_hermes_home() / config.yaml`` — point it here.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_flat_string_model(tmp_path, monkeypatch):
    """Regression: ``model: deepseek-v4-flash`` (flat string) used to crash
    the gateway ``/model X --global`` persist branch with TypeError. After
    the fix, the flat string is coerced to ``{"default": ...}`` and the new
    model+provider are persisted on top.
    """
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, "deepseek-v4-flash")

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    # Sanity: the handler returned a success-looking message (not a crash log).
    assert result is not None
    assert "gpt-5.5" in result

    # The persist block must have rewritten config.yaml as a nested dict.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict), (
        "model: should be coerced to a dict, got %r" % (written["model"],)
    )
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"
    assert "base_url" not in written["model"]


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_missing_model(tmp_path, monkeypatch):
    """Companion case: ``model:`` key absent entirely. setdefault would have
    worked here, but the coercion branch also has to handle this cleanly.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"providers": {}}), encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict)
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_proper_dict_model(tmp_path, monkeypatch):
    """Already-correct nested dict must still work — no regression on the
    common case.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_model_no_flag_does_not_persist_by_default(tmp_path, monkeypatch):
    """A plain ``/model X`` (no --global) is session-scoped by default on messaging platforms.

    This differs from CLI (which persists by default) because messaging
    platform sessions are typically ephemeral and per-chat, and unintended
    global config writes cause cross-session pollution. See #63083.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    runner = _make_runner()
    result = await runner._handle_model_command(
        _make_event("/model gpt-5.5")
    )

    assert result is not None
    assert "gpt-5.5" in result
    # The session override IS applied in-memory.
    assert runner._session_model_overrides
    assert any(
        ov.get("model") == "gpt-5.5"
        for ov in runner._session_model_overrides.values()
    )
    # But config.yaml is untouched — the override is in-memory only.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "old-model"
    assert written["model"]["provider"] == "openai-codex"


@pytest.mark.asyncio
async def test_model_session_flag_does_not_persist(tmp_path, monkeypatch):
    """``/model X --session`` opts out of persistence even under the new default."""
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --session")
    )

    assert result is not None
    assert "gpt-5.5" in result
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Config untouched — the session override is in-memory only.
    assert written["model"]["default"] == "old-model"
