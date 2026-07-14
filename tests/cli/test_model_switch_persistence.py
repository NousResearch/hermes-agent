"""Regression tests for /model persistence of provider endpoint tuples."""

from __future__ import annotations

import yaml

from hermes_cli.model_switch import ModelSwitchResult


class _PickerContext:
    user_providers = {}
    custom_providers = []

    def with_overrides(self, **kwargs):
        return self


class _StubCLI:
    agent = None
    model = "gpt-5.4"
    provider = "gpt55"
    requested_provider = "gpt55"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = "https://cc.auto-link.com.cn/pro/v1"
    _explicit_base_url = "https://cc.auto-link.com.cn/pro/v1"
    api_mode = "chat_completions"
    conversation_history = []
    _pending_model_switch_note = ""

    def _confirm_expensive_model_switch(self, result):
        return True


def _write_config(tmp_path, monkeypatch, model_value):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": model_value}), encoding="utf-8")

    monkeypatch.setattr("cli._hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


def _deepseek_result(provider_changed=False):
    return ModelSwitchResult(
        success=True,
        new_model="deepseek-v4-pro",
        target_provider="deepseek",
        provider_changed=provider_changed,
        api_key="sk-new",
        base_url="https://api.deepseek.com/v1",
        api_mode="chat_completions",
        provider_label="DeepSeek",
        is_global=True,
    )


def test_persist_model_switch_config_overwrites_stale_base_url(tmp_path, monkeypatch):
    cfg_path = _write_config(
        tmp_path,
        monkeypatch,
        {
            "default": "gpt-5.4",
            "provider": "gpt55",
            "base_url": "https://cc.auto-link.com.cn/pro/v1",
            "api_mode": "chat_completions",
        },
    )

    from hermes_cli.config import persist_model_switch_config

    persist_model_switch_config(
        new_model="deepseek-v4-pro",
        target_provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_mode="chat_completions",
    )

    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"] == {
        "default": "deepseek-v4-pro",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "api_mode": "chat_completions",
    }


def test_persist_model_switch_config_coerces_scalar_model(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, monkeypatch, "gpt-5.4")

    from hermes_cli.config import persist_model_switch_config

    persist_model_switch_config(
        new_model="deepseek-v4-pro",
        target_provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_mode="chat_completions",
    )

    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "deepseek-v4-pro"
    assert written["model"]["provider"] == "deepseek"
    assert written["model"]["base_url"] == "https://api.deepseek.com/v1"


def test_typed_model_switch_persists_full_tuple_even_without_provider_changed(tmp_path, monkeypatch):
    """The typed CLI path must not write only model.default/provider.

    This catches the real failure mode: switching to DeepSeek produced a result
    with the right base_url, but the typed /model path persisted only two fields,
    leaving the previous gpt55 endpoint in config.yaml.
    """
    cfg_path = _write_config(
        tmp_path,
        monkeypatch,
        {
            "default": "gpt-5.4",
            "provider": "gpt55",
            "base_url": "https://cc.auto-link.com.cn/pro/v1",
            "api_mode": "chat_completions",
        },
    )

    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: _PickerContext())
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: 1_000_000)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _deepseek_result(False))

    cli_mod.HermesCLI._handle_model_switch(_StubCLI(), "/model deepseek-v4-pro")

    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "deepseek-v4-pro"
    assert written["model"]["provider"] == "deepseek"
    assert written["model"]["base_url"] == "https://api.deepseek.com/v1"
    assert written["model"]["api_mode"] == "chat_completions"
