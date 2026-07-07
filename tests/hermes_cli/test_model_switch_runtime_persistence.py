"""Regression tests for global /model runtime persistence (#25106)."""
from __future__ import annotations

from typing import Any

from hermes_cli.model_switch import ModelSwitchResult


def _result(**overrides: Any) -> ModelSwitchResult:
    values: dict[str, Any] = dict(
        success=True,
        new_model="kimi-k2.6",
        target_provider="kimi-coding",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://api.kimi.com/coding",
        api_mode="anthropic_messages",
        warning_message="",
        provider_label="Kimi Coding",
        resolved_via_alias=False,
        capabilities=None,
        model_info=None,
        is_global=True,
    )
    values.update(overrides)
    return ModelSwitchResult(**values)  # type: ignore[arg-type]


def test_persist_model_switch_runtime_state_writes_complete_tuple(monkeypatch):
    import cli as cli_mod

    saved = {}
    preserve_keys = set()

    def fake_save_config(cfg, **kwargs):
        saved.update(cfg)
        preserve_keys.update(kwargs.get("preserve_keys") or set())

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "model": {
                "default": "old-model",
                "provider": "custom",
                "base_url": "https://stale.example/v1",
            },
            "display": {"skin": "plain"},
        },
    )
    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)

    assert cli_mod._persist_model_switch_runtime_state(_result()) is True

    assert saved == {
        "model": {
            "default": "kimi-k2.6",
            "provider": "kimi-coding",
            "base_url": "https://api.kimi.com/coding",
            "api_mode": "anthropic_messages",
        },
        "display": {"skin": "plain"},
    }
    assert ("model", "default") in preserve_keys
    assert ("model", "provider") in preserve_keys
    assert ("model", "base_url") in preserve_keys
    assert ("model", "api_mode") in preserve_keys


def test_persist_model_switch_runtime_state_clears_stale_runtime_fields(monkeypatch):
    import cli as cli_mod

    saved = {}

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "model": {
                "default": "old-model",
                "provider": "custom",
                "base_url": "https://stale.example/v1",
                "api_mode": "anthropic_messages",
            }
        },
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg, **_kwargs: saved.update(cfg))

    result = _result(
        new_model="gpt-5.5",
        target_provider="openai-codex",
        base_url="",
        api_mode="",
        api_key="",
        provider_label="OpenAI Codex",
    )

    assert cli_mod._persist_model_switch_runtime_state(result) is True
    assert saved["model"]["default"] == "gpt-5.5"
    assert saved["model"]["provider"] == "openai-codex"
    assert "base_url" not in saved["model"]
    assert "api_mode" not in saved["model"]


def test_persist_model_switch_runtime_state_refuses_managed_mode(monkeypatch):
    import cli as cli_mod

    called = {"save": False}

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: True)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"model": {}})

    def fake_save_config(*_args, **_kwargs):
        called["save"] = True

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)

    assert cli_mod._persist_model_switch_runtime_state(_result()) is False
    assert called["save"] is False


class _StubCLI:
    agent = None
    model = "old-model"
    provider = "custom"
    requested_provider = "custom"
    api_key = "old-key"
    _explicit_api_key = "old-key"
    base_url = "https://stale.example/v1"
    _explicit_base_url = "https://stale.example/v1"
    api_mode = "chat_completions"
    conversation_history = []
    _pending_model_switch_note = ""

    def _confirm_expensive_model_switch(self, _result):
        return True


def test_typed_global_model_switch_uses_complete_runtime_persistence(monkeypatch):
    """The user-facing typed /model path must not bypass the shared helper."""
    import cli as cli_mod

    saved = {}
    printed: list[str] = []

    monkeypatch.setattr(cli_mod, "_cprint", lambda s="", *a, **k: printed.append(str(s)))
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: _result())
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"default": "old-model", "provider": "custom", "base_url": "https://stale.example/v1"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg, **_kwargs: saved.update(cfg))

    cli_mod.HermesCLI._handle_model_switch(
        _StubCLI(),
        "/model kimi-k2.6 --provider kimi-coding --global",
    )

    assert saved["model"]["default"] == "kimi-k2.6"
    assert saved["model"]["provider"] == "kimi-coding"
    assert saved["model"]["base_url"] == "https://api.kimi.com/coding"
    assert saved["model"]["api_mode"] == "anthropic_messages"
    assert any("Saved to config.yaml" in line for line in printed)
