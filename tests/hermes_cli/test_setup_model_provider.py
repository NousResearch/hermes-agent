"""Regression tests for interactive setup provider/model persistence."""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.setup import setup_model_provider


def _read_env(home):
    env_path = home / ".env"
    data = {}
    if not env_path.exists():
        return data
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k] = v
    return data


def _clear_provider_env(monkeypatch):
    for key in (
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_setup_keep_current_custom_from_config_does_not_fall_through(tmp_path, monkeypatch):
    """Keep-current custom should not fall through to the generic model menu."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["model"] = {
        "default": "custom/model",
        "provider": "custom",
        "base_url": "https://example.invalid/v1",
    }
    save_config(config)

    calls = {"count": 0}

    def fake_prompt_choice(_question, choices, default=0):
        calls["count"] += 1
        if calls["count"] == 1:
            assert choices[-1] == "Keep current (Custom: https://example.invalid/v1)"
            return len(choices) - 1
        raise AssertionError("Model menu should not appear for keep-current custom")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["default"] == "custom/model"
    assert reloaded["model"]["base_url"] == "https://example.invalid/v1"
    assert calls["count"] == 1


def test_setup_keep_current_config_provider_uses_provider_specific_model_menu(tmp_path, monkeypatch):
    """Keep-current should respect config-backed providers, not fall back to OpenRouter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["model"] = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
    }
    save_config(config)

    captured = {"provider_choices": None, "model_choices": None}
    calls = {"count": 0}

    def fake_prompt_choice(_question, choices, default=0):
        calls["count"] += 1
        if calls["count"] == 1:
            captured["provider_choices"] = list(choices)
            assert choices[-1] == "Keep current (Anthropic)"
            return len(choices) - 1
        if calls["count"] == 2:
            captured["model_choices"] = list(choices)
            return len(choices) - 1  # keep current model
        raise AssertionError("Unexpected extra prompt_choice call")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: [])

    setup_model_provider(config)
    save_config(config)

    assert captured["provider_choices"] is not None
    assert captured["model_choices"] is not None
    assert captured["model_choices"][0] == "claude-opus-4-6"
    assert "anthropic/claude-opus-4.6 (recommended)" not in captured["model_choices"]
    assert calls["count"] == 2


def test_setup_switch_custom_to_codex_clears_custom_endpoint_and_updates_config(tmp_path, monkeypatch):
    """Switching from custom to Codex should clear custom endpoint overrides."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    save_env_value("OPENAI_BASE_URL", "https://example.invalid/v1")
    save_env_value("OPENAI_API_KEY", "sk-custom")
    save_env_value("OPENROUTER_API_KEY", "sk-or")

    config = load_config()
    config["model"] = {
        "default": "custom/model",
        "provider": "custom",
        "base_url": "https://example.invalid/v1",
    }
    save_config(config)

    picks = iter([1, 0])
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: next(picks))
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-access-token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda **kwargs: ["openai/gpt-5.3-codex", "openai/gpt-5-codex-mini"],
    )

    setup_model_provider(config)
    save_config(config)

    env = _read_env(tmp_path)
    reloaded = load_config()

    assert env.get("OPENAI_BASE_URL") == ""
    assert env.get("OPENAI_API_KEY") == ""
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "openai/gpt-5.3-codex"
    assert reloaded["model"]["base_url"] == "https://chatgpt.com/backend-api/codex"
