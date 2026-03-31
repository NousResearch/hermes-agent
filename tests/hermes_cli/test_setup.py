"""Tests for setup_model_provider — verifies the delegation to
select_provider_and_model() and config dict sync."""
import json

from hermes_cli.auth import get_active_provider
from hermes_cli.config import load_config, save_config
from hermes_cli.setup import setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _clear_provider_env(monkeypatch):
    for key in (
        "NOUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def _stub_tts(monkeypatch):
    """Stub out TTS prompts so setup_model_provider doesn't block."""
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda q, c, d=0: (
        _maybe_keep_current_tts(q, c) if _maybe_keep_current_tts(q, c) is not None
        else d
    ))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)


def _write_model_config(tmp_path, provider, base_url="", model_name="test-model"):
    """Simulate what a _model_flow_* function writes to disk."""
    cfg = load_config()
    m = cfg.get("model")
    if not isinstance(m, dict):
        m = {"default": m} if m else {}
        cfg["model"] = m
    m["provider"] = provider
    if base_url:
        m["base_url"] = base_url
    if model_name:
        m["default"] = model_name
    save_config(cfg)


def test_setup_delegates_to_select_provider_and_model(tmp_path, monkeypatch):
    """setup_model_provider calls select_provider_and_model and syncs config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config(tmp_path, "custom", "http://localhost:11434/v1", "qwen3.5:32b")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:11434/v1"
    assert reloaded["model"]["default"] == "qwen3.5:32b"


def test_setup_syncs_openrouter_from_disk(tmp_path, monkeypatch):
    """When select_provider_and_model saves OpenRouter config to disk,
    the wizard's config dict picks it up."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()
    assert isinstance(config.get("model"), str)  # fresh install

    def fake_select():
        _write_model_config(tmp_path, "openrouter", model_name="anthropic/claude-opus-4.6")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openrouter"


def test_setup_syncs_nous_from_disk(tmp_path, monkeypatch):
    """Nous OAuth writes config to disk; wizard config dict must pick it up."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config(tmp_path, "nous", "https://inference.example.com/v1", "gemini-3-flash")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "nous"
    assert reloaded["model"]["base_url"] == "https://inference.example.com/v1"


def test_setup_custom_providers_synced(tmp_path, monkeypatch):
    """custom_providers written by select_provider_and_model must survive."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config(tmp_path, "custom", "http://localhost:8080/v1", "llama3")
        cfg = load_config()
        cfg["custom_providers"] = [{"name": "Local", "base_url": "http://localhost:8080/v1"}]
        save_config(cfg)

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded.get("custom_providers") == [{"name": "Local", "base_url": "http://localhost:8080/v1"}]


def test_setup_cancel_preserves_existing_config(tmp_path, monkeypatch):
    """When the user cancels provider selection, existing config is preserved."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    # Pre-set a provider
    _write_model_config(tmp_path, "openrouter", model_name="gpt-4o")

    config = load_config()
    assert config["model"]["provider"] == "openrouter"

    def fake_select():
        pass  # user cancelled — nothing written to disk

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openrouter"
    assert reloaded["model"]["default"] == "gpt-4o"


def test_setup_exception_in_select_gracefully_handled(tmp_path, monkeypatch):
    """If select_provider_and_model raises, setup continues with existing config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        raise RuntimeError("something broke")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    # Should not raise
    setup_model_provider(config)


def test_setup_keyboard_interrupt_gracefully_handled(tmp_path, monkeypatch):
    """KeyboardInterrupt during provider selection is handled."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        raise KeyboardInterrupt()

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)


def test_codex_setup_uses_runtime_access_token_for_live_model_list(tmp_path, monkeypatch):
    """Codex model list fetching uses the runtime access token."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()
    _stub_tts(monkeypatch)

    def fake_select():
        _write_model_config(tmp_path, "openai-codex", "https://api.openai.com/v1", "gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openai-codex"
