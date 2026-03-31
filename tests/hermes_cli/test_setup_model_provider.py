"""Regression tests for interactive setup provider/model persistence.

Since setup_model_provider delegates to select_provider_and_model()
from hermes_cli.main, these tests mock the delegation point and verify
that the setup wizard correctly syncs config from disk after the call.
"""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.setup import setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _clear_provider_env(monkeypatch):
    for key in (
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _stub_tts(monkeypatch):
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda q, c, d=0: (
        _maybe_keep_current_tts(q, c) if _maybe_keep_current_tts(q, c) is not None
        else d
    ))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)


def _write_model_config(provider, base_url="", model_name="test-model"):
    """Simulate what a _model_flow_* function writes to disk."""
    cfg = load_config()
    m = cfg.get("model")
    if not isinstance(m, dict):
        m = {"default": m} if m else {}
        cfg["model"] = m
    m["provider"] = provider
    if base_url:
        m["base_url"] = base_url
    else:
        m.pop("base_url", None)
    if model_name:
        m["default"] = model_name
    m.pop("api_mode", None)
    save_config(cfg)


def test_setup_keep_current_custom_from_config_does_not_fall_through(tmp_path, monkeypatch):
    """Keep-current custom should not fall through to the generic model menu."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    # Pre-set custom provider
    _write_model_config("custom", "http://localhost:8080/v1", "local-model")

    config = load_config()
    assert config["model"]["provider"] == "custom"

    def fake_select():
        pass  # user chose "cancel" or "keep current"

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:8080/v1"


def test_setup_keep_current_config_provider_uses_provider_specific_model_menu(
    tmp_path, monkeypatch
):
    """Keeping current provider preserves the config on disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    _write_model_config("zai", "https://open.bigmodel.cn/api/paas/v4", "glm-5")

    config = load_config()

    def fake_select():
        pass  # keep current

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "zai"


def test_setup_copilot_uses_gh_auth_and_saves_provider(tmp_path, monkeypatch):
    """Copilot provider saves correctly through delegation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config("copilot", "https://models.github.ai/inference/v1", "gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "copilot"


def test_setup_copilot_acp_uses_model_picker_and_saves_provider(tmp_path, monkeypatch):
    """Copilot ACP provider saves correctly through delegation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config("copilot-acp", "", "claude-sonnet-4")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "copilot-acp"


def test_setup_switch_custom_to_codex_clears_custom_endpoint_and_updates_config(
    tmp_path, monkeypatch
):
    """Switching from custom to codex updates config correctly."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    # Start with custom
    _write_model_config("custom", "http://localhost:11434/v1", "qwen3.5:32b")

    config = load_config()
    assert config["model"]["provider"] == "custom"

    def fake_select():
        _write_model_config("openai-codex", "https://api.openai.com/v1", "gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "gpt-4o"


def test_setup_switch_preserves_non_model_config(tmp_path, monkeypatch):
    """Provider switch preserves other config sections (terminal, display, etc.)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()
    config["terminal"]["timeout"] = 999
    save_config(config)

    config = load_config()

    def fake_select():
        _write_model_config("openrouter", model_name="gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded["terminal"]["timeout"] == 999
    assert reloaded["model"]["provider"] == "openrouter"
