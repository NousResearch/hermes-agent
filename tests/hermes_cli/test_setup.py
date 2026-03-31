import json

from hermes_cli.auth import _update_config_for_provider, get_active_provider
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



def test_nous_oauth_setup_keeps_current_model_when_syncing_disk_provider(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 1  # Nous Portal
        if question == "Configure vision:":
            return len(choices) - 1
        if question == "Select default model:":
            assert choices[-1] == "Keep current (anthropic/claude-opus-4.6)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])

    def _fake_login_nous(*args, **kwargs):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))
        _update_config_for_provider("nous", "https://inference.example.com/v1")

    monkeypatch.setattr("hermes_cli.auth._login_nous", _fake_login_nous)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "api_key": "nous-key",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.fetch_nous_models",
        lambda *args, **kwargs: ["gemini-3-flash"],
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "nous"
    assert reloaded["model"]["base_url"] == "https://inference.example.com/v1"
    assert reloaded["model"]["default"] == "anthropic/claude-opus-4.6"


def test_custom_setup_clears_active_oauth_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)

    # _model_flow_custom uses builtins.input (URL, key, model, context_length)
    input_values = iter([
        "https://custom.example/v1",
        "custom-api-key",
        "custom/model",
        "",  # context_length (blank = auto-detect)
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {"models": ["m"], "probed_url": base_url + "/models"},
    )

    setup_model_provider(config)

    # Core assertion: switching to custom endpoint clears OAuth provider
    assert get_active_provider() is None

    # Simulate what the real setup wizard does: save_config(config) AFTER
    # setup_model_provider returns.  This is the step that previously
    # overwrote model.provider/base_url (#4172).
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded.get("model"), dict), (
        "model should be a dict after custom setup, not "
        + repr(type(reloaded.get("model")))
    )
    assert reloaded["model"].get("provider") == "custom"
    assert reloaded["model"].get("default") == "custom/model"
    assert "custom.example" in reloaded["model"].get("base_url", "")


def test_custom_setup_preserves_provider_after_wizard_save_config(
    tmp_path, monkeypatch
):
    """Regression test for #4172: the setup wizard's final save_config(config)
    must not overwrite model.provider/base_url that _model_flow_custom set.

    Simulates the full flow:
      1. load config (fresh install — model is a string)
      2. setup_model_provider picks custom
      3. wizard calls save_config(config) afterward
      4. verify resolve_requested_provider returns "custom"
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    # Sanity: fresh install has model as a string
    assert isinstance(config.get("model"), str) or config.get("model") is None

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3  # Custom endpoint
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)

    input_values = iter([
        "http://localhost:11434/v1",  # Ollama URL
        "",                           # no API key (local)
        "qwen3.5:32b",               # model name
        "",                           # context length (auto-detect)
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *a, **kw: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {"models": ["qwen3.5:32b"], "probed_url": base_url + "/models"},
    )

    # Full wizard cycle
    setup_model_provider(config)
    save_config(config)  # ← this is what the real wizard does

    # Verify config on disk
    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:11434/v1"
    assert reloaded["model"]["default"] == "qwen3.5:32b"
    assert "api_mode" not in reloaded["model"]

    # Verify the runtime resolver sees "custom", not "auto"
    from hermes_cli.runtime_provider import resolve_requested_provider
    assert resolve_requested_provider() == "custom"


def test_custom_setup_no_model_name_still_preserves_endpoint(
    tmp_path, monkeypatch
):
    """When the user enters a URL and key but skips the model name,
    model.provider and model.base_url must still survive the wizard's
    final save_config(config)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)

    input_values = iter([
        "http://192.168.1.50:8080/v1",  # URL
        "my-key",                        # API key
        "",                              # no model name
        "",                              # context length
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *a, **kw: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {"models": None, "probed_url": base_url + "/models"},
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://192.168.1.50:8080/v1"


def test_codex_setup_uses_runtime_access_token_for_live_model_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 2  # OpenAI Codex
        if question == "Select default model:":
            return 0
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-access-token",
        },
    )

    captured = {}

    def _fake_get_codex_model_ids(access_token=None):
        captured["access_token"] = access_token
        return ["gpt-5.2-codex", "gpt-5.2"]

    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        _fake_get_codex_model_ids,
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert captured["access_token"] == "codex-access-token"
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "gpt-5.2-codex"
    assert reloaded["model"]["base_url"] == "https://chatgpt.com/backend-api/codex"
