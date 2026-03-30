import json
import sys
import types

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

    # _model_flow_custom writes config via its own load/save cycle
    reloaded = load_config()
    if isinstance(reloaded.get("model"), dict):
        assert reloaded["model"].get("provider") == "custom"
        assert reloaded["model"].get("default") == "custom/model"


def test_codex_setup_uses_runtime_access_token_for_live_model_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 2  # OpenAI Codex
        if question == "Configure vision:":
            return len(choices) - 1
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


def test_nous_setup_sets_managed_openai_tts_when_unconfigured(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 1
        if question == "Configure vision:":
            return len(choices) - 1
        if question == "Select default model:":
            return len(choices) - 1
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])

    def _fake_login_nous(*args, **kwargs):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {"nous": {"access_token": "nous-token"}}}))
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

    out = capsys.readouterr().out
    assert config["tts"]["provider"] == "openai"
    assert "Nous subscription enables managed web tools" in out
    assert "OpenAI TTS via your Nous subscription" in out


def test_nous_setup_preserves_existing_tts_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["tts"] = {"provider": "elevenlabs"}

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 1
        if question == "Configure vision:":
            return len(choices) - 1
        if question == "Select default model:":
            return len(choices) - 1
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.auth._login_nous",
        lambda *args, **kwargs: (tmp_path / "auth.json").write_text(
            json.dumps({"active_provider": "nous", "providers": {"nous": {"access_token": "nous-token"}}})
        ),
    )
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

    assert config["tts"]["provider"] == "elevenlabs"


def test_modal_setup_can_use_nous_subscription_without_modal_creds(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 2
        if question == "Select how Modal execution should be billed:":
            return 0
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        assert "Modal Token" not in message
        raise AssertionError(f"Unexpected prompt call: {message}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup._prompt_container_resources", lambda config: None)
    monkeypatch.setattr(
        "hermes_cli.setup.get_nous_subscription_features",
        lambda config: type("Features", (), {"nous_auth_present": True})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.managed_tool_gateway",
        types.SimpleNamespace(
            is_managed_tool_gateway_ready=lambda vendor: vendor == "modal",
            resolve_managed_tool_gateway=lambda vendor: None,
        ),
    )

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    out = capsys.readouterr().out
    assert config["terminal"]["backend"] == "modal"
    assert config["terminal"]["modal_mode"] == "managed"
    assert "bill to your subscription" in out


def test_modal_setup_persists_direct_mode_when_user_chooses_their_own_account(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 2
        if question == "Select how Modal execution should be billed:":
            return 1
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    prompt_values = iter(["token-id", "token-secret", ""])

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_values))
    monkeypatch.setattr("hermes_cli.setup._prompt_container_resources", lambda config: None)
    monkeypatch.setattr(
        "hermes_cli.setup.get_nous_subscription_features",
        lambda config: type("Features", (), {"nous_auth_present": True})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.managed_tool_gateway",
        types.SimpleNamespace(
            is_managed_tool_gateway_ready=lambda vendor: vendor == "modal",
            resolve_managed_tool_gateway=lambda vendor: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "swe_rex", object())

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "modal"
    assert config["terminal"]["modal_mode"] == "direct"
