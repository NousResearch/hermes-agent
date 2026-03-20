"""Regression tests for interactive setup provider/model persistence."""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.setup import _print_setup_summary, setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


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


def test_setup_keep_current_custom_from_config_does_not_fall_through(tmp_path, monkeypatch):
    """Keep-current custom should not fall through to the generic model menu."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENAI_BASE_URL", "https://example.invalid/v1")
    save_env_value("OPENAI_API_KEY", "custom-key")

    config = load_config()
    config["model"] = {
        "default": "custom/model",
        "provider": "custom",
        "base_url": "https://example.invalid/v1",
    }
    save_config(config)

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[-1] == "Keep current (Custom: https://example.invalid/v1)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
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


def test_setup_custom_endpoint_saves_working_v1_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 3  # Custom endpoint
        if question == "Configure vision:":
            return len(choices) - 1  # Skip
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    # _model_flow_custom uses builtins.input (URL, key, model, context_length)
    input_values = iter([
        "http://localhost:8000",
        "local-key",
        "llm",
        "",  # context_length (blank = auto-detect)
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(input_values))

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url: {
            "models": ["llm"],
            "probed_url": "http://localhost:8000/v1/models",
            "resolved_base_url": "http://localhost:8000/v1",
            "suggested_base_url": "http://localhost:8000/v1",
            "used_fallback": True,
        },
    )

    setup_model_provider(config)

    env = _read_env(tmp_path)

    # _model_flow_custom saves env vars and config to disk
    assert env.get("OPENAI_BASE_URL") == "http://localhost:8000/v1"
    assert env.get("OPENAI_API_KEY") == "local-key"

    # The model config is saved as a dict by _model_flow_custom
    reloaded = load_config()
    model_cfg = reloaded.get("model", {})
    if isinstance(model_cfg, dict):
        assert model_cfg.get("provider") == "custom"
        assert model_cfg.get("default") == "llm"


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

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            captured["provider_choices"] = list(choices)
            assert choices[-1] == "Keep current (Anthropic)"
            return len(choices) - 1
        if question == "Configure vision:":
            assert question == "Configure vision:"
            assert choices[-1] == "Skip for now"
            return len(choices) - 1
        if question == "Select default model:":
            captured["model_choices"] = list(choices)
            return len(choices) - 1  # keep current model
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    assert captured["provider_choices"] is not None
    assert captured["model_choices"] is not None
    assert captured["model_choices"][0] == "claude-opus-4-6"
    assert "anthropic/claude-opus-4.6 (recommended)" not in captured["model_choices"]


def test_setup_keep_current_anthropic_can_configure_openai_vision_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    config["model"] = {
        "default": "claude-opus-4-6",
        "provider": "anthropic",
    }
    save_config(config)

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[-1] == "Keep current (Anthropic)"
            return len(choices) - 1
        if question == "Configure vision:":
            return 1
        if question == "Select vision model:":
            assert choices[-1] == "Use default (gpt-4o-mini)"
            return len(choices) - 1
        if question == "Select default model:":
            assert choices[-1] == "Keep current (claude-opus-4-6)"
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda message, *args, **kwargs: "sk-openai" if "OpenAI API key" in message else "",
    )
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: [])
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    env = _read_env(tmp_path)

    assert env.get("OPENAI_API_KEY") == "sk-openai"
    assert env.get("OPENAI_BASE_URL") == "https://api.openai.com/v1"
    assert env.get("AUXILIARY_VISION_MODEL") == "gpt-4o-mini"


def test_setup_copilot_uses_gh_auth_and_saves_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[14] == "GitHub Copilot (uses GITHUB_TOKEN or gh auth token)"
            return 14
        if question == "Select default model:":
            assert "gpt-4.1" in choices
            assert "gpt-5.4" in choices
            return choices.index("gpt-5.4")
        if question == "Select reasoning effort:":
            assert "low" in choices
            assert "high" in choices
            return choices.index("high")
        if question == "Configure vision:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        raise AssertionError(f"Unexpected prompt call: {message}")

    def fake_get_auth_status(provider_id):
        if provider_id == "copilot":
            return {"logged_in": True}
        return {"logged_in": False}

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", fake_get_auth_status)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "gh-cli-token",
            "base_url": "https://api.githubcopilot.com",
            "source": "gh auth token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key: [
            {
                "id": "gpt-4.1",
                "capabilities": {"type": "chat", "supports": {}},
                "supported_endpoints": ["/chat/completions"],
            },
            {
                "id": "gpt-5.4",
                "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                "supported_endpoints": ["/responses"],
            },
        ],
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    env = _read_env(tmp_path)
    reloaded = load_config()

    assert env.get("GITHUB_TOKEN") is None
    assert reloaded["model"]["provider"] == "copilot"
    assert reloaded["model"]["base_url"] == "https://api.githubcopilot.com"
    assert reloaded["model"]["default"] == "gpt-5.4"
    assert reloaded["model"]["api_mode"] == "codex_responses"
    assert reloaded["agent"]["reasoning_effort"] == "high"


def test_setup_copilot_acp_uses_model_picker_and_saves_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            assert choices[15] == "GitHub Copilot ACP (spawns `copilot --acp --stdio`)"
            return 15
        if question == "Select default model:":
            assert "gpt-4.1" in choices
            assert "gpt-5.4" in choices
            return choices.index("gpt-5.4")
        if question == "Configure vision:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt(message, *args, **kwargs):
        raise AssertionError(f"Unexpected prompt call: {message}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda provider_id: {"logged_in": provider_id == "copilot-acp"})
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": "copilot",
            "api_key": "gh-cli-token",
            "base_url": "https://api.githubcopilot.com",
            "source": "gh auth token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key: [
            {
                "id": "gpt-4.1",
                "capabilities": {"type": "chat", "supports": {}},
                "supported_endpoints": ["/chat/completions"],
            },
            {
                "id": "gpt-5.4",
                "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
                "supported_endpoints": ["/responses"],
            },
        ],
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert reloaded["model"]["provider"] == "copilot-acp"
    assert reloaded["model"]["base_url"] == "acp://copilot"
    assert reloaded["model"]["default"] == "gpt-5.4"
    assert reloaded["model"]["api_mode"] == "chat_completions"


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

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 1
        if question == "Select default model:":
            return 0
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth.detect_external_credentials", lambda: [])
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-...oken",
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


def test_setup_summary_marks_codex_auth_as_vision_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    (tmp_path / "auth.json").write_text(
        '{"active_provider":"openai-codex","providers":{"openai-codex":{"tokens":{"access_token": "***", "refresh_token": "***"}}}}'
    )

    monkeypatch.setattr("shutil.which", lambda _name: None)

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Vision (image analysis)" in output
    assert "missing run 'hermes setup' to configure" not in output
    assert "Mixture of Agents" in output
    assert "missing OPENROUTER_API_KEY" in output


def test_setup_summary_marks_anthropic_auth_as_vision_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: ["anthropic"])

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Vision (image analysis)" in output
    assert "missing run 'hermes setup' to configure" not in output
