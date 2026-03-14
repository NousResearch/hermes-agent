import json

from hermes_cli.auth import _update_config_for_provider, get_active_provider
from hermes_cli.config import load_config, save_config, get_env_value
from hermes_cli.setup import setup_model_provider


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

    prompt_choices = iter([0, 2])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")

    def _fake_login_nous(*args, **kwargs):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))
        _update_config_for_provider("nous", "https://inference.example.com/v1")

    monkeypatch.setattr("hermes_cli.auth._login_nous", _fake_login_nous)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda *args, **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "api_key": "nous-test-key",
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

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: 3)

    prompt_values = iter(
        [
            "https://custom.example/v1",
            "custom-api-key",
            "custom/model",
            "",
        ]
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_values),
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert get_active_provider() is None
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "https://custom.example/v1"
    assert reloaded["model"]["default"] == "custom/model"


def test_codex_setup_uses_runtime_access_token_for_live_model_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    prompt_choices = iter([1, 0])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
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


def test_local_setup_updates_llm_model_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    prompt_choices = iter([4, 0])  # provider=local, model=first recommended
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: True)

    monkeypatch.setattr(
        "hermes_cli.local_provider.detect_hardware",
        lambda: {"chip": "Apple M4", "ram_gb": 64.0},
    )
    monkeypatch.setattr(
        "hermes_cli.local_provider.recommend_models",
        lambda ram_gb: [
            {
                "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "name": "Qwen 2.5 Coder 7B (4-bit)",
                "description": "Good coding, fast",
                "min_ram_gb": 8,
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.local_provider.installed_model_ids", lambda mids: set())
    monkeypatch.setattr("hermes_cli.local_provider.list_cached_model_ids", lambda limit=200: [])
    monkeypatch.setattr("hermes_cli.local_provider.check_mlx_lm_installed", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.local_provider.start_server",
        lambda model_id, port: {"pid": 1234, "reused": True},
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "local"
    assert reloaded["model"]["default"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert reloaded["local"]["model_id"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert reloaded["local"]["port"] == 8899
    assert reloaded["local"]["auto_start"] is True
    assert get_env_value("LLM_MODEL") == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def test_local_setup_allows_selecting_cached_non_recommended_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = load_config()

    # provider=local, model index=1 (cached non-recommended model)
    prompt_choices = iter([4, 1])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *args, **kwargs: True)

    monkeypatch.setattr(
        "hermes_cli.local_provider.detect_hardware",
        lambda: {"chip": "Apple M4", "ram_gb": 64.0},
    )
    monkeypatch.setattr(
        "hermes_cli.local_provider.recommend_models",
        lambda ram_gb: [
            {
                "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "name": "Qwen 2.5 Coder 7B (4-bit)",
                "description": "Good coding, fast",
                "min_ram_gb": 8,
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.local_provider.installed_model_ids", lambda mids: set())
    monkeypatch.setattr(
        "hermes_cli.local_provider.list_cached_model_ids",
        lambda limit=200: ["mlx-community/My-Custom-Model-4bit"],
    )
    monkeypatch.setattr("hermes_cli.local_provider.check_mlx_lm_installed", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.local_provider.start_server",
        lambda model_id, port: {"pid": 1234, "reused": True},
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded["model"]["provider"] == "local"
    assert reloaded["model"]["default"] == "mlx-community/My-Custom-Model-4bit"
    assert reloaded["local"]["model_id"] == "mlx-community/My-Custom-Model-4bit"
    assert get_env_value("LLM_MODEL") == "mlx-community/My-Custom-Model-4bit"
