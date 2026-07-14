from pathlib import Path

import yaml

from hermes_cli.model_switch import ModelSwitchResult


class _StubCLI:
    agent = None
    model = "old-model"
    provider = "openai-codex"
    requested_provider = "openai-codex"
    api_key = ""
    _explicit_api_key = ""
    base_url = "https://chatgpt.com/backend-api/codex"
    _explicit_base_url = "https://chatgpt.com/backend-api/codex"
    api_mode = "codex_responses"
    _pending_model_switch_note = ""


def _apply_global_switch(monkeypatch, config_dir: Path, result: ModelSwitchResult) -> dict:
    import cli as cli_mod

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  provider: openai-codex\n"
        "  default: gpt-5.5\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_hermes_home", config_dir)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *args, **kwargs: None)

    cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, True)

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_global_builtin_switch_replaces_previous_provider_base_url(monkeypatch, tmp_path):
    config = _apply_global_switch(
        monkeypatch,
        tmp_path,
        ModelSwitchResult(
            success=True,
            new_model="deepseek-v4-pro",
            target_provider="ollama-cloud",
            provider_changed=True,
            api_key="ollama-key",
            base_url="https://ollama.com/v1",
            api_mode="chat_completions",
        ),
    )

    assert config["model"]["provider"] == "ollama-cloud"
    assert config["model"]["default"] == "deepseek-v4-pro"
    assert config["model"]["base_url"] == "https://ollama.com/v1"


def test_global_custom_switch_persists_explicit_base_url(monkeypatch, tmp_path):
    config = _apply_global_switch(
        monkeypatch,
        tmp_path,
        ModelSwitchResult(
            success=True,
            new_model="local-model",
            target_provider="custom",
            provider_changed=True,
            api_key="local-key",
            base_url="http://127.0.0.1:1234/v1",
            api_mode="chat_completions",
        ),
    )

    assert config["model"]["provider"] == "custom"
    assert config["model"]["default"] == "local-model"
    assert config["model"]["base_url"] == "http://127.0.0.1:1234/v1"


def test_global_switch_without_resolved_base_url_clears_stale_value(monkeypatch, tmp_path):
    config = _apply_global_switch(
        monkeypatch,
        tmp_path,
        ModelSwitchResult(
            success=True,
            new_model="bedrock-model",
            target_provider="bedrock",
            provider_changed=True,
            api_key="aws-sdk",
            base_url="",
            api_mode="bedrock_converse",
        ),
    )

    assert config["model"].get("base_url") is None
