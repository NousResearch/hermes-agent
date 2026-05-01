import importlib
import os
import sys
from pathlib import Path

from hermes_cli.env_loader import load_hermes_dotenv


def test_user_env_overrides_stale_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://new.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"


def test_project_env_overrides_stale_shell_values_when_user_env_missing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://project.example/v1"


def test_project_env_is_sanitized_before_loading(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "TELEGRAM_BOT_TOKEN=8356550917:AAGGEkzg06Hrc3Hjb3Sa1jkGVDOdU_lYy2Q"
        "ANTHROPIC_API_KEY=sk-ant-test123\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "8356550917:AAGGEkzg06Hrc3Hjb3Sa1jkGVDOdU_lYy2Q"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-test123"


def test_user_env_takes_precedence_over_project_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    project_env = tmp_path / ".env"
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\nOPENAI_API_KEY=project-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [user_env, project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    assert os.getenv("OPENAI_API_KEY") == "project-key"


def test_main_import_applies_user_env_over_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://new.example/v1\nHERMES_INFERENCE_PROVIDER=custom\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

    sys.modules.pop("hermes_cli.main", None)
    importlib.import_module("hermes_cli.main")

    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"
    assert os.getenv("HERMES_INFERENCE_PROVIDER") == "custom"


def test_decode_vault_broker_env_accepts_double_encoded_json():
    from hermes_cli.env_loader import _decode_vault_broker_env

    payload = '"{\\"env\\": {\\"TELEGRAM_BOT_TOKEN\\": \\"token-value\\"}}"'

    assert _decode_vault_broker_env(payload) == {"TELEGRAM_BOT_TOKEN": "token-value"}


def test_load_hermes_dotenv_injects_missing_telegram_token_from_vault(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "hermes"
    vault_bin = home / "hermes-agent" / "venv" / "bin" / "hermes-vault"
    vault_bin.parent.mkdir(parents=True)
    vault_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    (home / ".env").write_text("# TELEGRAM_BOT_TOKEN migrated to vault\n", encoding="utf-8")

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_VAULT_ENABLE_RUNTIME_INJECTION", "1")
    monkeypatch.setenv("HERMES_VAULT_POLICY", str(tmp_path / "policy.yaml"))
    monkeypatch.setattr(env_loader, "_VAULT_INJECTED", False)
    monkeypatch.setattr(env_loader.sys, "platform", "darwin")

    calls = []

    class Result:
        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:3] == ["security", "find-generic-password", "-s"]:
            return Result(stdout="vault-passphrase\n")
        assert cmd[:4] == [str(vault_bin), "broker", "env", "telegram"]
        assert kwargs["env"]["HERMES_VAULT_PASSPHRASE"] == "vault-passphrase"
        return Result(stdout='{"env": {"TELEGRAM_BOT_TOKEN": "vault-token"}}')

    monkeypatch.setattr(env_loader.subprocess, "run", fake_run)

    loaded = env_loader.load_hermes_dotenv(hermes_home=home)

    assert loaded == [home / ".env"]
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "vault-token"
    assert len(calls) == 2


def test_load_hermes_dotenv_skips_vault_when_token_already_present(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "existing-token")
    monkeypatch.setenv("HERMES_VAULT_ENABLE_RUNTIME_INJECTION", "1")
    monkeypatch.setattr(env_loader, "_VAULT_INJECTED", False)

    def fail_run(*args, **kwargs):
        raise AssertionError("vault broker should not run when env already has token")

    monkeypatch.setattr(env_loader.subprocess, "run", fail_run)

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "existing-token"


def test_load_hermes_dotenv_skips_vault_on_non_macos(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "hermes"
    vault_bin = home / "hermes-agent" / "venv" / "bin" / "hermes-vault"
    vault_bin.parent.mkdir(parents=True)
    vault_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_VAULT_ENABLE_RUNTIME_INJECTION", "1")
    monkeypatch.setattr(env_loader, "_VAULT_INJECTED", False)
    monkeypatch.setattr(env_loader.sys, "platform", "linux")

    def fail_run(*args, **kwargs):
        raise AssertionError("macOS keychain lookup should not run on non-macOS")

    monkeypatch.setattr(env_loader.subprocess, "run", fail_run)

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert "TELEGRAM_BOT_TOKEN" not in os.environ
