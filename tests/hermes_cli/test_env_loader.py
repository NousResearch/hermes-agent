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
        "TELEGRAM_BOT_TOKEN=0123456789:test"
        "ANTHROPIC_API_KEY=sk-ant-test123\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"
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


def test_config_yaml_terminal_backend_overrides_stale_env(tmp_path, monkeypatch):
    """Regression for #29186: a leftover TERMINAL_ENV=docker in ~/.hermes/.env
    must not silently override the user's choice in config.yaml.  config.yaml
    is the documented source of truth, so its value must win after load."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_config_yaml_terminal_backend_overrides_stale_shell(tmp_path, monkeypatch):
    """config.yaml must also beat a stale TERMINAL_ENV exported in the shell
    (e.g. set in ~/.zshrc when the user was experimenting with docker)."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )

    monkeypatch.setenv("TERMINAL_ENV", "docker")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_no_config_yaml_leaves_env_value_alone(tmp_path, monkeypatch):
    """When config.yaml doesn't exist we must NOT clobber the .env value —
    that's still the user's setting."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"


def test_config_yaml_terminal_omitted_key_does_not_clear_env(tmp_path, monkeypatch):
    """If config.yaml has a terminal block but no `backend`, the .env value
    must survive (don't unset env vars the user didn't override)."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "terminal:\n  timeout: 600\n", encoding="utf-8"
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"
    assert os.getenv("TERMINAL_TIMEOUT") == "600"
