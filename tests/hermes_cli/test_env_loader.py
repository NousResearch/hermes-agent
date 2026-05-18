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


def test_profile_env_overlays_canonical_user_env(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    root_home = user_home / ".hermes"
    profile_home = root_home / "profiles" / "skippy"
    profile_home.mkdir(parents=True)
    (root_home / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-v1-root-key\nTELEGRAM_BOT_TOKEN=root-token\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "OPENROUTER_API_KEY=xxx\nTELEGRAM_BOT_TOKEN=profile-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile_home)

    assert loaded == [root_home / ".env", profile_home / ".env"]
    assert os.getenv("OPENROUTER_API_KEY") == "sk-or-v1-root-key"
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "profile-token"


def test_profile_env_does_not_inherit_root_bot_token(tmp_path, monkeypatch):
    user_home = tmp_path / "user"
    root_home = user_home / ".hermes"
    profile_home = root_home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (root_home / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-v1-root-key\nTELEGRAM_BOT_TOKEN=root-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile_home)

    assert loaded == [root_home / ".env"]
    assert os.getenv("OPENROUTER_API_KEY") == "sk-or-v1-root-key"
    assert os.getenv("TELEGRAM_BOT_TOKEN") is None


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
