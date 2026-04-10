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


def test_cwd_env_fills_missing_values(tmp_path, monkeypatch):
    """CWD .env fills missing values when user env exists."""
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")

    cwd = tmp_path / "project"
    cwd.mkdir()
    cwd_env = cwd / ".env"
    cwd_env.write_text("OPENAI_BASE_URL=https://cwd.example/v1\nMY_PROJECT_KEY=cwd-key\n", encoding="utf-8")

    monkeypatch.delenv("MY_PROJECT_KEY", raising=False)
    monkeypatch.chdir(cwd)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [user_env, cwd_env]
    # User env wins for shared keys
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    # CWD env fills missing keys
    assert os.getenv("MY_PROJECT_KEY") == "cwd-key"


def test_cwd_env_overrides_when_no_user_env(tmp_path, monkeypatch):
    """CWD .env overrides shell values when no user env exists."""
    home = tmp_path / "hermes"
    # No .env in home

    cwd = tmp_path / "project"
    cwd.mkdir()
    cwd_env = cwd / ".env"
    cwd_env.write_text("OPENAI_BASE_URL=https://cwd.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.chdir(cwd)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [cwd_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://cwd.example/v1"


def test_cwd_env_skipped_when_same_as_user_env(tmp_path, monkeypatch):
    """CWD .env is not loaded twice if it resolves to the same file as user env."""
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    user_env.write_text("MY_KEY=user-value\n", encoding="utf-8")

    # CWD is the hermes home dir itself
    monkeypatch.chdir(home)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [user_env]


def test_cwd_env_disabled_by_default(tmp_path, monkeypatch):
    """CWD .env is not loaded when cwd_env=False (default)."""
    home = tmp_path / "hermes"

    cwd = tmp_path / "project"
    cwd.mkdir()
    cwd_env = cwd / ".env"
    cwd_env.write_text("MY_CWD_KEY=should-not-load\n", encoding="utf-8")

    monkeypatch.delenv("MY_CWD_KEY", raising=False)
    monkeypatch.chdir(cwd)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == []
    assert os.getenv("MY_CWD_KEY") is None


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
