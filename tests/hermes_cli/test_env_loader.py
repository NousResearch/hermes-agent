import importlib
import os
import sys

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


def test_null_bytes_in_user_env_are_stripped(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # Null bytes can be introduced when copy-pasting API keys.
    env_file.write_text("GLM_API_KEY=abc\x00\x00\nOPENAI_API_KEY=sk-123\n", encoding="utf-8")

    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GLM_API_KEY") == "abc"
    assert os.getenv("OPENAI_API_KEY") == "sk-123"


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


def test_env_d_files_loaded_and_override(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("API_KEY=base\nMODEL=gpt-4\n", encoding="utf-8")
    env_d = home / "env.d"
    env_d.mkdir()
    (env_d / "secrets.env").write_text("API_KEY=secret\n", encoding="utf-8")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert (home / ".env") in loaded
    assert (env_d / "secrets.env") in loaded
    assert os.getenv("API_KEY") == "secret"
    assert os.getenv("MODEL") == "gpt-4"


def test_env_d_multiple_files_alphabetical_order(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_d = home / "env.d"
    env_d.mkdir()
    (env_d / "a.env").write_text("KEY=first\n", encoding="utf-8")
    (env_d / "b.env").write_text("KEY=second\n", encoding="utf-8")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_d / "a.env", env_d / "b.env"]
    assert os.getenv("KEY") == "second"


def test_env_d_missing_dir_no_error(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == []


def test_env_d_empty_dir_no_error(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "env.d").mkdir()

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == []


def test_env_d_user_tier_still_overrides_project(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_d = home / "env.d"
    env_d.mkdir()
    (env_d / "secrets.env").write_text("API_KEY=from-secrets\n", encoding="utf-8")
    project_env = tmp_path / ".env"
    project_env.write_text("API_KEY=from-project\nOTHER=project-val\n", encoding="utf-8")

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert os.getenv("API_KEY") == "from-secrets"
    assert os.getenv("OTHER") == "project-val"
