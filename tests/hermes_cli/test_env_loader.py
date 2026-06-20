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


def test_resolved_secret_cache_overlays_unresolved_op_refs(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    runtime_dir = tmp_path / "runtime"
    cache_dir = runtime_dir / "hermes"
    cache_dir.mkdir(parents=True)
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=op://vault/telegram/token\n",
        encoding="utf-8",
    )
    (cache_dir / "resolved-secrets.env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:resolved-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "123456:resolved-token"


def test_resolved_secret_cache_does_not_override_plaintext_user_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    cache_dir = home / "state" / "secrets-cache"
    cache_dir.mkdir(parents=True)
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:plaintext-token\n",
        encoding="utf-8",
    )
    (cache_dir / "resolved.env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:cached-token\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "123456:plaintext-token"


def test_runtime_resolved_secret_cache_beats_persistent_cache(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_cache_dir = runtime_dir / "hermes"
    persistent_cache_dir = home / "state" / "secrets-cache"
    runtime_cache_dir.mkdir(parents=True)
    persistent_cache_dir.mkdir(parents=True)
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=op://vault/telegram/token\n",
        encoding="utf-8",
    )
    (runtime_cache_dir / "resolved-secrets.env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:runtime-token\n",
        encoding="utf-8",
    )
    (persistent_cache_dir / "resolved.env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:persistent-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "123456:runtime-token"


def test_resolved_secret_cache_ignores_blank_and_unresolved_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    cache_dir = home / "state" / "secrets-cache"
    cache_dir.mkdir(parents=True)
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=op://vault/telegram/token\n",
        encoding="utf-8",
    )
    (cache_dir / "resolved.env").write_text(
        "TELEGRAM_BOT_TOKEN=op://vault/telegram/cached\n"
        "OPENAI_API_KEY=\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "op://vault/telegram/token"
    assert os.getenv("OPENAI_API_KEY") is None


def test_managed_env_still_wins_after_cache_overlay(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    cache_dir = home / "state" / "secrets-cache"
    cache_dir.mkdir(parents=True)
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=op://vault/telegram/token\n",
        encoding="utf-8",
    )
    (cache_dir / "resolved.env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:cached-token\n",
        encoding="utf-8",
    )
    (managed_dir / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=123456:managed-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "123456:managed-token"
