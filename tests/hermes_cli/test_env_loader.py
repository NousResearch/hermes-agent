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
        "ANTHROPIC_API_KEY=«redacted:sk-…»\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"


def test_profile_env_overrides_rootEnv(tmp_path, monkeypatch):
    """#61046: a profile-loaded hermes must see shared root credentials
    but also be able to override them. Pre-fix, the profile's
    `hermes_home=<root>/profiles/<name>` made the loader load only
    `<root>/profiles/<name>/.env`, ignoring `<root>/.env` entirely.
    """
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "test"
    profile.mkdir(parents=True)

    (root / ".env").write_text(
        "QQ_APP_ID=root-app-id\n"
        "OPENAI_BASE_URL=https://root.example/v1\n",
        encoding="utf-8",
    )
    (profile / ".env").write_text(
        "QQ_APP_ID=profile-app-id-override\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("QQ_APP_ID", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    root_env = root / ".env"
    profile_env = profile / ".env"
    assert root_env in loaded
    assert profile_env in loaded
    # Profile-specific value overrides the shared root value.
    assert os.getenv("QQ_APP_ID") == "profile-app-id-override"
    # Shared value not overridden by profile is still available.
    assert os.getenv("OPENAI_BASE_URL") == "https://root.example/v1"


def test_profile_home_without_root_env_uses_only_profile_env(tmp_path, monkeypatch):
    """#61046: a profile home with no root .env falls back to the
    pre-fix behavior (only the profile .env is loaded).
    """
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "test"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("QQ_APP_ID=profile-only\n", encoding="utf-8")

    monkeypatch.delenv("QQ_APP_ID", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    profile_env = profile / ".env"
    # No root .env present → only the profile .env is loaded.
    assert loaded == [profile_env]
    assert os.getenv("QQ_APP_ID") == "profile-only"


def test_non_profile_home_does_not_touch_root(tmp_path, monkeypatch):
    """#61046: a non-profile hermes home (no `<root>/profiles/<name>`
    layout) must NOT load a parent .env. The root path is only honored
    when hermes_home itself is INSIDE a `profiles/` directory.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    sibling_root = tmp_path / "sibling_root"
    sibling_root.mkdir()
    (sibling_root / ".env").write_text("SHOULD_NOT_LOAD=wrong\n", encoding="utf-8")
    (home / ".env").write_text("QQ_APP_ID=user-level\n", encoding="utf-8")

    monkeypatch.delenv("SHOULD_NOT_LOAD", raising=False)
    monkeypatch.delenv("QQ_APP_ID", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    # Only the user_home .env is loaded — sibling_root/.env is unrelated.
    assert sibling_root / ".env" not in loaded
    assert os.getenv("QQ_APP_ID") == "user-level"
    assert os.getenv("SHOULD_NOT_LOAD") is None


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
