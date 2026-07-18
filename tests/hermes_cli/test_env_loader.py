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


def test_profile_home_loads_root_env_as_base_and_profile_env_as_override(tmp_path, monkeypatch):
    """When HERMES_HOME points to ~/.hermes/profiles/<name>, the root
    ~/.hermes/.env is loaded as a non-override base, and the profile's own
    .env overrides the base for keys it defines explicitly (#61046).
    """
    root = tmp_path / "hermes"
    profile = root / "profiles" / "test"
    profile.mkdir(parents=True)

    (root / ".env").write_text(
        "QQ_APP_ID=root-app\nSHARED_KEY=root-value\n",
        encoding="utf-8",
    )
    (profile / ".env").write_text(
        "QQ_APP_ID=profile-app\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("QQ_APP_ID", raising=False)
    monkeypatch.delenv("SHARED_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    # Both files should be reported as loaded, root first (base) then profile.
    assert loaded == [root / ".env", profile / ".env"]
    # Profile wins for keys it defines.
    assert os.getenv("QQ_APP_ID") == "profile-app"
    # Root fills in keys the profile did not set.
    assert os.getenv("SHARED_KEY") == "root-value"


def test_profile_home_without_root_env_still_loads_profile_env(tmp_path, monkeypatch):
    """Profile mode must not crash when the root ~/.hermes/.env is missing."""
    root = tmp_path / "hermes"
    profile = root / "profiles" / "test"
    profile.mkdir(parents=True)

    # Only the profile .env exists, not the root one.
    (profile / ".env").write_text("QQ_APP_ID=profile-app\n", encoding="utf-8")

    monkeypatch.delenv("QQ_APP_ID", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    assert loaded == [profile / ".env"]
    assert os.getenv("QQ_APP_ID") == "profile-app"


def test_non_profile_home_does_not_look_for_root_env_sibling(tmp_path, monkeypatch):
    """When HERMES_HOME is the hermes root itself (default), a sibling
    directory at the same level must not be mistaken for a profiles dir.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("QQ_APP_ID=root-app\n", encoding="utf-8")
    # A sibling named 'profiles' is unrelated and must be ignored.
    sibling_profiles = tmp_path / "profiles"
    sibling_profiles.mkdir()
    (sibling_profiles / ".env").write_text("QQ_APP_ID=should-not-load\n", encoding="utf-8")

    monkeypatch.delenv("QQ_APP_ID", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [home / ".env"]
    assert os.getenv("QQ_APP_ID") == "root-app"
