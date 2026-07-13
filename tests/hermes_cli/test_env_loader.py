import importlib
import os
import sys

from hermes_cli.env_loader import (
    classify_effective_credential_source,
    get_effective_credential_source,
    load_hermes_dotenv,
)


def test_effective_credential_source_prefers_matching_managed_value():
    source = classify_effective_credential_source(
        effective_value="managed-token",
        managed_value="managed-token",
        external_source="onepassword",
        profile_value="managed-token",
    )

    assert source == "managed_env"


def test_effective_credential_source_precedence_and_fallbacks():
    assert classify_effective_credential_source(
        effective_value="vault-token",
        managed_value="different-token",
        external_source="bitwarden",
        profile_value="vault-token",
    ) == "bitwarden"
    assert classify_effective_credential_source(
        effective_value="profile-token",
        profile_value="profile-token",
    ) == "profile_env"
    assert classify_effective_credential_source(
        effective_value="shell-token",
        profile_value="different-token",
    ) == "process_env"
    assert classify_effective_credential_source(
        effective_value=None,
        external_source="onepassword",
    ) == "missing"
    assert classify_effective_credential_source(
        effective_value="vault-token",
        external_source="custom-vault",
    ) == "unknown"


def test_get_effective_credential_source_detects_profile_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=profile-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.managed_scope.load_managed_env",
        lambda: {},
    )

    source = get_effective_credential_source(
        "TELEGRAM_BOT_TOKEN",
        effective_value="profile-token",
        hermes_home=home,
    )

    assert source == "profile_env"


def test_get_effective_credential_source_reads_legacy_encoded_profile_env(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_bytes(
        b"# caf\xe9\nTELEGRAM_BOT_TOKEN=profile-token\n"
    )
    monkeypatch.setattr(
        "hermes_cli.managed_scope.load_managed_env",
        lambda: {},
    )

    source = get_effective_credential_source(
        "TELEGRAM_BOT_TOKEN",
        effective_value="profile-token",
        hermes_home=home,
    )

    assert source == "profile_env"


def test_get_effective_credential_source_honors_context_local_profile(
    tmp_path,
    monkeypatch,
):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    (primary / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=primary-token\n",
        encoding="utf-8",
    )
    (secondary / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=secondary-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(primary))
    monkeypatch.setattr(
        "hermes_cli.managed_scope.load_managed_env",
        lambda: {},
    )

    token = set_hermes_home_override(secondary)
    try:
        source = get_effective_credential_source(
            "TELEGRAM_BOT_TOKEN",
            effective_value="secondary-token",
        )
    finally:
        reset_hermes_home_override(token)

    assert source == "profile_env"


def test_get_effective_credential_source_does_not_cross_profile_provenance(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import env_loader

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    (secondary / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=secondary-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.managed_scope.load_managed_env",
        lambda: {},
    )
    monkeypatch.setitem(
        env_loader._SECRET_SOURCES,
        "TELEGRAM_BOT_TOKEN",
        "onepassword",
    )
    monkeypatch.setitem(
        env_loader._SECRET_SOURCES_BY_HOME,
        (str(primary.resolve()), "TELEGRAM_BOT_TOKEN"),
        "onepassword",
    )

    source = get_effective_credential_source(
        "TELEGRAM_BOT_TOKEN",
        effective_value="secondary-token",
        hermes_home=secondary,
    )

    assert source == "profile_env"


def test_get_effective_credential_source_uses_same_profile_provenance(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import env_loader

    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=old-profile-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.managed_scope.load_managed_env",
        lambda: {},
    )
    monkeypatch.setitem(
        env_loader._SECRET_SOURCES,
        "TELEGRAM_BOT_TOKEN",
        "onepassword",
    )
    monkeypatch.setitem(
        env_loader._SECRET_SOURCES_BY_HOME,
        (str(home.resolve()), "TELEGRAM_BOT_TOKEN"),
        "onepassword",
    )

    source = get_effective_credential_source(
        "TELEGRAM_BOT_TOKEN",
        effective_value="vault-token",
        hermes_home=home,
    )

    assert source == "onepassword"


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
