"""Profile-scoped provider credentials must not inherit process-global values."""

from contextlib import contextmanager
import os

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _foreign_profile_scope(home, secrets):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in secrets.items()),
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(dict(secrets))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def test_setup_status_ignores_launch_profile_provider_key(tmp_path, monkeypatch):
    from hermes_cli.main import _has_any_provider_configured

    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-placeholder")
    with _foreign_profile_scope(tmp_path / "foreign", {}):
        assert _has_any_provider_configured() is False


def test_auto_provider_uses_profile_key_over_launch_profile_key(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-placeholder")
    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"ANTHROPIC_API_KEY": "profile-placeholder"},
    ):
        assert resolve_provider("auto") == "anthropic"


def test_api_key_provider_base_url_uses_profile_scope(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_api_key_provider_credentials

    monkeypatch.setenv("GLM_BASE_URL", "https://launch.invalid/v1")
    with _foreign_profile_scope(
        tmp_path / "foreign",
        {
            "GLM_API_KEY": "profile-placeholder",
            "GLM_BASE_URL": "https://profile.invalid/v1",
        },
    ):
        resolved = resolve_api_key_provider_credentials("zai")

    assert resolved["api_key"] == "profile-placeholder"
    assert resolved["base_url"] == "https://profile.invalid/v1"


def test_azure_runtime_uses_profile_key_over_launch_profile_key(tmp_path, monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "launch-profile-placeholder")
    monkeypatch.setattr(rp, "resolve_provider", lambda *args, **kwargs: "azure-foundry")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "azure-foundry",
            "base_url": "https://profile.openai.azure.com/openai/v1",
            "api_mode": "chat_completions",
            "default": "gpt-4.1",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)

    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"AZURE_FOUNDRY_API_KEY": "profile-placeholder"},
    ):
        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

    assert resolved["api_key"] == "profile-placeholder"


def test_launch_profile_save_still_updates_process_environment(monkeypatch):
    from hermes_cli.config import save_env_value

    monkeypatch.delenv("FAL_KEY", raising=False)
    save_env_value("FAL_KEY", "launch-profile-placeholder")
    assert os.environ["FAL_KEY"] == "launch-profile-placeholder"


def test_auth_getenv_keeps_launch_profile_process_environment(monkeypatch):
    from hermes_cli.auth import _getenv

    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-placeholder")
    assert _getenv("OPENROUTER_API_KEY") == "launch-profile-placeholder"


def test_profile_op_reference_does_not_seed_raw_or_launch_secret(tmp_path, monkeypatch):
    from agent.credential_pool import _seed_from_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-placeholder")
    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"OPENROUTER_API_KEY": "op://Example/Item/credential"},
    ):
        entries = []
        changed, active_sources = _seed_from_env("openrouter", entries)

    assert changed is False
    assert active_sources == set()
    assert entries == []


def test_profile_op_reference_is_not_accepted_as_runtime_api_key(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_api_key_provider_credentials

    monkeypatch.setenv("GLM_API_KEY", "launch-profile-placeholder")
    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"GLM_API_KEY": "op://Example/Item/credential"},
    ):
        resolved = resolve_api_key_provider_credentials("zai")

    assert resolved["api_key"] == ""
    assert resolved["source"] == "default"


def test_anthropic_runtime_ignores_launch_token_for_empty_profile(tmp_path, monkeypatch):
    from agent import anthropic_adapter as adapter

    monkeypatch.setenv("ANTHROPIC_TOKEN", "launch-profile-placeholder")
    monkeypatch.setattr(adapter, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(adapter, "_resolve_anthropic_pool_token", lambda: None)

    with _foreign_profile_scope(tmp_path / "foreign", {}):
        assert adapter.resolve_anthropic_token() is None


def test_anthropic_runtime_uses_foreign_profile_token(tmp_path, monkeypatch):
    from agent import anthropic_adapter as adapter

    monkeypatch.setenv("ANTHROPIC_TOKEN", "launch-profile-placeholder")
    monkeypatch.setattr(adapter, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(adapter, "_resolve_anthropic_pool_token", lambda: None)

    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"ANTHROPIC_TOKEN": "profile-placeholder"},
    ):
        assert adapter.resolve_anthropic_token() == "profile-placeholder"


def test_copilot_runtime_ignores_launch_token_for_empty_profile(tmp_path, monkeypatch):
    from hermes_cli import copilot_auth

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_launch_profile_placeholder")
    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", lambda: None)

    with _foreign_profile_scope(tmp_path / "foreign", {}):
        assert copilot_auth.resolve_copilot_token() == ("", "")


def test_copilot_runtime_uses_foreign_profile_token(tmp_path, monkeypatch):
    from hermes_cli import copilot_auth

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_launch_profile_placeholder")
    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", lambda: None)

    with _foreign_profile_scope(
        tmp_path / "foreign",
        {"COPILOT_GITHUB_TOKEN": "gho_profile_placeholder"},
    ):
        assert copilot_auth.resolve_copilot_token() == (
            "gho_profile_placeholder",
            "COPILOT_GITHUB_TOKEN",
        )
