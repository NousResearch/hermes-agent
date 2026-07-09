"""Tests for is_provider_explicitly_configured()."""

import json
import pytest


def _write_config(tmp_path, config: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump(config))


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


@pytest.fixture(autouse=True)
def _clean_anthropic_env(monkeypatch):
    """Strip Anthropic env vars so CI secrets don't leak into tests."""
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)


def test_returns_false_when_no_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is False


def test_returns_true_when_active_provider_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": "anthropic",
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is True


def test_returns_true_when_config_provider_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "anthropic", "default": "claude-sonnet-4-6"}})

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is True


def test_returns_false_when_config_provider_is_different(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "kimi-coding", "default": "kimi-k2"}})
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is False


def test_returns_true_when_anthropic_env_var_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-realkey")
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is True


def test_claude_code_oauth_token_does_not_count_as_explicit(tmp_path, monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN is set by Claude Code, not the user — must not gate."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-auto-token")
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is False


def test_ambient_pool_source_does_not_count_as_explicit(tmp_path, monkeypatch):
    """gh_cli-seeded Copilot pool entries are ambient, not explicit config (#56974)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "copilot": [{
                "id": "abc123",
                "source": "gh_cli",
                "auth_type": "api_key",
                "access_token": "ghu_sometoken",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("copilot") is False


def test_explicit_pool_source_counts_as_explicit(tmp_path, monkeypatch):
    """manual / device_code / PKCE pool entries reflect explicit Hermes flows."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "anthropic": [{
                "id": "def456",
                "source": "manual:key-1",
                "auth_type": "api_key",
                "access_token": "sk-ant-api03-key",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is True


def test_stale_env_pool_entry_does_not_count_when_var_unset(tmp_path, monkeypatch):
    """An env-seeded pool entry left in auth.json after the env var was removed
    must not mark the provider configured (#55790): the picker showed removed
    providers forever because the record existed even though no secret resolves."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "deepseek": [{
                "id": "aaa111",
                "source": "env:DEEPSEEK_API_KEY",
                "auth_type": "api_key",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("deepseek") is False


def test_env_pool_entry_counts_when_var_still_resolves(tmp_path, monkeypatch):
    """The same env-seeded pool entry IS explicit while the var still resolves."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-realkey-123456")
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "deepseek": [{
                "id": "aaa111",
                "source": "env:DEEPSEEK_API_KEY",
                "auth_type": "api_key",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("deepseek") is True


# ─── aws_sdk providers (Bedrock) ─────────────────────────────────────────
#
# Bedrock is registered with auth_type="aws_sdk" and an empty
# api_key_env_vars tuple, so the api_key env-var loop never sees it.
# Setting AWS_BEARER_TOKEN_BEDROCK (or an access-key pair) in .env is
# exactly as explicit as pasting ANTHROPIC_API_KEY, and must count —
# otherwise the desktop picker's explicit_only filter hides Bedrock even
# though list_authenticated_providers builds a full row for it.
#
# Ambient AWS credential sources (SSO profiles, EC2 IMDS, container
# credentials) must NOT count: aws_sdk detection here is env-var only,
# never boto3's full chain.


@pytest.fixture()
def _clean_aws_env(monkeypatch):
    """Strip AWS env vars so developer/CI AWS credentials don't leak in."""
    for key in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_bedrock_not_explicit_without_aws_env(tmp_path, monkeypatch, _clean_aws_env):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("bedrock") is False


def test_bedrock_bearer_token_counts_as_explicit(tmp_path, monkeypatch, _clean_aws_env):
    """AWS_BEARER_TOKEN_BEDROCK is Bedrock-specific — the user set it for
    exactly this provider, so it gates like a provider API key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKexample-bearer-token-value")

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("bedrock") is True


def test_bedrock_access_key_pair_counts_as_explicit(tmp_path, monkeypatch, _clean_aws_env):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE1234567890")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "examplesecretexamplesecretexample0000000")

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("bedrock") is True


def test_bedrock_access_key_without_secret_is_not_explicit(tmp_path, monkeypatch, _clean_aws_env):
    """A lone AWS_ACCESS_KEY_ID can't authenticate anything — require the pair."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE1234567890")

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("bedrock") is False


def test_bedrock_ambient_aws_profile_is_not_explicit(tmp_path, monkeypatch, _clean_aws_env):
    """AWS_PROFILE is ambient machine state (SSO / shared credentials file),
    not an explicit Hermes provider choice — same principle as gh_cli-seeded
    Copilot pool entries (#56974)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_PROFILE", "default")

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("bedrock") is False


def test_aws_env_does_not_leak_into_other_providers(tmp_path, monkeypatch, _clean_aws_env):
    """The aws_sdk env check must key on the provider's auth_type, not fire
    for every provider whenever AWS credentials happen to be present."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKexample-bearer-token-value")

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is False
