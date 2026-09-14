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


def test_vertex_adc_counts_as_explicit_when_config_present(tmp_path, monkeypatch):
    """A keyless Vertex provider is explicitly configured when the user pointed
    Hermes at it (VERTEX_PROJECT_ID / vertex.project_id / VERTEX_CREDENTIALS_PATH),
    even when it is NOT the current provider — otherwise it silently vanishes
    from explicit-only pickers (desktop chat model menu) unless already selected."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for var in ("VERTEX_PROJECT_ID", "VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(var, raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": None})

    from hermes_cli.auth import is_provider_explicitly_configured

    # vertex.project_id in config.yaml is a deliberate, Hermes-scoped signal.
    _write_config(tmp_path, {
        "model": {"provider": "anthropic", "default": "claude-opus-4-8"},
        "vertex": {"project_id": "my-gcp-project"},
    })
    assert is_provider_explicitly_configured("vertex") is True

    # No Hermes-scoped Vertex config at all → stays hidden.
    _write_config(tmp_path, {"model": {"provider": "anthropic", "default": "claude-opus-4-8"}})
    assert is_provider_explicitly_configured("vertex") is False


def test_vertex_ambient_google_creds_env_does_not_count_as_explicit(tmp_path, monkeypatch):
    """An ambient GOOGLE_APPLICATION_CREDENTIALS path (commonly set globally for
    unrelated GCP work) must NOT mark Vertex explicit — only Hermes-scoped
    signals do. Regression guard for the picker gate (PR review feedback)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
    _write_config(tmp_path, {"model": {"provider": "anthropic", "default": "claude-opus-4-8"}})
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": None})

    # A real, existing SA file pointed to ONLY by the ambient Google var.
    sa = tmp_path / "adc.json"
    sa.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("vertex") is False


def test_vertex_credentials_path_must_be_readable_file(tmp_path, monkeypatch):
    """VERTEX_CREDENTIALS_PATH must point to an actual readable file, not a directory
    or non-existent path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    _write_config(tmp_path, {"model": {"provider": "anthropic", "default": "claude-opus-4-8"}})
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": None})

    from hermes_cli.auth import is_provider_explicitly_configured

    # Valid readable file -> True
    sa_file = tmp_path / "vertex_sa.json"
    sa_file.write_text("{}")
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(sa_file))
    assert is_provider_explicitly_configured("vertex") is True

    # Directory -> False
    sa_dir = tmp_path / "vertex_dir"
    sa_dir.mkdir()
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(sa_dir))
    assert is_provider_explicitly_configured("vertex") is False

    # Non-existent file -> False
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(tmp_path / "nonexistent.json"))
    assert is_provider_explicitly_configured("vertex") is False


def test_bedrock_region_counts_as_explicit(tmp_path, monkeypatch):
    """Bedrock (AWS SDK auth, no API key) is explicitly configured once the
    user pins a region in config.yaml, mirroring the Vertex keyless case."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": None})

    from hermes_cli.auth import is_provider_explicitly_configured

    _write_config(tmp_path, {
        "model": {"provider": "anthropic", "default": "claude-opus-4-8"},
        "bedrock": {"region": "us-east-1"},
    })
    assert is_provider_explicitly_configured("bedrock") is True

    _write_config(tmp_path, {
        "model": {"provider": "anthropic", "default": "claude-opus-4-8"},
        "bedrock": {"region": ""},
    })
    assert is_provider_explicitly_configured("bedrock") is False


def test_returns_true_when_moa_reference_slot_uses_provider(tmp_path, monkeypatch):
    """MoA advisor slots are explicit provider selections for auth gating."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {
        "model": {"provider": "openai-codex", "default": "gpt-5.5"},
        "moa": {
            "presets": {
                "default": {
                    "reference_models": [
                        {"provider": "anthropic", "model": "claude-opus-4-8"},
                        {"provider": "opencode-go", "model": "glm-5.2"},
                    ],
                    "aggregator": {"provider": "openai-codex", "model": "gpt-5.5"},
                }
            }
        },
    })
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": "openai-codex"})

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
