"""Regression tests for issue #42269.

A credential added via `hermes auth add deepseek --type api-key` is stored
in the credential pool (auth.json -> credential_pool.deepseek). Before the
fix, resolve_api_key_provider_credentials() did not pick it up because:

  1. The exception from load_pool() was swallowed by `except Exception: pass`,
     making load failures invisible.
  2. Even when the key was found, the pool entry's base_url was silently
     dropped — resolve_api_key_provider_credentials() always fell back to
     pconfig.inference_base_url instead of using the URL stored with the key.

These tests exercise the full resolution chain without mocking load_pool,
writing a real credential pool entry to a temp HERMES_HOME (exactly what
`hermes auth add deepseek` does at runtime).
"""

import uuid

import pytest


@pytest.fixture(autouse=True)
def _clean_deepseek_env(monkeypatch):
    """Remove any real DEEPSEEK_API_KEY from env so pool is the only source."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)


def _seed_deepseek_pool(tmp_path: "Path", token: str = "sk-deepseek-FAKEKEY") -> None:
    """Mimic `hermes auth add deepseek --type api-key` — writes a manual pool entry."""
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY["deepseek"]

    pool = load_pool("deepseek")
    pool.add_entry(
        PooledCredential(
            provider="deepseek",
            id=uuid.uuid4().hex[:6],
            label="api-key-1",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url=pconfig.inference_base_url,
        )
    )


def test_resolve_api_key_credentials_finds_pool_key(tmp_path, monkeypatch):
    """End-to-end: pool key is returned by resolve_api_key_provider_credentials."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    _seed_deepseek_pool(tmp_path, token="sk-deepseek-TESTKEY123")

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")

    assert creds["api_key"] == "sk-deepseek-TESTKEY123", (
        "Pool credential not picked up — resolve_api_key_provider_credentials "
        "returned an empty api_key even though the key is in the pool."
    )
    assert creds["base_url"] == "https://api.deepseek.com/v1"
    assert "credential_pool" in creds["source"]


def test_resolve_finds_pool_key_over_empty_env(tmp_path, monkeypatch):
    """Pool wins when DEEPSEEK_API_KEY is unset (the bug's primary trigger)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    assert "DEEPSEEK_API_KEY" not in __import__("os").environ

    _seed_deepseek_pool(tmp_path, token="sk-pool-wins-abc")

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == "sk-pool-wins-abc"


def test_env_var_takes_priority_over_pool(tmp_path, monkeypatch):
    """Env var still wins when both env and pool are present."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-priority-xyz")
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    _seed_deepseek_pool(tmp_path, token="sk-pool-should-lose")

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == "sk-env-priority-xyz"
    assert creds["source"] == "DEEPSEEK_API_KEY"


def test_empty_pool_returns_empty_key(tmp_path, monkeypatch):
    """No env var, empty pool → empty key (not a crash)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == ""


def test_pool_base_url_propagated_to_credentials(tmp_path, monkeypatch):
    """Pool entry's base_url is used in the returned credentials dict."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    # Use a custom base_url to verify it propagates (not the pconfig default)
    custom_url = "https://custom-deepseek-endpoint.example.com/v1"
    pool = load_pool("deepseek")
    pool.add_entry(
        PooledCredential(
            provider="deepseek",
            id=uuid.uuid4().hex[:6],
            label="custom-endpoint",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token="sk-custom-endpoint-key",
            base_url=custom_url,
        )
    )

    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == "sk-custom-endpoint-key"
    assert creds["base_url"] == custom_url, (
        f"Expected pool base_url {custom_url!r} to be used, got {creds['base_url']!r}"
    )
