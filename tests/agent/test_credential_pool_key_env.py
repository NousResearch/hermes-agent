"""Regression tests for #79130 — _seed_custom_pool key_env resolution.

``_seed_custom_pool()`` must resolve ``key_env`` (and its ``api_key_env``
alias) to populate the credential pool for custom providers configured with
env-var-referenced keys.  Every other key_env resolution path in the codebase
already does this; ``_seed_custom_pool`` was the lone exception, leaving the
pool empty and breaking credential rotation, failover, and health tracking.

Resolution uses the same ``.env``-prefer-over-``os.environ`` pattern as the
sibling ``_seed_from_env`` function (``load_env()`` → ``_get_secret()``).
"""

from __future__ import annotations

import pytest

from agent.credential_pool import _seed_custom_pool


def _provider_config(
    *,
    key_env: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict:
    cfg: dict = {
        "name": "test-provider",
        "base_url": "https://api.example.com/v1",
    }
    if key_env:
        cfg["key_env"] = key_env
    if api_key:
        cfg["api_key"] = api_key
    if api_key_env:
        cfg["api_key_env"] = api_key_env
    return cfg


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Prevent the suppression gate and model-config seed from interfering."""
    monkeypatch.setattr(
        "hermes_cli.auth.is_source_suppressed", lambda _p, _s: False,
    )
    monkeypatch.setattr("agent.credential_pool._load_config_safe", lambda: None)


class TestSeedCustomPoolKeyEnv:
    def test_key_env_seeds_pool_when_api_key_absent(self, monkeypatch):
        """Core bug: key_env-configured custom providers got empty pools."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(key_env="MY_KEY_ENV"),
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_env",
            lambda: {"MY_KEY_ENV": "resolved_key_value_123"},
        )
        monkeypatch.setattr("agent.credential_pool._get_secret", lambda _k, _d: "")

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is True
        assert len(entries) == 1
        assert entries[0].access_token == "resolved_key_value_123"
        assert entries[0].base_url == "https://api.example.com/v1"

    def test_api_key_env_alias_seeds_pool(self, monkeypatch):
        """api_key_env is a documented alias for key_env (config.py:1313)."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(api_key_env="MY_ALIAS_ENV"),
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_env",
            lambda: {"MY_ALIAS_ENV": "alias_key_value_456"},
        )
        monkeypatch.setattr("agent.credential_pool._get_secret", lambda _k, _d: "")

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is True
        assert len(entries) == 1
        assert entries[0].access_token == "alias_key_value_456"

    def test_direct_api_key_preferred_over_key_env(self, monkeypatch):
        """When both api_key and key_env are set, the inline value wins."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(
                api_key="direct_inline_key_789", key_env="SHOULD_NOT_BE_USED",
            ),
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_env",
            lambda: {"SHOULD_NOT_BE_USED": "env_key_value_000"},
        )

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is True
        assert len(entries) == 1
        assert entries[0].access_token == "direct_inline_key_789"

    def test_key_env_with_unset_env_var_seeds_nothing(self, monkeypatch):
        """key_env set but env var empty/unset -> gracefully skip (no crash)."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(key_env="MISSING_ENV_VAR"),
        )
        monkeypatch.setattr("agent.credential_pool.load_env", lambda: {})
        monkeypatch.setattr("agent.credential_pool._get_secret", lambda _k, _d: "")

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is False
        assert len(entries) == 0

    def test_direct_api_key_still_works_without_key_env(self, monkeypatch):
        """Regression check: the existing direct api_key path is unchanged."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(api_key="existing_direct_key_abc"),
        )

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is True
        assert len(entries) == 1
        assert entries[0].access_token == "existing_direct_key_abc"

    def test_key_env_source_label_uses_provider_name(self, monkeypatch):
        """The seeded entry's source label uses the provider name."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(key_env="MY_KEY_ENV"),
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_env",
            lambda: {"MY_KEY_ENV": "resolved_key_value_123"},
        )
        monkeypatch.setattr("agent.credential_pool._get_secret", lambda _k, _d: "")

        entries: list = []
        _seed_custom_pool("custom:test-provider", entries)

        assert len(entries) == 1
        assert entries[0].source == "config:test-provider"
        assert entries[0].label == "test-provider"

    def test_key_env_falls_back_to_secret_store(self, monkeypatch):
        """When .env lacks the var, _get_secret (keychain) is consulted."""
        monkeypatch.setattr(
            "agent.credential_pool._get_custom_provider_config",
            lambda _pk: _provider_config(key_env="KEYCHAIN_ONLY"),
        )
        monkeypatch.setattr("agent.credential_pool.load_env", lambda: {})
        monkeypatch.setattr(
            "agent.credential_pool._get_secret",
            lambda key, _default: "secret_store_value_789" if key == "KEYCHAIN_ONLY" else "",
        )

        entries: list = []
        changed, _ = _seed_custom_pool("custom:test-provider", entries)

        assert changed is True
        assert len(entries) == 1
        assert entries[0].access_token == "secret_store_value_789"
