"""Regression tests for #93593: a malformed provider key in .env must not
silently shadow a valid credential-pool key.

Before the fix, ``_resolve_api_key_provider_secret`` returned the FIRST env
value that passed ``has_usable_secret`` (length + placeholder check only), so
an obviously malformed OPENROUTER_API_KEY (e.g. a truncated paste or another
provider's key) in ~/.hermes/.env won over a valid pool entry and produced
opaque ``401 Missing Authentication header`` errors.

The fix:
- providers with a declared key prefix (KNOWN_PROVIDER_KEY_PREFIXES) skip
  env values that don't match, with a WARNING naming the env var, and fall
  through to the next env var / credential pool;
- the credential-pool fallback iterates entries instead of only peek(), so a
  malformed pool entry doesn't block a valid one either;
- providers WITHOUT a declared prefix are fail-open (unchanged behavior);
- a VALID env key still wins over the pool (precedence unchanged).
"""

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_pconfig(provider_id, env_vars=None):
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars or [f"{provider_id.upper()}_API_KEY"]),
    )


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    return home


def _write_env_file(home: Path, **kwargs) -> None:
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mock_pool(*entries):
    pool = MagicMock()
    pool.has_credentials.return_value = bool(entries)
    pool.peek.return_value = entries[0] if entries else None
    pool.entries.return_value = list(entries)
    return pool


def _entry(token):
    e = MagicMock()
    e.access_token = token
    e.runtime_api_key = ""
    e.last_status = None
    return e


class TestMalformedEnvKeySkipped:
    """Declared-prefix mismatch in .env → warn + fall through to pool."""

    def test_malformed_openrouter_env_key_falls_back_to_pool(
        self, isolated_hermes_home, caplog
    ):
        _write_env_file(isolated_hermes_home, OPENROUTER_API_KEY="not-a-real-openrouter-key")
        pool = _mock_pool(_entry("sk-or-v1-valid-pool-key-abc123"))

        from hermes_cli.auth import _resolve_api_key_provider_secret
        with patch("agent.credential_pool.load_pool", return_value=pool):
            with caplog.at_level(logging.WARNING):
                key, source = _resolve_api_key_provider_secret(
                    provider_id="openrouter",
                    pconfig=_make_pconfig("openrouter"),
                )
        assert key == "sk-or-v1-valid-pool-key-abc123"
        assert source == "credential_pool:openrouter"
        warnings = [r for r in caplog.records if "OPENROUTER_API_KEY" in r.getMessage()]
        assert warnings, "expected a WARNING naming the malformed env var"
        assert "sk-or-" in warnings[0].getMessage()

    @pytest.mark.parametrize(
        "status,expected_key,expected_source",
        [
            ("exhausted", "sk-or-v1-exhausted-key", "credential_pool:openrouter"),
            ("dead", "", ""),
        ],
    )
    def test_raw_pool_fallback_only_reuses_exhausted_entries(
        self,
        isolated_hermes_home,
        status,
        expected_key,
        expected_source,
    ):
        """Exhausted keys may surface the upstream error; dead keys stay dead."""
        from agent.credential_pool import CredentialPool, PooledCredential
        from hermes_cli.auth import _resolve_api_key_provider_secret

        entry = PooledCredential(
            provider="openrouter",
            id=f"{status}-entry",
            label=status,
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token=f"sk-or-v1-{status}-key",
            last_status=status,
            last_status_at=time.time(),
            last_error_reset_at=time.time() + 3600,
        )
        pool = CredentialPool("openrouter", [entry])

        with patch("agent.credential_pool.load_pool", return_value=pool):
            key, source = _resolve_api_key_provider_secret(
                provider_id="openrouter",
                pconfig=_make_pconfig("openrouter"),
            )

        assert key == expected_key
        assert source == expected_source

    def test_malformed_env_key_with_empty_pool_returns_empty(
        self, isolated_hermes_home, caplog
    ):
        """Malformed env key + no pool → empty result, never the bad key."""
        _write_env_file(isolated_hermes_home, OPENROUTER_API_KEY="sk-proj-wrong-provider-key")
        pool = _mock_pool()

        from hermes_cli.auth import _resolve_api_key_provider_secret
        with patch("agent.credential_pool.load_pool", return_value=pool):
            with caplog.at_level(logging.WARNING):
                key, source = _resolve_api_key_provider_secret(
                    provider_id="openrouter",
                    pconfig=_make_pconfig("openrouter"),
                )
        assert key == ""
        assert source == ""

    def test_malformed_pool_entry_skipped_for_valid_one(self, isolated_hermes_home):
        """Pool iteration: a malformed first entry must not block a valid second."""
        pool = _mock_pool(
            _entry("garbage-pool-entry"),
            _entry("sk-or-v1-second-entry-good"),
        )

        from hermes_cli.auth import _resolve_api_key_provider_secret
        with patch("agent.credential_pool.load_pool", return_value=pool):
            key, source = _resolve_api_key_provider_secret(
                provider_id="openrouter",
                pconfig=_make_pconfig("openrouter"),
            )
        assert key == "sk-or-v1-second-entry-good"
        assert source == "credential_pool:openrouter"


class TestNoDeclaredPrefixUnaffected:
    """Providers without a declared prefix keep today's fail-open behavior."""

    def test_undeclared_provider_env_key_returned_verbatim(self, isolated_hermes_home):
        _write_env_file(isolated_hermes_home, DEEPSEEK_API_KEY="totally-unknown-format-key")

        from hermes_cli.auth import _resolve_api_key_provider_secret
        pool = _mock_pool(_entry("pool-key-should-not-win"))
        with patch("agent.credential_pool.load_pool", return_value=pool) as mp:
            key, source = _resolve_api_key_provider_secret(
                provider_id="deepseek",
                pconfig=_make_pconfig("deepseek"),
            )
        assert key == "totally-unknown-format-key"
        assert source == "DEEPSEEK_API_KEY"
        mp.assert_not_called()


class TestValidEnvKeyStillWins:
    """Precedence unchanged: a valid env key beats the credential pool."""

    def test_valid_openrouter_env_key_wins_over_pool(self, isolated_hermes_home):
        _write_env_file(isolated_hermes_home, OPENROUTER_API_KEY="sk-or-v1-env-key-wins")
        pool = _mock_pool(_entry("sk-or-v1-pool-key-loses"))

        from hermes_cli.auth import _resolve_api_key_provider_secret
        with patch("agent.credential_pool.load_pool", return_value=pool) as mp:
            key, source = _resolve_api_key_provider_secret(
                provider_id="openrouter",
                pconfig=_make_pconfig("openrouter"),
            )
        assert key == "sk-or-v1-env-key-wins"
        assert source == "OPENROUTER_API_KEY"
        mp.assert_not_called()

    def test_second_env_var_wins_when_first_is_malformed(self, isolated_hermes_home):
        """Malformed first env var falls through to a valid SECOND env var."""
        _write_env_file(
            isolated_hermes_home,
            OPENROUTER_API_KEY="malformed-first-var",
            OPENROUTER_KEY="sk-or-v1-second-var-good",
        )

        from hermes_cli.auth import _resolve_api_key_provider_secret
        key, source = _resolve_api_key_provider_secret(
            provider_id="openrouter",
            pconfig=_make_pconfig(
                "openrouter", env_vars=["OPENROUTER_API_KEY", "OPENROUTER_KEY"]
            ),
        )
        assert key == "sk-or-v1-second-var-good"
        assert source == "OPENROUTER_KEY"
