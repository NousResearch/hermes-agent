"""Tests for credential pool cleanup when env vars are deleted."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so tests never touch real auth.json."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class TestClearPoolEntriesForEnvVar:
    def test_removes_matching_entries(self, tmp_path):
        """Entries with source=env:{VAR} are removed."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )

        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                    "label": "DEEPSEEK_API_KEY",
                }
            ],
        )
        write_credential_pool(
            "openrouter",
            [
                {
                    "id": "or1",
                    "source": "env:OPENROUTER_API_KEY",
                    "auth_type": "api_key",
                    "label": "OPENROUTER_API_KEY",
                }
            ],
        )

        removed = _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        assert removed is True

        pool = read_credential_pool()
        assert "deepseek" not in pool
        assert "openrouter" in pool

    def test_noop_when_not_found(self, tmp_path):
        """A nonexistent env var leaves the pool untouched."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )

        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                }
            ],
        )
        removed = _clear_pool_entries_for_env_var("NONEXISTENT_KEY")
        assert removed is False

        pool = read_credential_pool()
        assert "deepseek" in pool

    def test_preserves_non_env_entries(self, tmp_path):
        """Non-env entries (gh_cli, device_code, etc.) survive cleanup."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )

        write_credential_pool(
            "copilot",
            [
                {
                    "id": "gh1",
                    "source": "gh_cli",
                    "auth_type": "api_key",
                    "label": "gh auth token",
                }
            ],
        )
        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                }
            ],
        )
        removed = _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        assert removed is True

        pool = read_credential_pool()
        assert "copilot" in pool
        assert "deepseek" not in pool

    def test_removes_provider_when_all_entries_gone(self, tmp_path):
        """When the last entry for a provider is removed, the provider key is deleted."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )

        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                }
            ],
        )
        _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        pool = read_credential_pool()
        assert "deepseek" not in pool

    def test_keeps_other_entries_in_provider(self, tmp_path):
        """If a provider has multiple entries, only the matching one is removed."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )

        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                },
                {
                    "id": "ds2",
                    "source": "manual",
                    "auth_type": "api_key",
                },
            ],
        )
        removed = _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        assert removed is True

        pool = read_credential_pool()
        assert "deepseek" in pool
        assert len(pool["deepseek"]) == 1
        assert pool["deepseek"][0]["id"] == "ds2"

    def test_returns_false_when_no_pool(self, tmp_path):
        """Returns False when there is no credential pool at all."""
        from hermes_cli.auth import _clear_pool_entries_for_env_var

        assert _clear_pool_entries_for_env_var("ANY_KEY") is False

    def test_clears_suppressed_source_flags(self, tmp_path):
        """Suppressed-source flags for the same env var should be cleared."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            is_source_suppressed,
            suppress_credential_source,
            write_credential_pool,
        )

        write_credential_pool(
            "deepseek",
            [
                {
                    "id": "ds1",
                    "source": "env:DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                }
            ],
        )
        suppress_credential_source("deepseek", "env:DEEPSEEK_API_KEY")
        assert is_source_suppressed("deepseek", "env:DEEPSEEK_API_KEY") is True

        _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        assert is_source_suppressed("deepseek", "env:DEEPSEEK_API_KEY") is False
