"""Test that DELETE /api/env also clears credential pool entries."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so tests never touch real files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class TestDeleteEnvClearsCredentialPool:
    def test_delete_env_removes_pool_entry(self, tmp_path, monkeypatch):
        """Deleting an env var should remove matching pool entries."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )
        from hermes_cli.config import remove_env_value, save_env_value

        # Seed the pool
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

        # Set the env var first so remove_env_value succeeds
        save_env_value("DEEPSEEK_API_KEY", "test-key")
        removed = remove_env_value("DEEPSEEK_API_KEY")
        assert removed is True

        # Now clear the pool (simulating what DELETE /api/env does)
        pool_cleaned = _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")
        assert pool_cleaned is True

        pool = read_credential_pool()
        assert "deepseek" not in pool

    def test_delete_env_preserves_other_providers(self, tmp_path, monkeypatch):
        """Deleting one env var should not touch other providers."""
        from hermes_cli.auth import (
            _clear_pool_entries_for_env_var,
            read_credential_pool,
            write_credential_pool,
        )
        from hermes_cli.config import remove_env_value, save_env_value

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
        write_credential_pool(
            "openrouter",
            [
                {
                    "id": "or1",
                    "source": "env:OPENROUTER_API_KEY",
                    "auth_type": "api_key",
                }
            ],
        )

        save_env_value("DEEPSEEK_API_KEY", "test-key")
        remove_env_value("DEEPSEEK_API_KEY")
        _clear_pool_entries_for_env_var("DEEPSEEK_API_KEY")

        pool = read_credential_pool()
        assert "deepseek" not in pool
        assert "openrouter" in pool
