"""Tests for agent.secret_sources.keepassxc."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.secret_sources.keepassxc import (
    FetchResult,
    _is_valid_env_name,
    apply_keepassxc_secrets,
    find_keepassxc_cli,
)


class TestIsValidEnvName:
    def test_valid_names(self):
        assert _is_valid_env_name("OPENAI_API_KEY")
        assert _is_valid_env_name("_PRIVATE_KEY")
        assert _is_valid_env_name("A")
        assert _is_valid_env_name("A123")
        assert _is_valid_env_name("_")

    def test_invalid_names(self):
        assert not _is_valid_env_name("")
        assert not _is_valid_env_name("123_STARTS_WITH_NUMBER")
        assert not _is_valid_env_name("HAS SPACES")
        assert not _is_valid_env_name("HAS-DASHES")
        assert not _is_valid_env_name("HAS.DOTS")


class TestFindKeepassxcCli:
    def test_env_var_override(self, monkeypatch, tmp_path):
        fake = tmp_path / "fake-keepassxc-cli"
        fake.write_text("#!/bin/sh\necho ok")
        fake.chmod(0o755)
        monkeypatch.setenv("KEEPASSXC_CLI_PATH", str(fake))
        result = find_keepassxc_cli()
        assert result == fake

    def test_env_var_missing_file(self, monkeypatch):
        monkeypatch.setenv("KEEPASSXC_CLI_PATH", "/nonexistent/keepassxc-cli")
        with patch("shutil.which", return_value=None):
            result = find_keepassxc_cli()
        assert result is None

    def test_system_path(self):
        result = find_keepassxc_cli()
        assert result is None or result.name == "keepassxc-cli"


class TestApplyKeepassxcSecrets:
    def test_disabled_returns_empty(self):
        result = apply_keepassxc_secrets(enabled=False, db_path="")
        assert result.ok
        assert result.secrets == {}
        assert result.applied == []

    def test_missing_db_path(self):
        result = apply_keepassxc_secrets(enabled=True, db_path="")
        assert not result.ok
        assert "db_path" in result.error.lower()

    def test_missing_mappings(self):
        result = apply_keepassxc_secrets(
            enabled=True, db_path="/tmp/test.kdbx", mappings={}
        )
        assert not result.ok
        assert "mappings" in result.error.lower()

    def test_missing_binary(self, monkeypatch):
        monkeypatch.setenv("KEEPASSXC_CLI_PATH", "/nonexistent/keepassxc-cli")
        with patch("shutil.which", return_value=None):
            result = apply_keepassxc_secrets(
                enabled=True,
                db_path="/tmp/test.kdbx",
                no_password=True,
                mappings={"TEST_KEY": "Dev/Test"},
            )
        assert not result.ok
        assert "not available" in result.error.lower() or "not found" in result.error.lower()

    def test_missing_password_and_no_key_file(self, monkeypatch):
        monkeypatch.delenv("KEEPASSXC_PASSWORD", raising=False)
        result = apply_keepassxc_secrets(
            enabled=True,
            db_path="/tmp/test.kdbx",
            mappings={"TEST_KEY": "Dev/Test"},
        )
        assert not result.ok
        assert "password" in result.error.lower() or "key_file" in result.error.lower()

    def test_applies_secrets_to_env(self, monkeypatch, tmp_path):
        """Integration test with a real keepassxc-cli and a test database."""
        binary = find_keepassxc_cli()
        if binary is None:
            pytest.skip("keepassxc-cli not installed")

        db_path = tmp_path / "test.kdbx"
        password = "testpassword123"

        # Create database — db-create wants password twice (enter + repeat)
        proc = subprocess.run(
            [str(binary), "db-create", "-p", str(db_path)],
            input=f"{password}\n{password}\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, f"db-create failed: {proc.stderr}"

        # Add an entry with a password — need to create the group first
        proc = subprocess.run(
            [str(binary), "mkdir", str(db_path), "Dev"],
            input=f"{password}\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, f"mkdir failed: {proc.stderr}"

        proc = subprocess.run(
            [str(binary), "add", "-u", "TestEntry", "-p", str(db_path),
             "Dev/TestEntry"],
            input=f"{password}\nsecretvalue123\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0, f"add failed: {proc.stderr}"

        monkeypatch.setenv("KEEPASSXC_PASSWORD", password)

        result = apply_keepassxc_secrets(
            enabled=True,
            db_path=str(db_path),
            password_env="KEEPASSXC_PASSWORD",
            mappings={"TEST_API_KEY": "Dev/TestEntry"},
            override_existing=True,
            cache_ttl_seconds=0,
        )

        assert result.ok, f"Error: {result.error}"
        assert "TEST_API_KEY" in result.applied
        assert os.environ.get("TEST_API_KEY") == "secretvalue123"

    def test_does_not_override_existing_by_default(self, monkeypatch, tmp_path):
        binary = find_keepassxc_cli()
        if binary is None:
            pytest.skip("keepassxc-cli not installed")

        db_path = tmp_path / "test2.kdbx"
        password = "testpassword456"

        proc = subprocess.run(
            [str(binary), "db-create", "-p", str(db_path)],
            input=f"{password}\n{password}\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0

        # Create group before adding entry
        proc = subprocess.run(
            [str(binary), "mkdir", str(db_path), "Dev"],
            input=f"{password}\n",
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0

        proc = subprocess.run(
            [str(binary), "add", "-u", "TestEntry", "-p", str(db_path),
             "Dev/Test"],
            input=f"{password}\nfromkeepass\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0

        monkeypatch.setenv("EXISTING_KEY", "from_shell")
        monkeypatch.setenv("KEEPASSXC_PASSWORD", password)

        result = apply_keepassxc_secrets(
            enabled=True,
            db_path=str(db_path),
            password_env="KEEPASSXC_PASSWORD",
            mappings={"EXISTING_KEY": "Dev/Test"},
            override_existing=False,
            cache_ttl_seconds=0,
        )

        assert result.ok
        assert "EXISTING_KEY" in result.skipped
        assert os.environ["EXISTING_KEY"] == "from_shell"

    def test_entry_not_found_warning(self, monkeypatch, tmp_path):
        """Non-existent entry path → warning, not error."""
        binary = find_keepassxc_cli()
        if binary is None:
            pytest.skip("keepassxc-cli not installed")

        db_path = tmp_path / "test3.kdbx"
        password = "testpassword789"

        proc = subprocess.run(
            [str(binary), "db-create", "-p", str(db_path)],
            input=f"{password}\n{password}\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 0

        monkeypatch.setenv("KEEPASSXC_PASSWORD", password)

        result = apply_keepassxc_secrets(
            enabled=True,
            db_path=str(db_path),
            password_env="KEEPASSXC_PASSWORD",
            mappings={"MISSING_KEY": "Dev/DoesNotExist"},
            override_existing=True,
            cache_ttl_seconds=0,
        )

        assert result.ok
        assert "MISSING_KEY" not in result.applied
        assert len(result.warnings) >= 1
        assert any("not found" in w.lower() for w in result.warnings)


class TestFetchResult:
    def test_ok_when_no_error(self):
        r = FetchResult()
        assert r.ok

    def test_not_ok_when_error(self):
        r = FetchResult(error="something went wrong")
        assert not r.ok

    def test_defaults(self):
        r = FetchResult()
        assert r.secrets == {}
        assert r.applied == []
        assert r.skipped == []
        assert r.warnings == []
        assert r.error is None
        assert r.binary_path is None
