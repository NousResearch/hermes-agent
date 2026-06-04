"""Tests for agent/file_safety.py read guards — env file blocking.

Run with:  python -m pytest tests/agent/test_file_safety.py -v
"""

import os
from unittest.mock import patch

import pytest

from agent.file_safety import (
    _BLOCKED_PROJECT_ENV_BASENAMES,
    get_read_block_error,
)


# ---------------------------------------------------------------------------
# Project-local .env file blocking (issue #20734)
# ---------------------------------------------------------------------------


class TestEnvFileReadBlocking:
    """Secret-bearing .env files must be blocked by get_read_block_error."""

    @pytest.mark.parametrize("basename", [
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".envrc",
    ])
    def test_blocked_env_basenames(self, basename):
        """All secret-bearing .env basenames are blocked regardless of directory."""
        path = f"/tmp/project/{basename}"
        error = get_read_block_error(path)
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "secret-bearing" in error.lower() or "environment file" in error.lower()

    def test_blocked_env_in_subdirectory(self):
        """Nested .env files are also blocked."""
        error = get_read_block_error("/home/user/app/services/api/.env.production")
        assert error is not None

    def test_blocked_env_absolute_path(self):
        """Absolute paths to .env files are blocked."""
        error = get_read_block_error("/opt/myapp/.env")
        assert error is not None

    def test_allowed_env_example(self):
        """"The .env.example file is explicitly allowed — it's documentation, not a secret."""
        error = get_read_block_error("/tmp/project/.env.example")
        assert error is None

    def test_allowed_env_sample(self):
        """Other .env variants like .env.sample are allowed."""
        error = get_read_block_error("/tmp/project/.env.sample")
        assert error is None

    def test_allowed_non_env_files(self):
        """Regular files are not affected by the env guard."""
        for path in ["/tmp/project/config.yaml", "/tmp/project/main.py",
                     "/tmp/project/README.md", "/tmp/project/.gitignore"]:
            error = get_read_block_error(path)
            assert error is None, f"{path} should be allowed"

    def test_allowed_hermes_env(self):
        """Hermes' own .env inside HERMES_HOME is NOT blocked by this rule
        (it's handled by other mechanisms). Only project-local .env is blocked."""
        # Note: hermes internal .env is in ~/.hermes/.env which is NOT a project-local
        # path, but the basename check applies to ANY .env. This is intentional —
        # even ~/.hermes/.env should not be readable via read_file.
        error = get_read_block_error(os.path.expanduser("~/.hermes/.env"))
        assert error is not None

    def test_blocked_set_is_lowercase(self):
        """All entries in the blocked set are lowercase for case-insensitive matching."""
        for name in _BLOCKED_PROJECT_ENV_BASENAMES:
            assert name == name.lower(), f"{name} should be lowercase"


# ---------------------------------------------------------------------------
# Existing cache-file blocking (regression — must still work)
# ---------------------------------------------------------------------------


class TestCacheFileReadBlocking:
    """Internal Hermes cache files must remain blocked."""

    def test_hub_index_cache_blocked(self, tmp_path):
        """Hub index-cache reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "data.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error

    def test_hub_directory_blocked(self, tmp_path):
        """Hub directory reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        hub = hermes_home / "skills" / ".hub" / "metadata.json"
        hub.parent.mkdir(parents=True)
        hub.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(hub))
            assert error is not None


# ---------------------------------------------------------------------------
# Combined: env guard + cache guard don't interfere
# ---------------------------------------------------------------------------


class TestCombinedGuards:
    """Both guards should work independently without interference."""

    def test_env_guard_works_regardless_of_hermes_home(self, tmp_path):
        """The env basename guard does not depend on HERMES_HOME resolution."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            # Regular project .env should still be blocked
            error = get_read_block_error("/workspace/.env")
            assert error is not None

            # .env.example should still be allowed
            error = get_read_block_error("/workspace/.env.example")
            assert error is None

    def test_cache_guard_still_works_with_env_guard(self, tmp_path):
        """Cache file blocking still works when env guard is active."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "x"
        cache.parent.mkdir(parents=True)
        cache.write_text("")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error


# ---------------------------------------------------------------------------
# External provider-CLI credential stores (sibling follow-up to #17656)
#
# #17656 / #30721 / #30972 only read-deny credential stores UNDER HERMES_HOME.
# The provider-CLI stores Hermes imports OAuth tokens from live OUTSIDE
# HERMES_HOME, so the hermes_dirs resolve-loop never reaches them — yet
# read_file can exfiltrate them. ~/.codex/auth.json is intentionally excluded
# (#12360 made Hermes stop touching it by design).
# ---------------------------------------------------------------------------


class TestExternalCredentialStoreReadBlocking:
    """External provider-CLI credential stores must be read-denied."""

    @pytest.mark.parametrize("rel", [
        ".claude/.credentials.json",
        ".claude.json",
        ".config/github-copilot/hosts.json",
        ".config/github-copilot/apps.json",
        ".minimax/credentials.json",
    ])
    def test_external_credential_stores_read_denied(self, monkeypatch, tmp_path, rel):
        """read_file on external provider-CLI credential stores is blocked."""
        # Isolate OS home and HERMES_HOME at separate tmp dirs so the
        # HERMES_HOME credential loop cannot coincidentally match (profile-safe
        # rule: tests that mock home must also pin HERMES_HOME).
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")

        error = get_read_block_error(str(target))
        assert error is not None, f"{rel} should be read-denied"
        assert "Access denied" in error
        assert "credential" in error.lower()

    def test_xdg_config_home_honored_for_copilot(self, monkeypatch, tmp_path):
        """Copilot store under a custom XDG_CONFIG_HOME is still read-denied."""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        target = xdg / "github-copilot" / "hosts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")

        assert get_read_block_error(str(target)) is not None

    def test_codex_auth_not_read_denied(self, monkeypatch, tmp_path):
        """~/.codex/auth.json is intentionally external (#12360) — not denied here."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        target = tmp_path / ".codex" / "auth.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")

        assert get_read_block_error(str(target)) is None

    def test_non_credential_file_in_claude_dir_readable(self, monkeypatch, tmp_path):
        """Negative: a non-credential file under ~/.claude stays readable."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        target = tmp_path / ".claude" / "CLAUDE.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("hi")

        assert get_read_block_error(str(target)) is None
