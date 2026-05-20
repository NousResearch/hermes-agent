"""Tests for Phase 002-A path resolver functions in hermes_constants.py.

Covers:
- get_users_root()  — HERMES_USERS_ROOT override + default
- get_user_home()   — delegates to get_users_root()
- get_memory_root() — user-scoped memory directory
- get_plans_root(), get_skills_root(), get_artifacts_root(), get_credentials_root()
- get_runtime_root() — HERMES_RUNTIME_ROOT override + default
- get_mcp_servers_config() — HERMES_MCP_SERVERS_CONFIG override + default
- get_memory_dir() in memory_tool — canonical-first with legacy fallback

All tests are hermetic: they use tmp_path and monkeypatch so no real
~/.hermes state is touched.
"""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_constants():
    """Re-import hermes_constants so env var changes take effect."""
    import importlib
    import hermes_constants as hc
    importlib.reload(hc)
    return hc


# ---------------------------------------------------------------------------
# get_users_root()
# ---------------------------------------------------------------------------

class TestGetUsersRoot:
    def test_default_under_hermes_home(self, tmp_path, monkeypatch):
        """Without HERMES_USERS_ROOT, returns $HERMES_HOME/users/."""
        monkeypatch.delenv("HERMES_USERS_ROOT", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        from hermes_constants import get_users_root
        result = get_users_root()
        assert result == tmp_path / "hermes" / "users"

    def test_env_var_override(self, tmp_path, monkeypatch):
        """HERMES_USERS_ROOT overrides the default."""
        override = tmp_path / "custom-users"
        monkeypatch.setenv("HERMES_USERS_ROOT", str(override))
        from hermes_constants import get_users_root
        assert get_users_root() == override

    def test_env_var_strips_whitespace(self, tmp_path, monkeypatch):
        """Leading/trailing whitespace in HERMES_USERS_ROOT is stripped."""
        override = tmp_path / "users-ws"
        monkeypatch.setenv("HERMES_USERS_ROOT", f"  {override}  ")
        from hermes_constants import get_users_root
        assert get_users_root() == override

    def test_empty_env_var_uses_default(self, tmp_path, monkeypatch):
        """An empty HERMES_USERS_ROOT falls back to default."""
        monkeypatch.setenv("HERMES_USERS_ROOT", "")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        from hermes_constants import get_users_root
        assert get_users_root() == tmp_path / "h" / "users"


# ---------------------------------------------------------------------------
# get_user_home()
# ---------------------------------------------------------------------------

class TestGetUserHome:
    def test_returns_users_root_slash_user_id(self, tmp_path, monkeypatch):
        """get_user_home('blake') == get_users_root() / 'blake'."""
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        from hermes_constants import get_user_home
        assert get_user_home("blake") == tmp_path / "blake"

    def test_inherits_users_root_override(self, tmp_path, monkeypatch):
        """Setting HERMES_USERS_ROOT propagates through get_user_home()."""
        root = tmp_path / "shared-users"
        monkeypatch.setenv("HERMES_USERS_ROOT", str(root))
        from hermes_constants import get_user_home
        assert get_user_home("alice") == root / "alice"

    def test_different_user_ids(self, tmp_path, monkeypatch):
        """Different user_ids produce different paths."""
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        from hermes_constants import get_user_home
        assert get_user_home("alice") != get_user_home("bob")


# ---------------------------------------------------------------------------
# User-scoped path functions
# ---------------------------------------------------------------------------

class TestUserScopedPaths:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        return tmp_path

    def test_get_memory_root(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import get_memory_root
        assert get_memory_root("blake") == tmp_path / "blake" / "memory"

    def test_get_plans_root(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import get_plans_root
        assert get_plans_root("blake") == tmp_path / "blake" / "plans"

    def test_get_skills_root(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import get_skills_root
        assert get_skills_root("blake") == tmp_path / "blake" / "skills"

    def test_get_artifacts_root(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import get_artifacts_root
        assert get_artifacts_root("blake") == tmp_path / "blake" / "artifacts"

    def test_get_credentials_root(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import get_credentials_root
        assert get_credentials_root("blake") == tmp_path / "blake" / "credentials"

    def test_all_paths_under_user_home(self, tmp_path, monkeypatch):
        """All user-scoped paths are subdirectories of get_user_home()."""
        self._setup(tmp_path, monkeypatch)
        from hermes_constants import (
            get_user_home,
            get_memory_root,
            get_plans_root,
            get_skills_root,
            get_artifacts_root,
            get_credentials_root,
        )
        home = get_user_home("blake")
        for fn in (get_memory_root, get_plans_root, get_skills_root,
                   get_artifacts_root, get_credentials_root):
            result = fn("blake")
            assert str(result).startswith(str(home)), (
                f"{fn.__name__}('blake') = {result} is not under {home}"
            )


# ---------------------------------------------------------------------------
# get_runtime_root()
# ---------------------------------------------------------------------------

class TestGetRuntimeRoot:
    def test_default_under_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_RUNTIME_ROOT", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        from hermes_constants import get_runtime_root
        assert get_runtime_root() == tmp_path / "hermes" / "runtime"

    def test_env_var_override(self, tmp_path, monkeypatch):
        override = tmp_path / "fast-runtime"
        monkeypatch.setenv("HERMES_RUNTIME_ROOT", str(override))
        from hermes_constants import get_runtime_root
        assert get_runtime_root() == override

    def test_empty_env_var_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_RUNTIME_ROOT", "")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        from hermes_constants import get_runtime_root
        assert get_runtime_root() == tmp_path / "h" / "runtime"


# ---------------------------------------------------------------------------
# get_mcp_servers_config()
# ---------------------------------------------------------------------------

class TestGetMcpServersConfig:
    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_MCP_SERVERS_CONFIG", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        from hermes_constants import get_mcp_servers_config
        assert get_mcp_servers_config() == tmp_path / "hermes" / "system" / "mcp-servers.json"

    def test_env_var_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "my-mcp.json"
        monkeypatch.setenv("HERMES_MCP_SERVERS_CONFIG", str(cfg_file))
        from hermes_constants import get_mcp_servers_config
        assert get_mcp_servers_config() == cfg_file

    def test_empty_env_var_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_MCP_SERVERS_CONFIG", "")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        from hermes_constants import get_mcp_servers_config
        assert get_mcp_servers_config() == tmp_path / "h" / "system" / "mcp-servers.json"


# ---------------------------------------------------------------------------
# get_memory_dir() in tools/memory_tool — canonical-first, legacy fallback
# ---------------------------------------------------------------------------

class TestGetMemoryDir:
    def test_legacy_fallback_without_user_id(self, tmp_path, monkeypatch):
        """Without HERMES_USER_ID, returns legacy ~/.hermes/memories/."""
        monkeypatch.delenv("HERMES_USER_ID", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        import importlib
        import tools.memory_tool as mt
        importlib.reload(mt)
        result = mt.get_memory_dir()
        assert result == tmp_path / "hermes" / "memories"

    def test_canonical_path_when_user_id_and_dir_exists(self, tmp_path, monkeypatch):
        """With HERMES_USER_ID set and canonical dir existing, returns canonical path."""
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        monkeypatch.setenv("HERMES_USER_ID", "blake")
        # Create the canonical directory so the existence check passes
        canonical = tmp_path / "blake" / "memory"
        canonical.mkdir(parents=True)
        import importlib
        import tools.memory_tool as mt
        importlib.reload(mt)
        result = mt.get_memory_dir()
        assert result == canonical

    def test_legacy_fallback_when_canonical_dir_missing(self, tmp_path, monkeypatch):
        """With HERMES_USER_ID set but canonical dir absent, falls back to legacy."""
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path / "users"))
        monkeypatch.setenv("HERMES_USER_ID", "blake")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        # Do NOT create canonical dir — it doesn't exist yet
        import importlib
        import tools.memory_tool as mt
        importlib.reload(mt)
        result = mt.get_memory_dir()
        assert result == tmp_path / "hermes" / "memories"

    def test_whitespace_only_user_id_treated_as_absent(self, tmp_path, monkeypatch):
        """HERMES_USER_ID=whitespace-only is treated as unset → legacy path."""
        monkeypatch.setenv("HERMES_USER_ID", "   ")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        import importlib
        import tools.memory_tool as mt
        importlib.reload(mt)
        result = mt.get_memory_dir()
        assert result == tmp_path / "hermes" / "memories"
