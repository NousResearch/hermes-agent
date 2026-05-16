"""Test role-based tool gating (Issue #26085)."""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ── Unit tests for gateway.role_gating ─────────────────────────────

class TestLoadRoleMap:
    """Tests for load_role_map()."""

    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROFILE_HOME", str(tmp_path))
        from gateway.role_gating import load_role_map, _cache
        _cache.clear()
        result = load_role_map()
        assert result == {}

    def test_valid_yaml_returns_mapping(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "role-map.yaml").write_text(yaml.dump({
            "roles": {
                "admin": {"identities": ["sha256:abc123", "user_id:42"]},
                "viewer": {"identities": ["user_id:99"]},
            }
        }))
        monkeypatch.setenv("HERMES_PROFILE_HOME", str(tmp_path))
        from gateway.role_gating import load_role_map, _cache
        _cache.clear()
        result = load_role_map()
        assert result == {
            "sha256:abc123": "admin",
            "user_id:42": "admin",
            "user_id:99": "viewer",
        }

    def test_empty_roles_returns_empty(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "role-map.yaml").write_text("roles: {}\n")
        monkeypatch.setenv("HERMES_PROFILE_HOME", str(tmp_path))
        from gateway.role_gating import load_role_map, _cache
        _cache.clear()
        result = load_role_map()
        assert result == {}


class TestLoadRoleTools:
    """Tests for load_role_tools()."""

    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_PROFILE_HOME", str(tmp_path))
        from gateway.role_gating import load_role_tools, _cache
        _cache.clear()
        result = load_role_tools()
        assert result == {}

    def test_valid_yaml_returns_rules(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "role-tools.yaml").write_text(yaml.dump({
            "global_deny": ["terminal", "file"],
            "roles": {
                "admin": {"allow": [], "deny": []},
                "viewer": {"allow": ["web", "search"], "deny": ["browser"]},
            }
        }))
        monkeypatch.setenv("HERMES_PROFILE_HOME", str(tmp_path))
        from gateway.role_gating import load_role_tools, _cache
        _cache.clear()
        result = load_role_tools()
        assert "global_deny" in result
        assert "terminal" in result["global_deny"]
        assert "roles" in result


class TestResolveRole:
    """Tests for resolve_role()."""

    def test_none_identity_returns_none(self):
        from gateway.role_gating import resolve_role
        assert resolve_role(None) is None

    def test_empty_identity_returns_none(self):
        from gateway.role_gating import resolve_role
        assert resolve_role("") is None

    def test_known_identity_returns_role(self):
        from gateway.role_gating import resolve_role
        role_map = {"sha256:abc": "admin"}
        assert resolve_role("sha256:abc", role_map=role_map) == "admin"

    def test_unknown_identity_returns_none(self):
        from gateway.role_gating import resolve_role
        role_map = {"sha256:abc": "admin"}
        assert resolve_role("sha256:xyz", role_map=role_map) is None


class TestFilterToolsetsByRole:
    """Tests for filter_toolsets_by_role()."""

    def test_no_role_returns_unchanged(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search"}
        assert filter_toolsets_by_role(toolsets, role=None) == toolsets

    def test_global_deny_applied(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search", "file"}
        role_tools = {
            "global_deny": ["terminal", "file"],
            "roles": {
                "viewer": {"allow": [], "deny": []},
            }
        }
        result = filter_toolsets_by_role(toolsets, role="viewer", role_tools=role_tools)
        assert "terminal" not in result
        assert "file" not in result
        assert "web" in result
        assert "search" in result

    def test_allow_list_intersects(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search", "browser"}
        role_tools = {
            "roles": {
                "viewer": {"allow": ["web", "search"]},
            }
        }
        result = filter_toolsets_by_role(toolsets, role="viewer", role_tools=role_tools)
        assert result == {"web", "search"}

    def test_deny_list_removes(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search"}
        role_tools = {
            "roles": {
                "admin": {"allow": [], "deny": ["terminal"]},
            }
        }
        result = filter_toolsets_by_role(toolsets, role="admin", role_tools=role_tools)
        assert result == {"web", "search"}

    def test_empty_allow_means_all(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search"}
        role_tools = {
            "roles": {
                "admin": {"allow": [], "deny": []},
            }
        }
        result = filter_toolsets_by_role(toolsets, role="admin", role_tools=role_tools)
        assert result == toolsets

    def test_unknown_role_returns_global_deny_only(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search"}
        role_tools = {
            "global_deny": ["terminal"],
            "roles": {},
        }
        result = filter_toolsets_by_role(toolsets, role="unknown_role", role_tools=role_tools)
        assert result == {"web", "search"}

    def test_combined_global_deny_and_role_deny(self):
        from gateway.role_gating import filter_toolsets_by_role
        toolsets = {"web", "terminal", "search", "browser", "file"}
        role_tools = {
            "global_deny": ["file"],
            "roles": {
                "viewer": {
                    "allow": ["web", "search", "browser"],
                    "deny": ["browser"],
                },
            }
        }
        result = filter_toolsets_by_role(toolsets, role="viewer", role_tools=role_tools)
        assert result == {"web", "search"}


class TestHashApiKey:
    """Tests for hash_api_key()."""

    def test_produces_sha256_prefix(self):
        from gateway.role_gating import hash_api_key
        result = hash_api_key("test-key")
        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" + 64 hex chars

    def test_deterministic(self):
        from gateway.role_gating import hash_api_key
        assert hash_api_key("same-key") == hash_api_key("same-key")

    def test_different_keys_different_hashes(self):
        from gateway.role_gating import hash_api_key
        assert hash_api_key("key1") != hash_api_key("key2")
