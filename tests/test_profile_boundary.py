"""Tests for profile boundary enforcement in hermes_constants.

Covers: is_profiled_mode(), get_profile_boundary(), is_within_profile_boundary().
These functions enforce cross-profile isolation — a named profile must not
access another profile's data directory, but should still have full access
to the filesystem for project work.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_constants import (
    is_profiled_mode,
    get_profile_boundary,
    is_within_profile_boundary,
)


@pytest.fixture()
def default_env(tmp_path, monkeypatch):
    """Default profile (admin) — unrestricted, full filesystem access."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def named_profile_env(tmp_path, monkeypatch):
    """Named profile (e.g. 'frog') — restricted from other profiles."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    profile_dir = hermes_root / "profiles" / "frog"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    return profile_dir


@pytest.fixture()
def multi_profile_env(tmp_path, monkeypatch):
    """Two named profiles to test cross-profile blocking."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    frog_dir = hermes_root / "profiles" / "frog"
    xiaoge_dir = hermes_root / "profiles" / "xiaoge"
    frog_dir.mkdir(parents=True)
    xiaoge_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(frog_dir))
    return {
        "root": hermes_root,
        "frog": frog_dir,
        "xiaoge": xiaoge_dir,
    }


# ===================================================================
# is_profiled_mode
# ===================================================================


class TestIsProfiledMode:
    def test_default_profile_returns_false(self, default_env):
        assert is_profiled_mode() is False

    def test_named_profile_returns_true(self, named_profile_env):
        assert is_profiled_mode() is True

    def test_no_hermes_home_returns_false(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert is_profiled_mode() is False

    def test_custom_path_not_in_profiles_returns_false(self, tmp_path, monkeypatch):
        """A custom HERMES_HOME like /opt/hermes is NOT a profile."""
        custom = tmp_path / "opt" / "hermes"
        custom.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert is_profiled_mode() is False

    def test_profiles_in_other_path_components(self, tmp_path, monkeypatch):
        """HERMES_HOME=/tmp/profiles/work → profiled."""
        p = tmp_path / "tmp" / "profiles" / "work"
        p.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(p))
        assert is_profiled_mode() is True

    def test_profiles_dir_name_but_not_structure(self, tmp_path, monkeypatch):
        """HERMES_HOME=.../profiles → not profiled (parent is not 'profiles')."""
        p = tmp_path / "profiles"
        p.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(p))
        assert is_profiled_mode() is False


# ===================================================================
# get_profile_boundary
# ===================================================================


class TestGetProfileBoundary:
    def test_default_profile_no_boundary(self, default_env):
        assert get_profile_boundary() is None

    def test_named_profile_returns_hermes_home(self, named_profile_env):
        boundary = get_profile_boundary()
        assert boundary is not None
        assert boundary == named_profile_env

    def test_no_hermes_home_no_boundary(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert get_profile_boundary() is None


# ===================================================================
# is_within_profile_boundary — DEFAULT profile (unrestricted)
# ===================================================================


class TestBoundaryDefaultProfile:
    """Default profile should allow ALL paths."""

    def test_allows_any_path(self, default_env):
        for path in ["/etc/passwd", "/tmp/foo", "~/bar", "/any/thing"]:
            ok, reason = is_within_profile_boundary(path)
            assert ok is True, f"Should allow {path}, got: {reason}"

    def test_allows_other_profiles(self, default_env, tmp_path):
        """Admin profile can access other profiles' data."""
        other = tmp_path / ".hermes" / "profiles" / "other" / "secret.md"
        ok, reason = is_within_profile_boundary(str(other))
        assert ok is True


# ===================================================================
# is_within_profile_boundary — NAMED profile (cross-profile isolation)
# ===================================================================


class TestBoundaryNamedProfile:
    """Named profile: block other profiles, allow everything else."""

    def test_allows_own_profile_files(self, named_profile_env):
        own_file = named_profile_env / "skills" / "test.md"
        ok, reason = is_within_profile_boundary(str(own_file))
        assert ok is True

    def test_allows_own_profile_nested(self, named_profile_env):
        deep = named_profile_env / "sessions" / "sub" / "dir" / "file.json"
        ok, reason = is_within_profile_boundary(str(deep))
        assert ok is True

    def test_blocks_other_profile_file(self, multi_profile_env):
        other = multi_profile_env["xiaoge"] / "memory.md"
        ok, reason = is_within_profile_boundary(str(other))
        assert ok is False
        assert "other profile" in reason

    def test_blocks_other_profile_nested(self, multi_profile_env):
        other = multi_profile_env["xiaoge"] / "sessions" / "deep" / "data.json"
        ok, reason = is_within_profile_boundary(str(other))
        assert ok is False

    def test_blocks_other_profile_root(self, multi_profile_env):
        """Even listing the other profile's root dir should be blocked."""
        ok, reason = is_within_profile_boundary(str(multi_profile_env["xiaoge"]))
        assert ok is False

    def test_allows_system_files(self, named_profile_env):
        ok, reason = is_within_profile_boundary("/etc/passwd")
        assert ok is True, reason

    def test_allows_tmp(self, named_profile_env):
        ok, reason = is_within_profile_boundary("/tmp/test.txt")
        assert ok is True, reason

    def test_allows_home_projects(self, named_profile_env, tmp_path):
        project = tmp_path / "projects" / "my-app" / "main.py"
        ok, reason = is_within_profile_boundary(str(project))
        assert ok is True, reason

    def test_allows_default_hermes_config(self, multi_profile_env):
        """~/.hermes/config.yaml is outside profiles/, so it's allowed."""
        config = multi_profile_env["root"] / "config.yaml"
        ok, reason = is_within_profile_boundary(str(config))
        assert ok is True, reason

    def test_allows_path_outside_hermes_entirely(self, named_profile_env):
        ok, reason = is_within_profile_boundary("/usr/lib/python3/site.py")
        assert ok is True, reason

    def test_tilde_expansion(self, named_profile_env, tmp_path, monkeypatch):
        """~ should expand correctly."""
        monkeypatch.setenv("HOME", str(tmp_path))
        ok, reason = is_within_profile_boundary("~/file.txt")
        # ~/file.txt is not under profiles/, so allowed
        assert ok is True, reason

    def test_symlink_inside_own_profile(self, named_profile_env):
        """A symlink pointing inside the profile should be allowed."""
        target = named_profile_env / "real_file.txt"
        target.touch()
        link = named_profile_env / "link.txt"
        link.symlink_to(target)
        ok, reason = is_within_profile_boundary(str(link))
        assert ok is True, reason

    def test_symlink_to_other_profile(self, multi_profile_env):
        """A symlink pointing to another profile should be BLOCKED."""
        target = multi_profile_env["xiaoge"] / "secret.txt"
        target.touch()
        link = multi_profile_env["frog"] / "sneaky_link.txt"
        link.symlink_to(target)
        ok, reason = is_within_profile_boundary(str(link))
        # resolve() follows symlinks → ends up in xiaoge → blocked
        assert ok is False, "Symlink to other profile should be blocked"

    def test_nonexistent_path_outside_profiles(self, named_profile_env):
        """Nonexistent paths outside profiles/ should be allowed."""
        ok, reason = is_within_profile_boundary("/nonexistent/path/file.txt")
        assert ok is True, reason

    def test_nonexistent_path_in_other_profile(self, multi_profile_env):
        """Nonexistent paths in other profiles should still be blocked."""
        other = multi_profile_env["xiaoge"] / "does_not_exist" / "file.txt"
        ok, reason = is_within_profile_boundary(str(other))
        assert ok is False

    def test_error_message_mentions_profile_name(self, multi_profile_env):
        other = multi_profile_env["xiaoge"] / "secret.md"
        _, reason = is_within_profile_boundary(str(other))
        # Should mention "frog" (current profile) in the message
        assert "frog" in reason

    def test_blocks_sibling_profile_same_depth(self, multi_profile_env):
        """A file at the same nesting level but in another profile."""
        frog_file = multi_profile_env["frog"] / "data.json"
        xiaoge_file = multi_profile_env["xiaoge"] / "data.json"
        # Make sure frog_file is allowed
        ok, _ = is_within_profile_boundary(str(frog_file))
        assert ok is True
        # But xiaoge_file is blocked
        ok, reason = is_within_profile_boundary(str(xiaoge_file))
        assert ok is False
