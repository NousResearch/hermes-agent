"""Tests for hermes_constants module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    display_hermes_home,
    get_default_hermes_root,
    get_hermes_dir,
    get_optional_skills_dir,
    is_container,
    parse_reasoning_effort,
)


class TestGetDefaultHermesRoot:
    """Tests for get_default_hermes_root() — Docker/custom deployment awareness."""

    def test_no_hermes_home_returns_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME is not set, returns ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert get_default_hermes_root() == tmp_path / ".hermes"

    def test_hermes_home_is_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME = ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        native.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(native))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_profile(self, tmp_path, monkeypatch):
        """When HERMES_HOME is a profile under ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        profile = native / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_docker(self, tmp_path, monkeypatch):
        """When HERMES_HOME points outside ~/.hermes (Docker), returns HERMES_HOME."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        assert get_default_hermes_root() == docker_home

    def test_hermes_home_is_custom_path(self, tmp_path, monkeypatch):
        """Any HERMES_HOME outside ~/.hermes is treated as the root."""
        custom = tmp_path / "my-hermes-data"
        custom.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert get_default_hermes_root() == custom

    def test_docker_profile_active(self, tmp_path, monkeypatch):
        """When a Docker profile is active (HERMES_HOME=<root>/profiles/<name>),
        returns the Docker root, not the profile dir."""
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == docker_root


class TestIsContainer:
    """Tests for is_container() — Docker/Podman detection."""

    def _reset_cache(self, monkeypatch):
        """Reset the cached detection result before each test."""
        monkeypatch.setattr(hermes_constants, "_container_detected", None)

    def test_detects_dockerenv(self, monkeypatch, tmp_path):
        """/.dockerenv triggers container detection."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
        assert is_container() is True

    def test_detects_containerenv(self, monkeypatch, tmp_path):
        """/run/.containerenv triggers container detection (Podman)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/run/.containerenv")
        assert is_container() is True

    def test_detects_cgroup_docker(self, monkeypatch, tmp_path):
        """/proc/1/cgroup containing 'docker' triggers detection."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/docker/abc123\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is True

    def test_negative_case(self, monkeypatch, tmp_path):
        """Returns False on a regular Linux host."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is False

    def test_caches_result(self, monkeypatch):
        """Second call uses cached value without re-probing."""
        monkeypatch.setattr(hermes_constants, "_container_detected", True)
        assert is_container() is True
        # Even if we make os.path.exists return False, cached value wins
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        assert is_container() is True


class TestParseReasoningEffort:
    """Tests for parse_reasoning_effort() — string → reasoning config dict."""

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_empty_or_whitespace_returns_none(self, value):
        """Empty / whitespace-only input falls back to caller default (None)."""
        assert parse_reasoning_effort(value) is None

    def test_none_disables_reasoning(self):
        """The literal "none" disables reasoning explicitly."""
        assert parse_reasoning_effort("none") == {"enabled": False}

    @pytest.mark.parametrize("level", list(VALID_REASONING_EFFORTS))
    def test_each_valid_level(self, level):
        """Every level listed in VALID_REASONING_EFFORTS is accepted as-is."""
        assert parse_reasoning_effort(level) == {"enabled": True, "effort": level}

    @pytest.mark.parametrize(
        "raw, expected_effort",
        [
            ("MEDIUM", "medium"),
            ("High", "high"),
            ("  low  ", "low"),
            ("\tXHIGH\n", "xhigh"),
            ("None", False),
        ],
    )
    def test_case_and_whitespace_normalized(self, raw, expected_effort):
        """Mixed case and surrounding whitespace are normalized before lookup."""
        result = parse_reasoning_effort(raw)
        if expected_effort is False:
            assert result == {"enabled": False}
        else:
            assert result == {"enabled": True, "effort": expected_effort}

    @pytest.mark.parametrize(
        "value",
        ["bogus", "very-high", "max", "0", "off", "true", "default"],
    )
    def test_unknown_levels_return_none(self, value):
        """Unrecognized strings fall back to the caller default (None)."""
        assert parse_reasoning_effort(value) is None

    def test_known_supported_levels_are_documented(self):
        """Guard against silently dropping a documented level.

        The docstring promises "minimal", "low", "medium", "high", "xhigh".
        If someone removes one from VALID_REASONING_EFFORTS without updating
        the docstring, this test will fail and force the call out.
        """
        documented = {"minimal", "low", "medium", "high", "xhigh"}
        assert documented.issubset(set(VALID_REASONING_EFFORTS))


class TestGetOptionalSkillsDir:
    """Tests for get_optional_skills_dir() — env override + default fallback."""

    def test_env_override_wins(self, tmp_path, monkeypatch):
        """HERMES_OPTIONAL_SKILLS, when set, takes precedence over everything."""
        override = tmp_path / "custom-optional"
        monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(override))
        assert get_optional_skills_dir() == override
        assert get_optional_skills_dir(default=tmp_path / "ignored") == override

    def test_env_override_is_stripped(self, tmp_path, monkeypatch):
        """Surrounding whitespace in the env var is trimmed before use."""
        override = tmp_path / "whitespace-optional"
        monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", f"  {override}\t\n")
        assert get_optional_skills_dir() == override

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_env_falls_through(self, tmp_path, monkeypatch, blank):
        """Empty / whitespace-only env var is treated as unset."""
        monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", blank)
        default = tmp_path / "from-default"
        assert get_optional_skills_dir(default=default) == default

    def test_default_used_when_env_unset(self, tmp_path, monkeypatch):
        """When env is unset and default provided, default wins."""
        monkeypatch.delenv("HERMES_OPTIONAL_SKILLS", raising=False)
        default = tmp_path / "packaged-optional"
        assert get_optional_skills_dir(default=default) == default

    def test_falls_back_to_hermes_home(self, tmp_path, monkeypatch):
        """No env, no default → HERMES_HOME / 'optional-skills'."""
        monkeypatch.delenv("HERMES_OPTIONAL_SKILLS", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_optional_skills_dir() == tmp_path / "optional-skills"


class TestGetHermesDir:
    """Tests for get_hermes_dir() — backward-compatible subdir resolver."""

    def test_returns_new_path_when_neither_exists(self, tmp_path, monkeypatch):
        """Fresh install: nothing on disk → new layout wins."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == tmp_path / "cache/images"

    def test_returns_old_path_when_legacy_dir_exists(self, tmp_path, monkeypatch):
        """Legacy install: old dir on disk → old layout preserved."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        legacy = tmp_path / "image_cache"
        legacy.mkdir()
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_legacy_wins_even_if_new_path_also_exists(self, tmp_path, monkeypatch):
        """When both exist, the legacy path keeps winning — no migration."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        legacy = tmp_path / "image_cache"
        legacy.mkdir()
        (tmp_path / "cache/images").mkdir(parents=True)
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_legacy_file_at_old_path_also_counts(self, tmp_path, monkeypatch):
        """``exists()`` matches files too — a stray file pins the old path."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "image_cache").write_text("legacy data")
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == tmp_path / "image_cache"

    def test_returned_path_is_absolute(self, tmp_path, monkeypatch):
        """Result is always rooted at HERMES_HOME, regardless of cwd."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = get_hermes_dir("nested/deep/dir", "old_dir")
        assert result.is_absolute()
        assert result == tmp_path / "nested/deep/dir"


class TestDisplayHermesHome:
    """Tests for display_hermes_home() — user-facing path formatter."""

    def test_default_home_uses_tilde_shorthand(self, tmp_path, monkeypatch):
        """No HERMES_HOME set → renders as ``~/.hermes``."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert display_hermes_home() == "~/.hermes"

    def test_profile_path_uses_tilde_shorthand(self, tmp_path, monkeypatch):
        """Profile under ~/ renders with the ``~/`` prefix preserved."""
        profile = tmp_path / ".hermes" / "profiles" / "coder"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert display_hermes_home() == "~/.hermes/profiles/coder"

    def test_custom_path_outside_home_returns_full_path(
        self, tmp_path, monkeypatch
    ):
        """Docker / custom installs fall back to the absolute path."""
        custom = tmp_path / "opt" / "hermes-custom"
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert display_hermes_home() == str(custom)

    def test_arbitrary_subdir_under_home(self, tmp_path, monkeypatch):
        """Any path under home (not just ``.hermes``) gets the shorthand."""
        non_default = tmp_path / "myhermes-data"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(non_default))
        assert display_hermes_home() == "~/myhermes-data"
