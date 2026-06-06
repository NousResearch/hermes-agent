"""Tests for hermes_constants module."""

import os
from pathlib import Path

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    _dir_has_data,
    get_default_hermes_root,
    get_hermes_dir,
    get_hermes_home,
    is_container,
    parse_reasoning_effort,
    secure_parent_dir,
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

    def test_no_hermes_home_returns_localappdata_root_on_windows(self, tmp_path, monkeypatch):
        """Native Windows falls back to %LOCALAPPDATA%\\hermes, not ~/.hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")

        assert get_default_hermes_root() == local_appdata / "hermes"

    def test_no_hermes_home_uses_windows_path_when_localappdata_missing(self, tmp_path, monkeypatch):
        """Windows fallback still uses AppData/Local/hermes without LOCALAPPDATA."""
        home = tmp_path / "Home"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")

        assert get_default_hermes_root() == home / "AppData" / "Local" / "hermes"


class TestGetHermesHome:
    """Tests for get_hermes_home() platform-aware fallback."""

    def test_windows_fallback_uses_localappdata(self, tmp_path, monkeypatch):
        """When HERMES_HOME is unset on Windows, use %LOCALAPPDATA%\\hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)

        assert get_hermes_home() == local_appdata / "hermes"


class TestDirHasData:
    """Tests for _dir_has_data() — used by get_hermes_dir() to disambiguate
    legacy vs new layout.  An empty directory must not count as data,
    otherwise a stray empty dir from a prior install can shadow the
    populated alternative.  #27602."""

    def test_missing_dir_is_not_data(self, tmp_path):
        assert _dir_has_data(tmp_path / "nope") is False

    def test_empty_dir_is_not_data(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _dir_has_data(d) is False

    def test_populated_dir_is_data(self, tmp_path):
        d = tmp_path / "populated"
        d.mkdir()
        (d / "anything.txt").write_text("x")
        assert _dir_has_data(d) is True

    def test_subdirectory_counts_as_data(self, tmp_path):
        d = tmp_path / "with_subdir"
        d.mkdir()
        (d / "subdir").mkdir()
        assert _dir_has_data(d) is True

    def test_path_pointing_at_file_is_not_data(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert _dir_has_data(f) is False


class TestGetHermesDir:
    """Tests for get_hermes_dir() — picks between legacy and new layout
    based on which one actually has data on disk, instead of "old wins if
    it exists at all".  #27602."""

    def test_neither_exists_returns_new(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        new = tmp_path / "platforms/pairing"
        old = tmp_path / "pairing"
        assert get_hermes_dir("platforms/pairing", "pairing") == new

    def test_old_empty_returns_new(self, tmp_path, monkeypatch):
        """The reported footgun: empty pairing/ shadows populated
        platforms/pairing/.  After the fix, the empty old dir loses."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        new = tmp_path / "platforms/pairing"
        old = tmp_path / "pairing"
        old.mkdir()
        assert get_hermes_dir("platforms/pairing", "pairing") == new

    def test_old_populated_returns_old(self, tmp_path, monkeypatch):
        """Backwards-compat: legacy install with real data keeps working."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        new = tmp_path / "platforms/pairing"
        old = tmp_path / "pairing"
        old.mkdir()
        (old / "feishu-approved.json").write_text("{}")
        assert get_hermes_dir("platforms/pairing", "pairing") == old

    def test_new_populated_wins_over_old(self, tmp_path, monkeypatch):
        """Both populated → new layout wins (the canonical direction)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        new = tmp_path / "platforms/pairing"
        old = tmp_path / "pairing"
        old.mkdir()
        new.mkdir(parents=True)
        (old / "old.json").write_text("{}")
        (new / "new.json").write_text("{}")
        assert get_hermes_dir("platforms/pairing", "pairing") == new

    def test_new_empty_old_populated_returns_old(self, tmp_path, monkeypatch):
        """Mid-migration: new dir scaffolded but empty, old has real data."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        new = tmp_path / "platforms/pairing"
        old = tmp_path / "pairing"
        old.mkdir()
        new.mkdir(parents=True)
        (old / "feishu-approved.json").write_text("{}")
        assert get_hermes_dir("platforms/pairing", "pairing") == old

    def test_works_with_image_cache_paths(self, tmp_path, monkeypatch):
        """Sanity check: the resolver isn't pair-specific, it's used
        elsewhere too (e.g. ``cache/images`` vs ``image_cache``)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # legacy wins because it has data
        old = tmp_path / "image_cache"
        old.mkdir()
        (old / "img.png").write_text("x")
        assert get_hermes_dir("cache/images", "image_cache") == old
        # new wins when both populated
        new = tmp_path / "cache" / "images"
        new.mkdir(parents=True)
        (new / "img.png").write_text("x")
        assert get_hermes_dir("cache/images", "image_cache") == new


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


class TestSecureParentDir:
    """Tests for secure_parent_dir() — prevents chmod on / or top-level dirs."""

    def test_safe_path_calls_chmod(self, tmp_path, monkeypatch):
        """Normal nested path (depth >= 3) should call os.chmod."""
        safe_dir = tmp_path / "home" / "user" / ".hermes"
        safe_dir.mkdir(parents=True)
        target = safe_dir / "auth.json"
        target.touch()

        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        secure_parent_dir(target)
        assert len(called_with) == 1
        assert called_with[0] == (str(safe_dir), 0o700)

    def test_root_dir_skipped(self, monkeypatch):
        """Parent resolving to / must NOT be chmod'd."""
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Path("/foo").parent == Path("/")
        secure_parent_dir(Path("/foo"))
        assert called_with == []

    def test_top_level_dir_skipped(self, monkeypatch):
        """Parent resolving to a top-level dir (depth 2) must NOT be chmod'd."""
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Path("/usr/foo").parent == Path("/usr") — depth 2
        secure_parent_dir(Path("/usr/foo"))
        assert called_with == []

    def test_two_component_path_skipped(self, monkeypatch):
        """Parent with < 3 resolved parts must NOT be chmod'd.

        Uses monkeypatch to avoid macOS firmlink resolution of /home.
        """
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Mock Path.resolve to return a short path regardless of OS quirks
        original_resolve = Path.resolve
        def mock_resolve(self):
            if str(self) == "/x/y":
                return Path("/x")
            return original_resolve(self)
        monkeypatch.setattr(Path, "resolve", mock_resolve)

        secure_parent_dir(Path("/x/y"))
        assert called_with == []

    def test_oserror_suppressed(self, tmp_path, monkeypatch):
        """OSError from chmod should be silently caught."""
        safe_dir = tmp_path / "a" / "b" / "c"
        safe_dir.mkdir(parents=True)
        target = safe_dir / "file.json"
        target.touch()

        def raise_oserror(p, m):
            raise OSError("permission denied")

        monkeypatch.setattr(os, "chmod", raise_oserror)
        # Should not raise
        secure_parent_dir(target)

    def test_symlink_resolved(self, tmp_path, monkeypatch):
        """Symlinks should be resolved before checking depth."""
        real_dir = tmp_path / "a" / "b"
        real_dir.mkdir(parents=True)
        target = real_dir / "file.json"
        target.touch()

        # Create a symlink with fewer path components
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        link_target = link / "file.json"

        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Even though /tmp/link has only 3 parts, the resolved path has 4
        # The resolved parent (real_dir) has depth 4, so it should be chmod'd
        secure_parent_dir(link_target)
        assert len(called_with) == 1
        assert called_with[0] == (str(real_dir), 0o700)

