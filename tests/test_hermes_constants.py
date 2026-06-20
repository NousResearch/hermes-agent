"""Tests for hermes_constants module."""

import os
from pathlib import Path

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    display_hermes_home,
    find_hermes_node_executable,
    get_default_hermes_root,
    get_hermes_home,
    iter_hermes_node_dirs,
    is_container,
    parse_reasoning_effort,
    secure_parent_dir,
    with_hermes_node_path,
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


class TestHermesManagedNode:
    def test_windows_node_dir_prefers_portable_root(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        node_dir = home / "node"
        bin_dir = node_dir / "bin"
        node_dir.mkdir(parents=True)
        bin_dir.mkdir()
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert iter_hermes_node_dirs() == [node_dir, bin_dir]

    def test_windows_finds_npm_cmd_before_path(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        node_dir = home / "node"
        node_dir.mkdir(parents=True)
        npm_cmd = node_dir / "npm.cmd"
        npm_cmd.write_text("@echo off\n")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert find_hermes_node_executable("npm") == str(npm_cmd)

    def test_with_hermes_node_path_prepends_existing_managed_dirs(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        node_dir = home / "node"
        bin_dir = node_dir / "bin"
        node_dir.mkdir(parents=True)
        bin_dir.mkdir()
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))

        env = with_hermes_node_path({"PATH": "system-node"})
        parts = env["PATH"].split(os.pathsep)

        assert parts[:2] == [str(node_dir), str(bin_dir)]
        assert parts[-1] == "system-node"


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
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/\n")
        mountinfo_file = tmp_path / "mountinfo"
        mountinfo_file.write_text("22 21 0:20 / /sys rw shared:7 - sysfs sysfs rw\n")
        _real_open = builtins.open

        def _fake_open(p, *a, **kw):
            if p == "/proc/1/cgroup":
                return _real_open(str(cgroup_file), *a, **kw)
            if p == "/proc/self/mountinfo":
                return _real_open(str(mountinfo_file), *a, **kw)
            return _real_open(p, *a, **kw)

        monkeypatch.setattr("builtins.open", _fake_open)
        assert is_container() is False

    def test_detects_kubernetes_env(self, monkeypatch):
        """KUBERNETES_SERVICE_HOST env var triggers detection (k8s/k3s pod)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
        assert is_container() is True

    def test_detects_cgroup_kubepods(self, monkeypatch, tmp_path):
        """/proc/1/cgroup containing 'kubepods' triggers detection."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/kubepods/besteffort/podabc\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is True

    def test_detects_cgroup_v2_via_mountinfo(self, monkeypatch, tmp_path):
        """cgroup v2 (0::/ only) falls back to containerd marker in mountinfo."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("0::/\n")  # cgroup v2 — no runtime marker
        mountinfo_file = tmp_path / "mountinfo"
        mountinfo_file.write_text(
            "1234 1233 0:42 /containerd/.../rootfs / rw - overlay overlay rw\n"
        )
        _real_open = builtins.open

        def _fake_open(p, *a, **kw):
            if p == "/proc/1/cgroup":
                return _real_open(str(cgroup_file), *a, **kw)
            if p == "/proc/self/mountinfo":
                return _real_open(str(mountinfo_file), *a, **kw)
            return _real_open(p, *a, **kw)

        monkeypatch.setattr("builtins.open", _fake_open)
        assert is_container() is True

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


class TestDisplayHermesHome:
    """Tests for display_hermes_home() display formatting.

    Regression coverage for https://github.com/NousResearch/hermes-agent/issues/49500
    — on Windows the relative tail must not leak backslashes into the
    ``~/`` shorthand, producing a mixed path like ``~/AppData\\Local\\hermes``.
    """

    def test_posix_home_uses_shorthand(self, tmp_path, monkeypatch):
        """Standard POSIX install renders the ``~/`` shorthand."""
        monkeypatch.setattr(
            hermes_constants, "get_hermes_home", lambda: tmp_path / ".hermes"
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert display_hermes_home() == "~/.hermes"

    def test_windows_relative_tail_uses_forward_slashes(self, monkeypatch):
        """Regression #49500: the Windows relative tail must render with
        forward slashes so the whole string is a uniform POSIX-style path,
        not the mixed ``~/AppData\\Local\\hermes`` that confused readers."""
        from pathlib import PureWindowsPath

        win_home = PureWindowsPath(r"C:\Users\tester\AppData\Local\hermes")
        win_user_home = PureWindowsPath(r"C:\Users\tester")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: win_home)
        monkeypatch.setattr(Path, "home", lambda: win_user_home)

        result = display_hermes_home()

        assert result == "~/AppData/Local/hermes"
        assert "\\" not in result

    def test_custom_path_outside_home_unchanged(self, monkeypatch):
        """Paths outside the user home still fall back to the absolute path."""
        from pathlib import PureWindowsPath

        custom = PureWindowsPath(r"D:\hermes-custom")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: custom)
        monkeypatch.setattr(Path, "home", lambda: PureWindowsPath(r"C:\Users\tester"))
        assert display_hermes_home() == str(custom)
