"""Tests for hermes_constants module."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    agent_browser_runnable,
    find_hermes_node_executable,
    find_node_executable,
    find_node_executable_on_path,
    get_default_hermes_root,
    get_hermes_dir,
    get_hermes_home,
    heal_hermes_managed_node,
    hermes_managed_node_tree_present,
    iter_hermes_node_dirs,
    is_container,
    node_tool_runnable,
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
        monkeypatch.setattr(hermes_constants, "node_tool_runnable", lambda path: True)

        assert find_hermes_node_executable("npm") == str(npm_cmd)

    def test_windows_path_fallback_prefers_npm_cmd(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "nodejs"
        bin_dir.mkdir()
        extensionless = bin_dir / "npm"
        powershell = bin_dir / "npm.ps1"
        npm_cmd = bin_dir / "npm.cmd"
        extensionless.write_text("#!/usr/bin/env node\n")
        powershell.write_text("Write-Output npm\n")
        npm_cmd.write_text("@echo off\n")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("PATH", str(bin_dir))

        assert find_node_executable_on_path("npm") == str(npm_cmd)

    def test_windows_node_executable_falls_back_to_safe_path_shim(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        home.mkdir()
        bin_dir = tmp_path / "nodejs"
        bin_dir.mkdir()
        extensionless = bin_dir / "npm"
        npm_cmd = bin_dir / "npm.cmd"
        extensionless.write_text("#!/usr/bin/env node\n")
        npm_cmd.write_text("@echo off\n")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(bin_dir))

        assert find_node_executable("npm") == str(npm_cmd)

    def test_windows_skips_broken_managed_npm_without_path_fallback(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_npm = home / "node" / "npm.cmd"
        managed_npm.parent.mkdir(parents=True)
        managed_npm.write_text("@echo off\n")
        bin_dir = tmp_path / "nodejs"
        bin_dir.mkdir()
        path_npm = bin_dir / "npm.cmd"
        path_npm.write_text("@echo off\n")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)
        monkeypatch.setattr(
            hermes_constants,
            "node_tool_runnable",
            lambda path: False,
        )

        assert hermes_managed_node_tree_present() is True
        assert find_node_executable("npm") is None
        assert find_node_executable("npm") != str(path_npm)

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


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stubs; Windows uses .cmd shims")
class TestNodeToolRunnable:
    """node_tool_runnable() rejects broken Hermes-managed npm/node wrappers."""

    def _stub(self, tmp_path, name, body, mode=0o755):
        path = tmp_path / name
        path.write_text(body)
        path.chmod(mode)
        return path

    def test_none_and_empty_rejected(self):
        assert node_tool_runnable(None) is False
        assert node_tool_runnable("") is False

    def test_runnable_stub_accepted(self, tmp_path):
        good = self._stub(tmp_path, "npm", "#!/bin/sh\necho '11.10.0'\nexit 0\n")
        assert node_tool_runnable(str(good)) is True

    def test_nonzero_exit_rejected(self, tmp_path):
        bad = self._stub(tmp_path, "npm", "#!/bin/sh\nexit 1\n")
        assert node_tool_runnable(str(bad)) is False

    def test_broken_managed_npm_heals_when_node_still_runs(self, tmp_path, monkeypatch):
        """npm can fail while node --version still succeeds (missing lib/cli.js)."""
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        self._stub(managed_bin, "node", "#!/bin/sh\necho '22.0.0'\nexit 0\n")
        broken_npm = self._stub(managed_bin, "npm", "#!/bin/sh\nexit 1\n")
        heal_called = {"value": False}

        system_bin = tmp_path / "system-bin"
        system_bin.mkdir()
        self._stub(system_bin, "npm", "#!/bin/sh\necho '11.10.0'\nexit 0\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", str(system_bin))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        def _heal():
            heal_called["value"] = True
            broken_npm.write_text("#!/bin/sh\necho '22.0.0'\nexit 0\n")
            broken_npm.chmod(0o755)
            return True

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        resolved = find_node_executable("npm")
        assert heal_called["value"] is True
        assert resolved == str(broken_npm)
        assert resolved != str(system_bin / "npm")

    def test_broken_managed_npm_heals_instead_of_path_fallback(self, tmp_path, monkeypatch):
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        broken_npm = self._stub(managed_bin, "npm", "#!/bin/sh\nexit 1\n")
        healed_npm = self._stub(managed_bin, "npm", "#!/bin/sh\necho '22.0.0'\nexit 0\n")

        system_bin = tmp_path / "system-bin"
        system_bin.mkdir()
        good_npm = self._stub(system_bin, "npm", "#!/bin/sh\necho '11.10.0'\nexit 0\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", str(system_bin))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        def _heal():
            broken_npm.write_text(healed_npm.read_text())
            broken_npm.chmod(0o755)
            return True

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        assert find_hermes_node_executable("npm") == str(healed_npm)
        assert find_node_executable("npm") == str(healed_npm)
        assert find_node_executable("npm") != str(good_npm)

    def test_broken_managed_npm_returns_none_when_heal_fails(self, tmp_path, monkeypatch):
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        self._stub(managed_bin, "npm", "#!/bin/sh\nexit 1\n")

        system_bin = tmp_path / "system-bin"
        system_bin.mkdir()
        self._stub(system_bin, "npm", "#!/bin/sh\necho '11.10.0'\nexit 0\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", str(system_bin))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)

        assert find_node_executable("npm") is None

    def test_healthy_managed_npm_still_preferred(self, tmp_path, monkeypatch):
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        managed_npm = self._stub(managed_bin, "npm", "#!/bin/sh\necho '22.0.0'\nexit 0\n")

        system_bin = tmp_path / "system-bin"
        system_bin.mkdir()
        self._stub(system_bin, "npm", "#!/bin/sh\necho '11.10.0'\nexit 0\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", str(system_bin))

        assert find_node_executable("npm") == str(managed_npm)


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

    @pytest.mark.parametrize("value", [False, "false", "FALSE", "disabled", " Disabled "])
    def test_false_aliases_disable_reasoning(self, value):
        """YAML `reasoning_effort: false`/`off`/`no` reaches loaders as a
        boolean; users also hand-write "false"/"disabled". All must mean
        disabled — not "unset, fall back to the default and keep thinking"."""
        assert parse_reasoning_effort(value) == {"enabled": False}

    @pytest.mark.parametrize("value", [None, True])
    def test_non_string_non_false_returns_none(self, value):
        """None and boolean True fall back to the caller default."""
        assert parse_reasoning_effort(value) is None

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
        ["bogus", "very-high", "0", "off", "true", "default"],
    )
    def test_unknown_levels_return_none(self, value):
        """Unrecognized strings fall back to the caller default (None)."""
        assert parse_reasoning_effort(value) is None

    def test_known_supported_levels_are_documented(self):
        """Guard against silently dropping a documented level.

        The docstring promises "minimal", "low", "medium", "high", "xhigh",
        "max". If someone removes one from VALID_REASONING_EFFORTS without
        updating the docstring, this test will fail and force the call out.
        """
        documented = {"minimal", "low", "medium", "high", "xhigh", "max"}
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stubs; Windows uses .cmd shims")
class TestAgentBrowserRunnable:
    """agent_browser_runnable() validates the resolved CLI actually runs.

    Regression coverage for issue #48521: a dangling global symlink left by
    agent-browser's npm postinstall is reported by ``which`` but fails at exec
    with exit 127, silently breaking every browser tool. The validator must
    reject it (and other non-runnable candidates) so callers fall through.
    """

    def _stub(self, tmp_path, name, body, mode=0o755):
        p = tmp_path / name
        p.write_text(body)
        p.chmod(mode)
        return p

    def test_none_and_empty_rejected(self):
        assert agent_browser_runnable(None) is False
        assert agent_browser_runnable("") is False

    def test_dangling_symlink_rejected(self, tmp_path):
        link = tmp_path / "agent-browser"
        link.symlink_to(tmp_path / "does-not-exist")
        # exists() follows the link → False, so it's rejected without exec.
        assert agent_browser_runnable(str(link)) is False

    def test_runnable_binary_accepted(self, tmp_path):
        good = self._stub(tmp_path, "agent-browser", "#!/bin/sh\necho 'agent-browser 0.27.1'\nexit 0\n")
        assert agent_browser_runnable(str(good)) is True

    def test_nonzero_exit_rejected(self, tmp_path):
        bad = self._stub(tmp_path, "agent-browser", "#!/bin/sh\nexit 127\n")
        assert agent_browser_runnable(str(bad)) is False

    def test_not_executable_rejected(self, tmp_path):
        noexec = self._stub(tmp_path, "agent-browser", "#!/bin/sh\necho hi\n", mode=0o644)
        assert agent_browser_runnable(str(noexec)) is False

    def test_npx_fallback_form_accepted(self):
        # The "npx agent-browser" command form is not a real file; npx resolves
        # the package at run time, so the validator trusts it without stat.
        assert agent_browser_runnable("npx agent-browser") is True
        assert agent_browser_runnable("/usr/local/bin/npx agent-browser") is True

    def test_version_probe_uses_windows_hide_flags(self, tmp_path, monkeypatch):
        good = self._stub(tmp_path, "agent-browser", "#!/bin/sh\necho hi\n")
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append((cmd, kwargs))
            return SimpleNamespace(returncode=0)

        import hermes_cli._subprocess_compat as subprocess_compat
        import subprocess as subprocess_mod

        monkeypatch.setattr(subprocess_compat, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(subprocess_mod, "run", fake_run)

        assert agent_browser_runnable(str(good)) is True
        assert captured[0][0] == [str(good), "--version"]
        assert captured[0][1]["creationflags"] == 0x08000000


    def test_node_tool_probe_uses_windows_hide_flags(self, tmp_path, monkeypatch):
        good = self._stub(tmp_path, "node", "#!/bin/sh\necho v22\n")
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append((cmd, kwargs))
            return SimpleNamespace(returncode=0)

        import hermes_cli._subprocess_compat as subprocess_compat
        import subprocess as subprocess_mod

        monkeypatch.setattr(subprocess_compat, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(subprocess_mod, "run", fake_run)

        assert node_tool_runnable(str(good)) is True
        assert captured[0][0] == [str(good), "--version"]
        assert captured[0][1]["creationflags"] == 0x08000000


class TestGetHermesDir:
    """Tests for ``get_hermes_dir(new_subpath, old_name)``.

    Contract: prefer the legacy ``<old_name>/`` location, but only when
    it has content. An empty legacy stub must fall through to the new
    layout so dormant install scaffolds don't orphan populated data at
    ``<new_subpath>/``. Regression guard for #27602.
    """

    def _set_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def test_neither_exists_returns_new(self, tmp_path, monkeypatch):
        self._set_home(tmp_path, monkeypatch)
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == tmp_path / "platforms/pairing"

    def test_legacy_populated_returns_legacy(self, tmp_path, monkeypatch):
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "image_cache"
        legacy.mkdir()
        (legacy / "cached.png").write_bytes(b"x")
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_legacy_populated_with_subdir_returns_legacy(self, tmp_path, monkeypatch):
        """Sub-directories count as content (e.g. nested cache layout)."""
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "matrix" / "store"
        legacy.mkdir(parents=True)
        (legacy / "session").mkdir()  # subdir, not a file
        result = get_hermes_dir("platforms/matrix/store", "matrix/store")
        assert result == legacy

    def test_legacy_empty_returns_new(self, tmp_path, monkeypatch):
        """The #27602 regression: empty legacy dir orphans populated new dir.

        Without the fix, the resolver returned the empty legacy path
        unconditionally, causing the pairing store to forget every
        previously-approved user when an empty ``pairing/`` stub had
        been pre-created at install time.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "pairing"
        legacy.mkdir()
        # Populated new layout — this is the data that must not be orphaned.
        new = tmp_path / "platforms" / "pairing"
        new.mkdir(parents=True)
        (new / "telegram-approved.json").write_text("[]")
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == new

    def test_legacy_empty_and_new_missing_returns_new(self, tmp_path, monkeypatch):
        """Empty legacy + no new yet — return the new path (will be created lazily).

        Slight behaviour change vs the old resolver (which would return the
        empty legacy dir): the new path is what every consumer mkdirs into
        when it doesn't exist, so the next write lands in the canonical
        location instead of perpetuating the empty stub.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "audio_cache"
        legacy.mkdir()
        result = get_hermes_dir("cache/audio", "audio_cache")
        assert result == tmp_path / "cache/audio"

    def test_legacy_is_file_treated_as_content(self, tmp_path, monkeypatch):
        """A non-directory file at the legacy path counts as occupied.

        Defensive against odd installs where the caller previously wrote a
        single file instead of a directory. We honour whatever's there.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "image_cache"
        legacy.write_bytes(b"sentinel")
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_unreadable_legacy_dir_kept(self, tmp_path, monkeypatch):
        """If we can't enumerate the legacy dir, assume occupied — never
        accidentally orphan legacy data on a transient permission error.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "whatsapp" / "session"
        legacy.mkdir(parents=True)
        # Populate the new path too. The point is to verify that an
        # OSError on iterdir does NOT fall through to the new layout.
        new = tmp_path / "platforms" / "whatsapp" / "session"
        new.mkdir(parents=True)
        (new / "creds.json").write_text("{}")

        real_iterdir = Path.iterdir

        def boom(self):
            if self == legacy:
                raise PermissionError("simulated")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        result = get_hermes_dir(
            "platforms/whatsapp/session", "whatsapp/session"
        )
        assert result == legacy

    def test_unstatable_legacy_dir_kept(self, tmp_path, monkeypatch):
        """A ``PermissionError`` raised by the existence check itself (e.g.
        an unreadable parent) must NOT be read as "absent".

        The old ``Path.exists()``/``Path.is_dir()`` gate swallowed
        ``PermissionError`` and returned ``False``, so an unreadable legacy
        dir fell through to the new layout and orphaned legacy data —
        contradicting the docstring's "assume occupied on errors" intent.
        With the ``lstat()``-based gate this raises and is caught as
        occupied. Regression guard for the #27602 follow-up.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "pairing"
        legacy.mkdir()
        # Populate the new path; it must NOT be selected.
        new = tmp_path / "platforms" / "pairing"
        new.mkdir(parents=True)
        (new / "telegram-approved.json").write_text("[]")

        real_lstat = Path.lstat

        def boom(self):
            if self == legacy:
                raise PermissionError("simulated unreadable parent")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", boom)
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == legacy

    def test_dangling_legacy_symlink_returns_new(self, tmp_path, monkeypatch):
        """A dangling legacy symlink must NOT shadow populated new-layout data.

        ``lstat()`` reports the link itself (not its missing target), so the
        helper must resolve the link and treat a broken target as absent —
        matching the old ``exists()`` gate, which followed the link and
        returned False for a dangling one. Otherwise a stale broken symlink
        would orphan real data (a stricter variant of the #27602 bug).
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "pairing"
        legacy.symlink_to(tmp_path / "does-not-exist")
        new = tmp_path / "platforms" / "pairing"
        new.mkdir(parents=True)
        (new / "discord-approved.json").write_text("[]")
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == new

    def test_symlink_to_populated_dir_returns_legacy(self, tmp_path, monkeypatch):
        """A legacy symlink pointing at a populated directory is honoured."""
        self._set_home(tmp_path, monkeypatch)
        real = tmp_path / "real_store"
        real.mkdir()
        (real / "cached.png").write_bytes(b"x")
        legacy = tmp_path / "image_cache"
        legacy.symlink_to(real)
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_symlink_to_empty_dir_returns_new(self, tmp_path, monkeypatch):
        """A legacy symlink pointing at an EMPTY directory falls through."""
        self._set_home(tmp_path, monkeypatch)
        empty = tmp_path / "empty_real"
        empty.mkdir()
        legacy = tmp_path / "audio_cache"
        legacy.symlink_to(empty)
        result = get_hermes_dir("cache/audio", "audio_cache")
        assert result == tmp_path / "cache/audio"
