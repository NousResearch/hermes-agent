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
    get_process_hermes_home,
    heal_hermes_managed_node,
    hermes_managed_node_tree_present,
    iter_hermes_node_dirs,
    is_container,
    node_tool_runnable,
    parse_reasoning_effort,
    reset_hermes_home_override,
    secure_parent_dir,
    set_hermes_home_override,
    with_hermes_node_path,
)


class TestGetDefaultHermesRoot:
    """Tests for get_default_hermes_root() — Docker/custom deployment awareness."""

    def test_no_hermes_home_returns_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME is not set, returns ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert get_default_hermes_root() == tmp_path / ".hermes"





    def test_docker_profile_active(self, tmp_path, monkeypatch):
        """When a Docker profile is active (HERMES_HOME=<root>/profiles/<name>),
        returns the Docker root, not the profile dir."""
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == docker_root

    @pytest.mark.windows_only
    def test_no_hermes_home_returns_localappdata_root_on_windows(self, tmp_path, monkeypatch):
        """Native Windows falls back to %LOCALAPPDATA%\\hermes, not ~/.hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")

        assert get_default_hermes_root() == local_appdata / "hermes"

    def test_result_memoised_until_env_or_home_changes(self, tmp_path, monkeypatch):
        """Repeated calls reuse the memo; HERMES_HOME / home changes invalidate.

        get_default_hermes_root() resolves HERMES_HOME against the native
        home (~80us of path resolution) and is called at 31+ sites — every
        _load_global_auth_store() (per provider row in the /model picker),
        kanban, backup, gateway, update. The memo is keyed on
        (native home, HERMES_HOME) compared for free each call.
        """
        # HERMES_HOME set to a Docker-profile path: every call resolves the
        # env path against the native home (the ~80us work the memo skips).
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Probe the expensive inner work: the memo check itself calls
        # _get_platform_default_hermes_home() on every call (even hits), so
        # count Path.resolve on the env path instead — only the actual
        # resolution branch pays it.
        resolve_calls = {"n": 0}
        orig_resolve = Path.resolve

        def counting_resolve(self, *a, **k):
            resolve_calls["n"] += 1
            return orig_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", counting_resolve)
        # raising=False: on pre-fix code the memo attribute doesn't exist
        # (that IS the fix); the reset is a no-op there so the measured-work
        # assertion below fails genuinely instead of erroring.
        monkeypatch.setattr(
            hermes_constants, "_default_hermes_root_memo", None, raising=False
        )

        first = get_default_hermes_root()
        first_count = resolve_calls["n"]
        for _ in range(10):
            get_default_hermes_root()
        assert resolve_calls["n"] == first_count, (
            "repeated calls must be memo hits (no path resolution on hits), "
            f"resolve went {first_count} -> {resolve_calls['n']}"
        )
        assert first == docker_root

        # HERMES_HOME change invalidates the memo (fresh resolution).
        other_profile = docker_root / "profiles" / "writer"
        other_profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(other_profile))
        before = resolve_calls["n"]
        assert get_default_hermes_root() == docker_root
        assert resolve_calls["n"] > before, (
            "HERMES_HOME change must force a fresh resolution"
        )





class TestGetHermesHome:
    """Tests for get_hermes_home() platform-aware fallback."""

    @pytest.mark.windows_only
    def test_windows_fallback_uses_localappdata(self, tmp_path, monkeypatch):
        """When HERMES_HOME is unset on Windows, use %LOCALAPPDATA%\\hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)

        assert get_hermes_home() == local_appdata / "hermes"


class TestGetProcessHermesHome:
    """Tests for get_process_hermes_home() — process launch scope.

    Contract: resolve only the process env / platform default, and never
    follow the context-local override that per-task profile scoping installs
    via set_hermes_home_override().
    """

    def test_env_set_returns_that_path(self, tmp_path, monkeypatch):
        home = tmp_path / "launch-home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert get_process_hermes_home() == home


    @pytest.mark.parametrize(
        "mountinfo",
        [
            "",
            "malformed",
            "35 2 0:52 / / rw containerd",
            "35 2 8:2 / / rw,relatime - ext4 /dev/sda2 rw",
            # Host root on a volume group that merely contains "docker" in the
            # backing device name — the marker is in the mount source field,
            # not in the fstype/super-options, so it must not be detected.
            "35 2 253:0 / / rw,relatime - ext4 /dev/mapper/docker--vg-root rw",
            "35 2 253:0 / / rw,relatime - xfs /dev/mapper/containers--vg-docker--pool rw",
        ],
    )
    def test_non_container_or_malformed_root_is_not_detected(self, mountinfo):
        assert hermes_constants._root_mount_has_container_runtime(mountinfo) is False

    def test_runtime_marker_in_super_options_is_detected_despite_device_source(self):
        # Marker lives in the super options (docker overlay upperdir) while the
        # mount source itself is a docker-named host device — still a container.
        mountinfo = (
            "35 2 253:0 / / rw - overlay /dev/mapper/docker--vg-root/lowerdir "
            "rw,upperdir=/var/lib/docker/overlay2/x/upper"
        )
        assert hermes_constants._root_mount_has_container_runtime(mountinfo) is True


class TestHermesManagedNode:
    @pytest.mark.windows_only
    def test_windows_node_dir_prefers_portable_root(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        node_dir = home / "node"
        bin_dir = node_dir / "bin"
        node_dir.mkdir(parents=True)
        bin_dir.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert iter_hermes_node_dirs() == [node_dir, bin_dir]

    @pytest.mark.windows_only
    def test_windows_finds_npm_cmd_before_path(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        node_dir = home / "node"
        node_dir.mkdir(parents=True)
        npm_cmd = node_dir / "npm.cmd"
        npm_cmd.write_text("@echo off\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(hermes_constants, "node_tool_runnable", lambda path: True)

        assert find_hermes_node_executable("npm") == str(npm_cmd)



    @pytest.mark.windows_only
    def test_windows_skips_broken_managed_npm_without_path_fallback(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_npm = home / "node" / "npm.cmd"
        managed_npm.parent.mkdir(parents=True)
        managed_npm.write_text("@echo off\n")
        bin_dir = tmp_path / "nodejs"
        bin_dir.mkdir()
        path_npm = bin_dir / "npm.cmd"
        path_npm.write_text("@echo off\n")
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

    def test_outdated_managed_node_heals_to_target_major(self, tmp_path, monkeypatch):
        """A healthy managed tree below the target major upgrades on next resolve."""
        target = hermes_constants._HERMES_NODE_TARGET_MAJOR
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        old_node = self._stub(
            managed_bin, "node", f"#!/bin/sh\necho 'v{target - 1}.20.0'\nexit 0\n"
        )
        heal_called = {"value": False}

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        def _heal():
            heal_called["value"] = True
            old_node.write_text(f"#!/bin/sh\necho 'v{target}.5.1'\nexit 0\n")
            old_node.chmod(0o755)
            return True

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        resolved = hermes_constants.find_hermes_node_executable("node")
        assert heal_called["value"] is True
        assert resolved == str(old_node)

    def test_outdated_managed_node_survives_failed_heal(self, tmp_path, monkeypatch):
        """Offline heal failure keeps serving the old tree — old Node beats no Node."""
        target = hermes_constants._HERMES_NODE_TARGET_MAJOR
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        old_node = self._stub(
            managed_bin, "node", f"#!/bin/sh\necho 'v{target - 1}.20.0'\nexit 0\n"
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)

        assert hermes_constants.find_hermes_node_executable("node") == str(old_node)

    def test_target_major_managed_node_does_not_heal(self, tmp_path, monkeypatch):
        """A tree already at the target major never triggers the heal."""
        target = hermes_constants._HERMES_NODE_TARGET_MAJOR
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        node = self._stub(
            managed_bin, "node", f"#!/bin/sh\necho 'v{target}.5.1'\nexit 0\n"
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        def _heal():
            raise AssertionError("heal must not run for an up-to-date tree")

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        assert hermes_constants.find_hermes_node_executable("node") == str(node)



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




    def test_detects_kubernetes_env(self, monkeypatch):
        """KUBERNETES_SERVICE_HOST env var triggers detection (k8s/k3s pod)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
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
        "max", "ultra". If someone removes one from VALID_REASONING_EFFORTS without
        updating the docstring, this test will fail and force the call out.
        """
        documented = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
        assert documented.issubset(set(VALID_REASONING_EFFORTS))


class TestResolvePerModelReasoningEffort:
    """Tests for resolve_per_model_reasoning_effort() — spelling-tolerant
    per-model override lookup from agent.reasoning_overrides dict.

    Contract: the override key the user writes in config.yaml should match
    regardless of how downstream consumers normalize the model string.
    normalize_model_for_provider() converts dots to dashes and
    adds/strips provider prefixes. Our resolver tolerates these
    variations so the user's intent ("this model always gets xhigh")
    is honored no matter which code path feeds the model string.
    """

    def test_exact_match(self):
        """Exact model string match returns the parsed override."""
        from hermes_constants import resolve_per_model_reasoning_effort
        overrides = {"claude-opus-4.5": "xhigh"}
        result = resolve_per_model_reasoning_effort("claude-opus-4.5", overrides)
        assert result == {"enabled": True, "effort": "xhigh"}





    def test_empty_model_returns_none(self):
        """Empty model string returns None."""
        from hermes_constants import resolve_per_model_reasoning_effort
        assert resolve_per_model_reasoning_effort("", {"gpt-5": "low"}) is None

    # --- Spelling tolerance layer ---






    def test_exact_match_wins_over_variant(self):
        """Ambiguity resolution: exact match takes priority over a variant.

        If both 'claude-opus-4.5' (exact) and 'claude-opus-4-5' (dashes
        variant) are keys, the exact input matches the exact key first.
        """
        from hermes_constants import resolve_per_model_reasoning_effort
        overrides = {"claude-opus-4.5": "high", "claude-opus-4-5": "xhigh"}
        result = resolve_per_model_reasoning_effort("claude-opus-4.5", overrides)
        assert result == {"enabled": True, "effort": "high"}





class TestResolveReasoningConfig:
    """Tests for resolve_reasoning_config() — the single shared chokepoint
    every surface (CLI, gateway, TUI, cron, /model switch, fallback) calls.

    Contract: per-model override > global agent.reasoning_effort; the raw
    global value passes through uncoerced (YAML False = disabled); an
    explicit model argument wins over the config's model.default.
    """

    def _cfg(self, effort: object = "medium", overrides=None, default_model="gpt-5"):
        return {
            "model": {"default": default_model},
            "agent": {
                "reasoning_effort": effort,
                "reasoning_overrides": overrides or {},
            },
        }

    def test_per_model_override_wins(self):
        from hermes_constants import resolve_reasoning_config
        cfg = self._cfg(overrides={"claude-opus-4.5": "xhigh"})
        result = resolve_reasoning_config(cfg, "claude-opus-4.5")
        assert result == {"enabled": True, "effort": "xhigh"}



    def test_empty_model_derives_from_config_default(self):
        from hermes_constants import resolve_reasoning_config
        cfg = self._cfg(overrides={"gpt-5": "high"}, default_model="gpt-5")
        assert resolve_reasoning_config(cfg) == {"enabled": True, "effort": "high"}








    def test_malformed_sections_tolerated(self):
        """Non-dict agent/model sections must not raise."""
        from hermes_constants import resolve_reasoning_config
        assert resolve_reasoning_config({"agent": "oops", "model": 42}) is None
        assert resolve_reasoning_config({"agent": None, "model": None}) is None
        assert resolve_reasoning_config({"agent": {"reasoning_overrides": "bad"}}) is None

    def test_invalid_override_value_falls_back_to_global(self):
        """A junk override value for the matching model falls through to global."""
        from hermes_constants import resolve_reasoning_config
        cfg = self._cfg(effort="medium", overrides={"gpt-5": "turbo-max"})
        assert resolve_reasoning_config(cfg, "gpt-5") == {"enabled": True, "effort": "medium"}


class TestReasoningOverridesDefaultConfig:
    """Tests for the agent.reasoning_overrides default config key (Task 2)."""

    def test_default_config_has_reasoning_overrides_key(self):
        """DEFAULT_CONFIG['agent'] contains 'reasoning_overrides' as an empty dict."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert "reasoning_overrides" in DEFAULT_CONFIG["agent"]
        assert DEFAULT_CONFIG["agent"]["reasoning_overrides"] == {}


    def test_spelling_tolerant_lookup_works_with_user_config(self):
        """resolve_per_model_reasoning_effort works with user-added overrides."""
        from hermes_constants import resolve_per_model_reasoning_effort
        # User config with one override, query uses different spelling
        overrides = {
            "anthropic/claude-opus-4.5": "xhigh",  # user wrote with dots
        }
        # Lookup with different spelling (bare, dashes) — should still match
        result = resolve_per_model_reasoning_effort("claude-opus-4-5", overrides)
        assert result == {"enabled": True, "effort": "xhigh"}

        # Another override, bare key
        overrides2 = {"gpt-5": "low"}
        # Lookup with provider prefix — should match
        result2 = resolve_per_model_reasoning_effort("openai/gpt-5", overrides2)
        assert result2 == {"enabled": True, "effort": "low"}


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

    def test_install_tree_skipped(self, monkeypatch):
        """Parent dir equal to (or inside) the install tree must NOT be chmod'd.

        Regression test for #93050: secure_parent_dir() chmod'd /opt/hermes to
        0700 because it has 3 path parts and passed the ``< 3`` guard, locking
        out UID 10000 (hermes user) from traversing the install dir.
        """
        install_root = Path(hermes_constants.__file__).resolve().parent

        # Directly under the install root (e.g. /opt/hermes/auth.json)
        target = install_root / "auth.json"
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))
        secure_parent_dir(target)
        assert called_with == [], "must not chmod the install root"

        # Inside a subdirectory of the install root
        sub = install_root / "subdir"
        target2 = sub / "auth.json"
        called_with2 = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with2.append((str(p), m)))
        secure_parent_dir(target2)
        assert called_with2 == [], "must not chmod dirs inside the install tree"

    def test_install_tree_siblings_still_hardened(self, monkeypatch):
        """Paths OUTSIDE the install tree must still be chmod'd.

        Negative boundary for the install-tree exclusion (#93050): the guard
        compares path components, so a sibling directory whose name merely
        starts with the install root's name (``<install_root>-data``) must
        still receive parent-dir hardening. Pins that the exclusion cannot
        silently widen into a string-prefix match.
        """
        install_root = Path(hermes_constants.__file__).resolve().parent

        # Prefix-named sibling of the install root (/opt/hermes-data/...).
        prefix_sibling = Path(str(install_root) + "-data")
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))
        secure_parent_dir(prefix_sibling / "auth.json")
        assert called_with == [(str(prefix_sibling), 0o700)], (
            "prefix-named siblings of the install root must still be hardened"
        )

        # Ordinary sibling next to the install root (same parent dir).
        sibling = install_root.parent / "unrelated-dir"
        if len(sibling.parts) >= 3 and install_root not in sibling.parents:
            called_with2 = []
            monkeypatch.setattr(
                os, "chmod", lambda p, m: called_with2.append((str(p), m))
            )
            secure_parent_dir(sibling / "auth.json")
            assert called_with2 == [(str(sibling), 0o700)], (
                "siblings of the install root must still be hardened"
            )

    @pytest.mark.require_symlinks
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



class TestWslPathTranslation:
    """Cross-boundary path translation for a Windows-host UI + WSL backend."""

    def test_windows_drive_to_wsl_mount(self):
        assert hermes_constants.windows_path_to_wsl(r"C:\Users\alex") == "/mnt/c/Users/alex"
        assert hermes_constants.windows_path_to_wsl("C:/Users/alex") == "/mnt/c/Users/alex"
        assert hermes_constants.windows_path_to_wsl("D:\\") == "/mnt/d/"

    def test_windows_drive_ignores_non_drive_paths(self):
        assert hermes_constants.windows_path_to_wsl("/home/alex") is None
        assert hermes_constants.windows_path_to_wsl("relative\\dir") is None




    def test_translate_maps_windows_and_unc_on_wsl(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_wsl", lambda: True)
        assert hermes_constants.translate_cwd_for_wsl_backend(r"C:\Users\alex") == "/mnt/c/Users/alex"
        assert hermes_constants.translate_cwd_for_wsl_backend(r"\\wsl.localhost\Ubuntu\home\alex") == "/home/alex"
        # Already-POSIX paths pass through untouched.
        assert hermes_constants.translate_cwd_for_wsl_backend("/home/alex") == "/home/alex"


class TestManagedNodeTreeInUse:
    """managed_node_tree_in_use() detects processes executing from the tree."""

    def _install_fake_psutil(self, monkeypatch, procs):
        import sys
        from types import SimpleNamespace

        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        fake = SimpleNamespace(
            process_iter=lambda fields: [
                SimpleNamespace(info=info) for info in procs
            ]
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)

    def test_always_false_off_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hermes_constants.sys, "platform", "darwin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert hermes_constants.managed_node_tree_in_use() is False

    def test_exe_under_node_dir_counts(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._install_fake_psutil(
            monkeypatch,
            [{"exe": str(home / "node" / "node.exe"), "cmdline": None}],
        )
        assert hermes_constants.managed_node_tree_in_use() is True

    def test_cmdline_arg_under_node_dir_counts(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._install_fake_psutil(
            monkeypatch,
            [
                {
                    "exe": r"C:\Windows\System32\cmd.exe",
                    "cmdline": [r"C:\Windows\System32\cmd.exe", "/d", "/s", "/c", str(home / "node" / "npm.cmd")],
                }
            ],
        )
        assert hermes_constants.managed_node_tree_in_use() is True

    def test_unrelated_process_does_not_count(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._install_fake_psutil(
            monkeypatch,
            [{"exe": r"C:\Program Files\nodejs\node.exe", "cmdline": None}],
        )
        assert hermes_constants.managed_node_tree_in_use() is False

    def test_missing_psutil_is_false(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # None in sys.modules makes `import psutil` raise ImportError.
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert hermes_constants.managed_node_tree_in_use() is False


class _FakeUrlResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _make_node_zip(major: int) -> tuple[str, bytes]:
    import io
    import zipfile

    name = f"node-v{major}.5.1-win-x64.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        root = f"node-v{major}.5.1-win-x64"
        archive.writestr(f"{root}/node.exe", b"fake-node")
        archive.writestr(f"{root}/npm.cmd", "@echo off\r\n")
    return name, buf.getvalue()


class TestWindowsHealStageSwap:
    """_heal_managed_node_windows() must never destroy the live tree before
    its replacement is fully staged, and must defer (return None) when the
    tree is in use instead of forcing the write (#80926)."""

    def _stub_env(
        self, monkeypatch, home, zip_name, zip_bytes, *, in_use=False
    ):
        import urllib.request

        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv(
            "HERMES_NODE_TARGET_MAJOR",
            str(hermes_constants._HERMES_NODE_TARGET_MAJOR),
        )
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(
            hermes_constants, "_managed_node_in_use_notice_printed", False
        )
        monkeypatch.setattr(
            hermes_constants,
            "managed_node_tree_in_use",
            lambda _home=None: in_use,
        )
        monkeypatch.setattr(
            hermes_constants, "node_tool_runnable", lambda path: True
        )

        index_html = f'<a href="./{zip_name}">{zip_name}</a>'.encode()

        def fake_urlopen(url, timeout=0):
            if str(url).endswith(".zip"):
                return _FakeUrlResponse(zip_bytes)
            return _FakeUrlResponse(index_html)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_in_use_defers_without_touching_tree(self, tmp_path, monkeypatch):
        import urllib.request

        home = tmp_path / "hermes"
        old = home / "node"
        old.mkdir(parents=True)
        (old / "node.exe").write_text("old", encoding="utf-8")
        (old / "npm.cmd").write_text("@echo off", encoding="utf-8")
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=True)

        def forbidden_urlopen(url, timeout=0):
            raise AssertionError(f"no download may start while the tree is in use: {url}")

        monkeypatch.setattr(urllib.request, "urlopen", forbidden_urlopen)

        result = hermes_constants._heal_managed_node_windows()

        assert result is None
        # The live tree is untouched, and no staging litter remains.
        assert (old / "node.exe").read_text(encoding="utf-8") == "old"
        assert list(home.glob("node.new-*")) == []
        assert list(home.glob("node.old-*")) == []

    def test_swap_replaces_tree_and_cleans_up(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        old = home / "node"
        old.mkdir(parents=True)
        (old / "node.exe").write_text("old", encoding="utf-8")
        (old / "old-marker").write_text("stale", encoding="utf-8")
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        result = hermes_constants._heal_managed_node_windows()

        assert result is True
        assert (home / "node" / "node.exe").exists()
        assert not (home / "node" / "old-marker").exists()
        assert list(home.glob("node.new-*")) == []
        assert list(home.glob("node.old-*")) == []

    def test_creates_tree_when_absent(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        home.mkdir()
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        result = hermes_constants._heal_managed_node_windows()

        assert result is True
        assert (home / "node" / "node.exe").exists()
        assert list(home.glob("node.new-*")) == []

    def test_rename_refusal_defers_and_preserves_tree(self, tmp_path, monkeypatch):
        import os as _os

        home = tmp_path / "hermes"
        old = home / "node"
        old.mkdir(parents=True)
        (old / "node.exe").write_text("old", encoding="utf-8")
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        real_replace = _os.replace
        state = {"calls": 0}

        def flaky_replace(src, dst):
            state["calls"] += 1
            if state["calls"] == 1:
                raise PermissionError(13, "Access is denied", str(src))
            return real_replace(src, dst)

        monkeypatch.setattr(_os, "replace", flaky_replace)

        result = hermes_constants._heal_managed_node_windows()

        assert result is None
        # The OS refused the swap — the live tree must survive intact.
        assert (old / "node.exe").read_text(encoding="utf-8") == "old"
        assert list(home.glob("node.new-*")) == []
        assert list(home.glob("node.old-*")) == []


    def test_touch_failure_does_not_abort_swap(self, tmp_path, monkeypatch):
        """The post-rename mtime touch is best-effort: a touch failure must
        not abort a swap that already succeeded."""
        import os as _os

        home = tmp_path / "hermes"
        old = home / "node"
        old.mkdir(parents=True)
        (old / "node.exe").write_text("old", encoding="utf-8")
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        calls = {"n": 0}

        def failing_utime(path, times=None):
            calls["n"] += 1
            raise PermissionError(13, "Access is denied", str(path))

        monkeypatch.setattr(_os, "utime", failing_utime)
        result = hermes_constants._heal_managed_node_windows()

        assert result is True
        assert calls["n"] >= 1
        # The new tree is in place despite the touch failure.
        assert (home / "node" / "node.exe").exists()
        assert list(home.glob("node.new-*")) == []
        # The old tree was swapped aside and then removed.
        assert list(home.glob("node.old-*")) == []

    def test_second_rename_failure_rolls_back(self, tmp_path, monkeypatch):
        """The staged->live rename failing must restore the live tree and
        remove the staged copy, reporting a genuine failure (not a deferral)."""
        import os as _os

        home = tmp_path / "hermes"
        old = home / "node"
        old.mkdir(parents=True)
        (old / "node.exe").write_text("old", encoding="utf-8")
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        real_replace = _os.replace
        state = {"calls": 0}

        def flaky_replace(src, dst):
            state["calls"] += 1
            if state["calls"] == 2:
                raise PermissionError(13, "Access is denied", str(src))
            return real_replace(src, dst)

        monkeypatch.setattr(_os, "replace", flaky_replace)

        result = hermes_constants._heal_managed_node_windows()

        assert result is False
        # The live tree was rolled back into place; the staged copy is gone.
        assert (old / "node.exe").read_text(encoding="utf-8") == "old"
        assert list(home.glob("node.new-*")) == []
        assert list(home.glob("node.old-*")) == []

    def test_stale_staging_litter_is_swept(self, tmp_path, monkeypatch):
        import os as _os
        import time as _time

        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        (home / "node" / "node.exe").write_text("old", encoding="utf-8")
        stale_backup = home / "node.old-deadbeef"
        stale_backup.mkdir()
        (stale_backup / "node.exe").write_text("stale", encoding="utf-8")
        stale_staged = home / "node.new-deadbeef"
        stale_staged.mkdir()
        (stale_staged / "node.exe").write_text("stale", encoding="utf-8")
        # The sweep only removes litter older than 10 minutes, so backdate.
        old_ts = _time.time() - 3600
        _os.utime(stale_backup, (old_ts, old_ts))
        _os.utime(stale_staged, (old_ts, old_ts))
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        result = hermes_constants._heal_managed_node_windows()

        assert result is True
        assert not stale_backup.exists()
        assert not stale_staged.exists()
        assert (home / "node" / "node.exe").exists()

    def test_fresh_staging_dirs_are_not_swept(self, tmp_path, monkeypatch):
        """A concurrent heal's in-flight backup must survive the sweep even
        when it was renamed from a long-lived tree (rename preserves mtime —
        the production code touches it after the rename, and the sweep must
        respect that)."""
        import os as _os
        import time as _time

        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        (home / "node" / "node.exe").write_text("old", encoding="utf-8")
        # Simulate a long-lived tree being renamed aside mid-swap: backdate
        # it, rename, then touch exactly like the production swap does.
        old_ts = _time.time() - 3600
        _os.utime(home / "node", (old_ts, old_ts))
        fresh_backup = home / "node.old-deadbeef"
        _os.replace(str(home / "node"), str(fresh_backup))
        _os.utime(fresh_backup, None)
        zip_name, zip_bytes = _make_node_zip(hermes_constants._HERMES_NODE_TARGET_MAJOR)
        self._stub_env(monkeypatch, home, zip_name, zip_bytes, in_use=False)

        result = hermes_constants._heal_managed_node_windows()

        assert result is True
        assert fresh_backup.exists()


class TestHealAttemptFlagSemantics:
    """An in-use deferral must not record the once-per-process heal attempt,
    so a later call can retry once the tree is free (#80926)."""

    def test_deferral_keeps_flag_clear_and_retries(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        (home / "node" / "node.exe").write_text("x", encoding="utf-8")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        calls = {"n": 0}

        def fake_heal():
            calls["n"] += 1
            return None

        monkeypatch.setattr(hermes_constants, "_heal_managed_node_windows", fake_heal)

        assert heal_hermes_managed_node() is False
        assert hermes_constants._managed_node_heal_attempted is False
        # The flag stayed clear, so the next call retries the heal.
        assert heal_hermes_managed_node() is False
        assert calls["n"] == 2

    def test_real_failure_records_attempt(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        (home / "node" / "node.exe").write_text("x", encoding="utf-8")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        calls = {"n": 0}

        def fake_heal():
            calls["n"] += 1
            return False

        monkeypatch.setattr(hermes_constants, "_heal_managed_node_windows", fake_heal)

        assert heal_hermes_managed_node() is False
        assert hermes_constants._managed_node_heal_attempted is True
        # The flag is set, so the once-per-process budget is spent.
        assert heal_hermes_managed_node() is False
        assert calls["n"] == 1
