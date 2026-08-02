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

    def test_no_hermes_home_returns_localappdata_root_on_windows(self, tmp_path, monkeypatch):
        """Native Windows falls back to %LOCALAPPDATA%\\hermes, not ~/.hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")

        assert get_default_hermes_root() == local_appdata / "hermes"



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
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        old_node = self._stub(managed_bin, "node", "#!/bin/sh\necho 'v22.20.0'\nexit 0\n")
        heal_called = {"value": False}

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        def _heal():
            heal_called["value"] = True
            old_node.write_text("#!/bin/sh\necho 'v26.5.1'\nexit 0\n")
            old_node.chmod(0o755)
            return True

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        resolved = hermes_constants.find_hermes_node_executable("node")
        assert heal_called["value"] is True
        assert resolved == str(old_node)

    def test_outdated_managed_node_survives_failed_heal(self, tmp_path, monkeypatch):
        """Offline heal failure keeps serving the old tree — old Node beats no Node."""
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        old_node = self._stub(managed_bin, "node", "#!/bin/sh\necho 'v22.20.0'\nexit 0\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)

        assert hermes_constants.find_hermes_node_executable("node") == str(old_node)

    def test_target_major_managed_node_does_not_heal(self, tmp_path, monkeypatch):
        """A tree already at the target major never triggers the heal."""
        profile_home = tmp_path / "profiles" / "assistant"
        managed_bin = profile_home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        node = self._stub(managed_bin, "node", "#!/bin/sh\necho 'v26.5.1'\nexit 0\n")

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
