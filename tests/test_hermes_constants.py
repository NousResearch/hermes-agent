"""Tests for hermes_constants module."""

import os
import sys
from pathlib import Path

import pytest

import hermes_constants
from hermes_platform.host import runtime as host_runtime
from hermes_constants import (
    agent_browser_runnable,
    get_default_hermes_root,
    get_hermes_dir,
    get_hermes_home,
    get_process_hermes_home,
    is_container,
    node_tool_runnable,
    parse_reasoning_effort,
    reset_hermes_home_override,
    secure_parent_dir,
    set_hermes_home_override,
)


class TestGetDefaultHermesRoot:
    """Tests for get_default_hermes_root() — Docker/custom deployment awareness."""

    @pytest.mark.platforms("linux")
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

    def test_expanded_custom_profile_returns_custom_root(self, tmp_path, monkeypatch):
        custom_root = tmp_path / "deployment"
        home_token = "$" + "HOME"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(
            "HERMES_HOME", f"{home_token}/deployment/profiles/research"
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "native-home")

        assert get_default_hermes_root() == custom_root

    @pytest.mark.platforms("windows")
    def test_no_hermes_home_returns_localappdata_root_on_windows(self, tmp_path, monkeypatch):
        """Native Windows falls back to %LOCALAPPDATA%\\hermes, not ~/.hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")

        assert get_default_hermes_root() == local_appdata / "hermes"






class TestGetHermesHome:
    """Tests for get_hermes_home() platform-aware fallback."""

    def test_warn_once_latch_engages_on_first_check_even_without_warning(self, tmp_path, monkeypatch):
        """Regression for #90065: the latch must engage on the first check even when there is
        nothing to warn about, otherwise every get_hermes_home() call re-stats active_profile."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)
        monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: tmp_path)

        get_hermes_home()
        assert hermes_constants._profile_fallback_warned is True

    @pytest.mark.platforms("windows")
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

    def test_process_and_context_homes_expand_environment_and_user_syntax(
        self, tmp_path, monkeypatch
    ):
        home_token = "$" + "HOME"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        for syntax in (home_token, "~"):
            process_home = tmp_path / "process-home"
            monkeypatch.setenv("HERMES_HOME", f"{syntax}/process-home")
            assert get_process_hermes_home() == process_home

            override_home = tmp_path / "override-home"
            token = set_hermes_home_override(f"{syntax}/override-home")
            try:
                assert get_hermes_home() == override_home
                assert get_process_hermes_home() == process_home
            finally:
                reset_hermes_home_override(token)





class TestNodeToolRunnable:
    """Empty executable paths cannot be probed."""

    def test_none_and_empty_rejected(self):
        assert node_tool_runnable(None) is False
        assert node_tool_runnable("") is False


class TestIsContainer:
    """Tests for is_container() — Docker/Podman detection."""

    def _reset_cache(self, monkeypatch):
        """Reset the cached detection result before each test."""
        monkeypatch.setattr(host_runtime, "_container_detected", None)

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



    def test_cgroup_v2_fallback_inspects_only_the_root_mount(self, tmp_path):
        """#58135: a host that merely RUNS containers exposes each container's overlay lowerdir
        (``lowerdir=/var/lib/containerd/...``) at non-root mount points; only the root ('/') line
        says whether *this* process lives in a runtime overlay."""
        markers = ("kubepods", "containerd", "crio")
        host = tmp_path / "host"
        host.write_text(
            "25 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
            "469 554 0:94 / /var/lib/docker/rootfs/overlayfs/7dda83 rw,relatime shared:247 - overlay overlay "
            "rw,lowerdir=/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/33509/fs\n"
        )
        container = tmp_path / "container"
        container.write_text(
            "1 0 0:50 / / rw,relatime - overlay overlay "
            "rw,lowerdir=/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/9/fs\n"
            "2 1 0:51 / /proc rw,nosuid - proc proc rw\n"
        )
        assert host_runtime._root_mount_has_marker(str(host), markers) is False
        assert host_runtime._root_mount_has_marker(str(container), markers) is True
        assert host_runtime._root_mount_has_marker(str(tmp_path / "missing"), markers) is False



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

    def test_prefixed_key_matches_bare_model(self):
        """A custom-provider prefixed key (``ollama-local/qwen3.6:27b``) applies to the bare runtime slug.

        Fallback entries and named custom providers feed ``agent.model`` without the provider
        prefix while the documented key spelling keeps ``provider/model``; a key for a different
        model must still miss.
        """
        from hermes_constants import resolve_per_model_reasoning_effort
        overrides = {"ollama-local/qwen3.6:27b-q4_k_m": "low"}
        assert resolve_per_model_reasoning_effort("qwen3.6:27b-q4_k_m", overrides) == {"enabled": True, "effort": "low"}
        assert resolve_per_model_reasoning_effort("llama3.2:3b", overrides) is None

    def test_direct_match_wins_over_reverse_lookup(self):
        """A direct/variant key match keeps priority over a prefixed reverse match."""
        from hermes_constants import resolve_per_model_reasoning_effort
        overrides = {"qwen3.6:27b": "medium", "ollama-local/qwen3.6:27b": "low"}
        assert resolve_per_model_reasoning_effort("qwen3.6:27b", overrides) == {"enabled": True, "effort": "medium"}


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

    def test_dict_form_passes_bespoke_tier_verbatim_globally_and_per_model(self):
        """#93238: providers with custom tiers (fast/thinking) need the dict form to send their
        real level; a bare non-ladder string stays rejected so typos never reach the wire."""
        from hermes_constants import parse_reasoning_effort, resolve_reasoning_config
        cfg = self._cfg(effort={"enabled": True, "effort": "thinking"},
                        overrides={"lumo-max": {"enabled": True, "effort": "fast"}})
        assert resolve_reasoning_config(cfg, "gpt-5") == {"enabled": True, "effort": "thinking"}
        assert resolve_reasoning_config(cfg, "my-relay/lumo-max") == {"enabled": True, "effort": "fast"}
        assert parse_reasoning_effort("thinking") is None

    def test_dict_form_disabled_or_empty_effort(self):
        """enabled:false disables regardless of level; a dict without a level is 'unset'."""
        from hermes_constants import parse_reasoning_effort
        assert parse_reasoning_effort({"enabled": False, "effort": "low"}) == {"enabled": False}
        assert parse_reasoning_effort({"enabled": True}) is None
        assert parse_reasoning_effort({"effort": 0}) is None


class TestReasoningOverridesDefaultConfig:
    """Tests for the agent.reasoning_overrides default config key (Task 2)."""



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


@pytest.mark.platforms("posix")  # POSIX shell stubs; Windows uses .cmd shims
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



    @pytest.mark.require_symlinks
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

    @pytest.mark.require_symlinks
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




class TestProjectVenvDirOutOfTree:
    """#116148: a checkout with no in-tree venv whose interpreter lives in ``$HERMES_HOME/venvs/<name>``
    (the layout the shipped Windows launchers pin) must resolve to the running interpreter's venv,
    never ``None`` — every updater call site turns ``None`` into a fabricated ``<checkout>/venv`` that
    uv cannot inspect, so tool dependencies are never refreshed."""

    @staticmethod
    def _running_from(monkeypatch, checkout, venv):
        monkeypatch.setattr(hermes_constants, "__file__", str(checkout / "hermes_constants.py"))
        monkeypatch.setattr(sys, "prefix", str(venv))
        monkeypatch.setattr(sys, "base_prefix", str(checkout / "no-such-base"))

    def test_out_of_tree_install_resolves_the_running_interpreter_venv(self, monkeypatch, tmp_path):
        checkout = tmp_path / "hermes-agent"
        checkout.mkdir()
        venv = tmp_path / "venvs" / "hermes"
        hermes_constants.venv_python_path(venv).parent.mkdir(parents=True)
        hermes_constants.venv_python_path(venv).write_text("", encoding="utf-8")
        self._running_from(monkeypatch, checkout, venv)

        assert hermes_constants.project_venv_dir(checkout) == venv

    def test_foreign_root_and_in_tree_venv_are_unchanged(self, monkeypatch, tmp_path):
        """A temp dir / another clone never claims the running venv; an in-tree venv still wins."""
        checkout = tmp_path / "hermes-agent"
        checkout.mkdir()
        venv = tmp_path / "venvs" / "hermes"
        hermes_constants.venv_python_path(venv).parent.mkdir(parents=True)
        hermes_constants.venv_python_path(venv).write_text("", encoding="utf-8")
        self._running_from(monkeypatch, checkout, venv)
        other = tmp_path / "not-our-checkout"
        other.mkdir()

        assert hermes_constants.project_venv_dir(other) is None
        (checkout / ".venv").mkdir()
        assert hermes_constants.project_venv_dir(checkout) == checkout / ".venv"


class TestFirstPartyModuleClassification:
    """Regression tests for #122328: all first-party modules ship in FIRST_PARTY_MODULE_ROOTS."""

    @pytest.mark.parametrize("name", [
        "acp_adapter",
        "agent",
        "batch_runner",
        "cli",
        "cron",
        "gateway",
        "mcp_serve",
        "mini_swe_runner",
        "model_tools",
        "plugins",
        "pm",
        "providers",
        "registration_lifecycle",
        "run_agent",
        "tools",
        "toolset_distributions",
        "toolsets",
        "trajectory_compressor",
        "tui_gateway",
        "utils",
        "hermes_cli",
        "hermes_bootstrap",
        "hermes_state",
    ])
    def test_first_party_modules_are_classified_correctly(self, name):
        assert hermes_constants.is_first_party_module(name) is True
        assert hermes_constants.is_first_party_module(f"{name}.submodule") is True

        hint = hermes_constants.partial_update_hint(ImportError("missing symbol", name=name))
        assert len(hint) > 0
        assert "hermes update" in "\n".join(hint)

    @pytest.mark.parametrize("name", [
        "agentops",
        "pytest",
        "requests",
        "pydantic",
        "pm_extra",
        "not_a_module",
        "",
        None,
    ])
    def test_third_party_and_lookalikes_are_not_first_party(self, name):
        assert hermes_constants.is_first_party_module(name) is False
        assert hermes_constants.partial_update_hint(ImportError("missing symbol", name=name)) == []

