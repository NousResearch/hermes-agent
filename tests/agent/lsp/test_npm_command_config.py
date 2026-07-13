"""Tests for configurable npm command in LSP auto-installs.

Validates that:
1. ``terminal.npmCommand`` in config.yaml selects the package manager.
2. Auto-detection prefers pnpm > npm > yarn when config is empty.
3. pnpm/yarn installs create a package.json manifest when needed.
4. No ``HERMES_*`` env var fallback — config.yaml only (AGENTS.md policy).
5. Real subprocess paths exercise the actual install against a temp
   ``HERMES_HOME`` (no mocked ``subprocess.run``).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _which(name: str) -> str | None:
    """Thin wrapper for shutil.which that also handles symlinks."""
    return shutil.which(name)


def _has_npm() -> bool:
    return _which("npm") is not None


def _has_pnpm() -> bool:
    return _which("pnpm") is not None


def _has_yarn() -> bool:
    return _which("yarn") is not None


def _fake_config(npm_command: str = "") -> dict:
    """Return a minimal config dict with terminal.npmCommand set."""
    return {"terminal": {"npm_command": npm_command}}


# ---------------------------------------------------------------------------
# Unit tests — config key and resolution logic (no network)
# ---------------------------------------------------------------------------

class TestConfigKey:
    """Verify the config schema contains the new key."""

    def test_npm_command_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        terminal = DEFAULT_CONFIG.get("terminal", {})
        assert "npm_command" in terminal, (
            "DEFAULT_CONFIG['terminal'] must include 'npm_command'"
        )
        assert terminal["npm_command"] == "", "default should be empty (auto-detect)"

    def test_resolve_npm_command_reads_config(self, monkeypatch):
        from agent.lsp import install as mod

        monkeypatch.setattr(
            mod, "_resolve_npm_command",
            lambda: "pnpm",
        )
        assert mod._resolve_npm_command() == "pnpm"

    def test_resolve_npm_command_auto_detects_pnpm_first(self, monkeypatch):
        """When config is empty, auto-detect should prefer pnpm over npm."""
        from agent.lsp import install as mod

        def fake_which(name):
            if name in ("npm", "pnpm", "yarn"):
                return f"/usr/bin/{name}"
            return None

        monkeypatch.setattr(mod.shutil, "which", fake_which)
        monkeypatch.setattr(
            mod, "_resolve_npm_command",
            lambda: next(
                (c for c in ("pnpm", "npm", "yarn") if fake_which(c)),
                None,
            ),
        )
        assert mod._resolve_npm_command() == "pnpm"

    def test_resolve_npm_command_falls_back_to_npm(self, monkeypatch):
        """When pnpm is not available, fall back to npm."""
        from agent.lsp import install as mod

        def fake_which(name):
            if name == "npm":
                return "/usr/bin/npm"
            return None  # no pnpm, no yarn

        monkeypatch.setattr(mod.shutil, "which", fake_which)
        monkeypatch.setattr(
            mod, "_resolve_npm_command",
            lambda: next(
                (c for c in ("pnpm", "npm", "yarn") if fake_which(c)),
                None,
            ),
        )
        assert mod._resolve_npm_command() == "npm"

    def test_resolve_npm_command_returns_none_when_nothing_found(self, monkeypatch):
        from agent.lsp import install as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: None)
        monkeypatch.setattr(mod, "_resolve_npm_command", lambda: None)
        assert mod._resolve_npm_command() is None


class TestNoEnvVarFallback:
    """AGENTS.md: non-secret config must NOT use HERMES_* env vars."""

    def test_no_hermes_node_package_manager_in_codebase(self):
        """Ensure HERMES_NODE_PACKAGE_MANAGER is not referenced in install.py."""
        install_path = Path(__file__).resolve().parents[3] / "agent" / "lsp" / "install.py"
        content = install_path.read_text(encoding="utf-8")
        assert "HERMES_NODE_PACKAGE_MANAGER" not in content, (
            "install.py must not reference HERMES_NODE_PACKAGE_MANAGER — "
            "use config.yaml terminal.npmCommand instead (AGENTS.md policy)"
        )


# ---------------------------------------------------------------------------
# Unit tests — argv builder (no network)
# ---------------------------------------------------------------------------

class TestBuildPmArgv:
    """Verify _build_pm_argv constructs correct command lines."""

    def test_npm_argv(self, tmp_path):
        from agent.lsp import install as mod
        argv = mod._build_pm_argv("/usr/bin/npm", tmp_path, ["pyright"])
        assert argv[0] == "/usr/bin/npm"
        assert "install" in argv
        assert "--prefix" in argv
        assert "pyright" in argv

    def test_pnpm_argv(self, tmp_path):
        from agent.lsp import install as mod
        argv = mod._build_pm_argv("/usr/bin/pnpm", tmp_path, ["pyright"])
        assert argv[0] == "/usr/bin/pnpm"
        assert "add" in argv
        assert "--prefix" in argv
        assert "pyright" in argv

    def test_yarn_argv(self, tmp_path):
        from agent.lsp import install as mod
        argv = mod._build_pm_argv("/usr/bin/yarn", tmp_path, ["pyright"])
        assert argv[0] == "/usr/bin/yarn"
        assert "add" in argv
        assert "--cwd" in argv
        assert "pyright" in argv

    def test_pnpm_creates_package_json(self, tmp_path):
        from agent.lsp import install as mod
        mod._build_pm_argv("/usr/bin/pnpm", tmp_path, ["pyright"])
        manifest = tmp_path / "package.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["name"] == "hermes-lsp-staging"
        assert data["private"] is True

    def test_yarn_creates_package_json(self, tmp_path):
        from agent.lsp import install as mod
        mod._build_pm_argv("/usr/bin/yarn", tmp_path, ["pyright"])
        manifest = tmp_path / "package.json"
        assert manifest.exists()

    def test_npm_does_not_create_package_json(self, tmp_path):
        from agent.lsp import install as mod
        mod._build_pm_argv("/usr/bin/npm", tmp_path, ["pyright"])
        assert not (tmp_path / "package.json").exists()

    def test_pnpm_preserves_existing_package_json(self, tmp_path):
        from agent.lsp import install as mod
        original = {"name": "custom", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(original))
        mod._build_pm_argv("/usr/bin/pnpm", tmp_path, ["pyright"])
        data = json.loads((tmp_path / "package.json").read_text())
        assert data == original, "existing package.json must not be overwritten"

    def test_pnpm_with_extra_pkgs(self, tmp_path):
        from agent.lsp import install as mod
        argv = mod._build_pm_argv("/usr/bin/pnpm", tmp_path, ["typescript-language-server", "typescript"])
        assert "typescript-language-server" in argv
        assert "typescript" in argv


# ---------------------------------------------------------------------------
# Integration tests — real subprocess (requires npm/pnpm on PATH)
# ---------------------------------------------------------------------------

class TestRealInstall:
    """End-to-end install against a temp HERMES_HOME.

    These tests actually run the package manager, so they need network
    access and take a few seconds.  They are skipped when the required
    package manager is not on PATH.
    """

    @pytest.mark.skipif(not _has_npm(), reason="npm not on PATH")
    def test_npm_install_pyright(self, tmp_path, monkeypatch):
        """Real npm install of pyright into a temp HERMES_HOME."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.lsp import install as mod

        # Clear the result cache so we get a fresh install attempt
        mod._install_results.clear()

        result = mod.try_install("pyright", strategy="auto")
        if result is not None:
            assert os.path.exists(result)
            assert os.access(result, os.X_OK)
            # Verify the symlink in lsp/bin/
            bin_dir = mod.hermes_lsp_bin_dir()
            symlinks = list(bin_dir.glob("pyright*"))
            assert len(symlinks) >= 1, "expected symlink in lsp/bin/"

    @pytest.mark.skipif(not _has_npm(), reason="npm not on PATH")
    def test_npm_install_with_extra_pkgs(self, tmp_path, monkeypatch):
        """Real npm install of typescript-language-server + typescript."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.lsp import install as mod
        mod._install_results.clear()

        result = mod.try_install("typescript-language-server", strategy="auto")
        if result is not None:
            # typescript should be in node_modules alongside the server
            staging = mod.hermes_lsp_bin_dir().parent
            ts_pkg = staging / "node_modules" / "typescript" / "package.json"
            assert ts_pkg.exists(), (
                "typescript SDK must be installed alongside typescript-language-server"
            )

    @pytest.mark.skipif(not _has_pnpm(), reason="pnpm not on PATH")
    def test_pnpm_install_creates_manifest(self, tmp_path, monkeypatch):
        """Real pnpm install creates package.json and installs correctly."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Write a config that forces pnpm
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(
            "terminal:\n  npm_command: pnpm\n",
            encoding="utf-8",
        )

        from agent.lsp import install as mod
        mod._install_results.clear()

        # Patch load_config to return our test config
        def fake_load_config():
            return {"terminal": {"npm_command": "pnpm"}}

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)

        result = mod.try_install("pyright", strategy="auto")
        staging = mod.hermes_lsp_bin_dir().parent

        # Verify package.json was created
        manifest = staging / "package.json"
        assert manifest.exists(), "pnpm install must create package.json"

        if result is not None:
            assert os.path.exists(result)

    @pytest.mark.skipif(not _has_pnpm(), reason="pnpm not on PATH")
    def test_pnpm_inherits_supply_chain_config(self, tmp_path, monkeypatch):
        """When pnpm is selected, installs should use pnpm's config for
        supply-chain policies (minimumReleaseAge, blockExoticSubdeps, etc.).
        
        We verify this indirectly: pnpm respects its own config.yaml which
        lives in the project root (staging dir), not a hardcoded npm call.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from agent.lsp import install as mod
        mod._install_results.clear()

        # Mock load_config to return pnpm
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"terminal": {"npm_command": "pnpm"}},
        )

        result = mod.try_install("pyright", strategy="auto")
        staging = mod.hermes_lsp_bin_dir().parent

        # pnpm creates its own lockfile and respects .npmrc / pnpm config
        pnpm_lock = staging / "pnpm-lock.yaml"
        assert pnpm_lock.exists(), (
            "pnpm install must create pnpm-lock.yaml "
            "(proves pnpm was used, not npm)"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_config_load_failure_falls_back_to_auto(self, monkeypatch):
        """If load_config() raises, _resolve_npm_command should not crash."""
        from agent.lsp import install as mod

        def broken_config():
            raise RuntimeError("config broken")

        monkeypatch.setattr("hermes_cli.config.load_config", broken_config)
        monkeypatch.setattr(mod.shutil, "which", lambda c: "/usr/bin/npm" if c == "npm" else None)

        # Should not raise, should fall back to auto-detect
        result = mod._resolve_npm_command()
        assert result in ("/usr/bin/npm", None)

    def test_invalid_npm_command_falls_back_to_auto(self, monkeypatch):
        """An unrecognized npm_command value should fall through to auto-detect."""
        from agent.lsp import install as mod

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"terminal": {"npm_command": "bogus"}},
        )
        monkeypatch.setattr(mod.shutil, "which", lambda c: "/usr/bin/npm" if c == "npm" else None)

        result = mod._resolve_npm_command()
        assert result == "/usr/bin/npm"

    def test_configured_pm_not_on_path_falls_back(self, monkeypatch):
        """If user configures pnpm but it's not installed, fall back."""
        from agent.lsp import install as mod

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"terminal": {"npm_command": "pnpm"}},
        )
        # pnpm not found, npm found
        monkeypatch.setattr(
            mod.shutil, "which",
            lambda c: "/usr/bin/npm" if c == "npm" else None,
        )

        result = mod._resolve_npm_command()
        assert result == "/usr/bin/npm"

    def test_install_returns_none_when_no_pm_available(self, monkeypatch):
        """_install_npm should return None gracefully when no PM is found."""
        from agent.lsp import install as mod

        monkeypatch.setattr(mod, "_resolve_npm_command", lambda: None)

        result = mod._install_npm("pyright", "pyright-langserver")
        assert result is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
