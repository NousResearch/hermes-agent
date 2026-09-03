"""Behavioral tests for the live-config test-isolation guard.

Forensic background (Sept 2026): External plugins and autonomous coding agents
running unit tests without hermetic isolation (or where HERMES_HOME was unset/
restored to live) resolved get_config_path() / save_config() to the developer's
REAL ~/.hermes/config.yaml, wiping live providers, credentials, and models.
The guard makes any pytest/unittest-context config write or config path resolution
that targets production fail hard instead of falling through.

These tests are behavioral: they exercise the actual config functions and guard
mechanisms without mocking internal source structure.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.config as hermes_config
from hermes_cli.config import (
    atomic_config_write,
    get_config_path,
    get_env_path,
    require_readable_config_before_write,
    save_config,
)

REAL_ROOT = (Path.home() / ".hermes").resolve()


class TestProductionConfigPathRefused:
    def test_explicit_production_config_write_raises(self):
        """atomic_config_write pointed at ~/.hermes/config.yaml must fail hard."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            atomic_config_write(REAL_ROOT / "config.yaml", {"model": {"default": "test"}})

    def test_production_profile_config_write_raises(self):
        """Profile configs under the real root are production too."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            atomic_config_write(
                REAL_ROOT / "profiles" / "work" / "config.yaml",
                {"model": {"default": "test"}},
            )

    def test_require_readable_production_config_raises(self):
        """Pre-write readability check on production config raises."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            require_readable_config_before_write(REAL_ROOT / "config.yaml")

    def test_unnormalized_production_path_raises(self):
        """Symlink-free but unnormalized spellings still resolve and refuse."""
        sneaky = Path.home() / "subdir" / ".." / ".hermes" / "config.yaml"
        with pytest.raises(RuntimeError, match="live-system guard"):
            atomic_config_write(sneaky, {"model": {"default": "test"}})

    def test_default_resolution_to_production_raises_on_get_config_path(self, monkeypatch):
        """get_config_path() raises when HERMES_HOME resolves to the live user root."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        with pytest.raises(RuntimeError, match="live-system guard"):
            get_config_path()

    def test_default_resolution_to_production_raises_on_get_env_path(self, monkeypatch):
        """get_env_path() raises when HERMES_HOME resolves to the live user root."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        with pytest.raises(RuntimeError, match="live-system guard"):
            get_env_path()

    def test_save_config_under_production_home_raises(self, monkeypatch):
        """save_config() fails closed if HERMES_HOME points to production."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        with pytest.raises(RuntimeError, match="live-system guard"):
            save_config({"agent": {"max_turns": 500}})


class TestHermeticConfigPathsAllowed:
    def test_tmp_config_write_works(self, tmp_path):
        """atomic_config_write to an isolated tmp path succeeds."""
        target = tmp_path / "config.yaml"
        atomic_config_write(target, {"model": {"default": "hermetic-model"}})
        assert target.exists()
        assert "hermetic-model" in target.read_text(encoding="utf-8")

    def test_tmp_hermes_home_save_config_works(self, tmp_path, monkeypatch):
        """save_config() under an isolated HERMES_HOME succeeds."""
        fake_home = tmp_path / "hermetic-home"
        fake_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(fake_home))
        path = get_config_path()
        assert str(fake_home) in str(path)
        save_config({"model": {"default": "hermetic-model"}})
        assert path.exists()
        assert "hermetic-model" in path.read_text(encoding="utf-8")


class TestBypassOptions:
    def test_env_var_bypass_allows_production_path(self, monkeypatch):
        """HERMES_ALLOW_TEST_WRITES_TO_REAL_HOME=1 bypasses the guard."""
        monkeypatch.setenv("HERMES_ALLOW_TEST_WRITES_TO_REAL_HOME", "1")
        # Direct check of the guard helper without performing an actual write
        hermes_config._ensure_config_test_isolation(REAL_ROOT / "config.yaml")

    @pytest.mark.live_system_guard_bypass
    def test_live_system_guard_bypass_marker_disables_guard(self):
        """@pytest.mark.live_system_guard_bypass marker bypasses the guard."""
        # Direct check of the guard helper
        hermes_config._ensure_config_test_isolation(REAL_ROOT / "config.yaml")


class TestExternalPluginReproductionScenario:
    def test_unisolated_plugin_test_fails_closed_before_write(self, monkeypatch):
        """Simulate the reported incident:
        A test helper temporarily set HERMES_HOME and restored it to live.
        A later test calls get_config_path() and attempts to truncate.
        The guard must raise at get_config_path() BEFORE any file write happens.
        """
        # Restore HERMES_HOME to real live user root (unisolated state)
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))

        with pytest.raises(RuntimeError, match="live-system guard"):
            cfg_path = get_config_path()
            # If get_config_path hadn't raised, this would wipe live config:
            cfg_path.write_text("agent:\n  max_turns: '500'\n")
