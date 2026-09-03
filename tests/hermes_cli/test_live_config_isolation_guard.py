"""Behavioral tests for the live-config test-isolation guard (write-boundary enforcement).

Forensic background (Sept 2026): External plugins and autonomous coding agents
running unit tests without hermetic isolation (or where HERMES_HOME was unset/
restored to live) resolved get_config_path() / save_config() to the developer's
REAL ~/.hermes/config.yaml, wiping live providers, credentials, and models.

Under Option A (write-boundary enforcement):
- Query accessors (get_config_path, get_env_path) remain pure and never raise.
- Any mutation path (save_config, atomic_config_write, save_env_value, etc.)
  fails hard with RuntimeError when targeted at the real production Hermes root
  while running in a test context.
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
    save_env_value,
)

REAL_ROOT = (Path.home() / ".hermes").resolve()


class TestPureQueryAccessorsUnrestricted:
    def test_get_config_path_under_production_home_does_not_raise(self, monkeypatch):
        """get_config_path() is a pure path query and does not raise even when pointing to live home."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        path = get_config_path()
        assert path.resolve() == (REAL_ROOT / "config.yaml").resolve()

    def test_get_env_path_under_production_home_does_not_raise(self, monkeypatch):
        """get_env_path() is a pure path query and does not raise even when pointing to live home."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        path = get_env_path()
        assert path.resolve() == (REAL_ROOT / ".env").resolve()


class TestProductionConfigMutationRefused:
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
        """Pre-write readability check on production config raises before any read-then-write."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            require_readable_config_before_write(REAL_ROOT / "config.yaml")

    def test_unnormalized_production_path_raises(self):
        """Symlink-free but unnormalized spellings still resolve and refuse."""
        sneaky = Path.home() / "subdir" / ".." / ".hermes" / "config.yaml"
        with pytest.raises(RuntimeError, match="live-system guard"):
            atomic_config_write(sneaky, {"model": {"default": "test"}})

    def test_save_config_under_production_home_raises(self, monkeypatch):
        """save_config() fails closed if HERMES_HOME points to production."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        with pytest.raises(RuntimeError, match="live-system guard"):
            save_config({"agent": {"max_turns": 500}})

    def test_save_env_value_under_production_home_raises(self, monkeypatch):
        """save_env_value() fails closed if HERMES_HOME points to production."""
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        with pytest.raises(RuntimeError, match="live-system guard"):
            save_env_value("DUMMY_KEY", "dummy_val")


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

    def test_tmp_hermes_home_save_env_value_works(self, tmp_path, monkeypatch):
        """save_env_value() under an isolated HERMES_HOME succeeds."""
        fake_home = tmp_path / "hermetic-home"
        fake_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(fake_home))
        save_env_value("TEST_SECRET_KEY", "test_secret_val")
        env_file = fake_home / ".env"
        assert env_file.exists()
        assert "TEST_SECRET_KEY" in env_file.read_text(encoding="utf-8")


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
    def test_unisolated_plugin_test_fails_closed_on_mutation(self, monkeypatch):
        """Simulate the reported incident:
        A test helper temporarily set HERMES_HOME and restored it to live.
        get_config_path() returns the path cleanly (pure resolution), but when
        the test calls save_config() or atomic_config_write(), the guard intercepts
        and raises BEFORE any file mutation happens.
        """
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))

        # Pure resolution succeeds without error:
        cfg_path = get_config_path()
        assert cfg_path.resolve() == (REAL_ROOT / "config.yaml").resolve()

        # Mutation attempts fail closed before modifying disk:
        with pytest.raises(RuntimeError, match="live-system guard"):
            save_config({"agent": {"max_turns": "500"}})

        with pytest.raises(RuntimeError, match="live-system guard"):
            atomic_config_write(cfg_path, {"agent": {"max_turns": "500"}})
