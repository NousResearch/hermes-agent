"""Tests for the bundled observability/llm-api-call-logger plugin."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "llm-api-call-logger"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "llm-api-call-logger"
        assert data["version"]
        assert "post_api_request" in data["hooks"]


# ---------------------------------------------------------------------------
# Plugin discovery: llm-api-call-logger is opt-in (not loaded unless
# explicitly enabled). This guards against accidentally making the plugin
# auto-load or requiring a per-hook load_config() gate.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Verify the plugin directory and yaml are on disk
        bundled_dir = plugins_mod.get_bundled_plugins_dir()
        obs_dir = bundled_dir / "observability" / "llm-api-call-logger"
        obs_yaml = obs_dir / "plugin.yaml"
        assert obs_dir.is_dir(), f"Plugin dir not found: {obs_dir}"
        assert obs_yaml.exists(), f"plugin.yaml not found: {obs_yaml}"

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # Debug: dump all keys
        all_keys = sorted(manager._plugins.keys())
        obs_keys = [k for k in all_keys if "observability" in k or "llm" in k]

        # observability/llm-api-call-logger appears in the plugin registry ...
        loaded = manager._plugins.get("observability/llm-api-call-logger")
        assert loaded is not None, (
            f"plugin not discovered. All observability keys: {obs_keys}. "
            f"All keys count: {len(all_keys)}"
        )
        # ... but is not loaded (opt-in default -> no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Data directory resolution logic
# ---------------------------------------------------------------------------

class TestResolveDataDir:
    """Test _resolve_data_dir directly by loading the module from its file path."""

    def _load_plugin_module(self):
        """Load the plugin module from its file path."""
        mod_name = "plugins.observability.llm_api_call_logger"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(
            mod_name,
            PLUGIN_DIR / "__init__.py",
        )
        assert spec is not None, "could not create module spec"
        assert spec.loader is not None, "module spec has no loader"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_fallback_to_default(self, monkeypatch):
        """When no env vars or config are set, falls back to ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LLM_API_CALL_DATA_DIR", raising=False)
        monkeypatch.delenv("OBSERVABILITY_DATA_DIR", raising=False)
        # Mock get_default_hermes_root to return a path without a config.yaml
        # so the global config fallback path doesn't read a real config.
        import hermes_constants
        monkeypatch.setattr(
            hermes_constants, "get_default_hermes_root",
            lambda: Path("/nonexistent-hermes-root"),
        )

        mod = self._load_plugin_module()
        path = mod._resolve_data_dir()
        assert path is not None
        assert path == Path("~/.hermes").expanduser()

    def test_env_var_llm_api_call_data_dir(self, monkeypatch, tmp_path):
        """LLM_API_CALL_DATA_DIR env var takes highest priority."""
        data_dir = tmp_path / "custom-llm-calls"
        data_dir.mkdir()
        monkeypatch.setenv("LLM_API_CALL_DATA_DIR", str(data_dir))
        # Set HERMES_HOME so module init doesn't try to read non-existent config
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        mod = self._load_plugin_module()
        path = mod._resolve_data_dir()
        assert path == data_dir

    def test_env_var_observability_data_dir(self, monkeypatch, tmp_path):
        """OBSERVABILITY_DATA_DIR env var is second priority."""
        data_dir = tmp_path / "custom-obs"
        data_dir.mkdir()
        monkeypatch.setenv("OBSERVABILITY_DATA_DIR", str(data_dir))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        mod = self._load_plugin_module()
        path = mod._resolve_data_dir()
        assert path == data_dir

    def test_config_profile_override(self, monkeypatch, tmp_path):
        """observability.llm-api-call-logger.data_dir in profile config is read."""
        home = tmp_path / ".hermes"
        home.mkdir()
        data_dir = tmp_path / "from-config"
        data_dir.mkdir()

        config_yaml = home / "config.yaml"
        config_yaml.write_text(
            f"observability:\n  llm-api-call-logger:\n    data_dir: {data_dir}\n"
        )

        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("LLM_API_CALL_DATA_DIR", raising=False)
        monkeypatch.delenv("OBSERVABILITY_DATA_DIR", raising=False)

        mod = self._load_plugin_module()
        path = mod._resolve_data_dir()
        assert path == data_dir
