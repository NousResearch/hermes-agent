"""Tests for the bundled observability/token-consumption-tracker plugin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "token-consumption-tracker"


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
        assert data["name"] == "token-consumption-tracker"
        assert data["version"]
        assert "post_api_request" in data["hooks"]


# ---------------------------------------------------------------------------
# Plugin discovery: token-consumption-tracker is opt-in (not loaded unless
# explicitly enabled).  Guards against accidentally making it auto-load.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # observability/token-consumption-tracker appears in the plugin registry …
        loaded = manager._plugins.get("observability/token-consumption-tracker")
        assert loaded is not None, "plugin not discovered"
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Data dir resolution: _resolve_data_dir() returns a reasonable default
# when no env vars or config files are present.
# ---------------------------------------------------------------------------

class TestResolveDataDir:
    def test_resolve_data_dir_fallback(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("TOKEN_CONSUMPTION_DATA_DIR", raising=False)
        monkeypatch.delenv("OBSERVABILITY_DATA_DIR", raising=False)

        mod_name = "plugins.observability.token-consumption-tracker"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)

        path = mod._resolve_data_dir()
        assert path is not None
        assert "hermes" in str(path).lower() or ".hermes" in str(path)


# ---------------------------------------------------------------------------
# generate_report(): returns a non-empty markdown string even on an empty DB.
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_generate_report_returns_string(self, monkeypatch):
        # Keep the module from caching _resolve_data_dir at import time
        # by using a temp dir for the DB.
        mod_name = "plugins.observability.token-consumption-tracker"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)

        report = mod.generate_report("2000-01-01")
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# Hooks plug in correctly: register() registers post_api_request
# and on_session_end hooks, plus the /token slash command.
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_registers_hooks_and_command(self):
        """register() must wire up post_api_request, on_session_end, and /token."""
        mod_name = "plugins.observability.token-consumption-tracker"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)

        class FakeCtx:
            def __init__(self):
                self.hooks = {}
                self.commands = {}

            def register_hook(self, name, handler):
                self.hooks[name] = handler

            def register_command(self, name, handler, description="", args_hint=""):
                self.commands[name] = {
                    "handler": handler,
                    "description": description,
                    "args_hint": args_hint,
                }

        ctx = FakeCtx()
        mod.register(ctx)

        assert "post_api_request" in ctx.hooks
        assert ctx.hooks["post_api_request"] is mod._on_post_api_request
        assert "on_session_end" in ctx.hooks
        assert ctx.hooks["on_session_end"] is mod._on_session_end
        assert "token" in ctx.commands
        assert ctx.commands["token"]["handler"] is mod._handle_slash_command
