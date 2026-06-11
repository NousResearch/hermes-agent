"""In-tree layout + discovery tests for the bundled hexis_appraisal plugin.

Modeled on tests/plugins/test_langfuse_plugin.py (TestManifest/TestDiscovery):
asserts the bundled directory layout, the loader-read manifest fields, and
that discovery treats the plugin as a standalone opt-in — discovered but NOT
loaded unless explicitly enabled.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "plugins" / "hexis_appraisal"

# The four hooks the plugin registers (manifest `provides_hooks` is asserted
# set-equal to the AST-collected ctx.register_hook names in test_anticreep.py;
# this is the in-tree manifest-side half).
EXPECTED_HOOKS = {
    "on_session_start",
    "pre_llm_call",
    "post_llm_call",
    "on_session_end",
}

SOURCE_MODULES = {
    "__init__.py",
    "appraisal.py",
    "config.py",
    "reflection.py",
    "render.py",
    "store.py",
}


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()
        assert (PLUGIN_DIR / "README.md").exists()

    def test_source_modules_present(self):
        present = {p.name for p in PLUGIN_DIR.glob("*.py")}
        assert present == SOURCE_MODULES

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "hexis_appraisal"
        assert data["version"]
        # Explicit standalone kind — bundled standalone plugins are opt-in.
        assert data["kind"] == "standalone"
        # Zero-dependency posture: stdlib + host surfaces only.
        assert data["pip_dependencies"] == []
        # `provides_hooks` is the key the loader reads
        # (hermes_cli/plugins.py _parse_manifest).
        assert set(data["provides_hooks"]) == EXPECTED_HOOKS


# ---------------------------------------------------------------------------
# Plugin discovery: hexis_appraisal is opt-in (not loaded unless explicitly
# enabled). Mirrors the langfuse discovery guard — a bundled standalone
# plugin must never auto-load.
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

        # hexis_appraisal appears in the plugin registry …
        loaded = manager._plugins.get("hexis_appraisal")
        assert loaded is not None, "plugin not discovered"
        assert loaded.manifest.source == "bundled"
        assert loaded.manifest.kind == "standalone"
        assert set(loaded.manifest.provides_hooks) == EXPECTED_HOOKS
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()
