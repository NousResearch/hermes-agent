"""Bundled backends must not import at startup (issue #52284).

Discovery used to ``import`` every bundled ``kind: backend`` plugin unconditionally, dragging each
provider SDK into every ``hermes`` invocation — 15-30s of startup on Windows. A bundled backend now
registers a one-shot loader on the registry it feeds (web search, image/video generation, browser,
dashboard auth); the module imports the first time that registry is actually read, and a placeholder
``LoadedPlugin(deferred=True)`` keeps it visible to discovery, enable/disable and ``hermes plugins
list`` until then.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml


# ── helpers ────────────────────────────────────────────────────────────────


def _write_backend_plugin(root: Path, name: str, *, register_body: str = "    pass\n"):
    """A synthetic bundled ``web/`` backend whose import is observable via ``_backend_probe``."""
    from hermes_cli.plugins import PluginManifest

    plugin_dir = root / "web" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.dump({"name": name, "kind": "backend", "version": "1.0.0"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "import _backend_probe\n"
        "_backend_probe.imports += 1\n"
        "\n"
        "\n"
        "def register(ctx):\n" + register_body,
        encoding="utf-8",
    )
    return PluginManifest(
        name=name, kind="backend", source="bundled", path=str(plugin_dir), key=f"web/{name}"
    )


@pytest.fixture
def probe(monkeypatch):
    mod = types.ModuleType("_backend_probe")
    mod.imports = 0
    monkeypatch.setitem(sys.modules, "_backend_probe", mod)
    return mod


@pytest.fixture
def clean_state():
    """Undo whatever a synthetic plugin left in the process-wide registries/modules."""
    from agent import web_search_registry

    before_modules = set(sys.modules)
    yield
    web_search_registry._reset_for_tests()
    for name in set(sys.modules) - before_modules:
        if name.startswith(("hermes_plugins.", "plugins.web.probe")):
            sys.modules.pop(name, None)


def _bundled_backend_keys(mgr, category: str):
    return sorted(
        key
        for key, plugin in mgr._plugins.items()
        if plugin.manifest.source == "bundled"
        and plugin.manifest.kind == "backend"
        and key.startswith(f"{category}/")
    )


# ── the reported symptom: no SDK import at discovery ───────────────────────


class TestDiscoveryDoesNotImportBackends:
    def test_bundled_backends_are_deferred_placeholders(self):
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()

        for category in ("web", "image_gen", "video_gen", "browser", "dashboard_auth"):
            keys = _bundled_backend_keys(mgr, category)
            assert keys, f"no bundled {category} backends discovered"
            for key in keys:
                plugin = mgr._plugins[key]
                assert plugin.deferred is True, f"{key} imported at discovery"
                assert plugin.module is None, f"{key} imported its module at discovery"

    def test_tool_only_backend_without_a_registry_stays_eager(self):
        """``spotify`` feeds no registry, so there is nothing to defer onto."""
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()

        spotify = mgr._plugins.get("spotify")
        assert spotify is not None
        assert spotify.deferred is False


# ── semantics preserved: the first registry read loads the backend ──────────


class TestRegistryReadMaterializesBackends:
    def test_web_registry_read_registers_bundled_providers(self):
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()

        tavily = mgr._plugins["web/tavily"]
        assert tavily.deferred is True

        names = {provider.name for provider in web_search_registry.list_providers()}

        assert {"tavily", "ddgs"} <= names
        assert mgr._plugins["web/tavily"].deferred is False
        assert mgr._plugins["web/tavily"].module is not None

    def test_active_provider_resolution_triggers_the_deferred_import(self):
        """``get_active_search_provider`` reads through ``merged()``, not just ``list_providers()``."""
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()

        assert mgr._plugins["web/ddgs"].deferred is True
        web_search_registry.get_active_search_provider()

        assert mgr._plugins["web/ddgs"].deferred is False

    def test_dashboard_auth_registry_read_materializes_providers(self):
        """The dashboard's fail-closed gate reads this registry — it must still see plugins."""
        from hermes_cli.dashboard_auth import registry as auth_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()

        assert mgr._plugins["dashboard_auth/basic"].deferred is True

        auth_registry.list_providers()

        assert mgr._plugins["dashboard_auth/basic"].deferred is False
        assert mgr._plugins["dashboard_auth/basic"].module is not None


# ── enable/disable + error isolation ───────────────────────────────────────


class TestDeferredBackendLifecycle:
    def test_unload_cancels_a_pending_loader(self, tmp_path, probe, clean_state):
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr._register_deferred_backend(_write_backend_plugin(tmp_path, "probeweb"))

        assert mgr._plugins["web/probeweb"].deferred is True
        assert probe.imports == 0

        mgr.unload()
        web_search_registry.list_providers()

        assert probe.imports == 0, "a disabled/unloaded backend was imported on a registry read"

    def test_disabled_backend_never_imports(self, tmp_path, probe, clean_state):
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        manifest = _write_backend_plugin(tmp_path, "probeweb")
        mgr._register_deferred_backend(manifest)

        # A disabled backend is dropped from the ledger, exactly like an unloaded one.
        owned = [r for r in mgr._registration_order if r.plugin_key == "web/probeweb"]
        mgr._dispose_registrations(owned)
        mgr._forget_registrations(owned)

        web_search_registry.list_providers()

        assert probe.imports == 0

    def test_failing_deferred_load_is_isolated(self, tmp_path, probe, clean_state):
        """One broken bundled backend must not break the registry read for every other one."""
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr.discover_and_load()
        mgr._register_deferred_backend(
            _write_backend_plugin(
                tmp_path, "probeweb", register_body="    raise RuntimeError('boom')\n"
            )
        )

        names = {provider.name for provider in web_search_registry.list_providers()}

        assert {"tavily", "ddgs"} <= names
        assert "boom" in (mgr._plugins["web/probeweb"].error or "")

    def test_deferred_load_runs_once(self, tmp_path, probe, clean_state):
        from agent import web_search_registry
        from hermes_cli.plugins import PluginManager

        mgr = PluginManager()
        mgr._register_deferred_backend(_write_backend_plugin(tmp_path, "probeweb"))

        web_search_registry.list_providers()
        web_search_registry.list_providers()

        assert probe.imports == 1
