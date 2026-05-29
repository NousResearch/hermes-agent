"""Regression coverage for dashboard plugin route overrides.

The dashboard discovers plugin manifests asynchronously in the browser. A plugin
that overrides `/` must not lose the initial home-page navigation to the
built-in `/sessions` redirect while manifests are still loading.
"""
from __future__ import annotations

from pathlib import Path


APP_TSX = Path(__file__).resolve().parents[2] / "web" / "src" / "App.tsx"
TITLE_RESOLVER = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "resolve-page-title.ts"


def _app_source() -> str:
    return APP_TSX.read_text(encoding="utf-8")


def _title_resolver_source() -> str:
    return TITLE_RESOLVER.read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    start_idx = source.index(start)
    end_idx = source.index(end, start_idx)
    return source[start_idx:end_idx]


def test_root_redirect_waits_for_plugin_manifest_load_window():
    """Root overrides must be able to claim `/` before RootRedirect fires."""
    source = _app_source()

    assert "function RootPending()" in source
    assert "pluginsLoading ? RootPending : RootRedirect" in source
    assert "buildRoutes(builtinRoutes, manifests, pluginsLoading)" in source


def test_override_plugins_remain_navigable_in_sidebar():
    """Override plugins should replace/insert nav items instead of disappearing."""
    source = _app_source()
    build_nav = _section(source, "function buildNavItems", "/** Split merged nav")

    assert "if (manifest.tab.override) continue;" not in build_nav
    assert "const itemPath = manifest.tab.override ?? manifest.tab.path" in build_nav
    assert "const existingIdx = items.findIndex((i) => i.path === itemPath)" in build_nav
    assert "items.splice(existingIdx, 1, pluginItem)" in build_nav


def test_sidebar_plugin_slot_is_rendered_in_cockpit_layout():
    """Declared `sidebar` slots must not be dead declarations."""
    source = _app_source()

    assert 'layoutVariant === "cockpit"' in source
    assert '<PluginSlot name="sidebar" />' in source


def test_override_plugin_tab_path_mounts_same_component():
    """A plugin with both `tab.override` and `tab.path` should render at both routes."""
    source = _app_source()
    build_routes = _section(source, "function buildRoutes", "const SIDEBAR_COLLAPSED_KEY")

    assert "const pluginPaths = [m.tab.override, m.tab.path]" in build_routes
    assert "for (const path of pluginPaths)" in build_routes
    assert "key: `plugin:${m.name}:${path}`" in build_routes


def test_root_title_prefers_plugin_override_before_sessions_default():
    """A plugin overriding `/` should own the chrome title as well as content."""
    source = _title_resolver_source()

    plugin_lookup = 'const plugin = pluginTabs.find((p) => p.path === normalized);'
    sessions_default = 'if (normalized === "/") {'
    assert source.index(plugin_lookup) < source.index(sessions_default)
    assert 'return plugin.label;' in source
