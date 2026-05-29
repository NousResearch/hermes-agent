"""Regression coverage for theme-driven dashboard navigation grouping.

Cockpit/product themes need a native way to present operator workspaces in the
left rail without CSS-hiding Hermes admin routes or creating fake redirect tabs.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_TSX = ROOT / "web" / "src" / "App.tsx"
THEME_TYPES = ROOT / "web" / "src" / "themes" / "types.ts"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_theme_schema_supports_navigation_config():
    source = _read(THEME_TYPES)

    assert "export interface ThemeNavigationItem" in source
    assert "export interface ThemeNavigationGroup" in source
    assert "export interface ThemeNavigationConfig" in source
    assert "navigation?: ThemeNavigationConfig" in source


def test_sidebar_applies_theme_navigation_without_dom_hacks():
    source = _read(APP_TSX)

    assert "function applyNavigationConfig" in source
    assert "theme.navigation" in source
    assert "sidebarNav.groups" in source
    assert "function SidebarNavGroup" in source
    assert "MutationObserver" not in source
    assert "querySelector(\"#app-sidebar\")" not in source


def test_nav_items_can_target_hash_workspaces_and_stay_active():
    source = _read(APP_TSX)

    assert "function isNavItemActive" in source
    assert "location.hash" in source
    assert "currentFull === targetFull" in source
    assert "item.match" in source


def test_backend_normalises_navigation_config():
    from hermes_cli.web_server import _normalise_theme_definition

    theme = _normalise_theme_definition({
        "name": "cockpit",
        "label": "Cockpit",
        "navigation": {
            "primary": [
                {"path": "/reinforce#home", "label": "Home", "icon": "Activity", "match": ["/reinforce", "/reinforce#home"]},
                {"path": "not-a-path", "label": "Bad"},
            ],
            "groups": [
                {
                    "id": "system",
                    "label": "System / Settings",
                    "items": [
                        {"path": "/sessions", "label": "Sessions", "icon": "MessageSquare"},
                        {"path": "docs", "label": "Bad docs"},
                    ],
                }
            ],
            "unlisted": "hide",
            "pluginSectionLabel": "Extensions",
        },
    })

    assert theme is not None
    nav = theme["navigation"]
    assert nav["primary"] == [
        {
            "path": "/reinforce#home",
            "label": "Home",
            "icon": "Activity",
            "match": ["/reinforce", "/reinforce#home"],
        }
    ]
    assert nav["groups"] == [
        {
            "id": "system",
            "label": "System / Settings",
            "items": [
                {"path": "/sessions", "label": "Sessions", "icon": "MessageSquare"},
            ],
        }
    ]
    assert nav["unlisted"] == "hide"
    assert nav["pluginSectionLabel"] == "Extensions"
