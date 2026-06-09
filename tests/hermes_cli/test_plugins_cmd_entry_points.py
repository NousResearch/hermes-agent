"""Tests for entry-point plugin discovery in hermes_cli.plugins_cmd.

Verifies that ``_scan_entry_point_plugins`` and ``_discover_all_plugins``
correctly surface pip-installed plugins that declare the
``hermes_agent.plugins`` entry-point group.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins_cmd import (
    _discover_all_plugins,
    _plugin_exists,
    _resolve_plugin_key,
    _scan_entry_point_plugins,
)


# ── _scan_entry_point_plugins ──────────────────────────────────────────


class TestScanEntryPointPlugins:
    """Unit tests for the entry-point scanner."""

    def test_adds_entry_point_plugin_to_seen(self):
        """A single entry-point plugin is added to the seen dict."""
        fake_ep = MagicMock()
        fake_ep.name = "my-custom-plugin"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [fake_ep]

        seen: dict = {}
        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            _scan_entry_point_plugins(seen)

        assert "my-custom-plugin" in seen
        entry = seen["my-custom-plugin"]
        assert entry[0] == "my-custom-plugin"  # name
        assert entry[3] == "entrypoint"         # source
        assert entry[5] == "my-custom-plugin"   # key

    def test_multiple_entry_points(self):
        """Multiple entry-point plugins are all discovered."""
        ep1 = MagicMock()
        ep1.name = "plugin-a"
        ep2 = MagicMock()
        ep2.name = "plugin-b"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [ep1, ep2]

        seen: dict = {}
        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            _scan_entry_point_plugins(seen)

        assert "plugin-a" in seen
        assert "plugin-b" in seen

    def test_does_not_overwrite_existing(self):
        """Entry-point plugins do not overwrite bundled/user plugins."""
        fake_ep = MagicMock()
        fake_ep.name = "existing-plugin"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [fake_ep]

        # Pre-populate seen with a bundled entry
        seen: dict = {
            "existing-plugin": (
                "existing-plugin", "1.0", "desc", "bundled",
                Path("/some/path"), "existing-plugin",
            )
        }
        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            _scan_entry_point_plugins(seen)

        # Should still be the bundled version
        assert seen["existing-plugin"][3] == "bundled"

    def test_graceful_on_import_error(self):
        """Scanner returns silently if entry_points() raises."""
        seen: dict = {}
        with patch("importlib.metadata.entry_points", side_effect=ImportError):
            _scan_entry_point_plugins(seen)
        assert len(seen) == 0

    def test_dict_fallback_for_older_python(self):
        """Falls back to eps.get() on dict-style entry points()."""
        fake_ep = MagicMock()
        fake_ep.name = "legacy-plugin"
        # Simulate Python <3.12 dict-style return
        fake_eps = {"hermes_agent.plugins": [fake_ep]}

        seen: dict = {}
        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            _scan_entry_point_plugins(seen)

        assert "legacy-plugin" in seen

    def test_no_entry_points_group(self):
        """Handles empty group gracefully (no matching entry points)."""
        fake_eps = MagicMock()
        fake_eps.select.return_value = []

        seen: dict = {}
        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            _scan_entry_point_plugins(seen)

        assert len(seen) == 0


# ── Integration: _discover_all_plugins + _plugin_exists ─────────────────


class TestDiscoverWithEntryPoints:
    """Integration tests for entry-point discovery in the full plugin pipeline."""

    def test_discover_includes_entry_point_plugins(self):
        """_discover_all_plugins returns entry-point plugins alongside bundled."""
        fake_ep = MagicMock()
        fake_ep.name = "pip-plugin"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [fake_ep]

        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            plugins = _discover_all_plugins()

        # Should have bundled plugins plus our fake entry-point plugin
        names = [p[0] for p in plugins]
        assert "pip-plugin" in names

    def test_plugin_exists_finds_entry_point(self):
        """_plugin_exists returns True for an entry-point-only plugin."""
        fake_ep = MagicMock()
        fake_ep.name = "ep-only-plugin"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [fake_ep]

        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            assert _plugin_exists("ep-only-plugin") is True

    def test_plugin_exists_false_for_unknown(self):
        """_plugin_exists returns False for a plugin that does not exist."""
        fake_eps = MagicMock()
        fake_eps.select.return_value = []

        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            assert _plugin_exists("nonexistent-plugin-xyz") is False

    def test_resolve_plugin_key_returns_entry_point_key(self):
        """_resolve_plugin_key resolves entry-point plugins to their canonical key."""
        fake_ep = MagicMock()
        fake_ep.name = "my-web-search"
        fake_eps = MagicMock()
        fake_eps.select.return_value = [fake_ep]

        with patch("importlib.metadata.entry_points", return_value=fake_eps):
            key = _resolve_plugin_key("my-web-search")

        assert key == "my-web-search"
