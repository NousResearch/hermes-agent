"""Regression tests for plugin-removal config cleanup (``hermes plugins uninstall``).

``cmd_remove`` / ``dashboard_remove_user_plugin`` delete the plugin tree, so they
must also drop the plugin out of ``plugins.enabled`` / ``plugins.disabled`` /
``plugins.entries`` — a stale ``plugins.entries.<key>`` record keeps a privileged
``allow_tool_override`` grant alive for whatever is installed at that key next.

The cleanup has to be scoped to identities the removed plugin demonstrably owns.
User plugins may be nested and their keys are path-derived, so a namespaced
``image_gen/openai`` shares its directory leaf with a wholly separate flat
``openai`` plugin. These tests drive real ``config.yaml`` I/O under a temporary
``HERMES_HOME`` rather than patching the helpers under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins_cmd import cmd_remove, dashboard_remove_user_plugin


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME, the bundled plugin tree, and entry-point plugins."""
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    bundled = tmp_path / "bundled"
    bundled.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))
    # Managed mode short-circuits save_config; keep it off unless a test asks.
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    # Discovery is deliberately real, but it must not see the repo's own
    # bundled plugins: Hermes ships plugins/image_gen/openai, whose key would
    # collide with the namespaced fixture below and make these tests depend on
    # what upstream happens to bundle. Entry-point plugins likewise depend on
    # what is pip-installed in the test environment.
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._discover_entrypoint_plugins", lambda: []
    )
    return home


def _install(home: Path, rel_path: str, manifest_name: str) -> Path:
    """Create a user plugin at ``plugins/<rel_path>`` with *manifest_name*."""
    plugin_dir = home / "plugins" / rel_path
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": manifest_name, "version": "1.0.0"}),
        encoding="utf-8",
    )
    return plugin_dir


def _write_plugins_config(home: Path, plugins_block: dict) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": plugins_block}), encoding="utf-8"
    )


def _read_plugins_config(home: Path) -> dict:
    raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    return raw.get("plugins") or {}


def _grant(*keys: str) -> dict:
    return {key: {"allow_tool_override": True} for key in keys}


class TestLeafCollision:
    """A namespaced removal must not strip a same-leaf flat plugin's state."""

    def test_removing_namespaced_plugin_spares_flat_plugin_sharing_the_leaf(
        self, plugin_env
    ):
        home = plugin_env
        # image_gen/openai (key "image_gen/openai") and a separate flat plugin
        # whose manifest name is also "openai" (key "openai").
        namespaced = _install(home, "image_gen/openai", "openai")
        flat = _install(home, "openai", "openai")
        _write_plugins_config(
            home,
            {
                "enabled": ["image_gen/openai", "openai"],
                "disabled": [],
                "entries": _grant("image_gen/openai", "openai"),
            },
        )

        cmd_remove("image_gen/openai")

        plugins_cfg = _read_plugins_config(home)
        # The bystander keeps its enabled state AND its override grant.
        assert "openai" in plugins_cfg["enabled"]
        assert plugins_cfg["entries"]["openai"]["allow_tool_override"] is True
        assert flat.is_dir()
        # The removed plugin loses both.
        assert "image_gen/openai" not in plugins_cfg["enabled"]
        assert "image_gen/openai" not in plugins_cfg["entries"]
        assert not namespaced.exists()

    def test_removing_flat_plugin_spares_namespaced_plugin_sharing_the_leaf(
        self, plugin_env
    ):
        home = plugin_env
        namespaced = _install(home, "image_gen/openai", "openai")
        flat = _install(home, "openai", "openai")
        _write_plugins_config(
            home,
            {
                "enabled": ["image_gen/openai", "openai"],
                "disabled": [],
                "entries": _grant("image_gen/openai", "openai"),
            },
        )

        cmd_remove("openai")

        plugins_cfg = _read_plugins_config(home)
        assert "image_gen/openai" in plugins_cfg["enabled"]
        assert plugins_cfg["entries"]["image_gen/openai"]["allow_tool_override"] is True
        assert namespaced.is_dir()
        assert "openai" not in plugins_cfg["enabled"]
        assert "openai" not in plugins_cfg["entries"]
        assert not flat.exists()


class TestOwnedStateIsCleaned:
    """The removed plugin's own identities are stripped from all three keys."""

    def test_flat_plugin_cleared_from_enabled_disabled_and_entries(self, plugin_env):
        home = plugin_env
        _install(home, "solo", "solo")
        _write_plugins_config(
            home,
            {
                "enabled": ["solo", "bystander"],
                "disabled": ["solo"],
                "entries": _grant("solo", "bystander"),
            },
        )

        cmd_remove("solo")

        plugins_cfg = _read_plugins_config(home)
        assert plugins_cfg["enabled"] == ["bystander"]
        assert plugins_cfg["disabled"] == []
        assert "solo" not in plugins_cfg["entries"]
        # An unrelated key we cannot prove anything about is left alone.
        assert plugins_cfg["entries"]["bystander"]["allow_tool_override"] is True

    def test_distinct_manifest_name_and_leaf_are_cleaned(self, plugin_env):
        home = plugin_env
        # Key is path-derived ("tools/relay_dir") while the manifest name is
        # "nemo_relay"; both forms, plus the bare leaf, can be in config.
        _install(home, "tools/relay_dir", "nemo_relay")
        _write_plugins_config(
            home,
            {
                "enabled": ["tools/relay_dir", "nemo_relay"],
                "disabled": ["relay_dir"],
                "entries": _grant("tools/relay_dir", "nemo_relay", "relay_dir"),
            },
        )

        cmd_remove("tools/relay_dir")

        plugins_cfg = _read_plugins_config(home)
        assert plugins_cfg["enabled"] == []
        assert plugins_cfg["disabled"] == []
        assert plugins_cfg["entries"] == {}

    def test_uninstalling_a_never_enabled_plugin_leaves_config_untouched(
        self, plugin_env
    ):
        home = plugin_env
        _install(home, "solo", "solo")
        _write_plugins_config(
            home, {"enabled": ["bystander"], "disabled": [], "entries": _grant("bystander")}
        )
        before = (home / "config.yaml").read_text(encoding="utf-8")

        cmd_remove("solo")

        assert (home / "config.yaml").read_text(encoding="utf-8") == before


class TestRemovalOrdering:
    """Config state is cleaned before the tree is deleted."""

    def test_config_write_failure_leaves_the_plugin_installed(
        self, plugin_env, monkeypatch
    ):
        home = plugin_env
        plugin_dir = _install(home, "solo", "solo")
        _write_plugins_config(
            home, {"enabled": ["solo"], "disabled": [], "entries": _grant("solo")}
        )

        def _boom(*_args, **_kwargs):
            raise OSError("config volume is read-only")

        monkeypatch.setattr("hermes_cli.config.save_config", _boom)

        with pytest.raises(OSError):
            cmd_remove("solo")

        # The tree survives, so the operator can retry rather than being left
        # with a deleted plugin still listed in plugins.enabled.
        assert plugin_dir.is_dir()


class TestDashboardRemoval:
    """The dashboard entry point cleans the same state as the CLI."""

    def test_dashboard_removal_clears_config_state(self, plugin_env):
        home = plugin_env
        plugin_dir = _install(home, "solo", "solo")
        _write_plugins_config(
            home,
            {
                "enabled": ["solo"],
                "disabled": [],
                "entries": _grant("solo"),
            },
        )

        result = dashboard_remove_user_plugin("solo")

        assert result["ok"] is True
        plugins_cfg = _read_plugins_config(home)
        assert plugins_cfg["enabled"] == []
        assert "solo" not in plugins_cfg["entries"]
        assert not plugin_dir.exists()

    def test_dashboard_removal_spares_flat_plugin_sharing_the_leaf(self, plugin_env):
        home = plugin_env
        _install(home, "image_gen/openai", "openai")
        flat = _install(home, "openai", "openai")
        _write_plugins_config(
            home,
            {
                "enabled": ["image_gen/openai", "openai"],
                "disabled": [],
                "entries": _grant("image_gen/openai", "openai"),
            },
        )

        result = dashboard_remove_user_plugin("image_gen/openai")

        assert result["ok"] is True
        plugins_cfg = _read_plugins_config(home)
        assert "openai" in plugins_cfg["enabled"]
        assert plugins_cfg["entries"]["openai"]["allow_tool_override"] is True
        assert flat.is_dir()


class TestEntryPointPlugins:
    """An entry-point plugin must not take part in directory matching.

    ``_discover_all_plugins`` stores the entry-point value ("pkg.mod:register")
    where directory-backed plugins store a ``Path``. Feeding that to
    ``Path.resolve()`` is meaningless everywhere and invalid path syntax on
    Windows, and a raise there would abandon the whole identity set and skip
    cleanup silently — reintroducing the bug this file guards.
    """

    @pytest.mark.parametrize(
        "ep_value",
        [
            "pkg.mod:register",
            # Stands in for any value Path() rejects outright; Windows reaches
            # the same branch with the ordinary colon-bearing value above.
            "pkg.mod:register\x00bad",
        ],
    )
    def test_entry_point_plugin_does_not_block_cleanup(
        self, plugin_env, monkeypatch, ep_value
    ):
        home = plugin_env
        _install(home, "solo", "solo")
        _write_plugins_config(
            home, {"enabled": ["solo"], "disabled": [], "entries": _grant("solo")}
        )
        monkeypatch.setattr(
            "hermes_cli.plugins_cmd._discover_entrypoint_plugins",
            lambda: [("ep_plugin", "1.0.0", "", ep_value)],
        )

        cmd_remove("solo")

        plugins_cfg = _read_plugins_config(home)
        assert plugins_cfg["enabled"] == []
        assert "solo" not in plugins_cfg["entries"]


class TestManagedMode:
    """Managed mode keeps main's behaviour: the tree goes, config is not written."""

    def test_managed_install_still_removes_the_tree(self, plugin_env):
        home = plugin_env
        plugin_dir = _install(home, "solo", "solo")
        _write_plugins_config(
            home, {"enabled": ["solo"], "disabled": [], "entries": _grant("solo")}
        )
        before = (home / "config.yaml").read_text(encoding="utf-8")
        # A managed install is laid out by the activation script, which creates
        # these before anything reads config; ensure_hermes_home() verifies them
        # rather than creating them under managed mode.
        for subdir in ("cron", "sessions", "logs", "memories"):
            (home / subdir).mkdir()
        (home / ".managed").write_text("", encoding="utf-8")

        cmd_remove("solo")

        # save_config prints the canonical managed error and returns without
        # writing, so the removal proceeds exactly as it does on main today.
        assert not plugin_dir.exists()
        assert (home / "config.yaml").read_text(encoding="utf-8") == before
