"""Migration 46→47: saved toolset lists lose the removed ``messaging``/``moa`` toolsets.

``hermes tools`` persists explicit per-platform toolset lists, and those lists
outlive the registry: `messaging` was removed with its agent-callable send_message
tool (#47856) and `moa` when MoA presets became selectable virtual models
(#46081). Unknown names contribute no tools but make every startup print
``Warning: Unknown toolsets: messaging, moa``. The 46→47 step prunes exactly
those names — never "anything the registry rejects", because plugin toolsets and
MCP server names are valid on a saved list but undiscoverable at migration time.
"""

import os
from unittest.mock import patch

import yaml


class TestRemovedToolsetsMigration:
    """Behaviour contract for ``_migrate_to_47`` driven through ``run_migrations``."""

    @staticmethod
    def _run_ladder(tmp_path, current_ver=46):
        from hermes_cli.config_migrations import run_migrations

        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            run_migrations(current_ver, results, quiet=True)
        return results

    @staticmethod
    def _write_config(tmp_path, config):
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )

    @staticmethod
    def _read_config(tmp_path):
        return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

    def test_stale_names_pruned_from_every_platform_list(self, tmp_path):
        self._write_config(
            tmp_path,
            {
                "_config_version": 46,
                "platform_toolsets": {
                    "cli": ["file", "messaging", "moa", "terminal", "web"],
                    "telegram": ["moa", "file"],
                },
            },
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"]["cli"] == ["file", "terminal", "web"]
        assert raw["platform_toolsets"]["telegram"] == ["file"]
        added = [e for e in results["config_added"] if "pruned" in e.lower()]
        assert len(added) == 1, results["config_added"]
        assert "messaging" in added[0] and "moa" in added[0]

    def test_top_level_toolsets_list_pruned(self, tmp_path):
        self._write_config(
            tmp_path,
            {"_config_version": 46, "toolsets": ["messaging", "hermes-cli"]},
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["toolsets"] == ["hermes-cli"]
        assert results["config_added"], results

    def test_valid_names_and_composites_untouched(self, tmp_path):
        """`hermes-messaging`-style composites share a prefix with a removed name —
        exact-match only; lists without the removed names rewrite nothing."""
        platform_toolsets = {
            "cli": ["hermes-cli"],
            "telegram": ["hermes-telegram", "file"],
        }
        self._write_config(
            tmp_path,
            {"_config_version": 46, "platform_toolsets": platform_toolsets},
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"] == platform_toolsets
        assert results["config_added"] == []

    def test_emptied_list_stays_explicitly_empty(self, tmp_path):
        """An explicit list that only held removed names stays an explicit empty list:
        the platform reader treats explicit-empty as authoritative, same as before."""
        self._write_config(
            tmp_path,
            {"_config_version": 46, "platform_toolsets": {"cli": ["messaging", "moa"]}},
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"]["cli"] == []
        assert results["config_added"], results

    def test_plugin_and_mcp_names_never_pruned(self, tmp_path):
        """Names that merely aren't built-ins (plugin toolsets, MCP servers) survive —
        validity-at-migration-time is not the prune predicate."""
        platform_toolsets = {"cli": ["messaging", "spotify", "my-mcp-server"]}
        self._write_config(
            tmp_path,
            {"_config_version": 46, "platform_toolsets": platform_toolsets},
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"]["cli"] == ["spotify", "my-mcp-server"]

    def test_rerun_is_a_noop(self, tmp_path):
        self._write_config(
            tmp_path,
            {"_config_version": 46, "platform_toolsets": {"cli": ["messaging", "file"]}},
        )

        self._run_ladder(tmp_path)
        after_first = self._read_config(tmp_path)
        second = self._run_ladder(tmp_path)
        after_second = self._read_config(tmp_path)

        assert second["config_added"] == []
        assert after_second == after_first

    def test_non_list_shapes_untouched(self, tmp_path):
        """A str-encoded list or null value is the legacy shape the runtime parser
        owns; the migration only rewrites real YAML lists."""
        self._write_config(
            tmp_path,
            {
                "_config_version": 46,
                "platform_toolsets": {"cli": '["messaging", "file"]', "web": None},
            },
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"]["cli"] == '["messaging", "file"]'
        assert raw["platform_toolsets"]["web"] is None
        assert results["config_added"] == []

    def test_full_migration_stamps_the_current_version(self, tmp_path):
        """A pre-47 config that takes this step ends at DEFAULT_CONFIG's version."""
        from hermes_cli.config import migrate_config
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        self._write_config(
            tmp_path,
            {"_config_version": 45, "platform_toolsets": {"cli": ["messaging", "file"]}},
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"]["cli"] == ["file"]
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
