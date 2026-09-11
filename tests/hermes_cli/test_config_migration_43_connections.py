"""Migration 42→43: saved ``platform_toolsets`` lists gain the ``connections`` toolset.

``hermes tools`` persists an explicit per-platform toolset list, and absence from
that list reads as "unchecked" — so a toolset that ships after the list was saved
stays off for picker users while composite (``[hermes-cli]``) users inherit it.
The 42→43 step turns ``connections`` on for stale lists, preserves an explicit
decline, and leaves composites/empty lists alone.
"""

import os
from unittest.mock import patch

import pytest
import yaml


class TestConnectionsToolsetMigration:
    """Behaviour contract for ``_migrate_to_43`` driven through ``run_migrations``."""

    @staticmethod
    def _run_ladder(tmp_path, current_ver=42):
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

    def test_stale_list_gains_connections_for_every_platform(self, tmp_path):
        """A list saved before the toolset shipped is offered it (and records the offer)."""
        platforms = {
            "cli": ["file", "terminal", "web"],
            "telegram": ["file", "web"],
        }
        self._write_config(
            tmp_path,
            {
                "_config_version": 42,
                "platform_toolsets": platforms,
                "known_builtin_toolsets": {
                    "cli": ["browser", "file", "memory", "skills", "terminal", "todo", "web"]
                },
            },
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert "connections" in raw["platform_toolsets"]["cli"]
        assert "connections" in raw["platform_toolsets"]["telegram"]
        # The offer is recorded so a later uncheck is a recorded decline, not another migration.
        assert "connections" in raw["known_builtin_toolsets"]["cli"]
        added = [entry for entry in results["config_added"] if "connections" in entry.lower()]
        assert len(added) == 1, results["config_added"]

    @pytest.mark.parametrize(
        "platform_toolsets, known",
        [
            pytest.param(  # (a) the user saw the checkbox and left it off
                {"cli": ["file", "terminal", "web"]},
                {"cli": ["browser", "connections", "file", "web"]},
                id="declined",
            ),
            pytest.param(  # (b) composite list already inherits every core tool
                {"cli": ["hermes-cli"]},
                {},
                id="composite",
            ),
            pytest.param(  # (c) empty picker selection — no configurable key to extend
                {"cli": []},
                {},
                id="empty",
            ),
        ],
    )
    def test_declines_composites_and_empty_lists_are_untouched(
        self, tmp_path, platform_toolsets, known
    ):
        self._write_config(
            tmp_path,
            {
                "_config_version": 42,
                "platform_toolsets": platform_toolsets,
                "known_builtin_toolsets": known,
            },
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"] == platform_toolsets
        assert results["config_added"] == []

    @pytest.mark.parametrize("disabled", [["browser", "connections", "web"], '["connections"]'], ids=["list", "json-string"])
    def test_global_disable_is_not_overridden_or_claimed(self, tmp_path, disabled):
        """Blank Slate and `hermes tools --disable` write agent.disabled_toolsets, which the resolver
        subtracts last; appending to the platform list would print an enable that never takes effect."""
        platform_toolsets = {"cli": ["file", "skills", "terminal", "vision"]}
        self._write_config(
            tmp_path,
            {
                "_config_version": 42,
                "platform_toolsets": platform_toolsets,
                "agent": {"disabled_toolsets": disabled},
            },
        )

        results = self._run_ladder(tmp_path)
        raw = self._read_config(tmp_path)

        assert raw["platform_toolsets"] == platform_toolsets
        assert raw["agent"]["disabled_toolsets"] == disabled
        assert results["config_added"] == []

    def test_rerun_adds_nothing_and_keeps_one_connections(self, tmp_path):
        """Running the step twice is a no-op; the toolset appears exactly once."""
        self._write_config(
            tmp_path,
            {
                "_config_version": 42,
                "platform_toolsets": {"cli": ["file", "terminal", "web"]},
                "known_builtin_toolsets": {"cli": ["file", "terminal", "web"]},
            },
        )

        self._run_ladder(tmp_path)
        after_first = self._read_config(tmp_path)["platform_toolsets"]["cli"]
        second = self._run_ladder(tmp_path)
        after_second = self._read_config(tmp_path)["platform_toolsets"]["cli"]

        # Running the step again rewrites nothing and never appends a duplicate.
        assert second["config_added"] == []
        assert after_second == after_first
        assert after_second.count("connections") <= 1
