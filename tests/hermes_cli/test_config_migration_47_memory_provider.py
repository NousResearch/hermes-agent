"""Migration 46→47: the legacy ``memory.provider: false`` literal becomes ``''``.

Older config writers emitted ``false`` to mean "unset". The runtime reader treats falsy as
no-provider, but the PM validator (``read_home_selection``) rejects the literal, so an upgrade
leaves the host unable to run ``hermes pm repair`` — the one tool that could heal the config.
The step rewrites only the legacy literal; an explicit provider name is never touched.
"""

import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml


class TestLegacyFalseProviderMigration:
    """Behaviour contract for the 46→47 step driven through ``run_migrations``."""

    @staticmethod
    def _run_ladder(tmp_path, current_ver=46):
        from hermes_cli.config_migrations import run_migrations

        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            run_migrations(current_ver, results, quiet=True)
        return results

    @staticmethod
    def _write_config(tmp_path, config):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    @staticmethod
    def _read_config(tmp_path):
        return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

    def test_legacy_false_becomes_empty_string(self, tmp_path):
        self._write_config(
            tmp_path,
            {"_config_version": 46, "memory": {"memory_enabled": True, "provider": False}},
        )

        results = self._run_ladder(tmp_path)

        raw = self._read_config(tmp_path)
        assert raw["memory"]["provider"] == ""
        assert raw["memory"]["memory_enabled"] is True
        added = [e for e in results["config_added"] if "memory.provider" in e]
        assert len(added) == 1, results["config_added"]

    def test_explicit_provider_name_is_preserved(self, tmp_path):
        self._write_config(
            tmp_path,
            {"_config_version": 46, "memory": {"provider": "mnemosyne"}},
        )

        self._run_ladder(tmp_path)

        raw = self._read_config(tmp_path)
        assert raw["memory"]["provider"] == "mnemosyne"

    def test_unset_key_is_left_alone(self, tmp_path):
        self._write_config(tmp_path, {"_config_version": 46, "memory": {"memory_enabled": True}})

        self._run_ladder(tmp_path)

        raw = self._read_config(tmp_path)
        assert "provider" not in raw["memory"]

    def test_falsey_zero_and_other_non_strings_are_not_migrated(self, tmp_path):
        """Only the boolean literal the old writers emitted is rewritten — a numeric or string
        value the user actually set is preserved (the PM validator still rejects non-strings,
        but a migration must not silently rewrite them)."""
        self._write_config(
            tmp_path,
            {"_config_version": 46, "memory": {"provider": 0}},
        )

        self._run_ladder(tmp_path)

        raw = self._read_config(tmp_path)
        assert raw["memory"]["provider"] == 0

    def test_unversioned_config_still_gets_the_legacy_repair(self, tmp_path):
        """An unstamped config carries the legacy literal too; the step is in LEGACY_KEY_STEPS."""
        self._write_config(
            tmp_path,
            {"memory": {"memory_enabled": True, "provider": False}},
        )

        from hermes_cli.config_migrations import run_migrations

        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            run_migrations(0, results, quiet=True, unversioned=True)

        raw = self._read_config(tmp_path)
        assert raw["memory"]["provider"] == ""

    def test_default_version_is_47(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["_config_version"] == 47
