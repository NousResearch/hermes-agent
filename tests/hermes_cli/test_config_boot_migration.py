"""Tests for raw-config version detection and the Docker-boot migration pass.

Covers the #35406 fix: Docker image updates must run config migration with the
same schema-safety guarantees as non-Docker ``hermes update``. The two moving
parts are:

  * ``check_config_version`` reading the *raw* on-disk ``_config_version``
    (not the deep-merged effective config), so an unversioned/legacy file is
    correctly treated as "never migrated" instead of inheriting the latest
    version from ``DEFAULT_CONFIG``.
  * ``run_boot_config_migration`` running migration non-interactively, honoring
    the ``HERMES_SKIP_CONFIG_MIGRATION`` opt-out and backing up first.
"""

import os
from unittest.mock import patch

import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    check_config_version,
    get_config_path,
    read_raw_config,
    run_boot_config_migration,
)

LATEST = DEFAULT_CONFIG["_config_version"]


def _write_config(data: dict) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class TestCheckConfigVersion:
    def test_unversioned_raw_config_reports_version_zero(self):
        # A raw file with no _config_version must NOT inherit the latest
        # version from DEFAULT_CONFIG via load_config()'s deep-merge (#35406).
        _write_config({"model": {"default": "anthropic/claude-opus-4.6"}})
        current, latest = check_config_version()
        assert current == 0
        assert latest == LATEST

    def test_versioned_raw_config_reports_its_version(self):
        _write_config({"_config_version": 5, "model": {"default": "x"}})
        current, latest = check_config_version()
        assert current == 5
        assert latest == LATEST

    def test_missing_config_file_reports_version_zero(self):
        # No config.yaml on disk → unversioned.
        path = get_config_path()
        if path.exists():
            path.unlink()
        current, _ = check_config_version()
        assert current == 0

    def test_non_int_version_treated_as_unversioned(self):
        _write_config({"_config_version": "garbage", "model": {"default": "x"}})
        current, _ = check_config_version()
        assert current == 0


class TestRunBootConfigMigration:
    def test_optout_env_var_skips_migration(self, monkeypatch):
        _write_config({"model": {"default": "x"}})
        monkeypatch.setenv("HERMES_SKIP_CONFIG_MIGRATION", "1")
        with patch("hermes_cli.config.migrate_config") as mig:
            outcome = run_boot_config_migration(quiet=True)
        assert outcome["status"] == "skipped_optout"
        mig.assert_not_called()

    def test_no_config_file_is_noop(self):
        path = get_config_path()
        if path.exists():
            path.unlink()
        with patch("hermes_cli.config.migrate_config") as mig:
            outcome = run_boot_config_migration(quiet=True)
        assert outcome["status"] == "no_config"
        mig.assert_not_called()

    def test_current_config_is_up_to_date(self):
        # Seed a config already stamped at the latest version with no missing
        # required fields → migration must not run.
        _write_config({"_config_version": LATEST, "model": {"default": "x"}})
        with (
            patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
            patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
            patch("hermes_cli.config.migrate_config") as mig,
        ):
            outcome = run_boot_config_migration(quiet=True)
        assert outcome["status"] == "up_to_date"
        mig.assert_not_called()

    def test_unversioned_config_is_migrated_and_backed_up(self):
        _write_config({"model": {"default": "x"}})
        with patch("hermes_cli.backup.create_pre_migration_backup") as backup:
            backup.return_value = None
            outcome = run_boot_config_migration(quiet=True)
        # Migration ran (status ok) and stamped the on-disk version forward.
        assert outcome["status"] == "ok"
        backup.assert_called_once()
        assert read_raw_config().get("_config_version") == LATEST
