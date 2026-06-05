"""Regression tests for bug #38798 — config migration v25→v26 platform_toolsets.

Why: The v25→v26 config version bump (display.interface addition) had no
dedicated migration step.  It relied on runtime deep-merge to seed the new key,
which meant the value was never written to the raw YAML file.  Additionally,
a related transient commit window caused platform_toolsets to be corrupted
during the migration (hermes-cli → hermes, other platforms stripped).

The fix adds an explicit v25→v26 migration step that:
  1. Writes display.interface to the raw config so it appears on disk.
  2. Validates platform_toolsets and warns on unrecognized toolset names.

Test strategy: create a temp config.yaml at v25 with known platform_toolsets,
run migrate_config(), then read the raw YAML back and assert:
  - _config_version is 26
  - display.interface was written to the raw file
  - platform_toolsets is exactly preserved (no corruption, no stripping)
  - corrupted toolset names trigger a warning
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_v25_config(path: Path, extra: str = "") -> None:
    """Write a minimal v25 config.yaml to *path*."""
    path.write_text(
        "".join([
            "_config_version: 25\n",
            "model: nous-hermes-3\n",
            "platform_toolsets:\n",
            "  cli:\n",
            "  - hermes-cli\n",
            "  telegram:\n",
            "  - hermes-telegram\n",
            "  discord:\n",
            "  - hermes-discord\n",
            "  slack:\n",
            "  - hermes-slack\n",
            extra,
        ])
    )


# ---------------------------------------------------------------------------
# v25→v26 migration: display.interface seeded in raw config
# ---------------------------------------------------------------------------

class TestV25ToV26MigrationSeedsDisplayInterface:
    """The migration must write display.interface to the raw YAML file.

    Why: Before the fix, load_config() deep-merged the key at runtime but
    never persisted it, leaving the raw file without the key.  Any code that
    saved back from raw_config() would then lose the key on the next load.
    """

    def test_migration_writes_display_interface_to_raw_file(self, tmp_path):
        """Why: the raw YAML must contain display.interface after migration.
        What: migrate_config() runs the v25→v26 step and writes the key.
        Test: create v25 config, run migration, read raw YAML back, assert
        display.interface is present with the default value 'cli'.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_v25_config(config_path)
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        assert raw.get("display", {}).get("interface") == "cli", (
            "display.interface should be 'cli' in raw config after migration"
        )

    def test_migration_does_not_overwrite_existing_display_interface(self, tmp_path):
        """Why: users may have set display.interface to 'tui'; migration must
        not clobber that.
        What: migration skips writing display.interface when it already exists.
        Test: write config with display.interface: tui at v25, run migration,
        assert value stays 'tui'.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_v25_config(
            config_path,
            extra="display:\n  interface: tui\n",
        )
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        assert raw.get("display", {}).get("interface") == "tui", (
            "migration must not overwrite a user-set display.interface"
        )

    def test_migration_bumps_config_version_to_26(self, tmp_path):
        """Why: the version bump is the signal for subsequent runs to skip the step.
        What: _config_version is 26 in the raw file after migration.
        Test: standard v25 config; assert version 26 after migrate_config().
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_v25_config(config_path)
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        assert raw.get("_config_version") == 26


# ---------------------------------------------------------------------------
# v25→v26 migration: platform_toolsets preserved verbatim
# ---------------------------------------------------------------------------

class TestV25ToV26MigrationPreservesPlatformToolsets:
    """Migration must not corrupt or strip platform_toolsets entries.

    Why: A transient commit window caused hermes-cli to become hermes and all
    platforms except cli to be stripped.  The migration must be a pure read-
    modify-save that does not touch platform_toolsets content.
    """

    def test_all_platform_toolsets_preserved_after_migration(self, tmp_path):
        """Why: the corruption dropped telegram, discord, slack completely.
        What: all four test platforms survive the migration unchanged.
        Test: standard v25 config with 4 platforms; assert all 4 present and
        their toolset lists unchanged after migration.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_v25_config(config_path)
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        pt = raw.get("platform_toolsets", {})
        assert pt.get("cli") == ["hermes-cli"], (
            f"cli platform_toolsets corrupted: {pt.get('cli')!r}"
        )
        assert pt.get("telegram") == ["hermes-telegram"], (
            f"telegram platform_toolsets stripped or corrupted: {pt.get('telegram')!r}"
        )
        assert pt.get("discord") == ["hermes-discord"], (
            f"discord platform_toolsets stripped or corrupted: {pt.get('discord')!r}"
        )
        assert pt.get("slack") == ["hermes-slack"], (
            f"slack platform_toolsets stripped or corrupted: {pt.get('slack')!r}"
        )

    def test_hermes_cli_toolset_name_not_truncated(self, tmp_path):
        """Why: the specific corruption changed 'hermes-cli' to 'hermes'.
        What: migration must preserve the full toolset name including the
        -cli suffix.
        Test: single platform cli with hermes-cli; assert it is not 'hermes'
        after migration.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "_config_version: 25\nplatform_toolsets:\n  cli:\n  - hermes-cli\n"
        )
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        cli_toolsets = raw.get("platform_toolsets", {}).get("cli", [])
        assert "hermes-cli" in cli_toolsets, (
            f"hermes-cli was lost from platform_toolsets.cli: {cli_toolsets!r}"
        )
        assert "hermes" not in cli_toolsets or "hermes-cli" in cli_toolsets, (
            f"hermes-cli was truncated to 'hermes': {cli_toolsets!r}"
        )

    def test_already_at_v26_no_migration_runs(self, tmp_path):
        """Why: idempotency — running migration again on a v26 config must not
        corrupt platform_toolsets.
        What: migrate_config() on a v26 config is a no-op for platform_toolsets.
        Test: write v26 config with valid toolsets; run migration; assert unchanged.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "_config_version: 26\n"
            "display:\n  interface: cli\n"
            "platform_toolsets:\n"
            "  cli:\n  - hermes-cli\n"
            "  telegram:\n  - hermes-telegram\n"
        )
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            cfg.migrate_config(interactive=False, quiet=True)
            cfg._LOAD_CONFIG_CACHE.clear()
            raw = cfg.read_raw_config()

        assert raw.get("platform_toolsets", {}).get("cli") == ["hermes-cli"]
        assert raw.get("platform_toolsets", {}).get("telegram") == ["hermes-telegram"]


# ---------------------------------------------------------------------------
# v25→v26 migration: warning on unrecognized toolset names
# ---------------------------------------------------------------------------

class TestV25ToV26MigrationWarnsOnBadToolsets:
    """Migration must warn when platform_toolsets contains unrecognized names.

    Why: The corruption was silent — no error, no warning. Tools stopped
    working with no diagnostic output.  This test ensures the validation
    path added to the migration emits a warning for unrecognized names.
    """

    def test_warns_on_corrupted_hermes_toolset_name(self, tmp_path):
        """Why: the corruption changed 'hermes-cli' to 'hermes'.  A warning
        should have made this immediately visible.
        What: when platform_toolsets contains 'hermes' (no suffix), migrate
        emits a warning and includes it in results['warnings'].
        Test: write v25 config with cli: [hermes], run migration, assert
        warning in results and/or log output.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "_config_version: 25\n"
            "platform_toolsets:\n"
            "  cli:\n"
            "  - hermes\n"  # corrupted name — should be hermes-cli
        )
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            results = cfg.migrate_config(interactive=False, quiet=False)

        # 'hermes' is not in TOOLSETS and doesn't start with 'hermes-'
        has_warning = any("hermes" in w for w in results.get("warnings", []))
        assert has_warning, (
            f"Expected warning about 'hermes' toolset name in results, got: "
            f"{results.get('warnings')!r}"
        )

    def test_valid_toolset_names_produce_no_warning(self, tmp_path):
        """Why: valid names like hermes-cli, hermes-telegram should not warn.
        What: migration with a clean v25 config has no warnings in results.
        Test: write standard v25 config; assert results['warnings'] is empty
        after migration.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_v25_config(config_path)
        cfg._LOAD_CONFIG_CACHE.clear()

        with patch.object(cfg, "get_config_path", return_value=config_path), \
             patch.object(cfg, "ensure_hermes_home"):
            results = cfg.migrate_config(interactive=False, quiet=True)

        # Warnings list should contain no toolset-related entries
        toolset_warnings = [
            w for w in results.get("warnings", [])
            if "platform_toolsets" in w or "toolset" in w.lower()
        ]
        assert not toolset_warnings, (
            f"Unexpected toolset warnings for valid config: {toolset_warnings!r}"
        )
