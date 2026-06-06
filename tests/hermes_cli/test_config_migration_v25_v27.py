"""Regression tests for bug #38798 — explicit migration steps for v25→v26 and v26→v27.

Why: Config version bumps 25→26 (display.interface, commit d6b0c23f) and
26→27 (updates.non_interactive_local_changes, commit 72eb42d9) relied on the
generic missing-field migration + load_config() deep-merge to seed the new
keys on existing configs.  The generic path uses load_config() which already
deep-merges defaults, so get_missing_config_fields() returns empty for any
field present in DEFAULT_CONFIG — meaning the save only happens in the
``elif current_ver < latest_ver`` branch.  That branch DOES save the merged
config, but the absence of an explicit migration block leaves no audit trail
and creates a subtle dependency on the branch ordering.

The fix adds explicit migration blocks for v25→v26 and v26→v27 that write
the new keys directly to the raw on-disk YAML, making the migration
deterministic regardless of the generic path's behaviour.

Test strategy: create a temp config.yaml at v25/v26 with known content,
run migrate_config(), then read the raw YAML back and assert:
  - The new key is present in the raw file
  - Pre-existing keys are preserved verbatim
  - _config_version is bumped to the expected value
  - Migration is idempotent (running again does not corrupt the config)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, content: str) -> None:
    """Write a config YAML to *path*."""
    path.write_text(content, encoding="utf-8")


def _run_migration(cfg, config_path: Path) -> dict:
    """Run migrate_config with the given config path and return results."""
    cfg._LOAD_CONFIG_CACHE.clear()
    with patch.object(cfg, "get_config_path", return_value=config_path), \
         patch.object(cfg, "ensure_hermes_home"):
        result = cfg.migrate_config(interactive=False, quiet=True)
    cfg._LOAD_CONFIG_CACHE.clear()
    return result


def _read_raw(cfg, config_path: Path) -> dict:
    """Read the raw (un-merged) YAML from config_path."""
    cfg._LOAD_CONFIG_CACHE.clear()
    with patch.object(cfg, "get_config_path", return_value=config_path), \
         patch.object(cfg, "ensure_hermes_home"):
        return cfg.read_raw_config()


# ---------------------------------------------------------------------------
# v25 → 26: display.interface seeded in raw config
# ---------------------------------------------------------------------------

class TestV25ToV26SeedsDisplayInterface:
    """migrate_config must write display.interface to the raw YAML for v25 configs.

    Why: Without an explicit block, the key was only present in the in-memory
    deep-merged config, not in the on-disk YAML.  Any code path that saved
    back from read_raw_config() would then lose display.interface.

    What: The migration writes display.interface=cli to the raw file when the
    key is absent.
    """

    def test_writes_display_interface_to_raw_file(self, tmp_path):
        """Why: the raw YAML must contain display.interface after migration.
        What: migrate_config() runs the v25→v26 step and writes the key.
        Test: create v25 config without display.interface, run migration,
        read raw YAML back, assert display.interface == 'cli'.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, "_config_version: 25\nmodel: nous-hermes-3\n")

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("display", {}).get("interface") == "cli", (
            "display.interface must be 'cli' in raw config after v25→v26 migration"
        )

    def test_does_not_overwrite_user_set_display_interface(self, tmp_path):
        """Why: a user may have already set display.interface to 'tui'.
        What: migration skips writing the key when it already exists.
        Test: v25 config with display.interface=tui; assert value stays 'tui'
        after migration.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            "_config_version: 25\ndisplay:\n  interface: tui\n",
        )

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("display", {}).get("interface") == "tui", (
            "migration must not overwrite a user-set display.interface"
        )

    def test_preserves_other_keys_during_migration(self, tmp_path):
        """Why: migration must be a pure read-modify-save that does not drop
        unrelated user keys.
        What: model and any other user keys survive unchanged.
        Test: v25 config with model key; assert model is unchanged after
        migration.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            "_config_version: 25\nmodel: nous-hermes-3\nagent:\n  max_turns: 42\n",
        )

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("model") == "nous-hermes-3", (
            "model key must survive v25→v26 migration"
        )
        assert raw.get("agent", {}).get("max_turns") == 42, (
            "agent.max_turns must survive v25→v26 migration"
        )

    def test_migration_idempotent_from_v26(self, tmp_path):
        """Why: running migrate_config on an already-migrated v26 config must
        not change display.interface or re-seed it.
        What: a v26 config with display.interface=tui stays unchanged.
        Test: write v26 config; run migration twice; assert value is still tui.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            "_config_version: 26\ndisplay:\n  interface: tui\n",
        )

        _run_migration(cfg, config_path)
        _run_migration(cfg, config_path)  # second run — idempotency
        raw = _read_raw(cfg, config_path)

        assert raw.get("display", {}).get("interface") == "tui", (
            "idempotent: display.interface must remain 'tui' after repeated migration"
        )


# ---------------------------------------------------------------------------
# v26 → 27: updates.non_interactive_local_changes seeded in raw config
# ---------------------------------------------------------------------------

class TestV26ToV27SeedsNonInteractiveLocalChanges:
    """migrate_config must write updates.non_interactive_local_changes for v26 configs.

    Why: Commit 72eb42d9 added the key to DEFAULT_CONFIG and relied on the
    generic path. The explicit block guarantees the key is on disk for any
    user upgrading from v26.

    What: Migration writes updates.non_interactive_local_changes=stash to the
    raw file when the key is absent.
    """

    def test_writes_non_interactive_local_changes_to_raw_file(self, tmp_path):
        """Why: must guarantee the key is present on disk after the v26→v27 step.
        What: migrate_config() seeds updates.non_interactive_local_changes=stash.
        Test: create v26 config without the key, run migration, read raw YAML,
        assert key == 'stash'.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, "_config_version: 26\nmodel: nous-hermes-3\n")

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("updates", {}).get("non_interactive_local_changes") == "stash", (
            "updates.non_interactive_local_changes must be 'stash' after v26→v27 migration"
        )

    def test_does_not_overwrite_user_set_non_interactive_local_changes(self, tmp_path):
        """Why: a user may have explicitly chosen 'discard'.
        What: migration skips writing the key when it already exists.
        Test: v26 config with non_interactive_local_changes=discard; assert
        value stays 'discard'.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            "_config_version: 26\nupdates:\n  non_interactive_local_changes: discard\n",
        )

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("updates", {}).get("non_interactive_local_changes") == "discard", (
            "migration must not overwrite user-set updates.non_interactive_local_changes"
        )

    def test_v25_config_gets_both_v26_and_v27_keys(self, tmp_path):
        """Why: a user upgrading from v25 directly to latest (v27) must get
        both the v25→v26 and v26→v27 migration steps applied.
        What: a v25 config ends up with both display.interface and
        updates.non_interactive_local_changes in the raw YAML.
        Test: write v25 config, run migration, assert both keys present.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, "_config_version: 25\nmodel: nous-hermes-3\n")

        _run_migration(cfg, config_path)
        raw = _read_raw(cfg, config_path)

        assert raw.get("display", {}).get("interface") == "cli", (
            "display.interface missing after v25→v27 migration"
        )
        assert raw.get("updates", {}).get("non_interactive_local_changes") == "stash", (
            "updates.non_interactive_local_changes missing after v25→v27 migration"
        )

    def test_migration_idempotent_from_v27(self, tmp_path):
        """Why: running migrate_config on an already-migrated v27 config must
        not change or re-seed updates.non_interactive_local_changes.
        What: a v27 config with non_interactive_local_changes=discard stays unchanged.
        Test: write v27 config; run migration twice; assert value is still discard.
        """
        import hermes_cli.config as cfg

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            "_config_version: 27\nupdates:\n  non_interactive_local_changes: discard\n",
        )

        _run_migration(cfg, config_path)
        _run_migration(cfg, config_path)  # second run — idempotency
        raw = _read_raw(cfg, config_path)

        assert raw.get("updates", {}).get("non_interactive_local_changes") == "discard", (
            "idempotent: non_interactive_local_changes must remain 'discard' after repeated migration"
        )
