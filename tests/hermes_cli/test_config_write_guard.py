"""Tests for the config-write guard: a written config.yaml must be loadable.

Incident: a duplicate ``platform_toolsets.telegram:`` key in a profile's
``config.yaml`` made the file unparseable. Every worker for that profile died at
startup, the multiplexer skipped the profile, and the crash loop produced an
85-message DM storm. The guard proves a written config parses BEFORE it counts as
written, and restores the previous good file when it does not.

Two defect classes are covered:

* duplicate key (the incident itself), and
* a silent top-level key deletion (the precursor bug: a regex edit dropped the key
  and then re-added a second copy of the same block).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# The exact duplicate from the incident (profiles/ultron/config.yaml, 2026-10-28).
DUPLICATE_KEY_CONFIG = """\
platform_toolsets:
  telegram:
    - hermes-telegram
    - messaging
  telegram:
    - hermes-telegram
    - messaging
"""

GOOD_CONFIG = """\
model:
  default: claude-opus-4-7
platform_toolsets:
  telegram:
    - hermes-telegram
    - messaging
"""


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "config.yaml"


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.delenv("HERMES_CONFIG_WRITE_GUARD", raising=False)
    yield
    monkeypatch.delenv("HERMES_CONFIG_WRITE_GUARD", raising=False)


class TestDuplicateKeyDetection:
    """A duplicate key must be REFUSED, and the previous good file restored."""

    def test_duplicate_key_config_does_not_parse(self):
        """The pre-fix file is unparseable under ruamel (PyYAML would not notice)."""
        import hermes_yaml as yaml

        with pytest.raises(Exception) as exc:
            yaml.safe_load(DUPLICATE_KEY_CONFIG)
        assert "duplicate" in str(exc.value).lower()

    def test_top_level_keys_returns_none_for_duplicate(self):
        from hermes_cli.config_write_guard import top_level_keys

        assert top_level_keys(DUPLICATE_KEY_CONFIG) is None

    def test_top_level_keys_reads_a_good_config(self):
        from hermes_cli.config_write_guard import top_level_keys

        assert top_level_keys(GOOD_CONFIG) == {"model", "platform_toolsets"}

    def test_verify_rejects_duplicate_key_and_restores(self, config_path, guard_on):
        from hermes_cli.config_write_guard import (
            ConfigWriteRejected, snapshot_config, verify_written_config,
        )

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")
        snapshot = snapshot_config(config_path)

        # Simulate a botched write that landed a duplicate key on disk.
        config_path.write_text(DUPLICATE_KEY_CONFIG, encoding="utf-8")

        with pytest.raises(ConfigWriteRejected) as exc:
            verify_written_config(
                config_path, previous_bytes=snapshot[0], previous_keys=snapshot[1])

        assert "formatting error" in str(exc.value)
        # The previous good file is back on disk, byte for byte.
        assert config_path.read_text(encoding="utf-8") == GOOD_CONFIG

    def test_verify_accepts_a_good_write(self, config_path, guard_on):
        from hermes_cli.config_write_guard import (
            snapshot_config, verify_written_config,
        )

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")
        snapshot = snapshot_config(config_path)
        config_path.write_text(GOOD_CONFIG + "display:\n  skin: default\n", encoding="utf-8")

        verify_written_config(
            config_path, previous_bytes=snapshot[0], previous_keys=snapshot[1])

        assert "display" in config_path.read_text(encoding="utf-8")


class TestKeyDeletionDetection:
    """The precursor bug: a regex edit that silently dropped a top-level key."""

    def test_deleted_top_level_keys_finds_the_drop(self):
        from hermes_cli.config_write_guard import deleted_top_level_keys

        assert deleted_top_level_keys({"model", "gateway"}, {"model"}) == {"gateway"}

    def test_deleted_top_level_keys_ignores_additions(self):
        from hermes_cli.config_write_guard import deleted_top_level_keys

        assert deleted_top_level_keys({"model"}, {"model", "gateway"}) == set()

    def test_deleted_top_level_keys_tolerates_unreadable_key_set(self):
        """An unreadable key set is the duplicate-key case, not a deletion case."""
        from hermes_cli.config_write_guard import deleted_top_level_keys

        assert deleted_top_level_keys(None, {"model"}) == set()
        assert deleted_top_level_keys({"model"}, None) == set()

    def test_verify_rejects_a_dropped_key_and_restores(self, config_path, guard_on):
        from hermes_cli.config_write_guard import (
            ConfigWriteRejected, snapshot_config, verify_written_config,
        )

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")
        snapshot = snapshot_config(config_path)

        # The regex edit dropped ``platform_toolsets:`` entirely.
        config_path.write_text("model:\n  default: claude-opus-4-7\n", encoding="utf-8")

        with pytest.raises(ConfigWriteRejected) as exc:
            verify_written_config(
                config_path, previous_bytes=snapshot[0], previous_keys=snapshot[1])

        assert "platform_toolsets" in str(exc.value)
        assert config_path.read_text(encoding="utf-8") == GOOD_CONFIG

    def test_verify_allows_a_deliberate_deletion(self, config_path, guard_on):
        from hermes_cli.config_write_guard import (
            snapshot_config, verify_written_config,
        )

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")
        snapshot = snapshot_config(config_path)
        config_path.write_text("model:\n  default: claude-opus-4-7\n", encoding="utf-8")

        verify_written_config(
            config_path, previous_bytes=snapshot[0], previous_keys=snapshot[1],
            allow_key_deletion=True)


class TestWritePathIntegration:
    """The guard is wired into the ONE on-disk write primitive, so every writer gets it."""

    def test_atomic_config_write_refuses_to_persist_a_duplicate_key(self, config_path, guard_on, monkeypatch):
        """A write that would leave a duplicate key on disk is refused and rolled back."""
        from hermes_cli.config import atomic_config_replace

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")

        # Force the writer to emit a duplicate key: the merged document ends up with the
        # same top-level key twice.
        import utils

        original = utils._roundtrip_dump

        def _dup_dump(path, yaml_rt, config, *, extra_content=None):
            original(path, yaml_rt, config, extra_content=extra_content)
            path.write_text(DUPLICATE_KEY_CONFIG, encoding="utf-8")

        monkeypatch.setattr(utils, "_roundtrip_dump", _dup_dump)

        with pytest.raises(Exception) as exc:
            atomic_config_replace(config_path, {"model": {"default": "x"}})

        assert "duplicate" in str(exc.value).lower() or "formatting error" in str(exc.value)
        # Rolled back to the previous good file.
        assert config_path.read_text(encoding="utf-8") == GOOD_CONFIG

    def test_atomic_config_write_still_writes_a_good_config(self, config_path, guard_on):
        from hermes_cli.config import atomic_config_write

        config_path.write_text(GOOD_CONFIG, encoding="utf-8")
        atomic_config_write(config_path, {
            "model": {"default": "claude-opus-4-7"},
            "platform_toolsets": {"telegram": ["hermes-telegram", "messaging"]},
        })

        import hermes_yaml as yaml

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert parsed["model"]["default"] == "claude-opus-4-7"

    def test_guard_can_be_disabled_for_a_hand_repair(self, config_path, monkeypatch):
        monkeypatch.setenv("HERMES_CONFIG_WRITE_GUARD", "0")
        from hermes_cli.config_write_guard import guard_enabled

        assert guard_enabled() is False


class TestProfileIsolation:
    """A broken profile config must not take a sibling profile down with it."""

    def test_broken_secondary_profile_is_skipped_alone(self, tmp_path, monkeypatch):
        """Ultron's broken config must not break the DEFAULT profile's plugin union."""
        import pm.plugins_state as plugins_state

        default_home = tmp_path / "default"
        broken_home = tmp_path / "profiles" / "ultron"
        good_home = tmp_path / "profiles" / "friday"
        for home in (default_home, broken_home, good_home):
            home.mkdir(parents=True)
        (default_home / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - hermes-telegram\n", encoding="utf-8")
        # The incident's duplicate key.
        (broken_home / "config.yaml").write_text(DUPLICATE_KEY_CONFIG, encoding="utf-8")
        (good_home / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - messaging\n", encoding="utf-8")

        monkeypatch.setattr(plugins_state, "dependency_homes",
                            lambda: [default_home, broken_home, good_home])

        with pytest.raises(ValueError):
            # The default home's own breakage still raises (fail closed).
            plugins_state.enabled_plugins_ordered()

        out = plugins_state.enabled_plugins_ordered(skip_invalid_secondary=True)
        assert default_home / "plugins" in out
        assert good_home / "plugins" in out
        # The broken profile contributed NOTHING — and did not break its siblings.
        assert broken_home / "plugins" not in out
