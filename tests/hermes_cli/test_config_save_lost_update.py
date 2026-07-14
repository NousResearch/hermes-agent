"""save_config lost-update protection: three-way merge-preserve + CAS refusal.

``save_config`` historically reserialized the caller's whole config dict over
``config.yaml``. Every caller mutates a snapshot from ``load_config()``; any
disk-side edit made after that snapshot (a human hand-edit, another process,
``hermes config set`` from a second terminal) was silently reverted on the
next save — the classic lost update.

These tests pin the fix:

* disk-side edits the caller never saw survive a save (two-writer race),
* the same key changed on both sides is a LOUD refusal
  (``ConfigWriteConflictError``), never silent loss,
* duplicate mapping keys in the on-disk YAML reject the write
  (``ConfigDuplicateKeyError``) instead of silently collapsing the dup,
* unknown keys inside DEFAULT_CONFIG-defined subtrees warn (never block),
  while top-level keys absent from DEFAULT_CONFIG stay silent —
  DEFAULT_CONFIG is not a complete schema,
* the pre-existing behaviors at the same seam are unchanged:
  ``${ENV}`` template preservation, schema-default stripping, and the
  legacy overwrite path when the on-disk YAML is unparseable (the
  last-known-good retention seam).
"""

import logging
import os
import time

import pytest
import yaml

from hermes_cli import config as cfg_mod
from hermes_cli.config import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
)
from unittest.mock import patch

ConfigWriteConflictError = getattr(cfg_mod, "ConfigWriteConflictError", None)
ConfigDuplicateKeyError = getattr(cfg_mod, "ConfigDuplicateKeyError", None)


@pytest.fixture(autouse=True)
def _reset_config_module_state():
    """Clear config.py module-level caches between tests in this file.

    Each test file runs in its own process, but tests within this file share
    module state; every test here uses a distinct tmp_path so path-keyed
    caches cannot collide — this clear is belt-and-braces plus the
    non-path-keyed warn sets.
    """
    for name in (
        "_RAW_CONFIG_CACHE",
        "_LOAD_CONFIG_CACHE",
        "_LAST_EXPANDED_CONFIG_BY_PATH",
        "_CONFIG_PARSE_WARNED",
        "_DISK_STATE_OBSERVATIONS",
        "_UNKNOWN_KEY_WARNED",
    ):
        state = getattr(cfg_mod, name, None)
        if state is not None:
            state.clear()
    yield


def _write_config(tmp_path, data):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    # (mtime_ns, size)-keyed caches need the mtime to move between writes.
    time.sleep(0.05)


def _read_disk(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def _v0(**extra):
    base = {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"default": "test/model-a"},
    }
    base.update(extra)
    return base


class TestLostUpdateMergePreserve:
    def test_hand_edit_survives_two_writer_race(self, tmp_path):
        """THE c002 repro: a disk-side key added after the caller's load must
        survive the caller's save. Unpatched save_config clobbers it."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()  # caller's (soon stale) snapshot

            # External writer: a human hand-edit adds a top-level key the
            # caller has never seen (the pins/session_reset/plugins shape).
            _write_config(tmp_path, _v0(custom_pins={"x": 1}))

            caller_cfg["memory"] = {"user_char_limit": 2200}
            save_config(caller_cfg)

            raw = _read_disk(tmp_path)
            assert raw.get("custom_pins") == {"x": 1}, (
                "lost update: the hand-edited key was clobbered by a stale "
                "whole-file reserialization"
            )
            assert raw["memory"]["user_char_limit"] == 2200
            assert raw["model"]["default"] == "test/model-a"

    def test_foreign_value_change_survives_when_caller_did_not_touch_it(
        self, tmp_path
    ):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0(memory={"user_char_limit": 1111}))
            caller_cfg = load_config()

            _write_config(tmp_path, _v0(memory={"user_char_limit": 2222}))

            caller_cfg["model"]["default"] = "test/model-b"
            save_config(caller_cfg)

            raw = _read_disk(tmp_path)
            assert raw["memory"]["user_char_limit"] == 2222, (
                "lost update: disk-side value change reverted by stale save"
            )
            assert raw["model"]["default"] == "test/model-b"

    def test_foreign_edit_survives_even_after_intervening_load(self, tmp_path):
        """A long-running process (gateway) reloads config constantly. A
        reload AFTER the foreign edit must not launder the stale snapshot's
        clobber — the disk-state history, not just the latest observation,
        is the merge base."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()

            _write_config(tmp_path, _v0(custom_pins={"x": 1}))
            load_config()  # unrelated in-process reload absorbs the edit

            caller_cfg["memory"] = {"user_char_limit": 2200}
            save_config(caller_cfg)

            raw = _read_disk(tmp_path)
            assert raw.get("custom_pins") == {"x": 1}
            assert raw["memory"]["user_char_limit"] == 2200

    def test_foreign_deletion_is_not_resurrected(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0(legacy_block={"a": 1}))
            caller_cfg = load_config()

            _write_config(tmp_path, _v0())  # external writer deleted the key

            caller_cfg["memory"] = {"user_char_limit": 2200}
            save_config(caller_cfg)

            raw = _read_disk(tmp_path)
            assert "legacy_block" not in raw, (
                "stale save resurrected a key deleted on disk"
            )
            assert raw["memory"]["user_char_limit"] == 2200

    def test_in_process_sequential_saves_stay_last_write_wins(self, tmp_path):
        """A process may overwrite its OWN earlier save with older values —
        the fallback_add pattern: the model picker saves a new primary, then
        the command restores the pre-picker primary from an older snapshot.
        Only FOREIGN disk transitions get lost-update protection."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            snapshot_before = load_config()

            picker_cfg = load_config()
            picker_cfg["model"] = {"default": "test/model-b"}
            save_config(picker_cfg)  # this process's own intermediate write
            time.sleep(0.05)

            snapshot_before["fallback_providers"] = [
                {"provider": "test", "model": "test/model-b"}
            ]
            save_config(snapshot_before)  # deliberately restores model-a

            raw = _read_disk(tmp_path)
            assert raw["model"]["default"] == "test/model-a", (
                "in-process restore of the process's own write was wrongly "
                "suppressed as a stale snapshot"
            )
            assert raw["fallback_providers"][0]["provider"] == "test"

    def test_own_merge_preserved_foreign_value_stays_protected(self, tmp_path):
        """A save that merge-PRESERVED a foreign edit must not launder that
        value into 'our own write': a later stale snapshot still cannot
        clobber it."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            stale_cfg = load_config()

            # Foreign hand-edit lands on disk.
            _write_config(tmp_path, _v0(custom_pins={"x": 1}))

            # Our process saves an unrelated change; the merge preserves the
            # foreign key — the write is self-authored but custom_pins is not.
            mid_cfg = load_config()
            mid_cfg["memory"] = {"user_char_limit": 2200}
            save_config(mid_cfg)
            assert _read_disk(tmp_path).get("custom_pins") == {"x": 1}
            time.sleep(0.05)

            # A stale snapshot from before the foreign edit saves again.
            stale_cfg["display"] = {"show_thinking": True}
            save_config(stale_cfg)

            raw = _read_disk(tmp_path)
            assert raw.get("custom_pins") == {"x": 1}, (
                "merge-preserved foreign value was laundered into a "
                "self-written state and clobbered by a stale snapshot"
            )

    def test_caller_deletion_on_quiet_disk_still_deletes(self, tmp_path):
        """Fresh-snapshot deletions keep working: load, delete, save."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0(legacy_block={"a": 1}))
            caller_cfg = load_config()
            del caller_cfg["legacy_block"]
            save_config(caller_cfg)

            raw = _read_disk(tmp_path)
            assert "legacy_block" not in raw


class TestCasConflictRefusal:
    def test_same_key_changed_both_sides_refuses_loudly(self, tmp_path):
        assert ConfigWriteConflictError is not None, (
            "patch not applied: hermes_cli.config.ConfigWriteConflictError "
            "does not exist"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()

            _write_config(tmp_path, _v0() | {"model": {"default": "test/model-b"}})

            caller_cfg["model"]["default"] = "test/model-c"
            with pytest.raises(ConfigWriteConflictError) as exc_info:
                save_config(caller_cfg)

            # Loud, names the conflicted path, and fail-closed: disk keeps
            # the concurrent writer's value, nothing was written.
            assert "model.default" in str(exc_info.value)
            assert _read_disk(tmp_path)["model"]["default"] == "test/model-b"

    def test_conflict_message_names_paths_not_values(self, tmp_path):
        """Config values can be secrets — the refusal names key paths only."""
        assert ConfigWriteConflictError is not None
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0(mcp_servers={"srv": {"api_key": "old-secret"}}))
            caller_cfg = load_config()

            _write_config(
                tmp_path, _v0(mcp_servers={"srv": {"api_key": "disk-secret"}})
            )

            caller_cfg["mcp_servers"]["srv"]["api_key"] = "caller-secret"
            with pytest.raises(ConfigWriteConflictError) as exc_info:
                save_config(caller_cfg)

            msg = str(exc_info.value)
            assert "mcp_servers.srv.api_key" in msg
            for leaked in ("old-secret", "disk-secret", "caller-secret"):
                assert leaked not in msg


class TestDuplicateKeyRejection:
    def test_duplicate_key_on_disk_rejects_write(self, tmp_path):
        assert ConfigDuplicateKeyError is not None, (
            "patch not applied: hermes_cli.config.ConfigDuplicateKeyError "
            "does not exist"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()

            dup_text = (
                "_config_version: {v}\n"
                "model:\n  default: test/model-a\n"
                "memory:\n  user_char_limit: 1111\n"
                "memory:\n  user_char_limit: 2222\n"
            ).format(v=DEFAULT_CONFIG["_config_version"])
            (tmp_path / "config.yaml").write_text(dup_text, encoding="utf-8")
            time.sleep(0.05)

            caller_cfg["model"]["default"] = "test/model-b"
            with pytest.raises(ConfigDuplicateKeyError, match="memory"):
                save_config(caller_cfg)

            # Fail-closed: the ambiguous file was not touched.
            assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == dup_text


class TestUnknownKeyWarn:
    def test_unknown_key_in_defined_subtree_warns_not_blocks(
        self, tmp_path, caplog
    ):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()
            caller_cfg.setdefault("display", {})["tool_progress_grouXping"] = True

            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                save_config(caller_cfg)

            assert any(
                "display.tool_progress_grouXping" in rec.message
                for rec in caplog.records
            ), "unknown key inside a DEFAULT_CONFIG subtree should WARN"
            # warn-only: the write went through with the key intact
            assert _read_disk(tmp_path)["display"]["tool_progress_grouXping"] is True

    def test_real_estate_garble_shape_warns(self, tmp_path, caplog):
        """The live defect this catches: an em-dash garble of a real key
        (display.tool_progress_gr―ouping) sitting next to nothing — it
        near-matches the known key and warns."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()
            caller_cfg.setdefault("display", {})["tool_progress_gr———ouping"] = True

            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                save_config(caller_cfg)

            assert any(
                "tool_progress_gr———ouping" in rec.message for rec in caplog.records
            )

    def test_benign_novel_key_in_defined_subtree_is_silent(
        self, tmp_path, caplog
    ):
        """DEFAULT_CONFIG under-enumerates even second-level keys (live
        profiles carry ~20 legit ones like agent.verbose). A novel key that
        does NOT resemble any known/sibling key must stay silent —
        otherwise the warning is noise nobody reads."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()
            caller_cfg.setdefault("agent", {})["verbose"] = True

            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                save_config(caller_cfg)

            assert not any("agent.verbose" in rec.message for rec in caplog.records)
            assert _read_disk(tmp_path)["agent"]["verbose"] is True

    def test_unknown_top_level_key_is_silent(self, tmp_path, caplog):
        """DEFAULT_CONFIG is NOT a complete schema — legit estate keys like
        session_reset/plugins/mcp_servers are absent from it. Top-level
        unknowns must stay silent."""
        assert "session_reset" not in DEFAULT_CONFIG
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()
            caller_cfg["session_reset"] = {"mode": "both"}

            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                save_config(caller_cfg)

            assert not any(
                "session_reset" in rec.message for rec in caplog.records
            )
            assert _read_disk(tmp_path)["session_reset"] == {"mode": "both"}


class TestExistingSeamBehaviorsUnchanged:
    def test_env_template_preserved_through_two_writer_merge(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_TEST_LOST_UPDATE_KEY", "expanded-secret-value")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(
                tmp_path,
                _v0(mcp_servers={"srv": {"api_key": "${HERMES_TEST_LOST_UPDATE_KEY}"}}),
            )
            caller_cfg = load_config()
            assert (
                caller_cfg["mcp_servers"]["srv"]["api_key"]
                == "expanded-secret-value"
            )

            # concurrent disk-side hand-edit, unrelated to the template
            _write_config(
                tmp_path,
                _v0(
                    mcp_servers={"srv": {"api_key": "${HERMES_TEST_LOST_UPDATE_KEY}"}},
                    custom_pins={"x": 1},
                ),
            )

            caller_cfg["memory"] = {"user_char_limit": 2200}
            save_config(caller_cfg)

            text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
            assert "${HERMES_TEST_LOST_UPDATE_KEY}" in text
            assert "expanded-secret-value" not in text
            raw = _read_disk(tmp_path)
            assert raw.get("custom_pins") == {"x": 1}

    def test_default_strip_unchanged(self, tmp_path):
        """save(load()) still writes only user-authored keys — schema
        defaults are not materialized by the merge."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(
                tmp_path,
                {
                    "_config_version": DEFAULT_CONFIG["_config_version"],
                    "memory": {"user_char_limit": 2200},
                },
            )
            save_config(load_config())
            raw = _read_disk(tmp_path)

            assert raw["memory"]["user_char_limit"] == 2200
            assert "agent" not in raw
            assert "gateway" not in raw

    def test_corrupt_disk_yaml_keeps_legacy_overwrite_path(self, tmp_path):
        """When the on-disk YAML is unparseable the disk state is UNKNOWN:
        no merge, no conflict — save falls back to the legacy overwrite
        (the corrupt content is .bak-snapshotted by the parse-failure
        warner at the last-known-good seam)."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _write_config(tmp_path, _v0())
            caller_cfg = load_config()

            (tmp_path / "config.yaml").write_text("\tbroken:\n", encoding="utf-8")
            time.sleep(0.05)

            caller_cfg["memory"] = {"user_char_limit": 2200}
            save_config(caller_cfg)  # must not raise

            raw = _read_disk(tmp_path)
            assert raw["memory"]["user_char_limit"] == 2200
            assert raw["model"]["default"] == "test/model-a"

    def test_absent_file_still_created(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config({"model": {"default": "test/model-a"}})
            raw = _read_disk(tmp_path)
            assert raw["model"]["default"] == "test/model-a"
