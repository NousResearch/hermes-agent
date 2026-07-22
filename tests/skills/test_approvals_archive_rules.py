"""TDD tests for migrate_approvals_config archive condition bug.

Bug: `if len(approvals) > 1` only fires when the top-level approvals dict has 2+
keys.  The typical OpenClaw layout is `{"exec": {mode, rules, requireReason,
timeout}}` which has len == 1, so the archive is never created and
exec.rules / requireReason / timeout are silently dropped.

Fix: change the condition to `if approvals and self.archive_dir` so that any
non-empty approvals block gets archived (the mode is still migrated separately
to config.yaml).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "optional-skills" / "migration"
        / "openclaw-migration" / "scripts"),
)
import openclaw_to_hermes as mod


def _make_migrator(source: Path, target: Path) -> mod.Migrator:
    return mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=target / "migration-report",
    )


# ---------------------------------------------------------------------------
# RED tests (demonstrate the bug as it existed before the fix)
# ---------------------------------------------------------------------------


class TestApprovalsArchiveRulesBug:
    """Confirm that exec.rules / requireReason / timeout survive migration."""

    def test_exec_rules_archived_when_only_exec_key(self, tmp_path: Path) -> None:
        """Typical config: approvals = {exec: {mode, rules, requireReason, timeout}}.

        len(approvals) == 1  -> old buggy condition skipped the archive.
        After fix: archive MUST be created with exec.rules intact.
        """
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        config = {
            "approvals": {
                "exec": {
                    "mode": "auto",
                    "rules": [{"pattern": "rm -rf", "action": "deny"}],
                    "requireReason": True,
                    "timeout": 30,
                }
            }
        }
        (source / "openclaw.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (target / "config.yaml").write_text("", encoding="utf-8")

        migrator = _make_migrator(source, target)
        migrator.migrate_approvals_config()

        archive_file = migrator.archive_dir / "approvals-config.json"
        # After fix: archive must exist
        assert archive_file.exists(), (
            "approvals-config.json archive was not created for config with "
            "approvals.exec.rules - exec.rules would be silently dropped"
        )

        archived = json.loads(archive_file.read_text(encoding="utf-8"))
        exec_block = archived.get("exec", {})

        assert exec_block.get("rules") == [{"pattern": "rm -rf", "action": "deny"}], (
            "exec.rules missing from archive"
        )
        assert exec_block.get("requireReason") is True, (
            "exec.requireReason missing from archive"
        )
        assert exec_block.get("timeout") == 30, (
            "exec.timeout missing from archive"
        )

    def test_mode_still_migrated_to_config_yaml(self, tmp_path: Path) -> None:
        """Mode migration to config.yaml must still work after the archive fix."""
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        config = {
            "approvals": {
                "exec": {
                    "mode": "auto",
                    "rules": [{"pattern": "sudo *", "action": "require-reason"}],
                    "timeout": 60,
                }
            }
        }
        (source / "openclaw.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (target / "config.yaml").write_text("", encoding="utf-8")

        migrator = _make_migrator(source, target)
        migrator.migrate_approvals_config()

        config_yaml = (target / "config.yaml").read_text(encoding="utf-8")
        # auto -> off in Hermes
        assert "off" in config_yaml, (
            f"Expected mode 'off' in config.yaml but got: {config_yaml!r}"
        )

    def test_archive_created_with_only_mode_no_rules(self, tmp_path: Path) -> None:
        """Even exec with only mode (no rules) should be archived if non-empty."""
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        config = {"approvals": {"exec": {"mode": "smart"}}}
        (source / "openclaw.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (target / "config.yaml").write_text("", encoding="utf-8")

        migrator = _make_migrator(source, target)
        migrator.migrate_approvals_config()

        # With fix: any non-empty approvals block should be archived
        archive_file = migrator.archive_dir / "approvals-config.json"
        assert archive_file.exists(), (
            "Archive not created for non-empty approvals even without rules"
        )

    def test_multi_key_approvals_still_archived(self, tmp_path: Path) -> None:
        """When approvals has 2 top-level keys (exec + shell), archive must exist.

        This was the ONLY case the old buggy condition handled correctly.
        Ensure the fix does not break it.
        """
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        config = {
            "approvals": {
                "exec": {"mode": "always", "rules": []},
                "shell": {"mode": "manual"},
            }
        }
        (source / "openclaw.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (target / "config.yaml").write_text("", encoding="utf-8")

        migrator = _make_migrator(source, target)
        migrator.migrate_approvals_config()

        archive_file = migrator.archive_dir / "approvals-config.json"
        assert archive_file.exists(), (
            "Archive not created for multi-key approvals (regression in fix)"
        )

    def test_empty_approvals_not_archived(self, tmp_path: Path) -> None:
        """Empty/absent approvals block must still be skipped (no false archive)."""
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        config: dict = {}  # no approvals key at all
        (source / "openclaw.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (target / "config.yaml").write_text("", encoding="utf-8")

        migrator = _make_migrator(source, target)
        migrator.migrate_approvals_config()

        archive_file = migrator.archive_dir / "approvals-config.json"
        assert not archive_file.exists(), (
            "Archive should not be created for empty approvals"
        )
