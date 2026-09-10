"""The Windows exe hand-off child must not re-run the full pre-update backup.

On Windows the ``hermes.exe`` shim cannot replace itself, so mid-update it hands
off to the venv interpreter, which re-runs ``hermes update`` and re-enters the
pre-update backup path. Without a session-level guard both legs write a full
~172 MB pre-update zip for one user-visible update (#107543). The child reuses
the parent's zip (taken before the code swap — the better rollback point), which
the parent publishes into the environment the hand-off inherits.
"""

import types

import hermes_cli.update_cmd_maint as m
from hermes_cli.main_install_repair import _UPDATE_REEXEC_ENV


class TestReuseHandoffPreUpdateBackup:
    def test_child_reuses_existing_parent_backup(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "backups" / "pre-update-2026-09-10-214152.zip"
        zip_path.parent.mkdir(parents=True)
        zip_path.write_bytes(b"PK\x03\x04")  # existence => complete (atomic rename)

        monkeypatch.setenv(_UPDATE_REEXEC_ENV, "1")
        monkeypatch.setenv(m._PRE_UPDATE_BACKUP_ENV, str(zip_path))

        assert m._reuse_handoff_pre_update_backup() == zip_path

    def test_non_handoff_process_never_reuses(self, tmp_path, monkeypatch):
        """The parent (not a re-exec child) always takes its own backup even if the
        path env var is somehow present — no accidental skip of the real backup."""
        zip_path = tmp_path / "pre-update.zip"
        zip_path.write_bytes(b"PK\x03\x04")
        monkeypatch.delenv(_UPDATE_REEXEC_ENV, raising=False)
        monkeypatch.setenv(m._PRE_UPDATE_BACKUP_ENV, str(zip_path))

        assert m._reuse_handoff_pre_update_backup() is None

    def test_child_without_published_path_takes_fresh_backup(self, monkeypatch):
        monkeypatch.setenv(_UPDATE_REEXEC_ENV, "1")
        monkeypatch.delenv(m._PRE_UPDATE_BACKUP_ENV, raising=False)

        assert m._reuse_handoff_pre_update_backup() is None

    def test_child_with_missing_parent_backup_takes_fresh_backup(self, tmp_path, monkeypatch):
        """Parent backup failed / was pruned: the child must fall through to a fresh
        full backup rather than silently leaving no rollback point."""
        monkeypatch.setenv(_UPDATE_REEXEC_ENV, "1")
        monkeypatch.setenv(m._PRE_UPDATE_BACKUP_ENV, str(tmp_path / "gone.zip"))

        assert m._reuse_handoff_pre_update_backup() is None


class TestRunPreUpdateBackupFullLeg:
    def _full_mode_args(self, monkeypatch):
        monkeypatch.setattr(m, "_resolve_pre_update_backup_mode", lambda _a: "full")
        monkeypatch.setattr(m, "_run_quick_snapshots", lambda: "snap-1")

    def test_handoff_child_skips_full_backup(self, tmp_path, monkeypatch):
        self._full_mode_args(monkeypatch)
        zip_path = tmp_path / "pre-update.zip"
        zip_path.write_bytes(b"PK\x03\x04")
        monkeypatch.setenv(_UPDATE_REEXEC_ENV, "1")
        monkeypatch.setenv(m._PRE_UPDATE_BACKUP_ENV, str(zip_path))

        calls = []
        monkeypatch.setattr(m, "_run_full_backup", lambda: calls.append(1))

        snap = m._run_pre_update_backup(types.SimpleNamespace())
        assert snap == "snap-1"          # quick snapshot still runs (self-prunes)
        assert calls == []               # but the full zip is NOT re-taken

    def test_parent_still_takes_full_backup(self, monkeypatch):
        self._full_mode_args(monkeypatch)
        monkeypatch.delenv(_UPDATE_REEXEC_ENV, raising=False)

        calls = []
        monkeypatch.setattr(m, "_run_full_backup", lambda: calls.append(1))

        m._run_pre_update_backup(types.SimpleNamespace())
        assert calls == [1]


class TestFullBackupPublishesPath:
    def test_successful_backup_publishes_path_env(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "backups" / "pre-update.zip"
        zip_path.parent.mkdir(parents=True)
        zip_path.write_bytes(b"PK\x03\x04")

        import hermes_cli.backup as backup
        monkeypatch.setattr(backup, "create_pre_update_backup", lambda **_k: zip_path)
        monkeypatch.setattr(m, "_load_updates_cfg", lambda: {"backup_keep": 5})
        monkeypatch.delenv(m._PRE_UPDATE_BACKUP_ENV, raising=False)

        m._run_full_backup()

        import os
        assert os.environ.get(m._PRE_UPDATE_BACKUP_ENV) == str(zip_path)

    def test_failed_backup_does_not_publish_path(self, monkeypatch):
        import hermes_cli.backup as backup
        monkeypatch.setattr(backup, "create_pre_update_backup", lambda **_k: None)
        monkeypatch.setattr(m, "_load_updates_cfg", lambda: {"backup_keep": 5})
        monkeypatch.delenv(m._PRE_UPDATE_BACKUP_ENV, raising=False)

        m._run_full_backup()

        import os
        assert m._PRE_UPDATE_BACKUP_ENV not in os.environ
