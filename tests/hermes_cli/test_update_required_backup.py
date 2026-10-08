"""Unattended updates require complete, recorded archives before mutation."""

from argparse import Namespace
from contextlib import closing
from pathlib import Path
import sqlite3
import zipfile

import pytest

from hermes_cli import backup, update_receipt, update_required_backup
from hermes_cli.update_cmd_maint import _run_pre_update_backup


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    named = root / "profiles" / "work"
    for home in (root, named):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("updates:\n  pre_update_backup: off\n", encoding="utf-8")
    (root / "profiles" / "ghost").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(named))
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        yield root, named


def _args(**kwargs):
    return Namespace(require_backup=True, no_backup=False, backup=False, **kwargs)


def test_required_backups_cover_root_and_live_profiles_with_real_wal(homes, tmp_path):
    root, named = homes
    with closing(sqlite3.connect(named / "state.db")) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE messages (body TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO messages VALUES ('only in WAL')")
        writer.commit()
        snapshot = _run_pre_update_backup(_args())
        assert snapshot, "the canonical quick snapshot still feeds recovery"
        run = update_receipt.read_run_record(update_receipt.current_correlation_id())[1]
        assert set(run["required_backups"]) == {str(root), str(named)}
        assert any(step["name"] == "required_backup" and step["ok"] for step in run["steps"])
        assert not (root / "profiles" / "ghost" / "backups").exists()
        for home, archive in run["required_backups"].items():
            assert Path(archive).parent == Path(home) / "backups"
            with zipfile.ZipFile(archive) as zipped:
                assert zipped.read("config.yaml")
                if home == str(named):
                    restored = tmp_path / "restored.db"
                    restored.write_bytes(zipped.read("state.db"))
                    with closing(sqlite3.connect(restored)) as reader:
                        assert reader.execute("SELECT body FROM messages").fetchall() == [("only in WAL",)]


@pytest.mark.parametrize("failure", ["missing", "error", "incomplete"])
def test_incomplete_required_archive_stops_before_quick_snapshots(homes, monkeypatch, failure):
    root, _ = homes

    def failed_backup(**kwargs):
        if failure == "error":
            raise OSError("device full")
        if failure == "missing":
            return root / "backups" / "missing.zip"
        return None

    monkeypatch.setattr(backup, "create_pre_update_backup", failed_backup)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11
    assert not (root / "state-snapshots").exists()
    run = update_receipt.read_run_record(update_receipt.current_correlation_id())[1]
    assert run["stages"][-1]["name"] == "required_backup"
    assert run["stages"][-1]["outcome"] == "failed"


def test_profile_enumeration_failure_is_not_an_empty_roster(homes, monkeypatch):
    root, _ = homes
    real_iterdir = Path.iterdir

    def denied(path):
        if path == root / "profiles":
            raise PermissionError("profile roster denied")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11
    assert not (root / "backups").exists()


def test_strict_archive_rejects_unreadable_subtree_default_stays_best_effort(homes, monkeypatch):
    root, _ = homes
    protected = root / "protected"
    protected.mkdir()
    (protected / "important.txt").write_text("preserve me", encoding="utf-8")
    real_scandir = backup.os.scandir

    def denied(path):
        if Path(path) == protected:
            raise PermissionError("unreadable subtree")
        return real_scandir(path)

    monkeypatch.setattr(backup.os, "scandir", denied)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11
    # The opt-in gate must not change the existing interactive update policy.
    assert backup.create_pre_update_backup(hermes_home=root) is not None


def test_required_archive_refuses_partial_file_write(homes, monkeypatch):
    real_write = backup._write_zip_file

    def denied(zipped, path, arcname):
        if Path(path).name == "config.yaml":
            raise OSError("cannot read config")
        return real_write(zipped, path, arcname)

    monkeypatch.setattr(backup, "_write_zip_file", denied)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11


def test_profile_created_during_backups_requires_retry(homes, monkeypatch):
    root, _ = homes
    real_backup = backup.create_pre_update_backup

    def create_profile(**kwargs):
        archived = real_backup(**kwargs)
        added = root / "profiles" / "newcomer"
        added.mkdir(exist_ok=True)
        (added / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        return archived

    monkeypatch.setattr(backup, "create_pre_update_backup", create_profile)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11


def test_required_backup_must_be_durable_in_own_receipt(homes, monkeypatch):
    from hermes_cli import runtime_state

    original = runtime_state._atomic_bytes

    def denied(path, content):
        if "update_receipts" in Path(path).parts:
            raise OSError("receipt store full")
        return original(path, content)

    monkeypatch.setattr(runtime_state, "_atomic_bytes", denied)
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(_args())
    assert exc.value.code == 11


def test_no_backup_conflict_refuses_before_writing(homes):
    root, _ = homes
    with pytest.raises(SystemExit) as exc:
        _run_pre_update_backup(Namespace(require_backup=True, no_backup=True, backup=False))
    assert exc.value.code == 11
    assert not (root / "backups").exists()


def test_normal_update_does_not_call_strict_gate(homes, monkeypatch):
    def unexpected(_args):
        pytest.fail("ordinary update invoked the required-backup gate")

    monkeypatch.setattr(update_required_backup, "require_pre_update_backups", unexpected)
    assert _run_pre_update_backup(Namespace(no_backup=False, backup=False)) is None
