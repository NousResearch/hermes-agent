import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backup-retention.py"


def _module():
    spec = importlib.util.spec_from_file_location("backup_retention", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_archive(root: Path, day: int, *, bad_checksum: bool = False) -> Path:
    stem = f"kensei-202607{day:02d}-0000"
    archive = root / f"{stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in (("config.yaml", b"cron: {}\n"), ("kanban.db", b"sqlite")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    with tarfile.open(archive, "r:gz") as tar:
        file_count = len(tar.getnames())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "sha256": "0" * 64 if bad_checksum else checksum,
        "file_count": file_count,
    }
    (root / f"{stem}.manifest.json").write_text(json.dumps(manifest))
    return archive


def test_retention_fails_closed_when_keep_set_has_bad_checksum(tmp_path, capsys):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for day in range(1, 17):
        _make_archive(backup_root, day, bad_checksum=(day == 16))

    setattr(module, "BACKUP_ROOT", backup_root)
    setattr(module, "LEDGER_PATH", tmp_path / "retention-ledger.jsonl")

    result = module.main()

    assert result == 2
    assert len(list(backup_root.glob("kensei-*.tar.gz"))) == 16
    assert "ALERT: retention set not fully verified" in capsys.readouterr().out


def test_verification_returns_error_when_archive_disappears(tmp_path):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    archive = _make_archive(backup_root, 1)
    archive.unlink()

    error = module._verify_archive(archive, backup_root)

    assert error.startswith("Cannot verify archive:")


def test_retention_stops_cleanly_if_delete_candidate_disappears(tmp_path, capsys):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for day in range(1, 17):
        _make_archive(backup_root, day)
    disappearing = backup_root / "kensei-20260702-0000.tar.gz"
    original_verify = module._verify_archive

    def verify_and_remove(archive, root):
        result = original_verify(archive, root)
        if archive.name == "kensei-20260703-0000.tar.gz":
            disappearing.unlink()
        return result

    setattr(module, "_verify_archive", verify_and_remove)
    result = module.main(["--backup-root", str(backup_root), "--ledger", str(tmp_path / "ledger")])

    assert result == 1
    assert (backup_root / "kensei-20260701-0000.tar.gz").exists()
    assert "ALERT: Failed to plan removal" in capsys.readouterr().out


def test_dry_run_keeps_archives_and_records_timestamped_plan(tmp_path, capsys):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for day in range(1, 17):
        _make_archive(backup_root, day)
    ledger_path = tmp_path / "retention-ledger.jsonl"

    result = module.main(
        ["--backup-root", str(backup_root), "--ledger", str(ledger_path), "--dry-run"]
    )

    assert result == 0
    assert len(list(backup_root.glob("kensei-*.tar.gz"))) == 16
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(entries) == 2
    assert all(entry["dry_run"] is True for entry in entries)
    assert all(entry["timestamp"] for entry in entries)
    assert capsys.readouterr().out == "DRY RUN: would remove 2, retain 14\n"


def test_verified_retention_deletes_only_archives_beyond_floor(tmp_path, capsys):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for day in range(1, 17):
        _make_archive(backup_root, day)
    ledger_path = tmp_path / "retention-ledger.jsonl"

    result = module.main(["--backup-root", str(backup_root), "--ledger", str(ledger_path)])

    assert result == 0
    retained = sorted(path.name for path in backup_root.glob("kensei-*.tar.gz"))
    assert retained == [f"kensei-202607{day:02d}-0000.tar.gz" for day in range(3, 17)]
    assert not (backup_root / "kensei-20260701-0000.manifest.json").exists()
    assert not (backup_root / "kensei-20260702-0000.manifest.json").exists()
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [entry["archive"] for entry in entries] == [
        "kensei-20260702-0000.tar.gz",
        "kensei-20260701-0000.tar.gz",
    ]
    assert all(entry["manifest_deleted"] is True for entry in entries)
    assert capsys.readouterr().out == "Retention: removed 2, retained 14\n"


def test_noop_under_retention_floor_is_silent(tmp_path, capsys):
    module = _module()
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    for day in range(1, 7):
        _make_archive(backup_root, day)

    result = module.main(["--backup-root", str(backup_root), "--ledger", str(tmp_path / "ledger")])

    assert result == 0
    assert len(list(backup_root.glob("kensei-*.tar.gz"))) == 6
    assert capsys.readouterr().out == ""
