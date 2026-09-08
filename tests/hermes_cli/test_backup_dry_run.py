"""Full-backup audits are complete, policy-identical and read-only in preview mode."""

import hashlib
import json
import os
import socket
import sqlite3
import stat
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import backup


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "home"
    root.mkdir()
    (root / "config.yaml").write_text("model: test\n")
    (root / ".env").write_text("TEST_SECRET=must-not-appear-in-audit\n")
    skill = root / "skills" / "custom"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Custom\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


def _inventory(root):
    """Ignore access times: reading metadata/content may update them on some mounts."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
            info = path.lstat()
            content = None
            if stat.S_ISREG(info.st_mode):
                content = hashlib.sha256(path.read_bytes()).hexdigest()
            elif stat.S_ISLNK(info.st_mode):
                content = os.readlink(path)
            result[str(path.relative_to(root))] = (
                info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, content)
    return result


def _audit(path):
    return json.loads(path.read_text())


def test_cli_dry_run_preserves_home_wal_and_output_directory(home, tmp_path):
    """Exercise the actual CLI startup, not just the backup function's write guards."""
    conn = sqlite3.connect(home / "state.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE evidence (value TEXT)")
    conn.execute("INSERT INTO evidence VALUES ('committed in WAL')")
    conn.commit()
    report = tmp_path / "audit.json"
    report.write_text("replace this report")
    report.chmod(0o644)
    output = tmp_path / "absent" / "nested" / "archive.zip"
    before = _inventory(home)
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               PYTHONDONTWRITEBYTECODE="1")
    try:
        run = subprocess.run(
            [sys.executable, "-B", "-m", "hermes_cli.main", "backup", "--dry-run",
             "--report", str(report), "-o", str(output)],
            cwd=Path(__file__).resolve().parents[2], env=env,
            capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        assert _inventory(home) == before
        assert not output.parent.parent.exists()
        data = _audit(report)
        assert data["mode"] == "dry_run"
        assert data["schema_version"] == 1
        assert data["root"] == str(home.resolve())
        entries = {entry["path"]: entry for entry in data["entries"]}
        assert entries["state.db"]["action"] == "included"
        assert entries["state.db-wal"]["action"] == "excluded"
        assert entries["state.db-shm"]["action"] == "excluded"
        assert "must-not-appear-in-audit" not in report.read_text()
        if os.name == "posix":
            assert stat.S_IMODE(report.stat().st_mode) == 0o600
    finally:
        conn.close()


@pytest.mark.linux_only
def test_preview_and_archive_share_selection_with_links_special_files_and_old_timestamps(
    home, tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("external secret")
    (home / "linked-directory").symlink_to(outside, target_is_directory=True)
    (home / "linked-file").symlink_to(outside / "secret")
    (home / "dangling").symlink_to(outside / "missing")
    # Even links whose names match exclusions must be accounted for.
    (home / "models").symlink_to(outside, target_is_directory=True)
    (home / "cache.pyc").symlink_to(outside / "secret")
    os.mkfifo(home / "runtime.pipe")
    monkeypatch.chdir(home)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind("runtime.sock")
    (home / "node_modules").mkdir()
    (home / "node_modules" / "never-enumerated").write_text("dependency")
    (home / "skills" / "custom" / "models").mkdir()
    ancient = home / "skills" / "custom" / "models" / "ancient.txt"
    ancient.write_text("old user data")
    os.utime(ancient, (1, 1))
    relocated = tmp_path / "relocated-home"
    relocated.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(relocated))
    output = tmp_path / "backup.zip"
    preview = tmp_path / "preview.json"
    actual = tmp_path / "actual.json"
    try:
        assert backup.run_backup(Namespace(output=str(output), dry_run=True, report=str(preview)))
        assert not output.exists()
        assert backup.run_backup(Namespace(output=str(output), report=str(actual)))
    finally:
        sock.close()
    planned = {entry["path"]: entry for entry in _audit(preview)["entries"]}
    written = {entry["path"]: entry for entry in _audit(actual)["entries"]}
    selected = {path for path, entry in planned.items() if entry["action"] == "included"}
    assert selected == {path for path, entry in written.items() if entry["action"] == "included"}
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == selected
        assert archive.read("skills/custom/models/ancient.txt") == b"old user data"
        assert archive.getinfo("skills/custom/models/ancient.txt").date_time == (1980, 1, 1, 0, 0, 0)
    for name in ("linked-directory", "linked-file", "dangling", "models", "cache.pyc"):
        assert planned[name]["action"] == "excluded"
        assert planned[name]["reason"] == "symlink"
        assert planned[name]["target"] == os.readlink(home / name)
        assert written[name] == planned[name]
    for name in ("runtime.pipe", "runtime.sock"):
        assert planned[name]["reason"] == "non_regular_file"
        assert planned[name]["action"] == "excluded"
    assert planned["node_modules"]["reason"] == "excluded_directory"
    assert not any(path.startswith("node_modules/") for path in planned)
    assert not any(path.startswith("linked-directory/") for path in planned)


def test_report_keeps_every_failure_and_cli_exits_nonzero_with_salvage(home, tmp_path, monkeypatch):
    from hermes_cli.main import cmd_backup

    failed_names = {f"unreadable-{number}.txt" for number in range(17)}
    for name in failed_names:
        (home / name).write_text("not readable during archive write")
    real_write = zipfile.ZipFile.write

    def fail_read(self, filename, *args, **kwargs):
        if Path(filename).name in failed_names:
            raise PermissionError("injected read denial")
        return real_write(self, filename, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_read)
    output = tmp_path / "salvage.zip"
    report = tmp_path / "errors.json"
    with pytest.raises(SystemExit) as error:
        cmd_backup(Namespace(quick=False, output=str(output), report=str(report)))
    assert error.value.code == 1
    data = _audit(report)
    assert data["mode"] == "backup"
    errors = {entry["path"]: entry for entry in data["entries"] if entry["action"] == "error"}
    assert set(errors) == failed_names
    assert all(entry["source"] == str(home / name) for name, entry in errors.items())
    assert all("injected read denial" in entry["reason"] for entry in errors.values())
    with zipfile.ZipFile(output) as archive:
        assert archive.read("config.yaml") == (home / "config.yaml").read_bytes()
        assert failed_names.isdisjoint(archive.namelist())
        assert {entry["path"] for entry in data["entries"] if entry["action"] == "included"} == set(archive.namelist())


def test_metadata_and_directory_read_errors_are_not_silent(home, tmp_path, monkeypatch):
    (home / "gone.txt").write_text("vanishes during scan")
    blocked = home / "unreadable-tree"
    blocked.mkdir()
    (blocked / "hidden.txt").write_text("not enumerable")
    real_lstat, real_scandir = Path.lstat, os.scandir

    def fail_stat(path, *args, **kwargs):
        if path.name == "gone.txt":
            raise FileNotFoundError("removed during scan")
        return real_lstat(path, *args, **kwargs)

    def fail_scandir(path):
        if Path(path) == blocked:
            raise PermissionError("cannot enumerate subtree")
        return real_scandir(path)

    monkeypatch.setattr(Path, "lstat", fail_stat)
    monkeypatch.setattr(os, "scandir", fail_scandir)
    report = tmp_path / "errors.json"
    output = tmp_path / "absent" / "backup.zip"
    assert backup.run_backup(Namespace(output=str(output), dry_run=True, report=str(report))) is False
    errors = {entry["path"] for entry in _audit(report)["entries"] if entry["action"] == "error"}
    assert errors == {"gone.txt", "unreadable-tree"}
    assert not output.parent.exists()
    assert not (home / ".backup.lock").exists()


def test_cli_dry_run_rejects_quick_without_writes(home, tmp_path):
    before = _inventory(home)
    report = tmp_path / "must-not-exist.json"
    run = subprocess.run(
        [sys.executable, "-B", "-m", "hermes_cli.main", "backup", "--dry-run", "--quick",
         "--report", str(report)], cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, HERMES_HOME=str(home), HOME=str(tmp_path), USERPROFILE=str(tmp_path),
                 PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=60)
    assert run.returncode == 2, run.stdout + run.stderr
    assert _inventory(home) == before
    assert not report.exists()


@pytest.mark.linux_only
def test_external_provider_paths_use_same_safe_audit(home, tmp_path, monkeypatch):
    external = tmp_path / "provider"
    external.mkdir()
    (external / "config.json").write_text('{"peer":"example"}')
    (external / "dangling").symlink_to(external / "missing")
    (external / "linked-dir").symlink_to(home, target_is_directory=True)
    os.mkfifo(external / "control.pipe")
    # The supplied HOME is the provider's parent; use a declared path above it.
    monkeypatch.setattr(backup, "_collect_memory_provider_external_paths",
                        lambda: [external, tmp_path.parent, external / "dangling"])
    output, report = tmp_path / "external.zip", tmp_path / "external.json"
    assert backup.run_backup(Namespace(output=str(output), dry_run=True, report=str(report)))
    preview = _audit(report)
    assert backup.run_backup(Namespace(output=str(output), report=str(report)))
    written = _audit(report)
    assert {entry["path"] for entry in preview["entries"] if entry["action"] == "included"} == {
        entry["path"] for entry in written["entries"] if entry["action"] == "included"}
    entries = {entry["source"]: entry for entry in preview["entries"]}
    assert entries[str(tmp_path.parent)]["reason"] == "outside_home"
    assert entries[str(external / "dangling")]["reason"] == "symlink"
    assert entries[str(external / "linked-dir")]["reason"] == "symlink"
    assert entries[str(external / "control.pipe")]["reason"] == "non_regular_file"
    with zipfile.ZipFile(output) as archive:
        external_names = {name for name in archive.namelist() if name.startswith("_external/")}
        assert external_names == {"_external/provider/config.json"}


@pytest.mark.linux_only
def test_report_refuses_symlink_and_archive_destination_before_source_writes(home, tmp_path):
    target = tmp_path / "private.json"
    target.write_text("preserve existing data")
    link = tmp_path / "report-link.json"
    link.symlink_to(target)
    output = tmp_path / "new" / "archive.zip"
    before = _inventory(home)
    with pytest.raises(SystemExit) as error:
        backup.run_backup(Namespace(output=str(output), dry_run=True, report=str(link)))
    assert error.value.code == 1
    assert target.read_text() == "preserve existing data"
    with pytest.raises(SystemExit) as error:
        backup.run_backup(Namespace(output=str(output), report=str(output)))
    assert error.value.code == 1
    assert _inventory(home) == before
    assert not output.parent.exists()
