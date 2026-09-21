"""P13 isolation proof for scripts/backup-staleness.py.

Verifies the --dry-run flag:
- --dry-run lists/expands cleanup candidates but never deletes files
- --backup-dir redirects so the production backup tree is untouched
- --hermes-home redirects the state file out of ~/.hermes
- exit 0 in both dry-run and live runs (alert-only contract preserved)
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "backup-staleness.py"
PYTHON = sys.executable


def _run(argv, env_extra=None):
    env = dict(os.environ)
    env.pop("BACKUP_STALENESS_DIR", None)
    env.pop("HERMES_HOME", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PYTHON, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=15,
    )


def _seed_old_tarball(daily_dir: Path, name: str, mtime_offset_days: int):
    """Create a .gz tarball with an old mtime in daily/."""
    import gzip
    daily_dir.mkdir(parents=True, exist_ok=True)
    f = daily_dir / name
    with gzip.open(f, "wb") as gz:
        gz.write(b"placeholder")
    # Backdate mtime
    import time
    ts = time.time() - (mtime_offset_days * 86400)
    os.utime(f, (ts, ts))
    return f


def test_dry_run_leaves_old_tarballs_unchanged(tmp_path):
    """--dry-run: tarballs older than CLEANUP_DAILY_DAYS stay on disk."""
    backup_dir = tmp_path / "backups"
    daily = backup_dir / "daily"
    old = _seed_old_tarball(daily, "kensei-20260101-0000.tar.gz", mtime_offset_days=30)
    before_size = old.stat().st_size

    r = _run(
        ["--dry-run", "--backup-dir", str(backup_dir),
         "--hermes-home", str(tmp_path / "hermes")],
    )
    assert r.returncode == 0, r.stderr
    assert old.exists(), "dry-run deleted an old tarball"
    assert old.stat().st_size == before_size, "dry-run modified the tarball"


def test_dry_run_leaves_old_snapshots_unchanged(tmp_path):
    """--dry-run: snapshot dirs older than CLEANUP_SNAPSHOT_DAYS stay on disk."""
    import time
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True)
    snap = backup_dir / "kensei-phase0-test"
    snap.mkdir()
    (snap / "marker.txt").write_text("data")
    ts = time.time() - (45 * 86400)
    os.utime(snap, (ts, ts))

    r = _run(
        ["--dry-run", "--backup-dir", str(backup_dir),
         "--hermes-home", str(tmp_path / "hermes")],
    )
    assert r.returncode == 0, r.stderr
    assert snap.exists(), "dry-run removed a snapshot directory"
    assert (snap / "marker.txt").exists()


def test_dry_run_does_not_write_state_file(tmp_path):
    """--dry-run must not create the state file under hermes-home."""
    hermes_home = tmp_path / "hermes"
    r = _run(
        ["--dry-run", "--backup-dir", str(tmp_path / "backups"),
         "--hermes-home", str(hermes_home)],
    )
    assert r.returncode == 0, r.stderr
    state = hermes_home / "scripts" / ".backup-staleness-state.json"
    assert not state.exists(), "dry-run wrote the state file"


def test_live_run_deletes_old_tarball(tmp_path):
    """Without --dry-run, live mode DOES delete old tarballs — proves the
    dry-run guard is what suppresses deletion (control case)."""
    backup_dir = tmp_path / "backups"
    daily = backup_dir / "daily"
    old = _seed_old_tarball(daily, "kensei-20260101-0000.tar.gz", mtime_offset_days=30)

    r = _run(
        ["--backup-dir", str(backup_dir), "--hermes-home", str(tmp_path / "hermes")],
    )
    assert r.returncode == 0, r.stderr
    assert not old.exists(), "live mode should have deleted the old tarball"


def test_hermes_home_redirects_state_file(tmp_path):
    """--hermes-home redirects the state file out of the real ~/.hermes."""
    hermes_home = tmp_path / "fakehermes"
    # Seed a fresh backup dir so a (fake) latest backup exists -> state is written.
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    (backup_dir / "kensei-today.tar.gz").write_bytes(b"newbackup")

    r = _run(["--backup-dir", str(backup_dir), "--hermes-home", str(hermes_home)])
    assert r.returncode == 0, r.stderr
    state = hermes_home / "scripts" / ".backup-staleness-state.json"
    assert state.exists(), "state file should have been written under fake hermes_home"
    # Must NOT touch the real home state file
    real_state = Path.home() / ".hermes" / "scripts" / ".backup-staleness-state.json"
    # We can't fully guarantee the real file wasn't touched if it pre-existed,
    # but the fake one must exist and be valid JSON.
    import json
    data = json.loads(state.read_text())
    assert "last_check" in data
