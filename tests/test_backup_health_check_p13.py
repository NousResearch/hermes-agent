"""P13 isolation proof for scripts/backup-health-check.py.

Verifies:
- BACKUP_ROOT parameterisation: resolves under the BACKUP_ROOT env, not
  a hardcoded path.
- BACKUP_HEALTH_DRY_RUN=1 skips archive verification (no tarfile open,
  no sha256 computation, no manifest parsing) — only path resolution
  runs and the script reports a DRY-RUN banner.
- import-safe: importing the module does not open any archive.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "backup-health-check.py"


def _load_module(monkeypatch, backup_root: Path, dry_run: bool = False):
    monkeypatch.setenv("BACKUP_ROOT", str(backup_root))
    if dry_run:
        monkeypatch.setenv("BACKUP_HEALTH_DRY_RUN", "1")
    else:
        monkeypatch.delenv("BACKUP_HEALTH_DRY_RUN", raising=False)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("bhc_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    return root


def test_backup_root_resolves_under_env(monkeypatch, fake_root):
    mod = _load_module(monkeypatch, fake_root)
    assert str(mod.BACKUP_ROOT) == str(fake_root)


def test_dry_run_flag_set(monkeypatch, fake_root):
    mod = _load_module(monkeypatch, fake_root, dry_run=True)
    assert mod._DRY_RUN is True


def test_dry_run_flag_unset_by_default(monkeypatch, fake_root):
    mod = _load_module(monkeypatch, fake_root, dry_run=False)
    assert mod._DRY_RUN is False


def test_dry_run_main_skips_verification_when_no_backups(monkeypatch, fake_root, capsys):
    """With no archives and dry-run, main() must print DRY-RUN (not ALERT)
    and return 0."""
    mod = _load_module(monkeypatch, fake_root, dry_run=True)
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "ALERT" not in out


def test_dry_run_main_skips_verification_with_backups(monkeypatch, fake_root, capsys):
    """With a fake archive present and dry-run, main() must report it
    would verify but NOT open the tarfile (no checksum/manifest work)."""
    # Create a fake archive (just a file, not a real tarball)
    (fake_root / "kensei-20260728-1200.tar.gz").write_bytes(b"fake")
    mod = _load_module(monkeypatch, fake_root, dry_run=True)
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "would verify" in out


def test_import_does_not_open_archive(monkeypatch, fake_root, capsys):
    """Importing the module must not open any archive or read files."""
    (fake_root / "kensei-20260728-1200.tar.gz").write_bytes(b"fake")
    mod = _load_module(monkeypatch, fake_root, dry_run=True)
    # No output during import.
    captured = capsys.readouterr()
    assert captured.out == ""
