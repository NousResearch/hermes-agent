import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash installer test")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "team-shortcuts" / "install-shortcuts.sh"


def build_fake_installer(tmp_path: Path, registry: str = "registry v1\n", ref: str = "ref v1\n"):
    team_dir = tmp_path / "team-shortcuts"
    payload = team_dir / "payload"
    registry_path = payload / "ai-context" / "prompt-shortcut-registry.md"
    ref_path = payload / "skills" / "prompt-shortcuts" / "references" / "a.md"

    team_dir.mkdir()
    shutil.copy2(SCRIPT, team_dir / "install-shortcuts.sh")
    registry_path.parent.mkdir(parents=True)
    ref_path.parent.mkdir(parents=True)
    registry_path.write_text(registry)
    ref_path.write_text(ref)
    return team_dir


def run_installer(team_dir: Path, tmp_path: Path, *args: str):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["HERMES_SHORTCUTS_DEST"] = str(tmp_path / "vault")
    return subprocess.run(
        ["bash", str(team_dir / "install-shortcuts.sh"), *args],
        cwd=team_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def vault_file(tmp_path: Path, relative: str) -> Path:
    return tmp_path / "vault" / relative


def backup_dirs(tmp_path: Path):
    return sorted((tmp_path / "vault").glob(".backup-shortcuts-*"))


def test_fresh_install_copies_payload_to_destination(tmp_path: Path):
    team_dir = build_fake_installer(tmp_path)

    result = run_installer(team_dir, tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert vault_file(tmp_path, "ai-context/prompt-shortcut-registry.md").read_text() == "registry v1\n"
    assert vault_file(tmp_path, "skills/prompt-shortcuts/references/a.md").read_text() == "ref v1\n"


def test_newer_different_destination_blocks_without_force(tmp_path: Path):
    team_dir = build_fake_installer(tmp_path)
    assert run_installer(team_dir, tmp_path).returncode == 0

    dest = vault_file(tmp_path, "skills/prompt-shortcuts/references/a.md")
    dest.write_text("owner newer work\n")
    future = time.time() + 60
    os.utime(dest, (future, future))

    result = run_installer(team_dir, tmp_path)

    assert result.returncode == 2
    assert dest.read_text() == "owner newer work\n"
    assert "ไฟล์ปลายทางใหม่กว่าชุดติดตั้ง" in result.stdout
    assert "skills/prompt-shortcuts/references/a.md" in result.stdout


def test_force_overwrites_newer_destination_and_creates_backup(tmp_path: Path):
    team_dir = build_fake_installer(tmp_path)
    assert run_installer(team_dir, tmp_path).returncode == 0

    dest = vault_file(tmp_path, "skills/prompt-shortcuts/references/a.md")
    dest.write_text("owner newer work\n")
    future = time.time() + 60
    os.utime(dest, (future, future))

    result = run_installer(team_dir, tmp_path, "--force")

    assert result.returncode == 0, result.stderr + result.stdout
    assert dest.read_text() == "ref v1\n"
    backups = backup_dirs(tmp_path)
    assert len(backups) == 1
    assert (backups[0] / "skills/prompt-shortcuts/references/a.md").read_text() == "owner newer work\n"


def test_rerun_unchanged_payload_does_not_create_second_backup(tmp_path: Path):
    team_dir = build_fake_installer(tmp_path)
    assert run_installer(team_dir, tmp_path).returncode == 0

    dest = vault_file(tmp_path, "skills/prompt-shortcuts/references/a.md")
    dest.write_text("changed but older\n")
    past = time.time() - 60
    os.utime(dest, (past, past))
    assert run_installer(team_dir, tmp_path).returncode == 0
    assert len(backup_dirs(tmp_path)) == 1

    result = run_installer(team_dir, tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert len(backup_dirs(tmp_path)) == 1
