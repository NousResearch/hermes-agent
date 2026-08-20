#!/usr/bin/env python3
"""Isolated behavioral tests for the atomic Hermes installer."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(installer: Path, hermes_home: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HERMES_HOME=str(hermes_home))
    result = subprocess.run(
        ["bash", str(installer), *args],
        cwd=installer.parent,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == expect, (args, result.returncode, result.stdout, result.stderr)
    return result


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in relative or relative.endswith(".pyc"):
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_source(destination: Path) -> None:
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", ".github", "out", "dist", "__pycache__", ".pytest_cache", "usage", "reports"
        ),
    )


def test_installer_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        source = base / "source"
        hermes = base / "hermes"
        copy_source(source)
        installer = source / "install.sh"
        destination = hermes / "skills/kubrick"

        version = run(installer, hermes, "--version").stdout.strip()
        assert version == (ROOT / "VERSION").read_text(encoding="utf-8").strip()

        dry = run(installer, hermes, "--dry-run")
        assert "stage, validate, and atomically install" in dry.stdout
        assert not hermes.exists(), "dry-run wrote to HERMES_HOME"

        fresh = run(installer, hermes)
        assert "Validated before activation" in fresh.stdout
        assert destination.is_dir()
        assert not any(destination.rglob("*.pyc"))
        assert not any(path.name == "__pycache__" for path in destination.rglob("*"))
        run(destination / "install.sh", hermes, "--version")
        validation = subprocess.run(
            ["python3", str(destination / "scripts/validate_hermes_skill.py")],
            cwd=base,
            text=True,
            capture_output=True,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr
        baseline = tree_digest(destination)

        (source / "upgrade-marker.txt").write_text("upgrade\n", encoding="utf-8")
        run(installer, hermes)
        assert (destination / "upgrade-marker.txt").is_file()
        backup = Path((hermes / "receipts/kubrick-last-backup").read_text(encoding="utf-8").strip())
        assert backup.is_dir() and not (backup / "upgrade-marker.txt").exists()
        receipts = sorted((hermes / "receipts").glob("kubrick-install-*.json"))
        install_receipt = json.loads(receipts[-1].read_text(encoding="utf-8"))
        assert install_receipt["status"] == "INSTALLED"
        assert install_receipt["validated_before_activation"] is True

        run(installer, hermes, "--rollback")
        assert tree_digest(destination) == baseline
        assert not (destination / "upgrade-marker.txt").exists()

        active_before_failure = tree_digest(destination)
        (source / "SKILL.md").unlink()
        failed = run(installer, hermes, expect=1)
        assert failed.stdout or failed.stderr, "failed staged validation emitted no diagnostic"
        assert tree_digest(destination) == active_before_failure
        assert not any((hermes / "staging").iterdir()), "failed staging directory was not cleaned"


def main() -> None:
    test_installer_lifecycle()
    print("atomic Hermes installer lifecycle: PASS")


if __name__ == "__main__":
    main()
