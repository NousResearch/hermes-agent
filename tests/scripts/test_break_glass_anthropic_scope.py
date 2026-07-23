"""Tests for scripts/break_glass_anthropic_scope.py."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "break_glass_anthropic_scope.py"


def run_bg(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_requires_yes(tmp_path):
    root = tmp_path / "h"
    root.mkdir()
    (root / "shared").mkdir()
    proc = run_bg("--root", str(root), "--backup-dir", str(tmp_path / "b"))
    assert proc.returncode == 2


def test_removes_marker_leaves_auth(tmp_path):
    root = tmp_path / "h"
    shared = root / "shared"
    shared.mkdir(parents=True)
    auth = root / "auth.json"
    auth_bytes = b'{"version": 1, "providers": {}}\n'
    auth.write_bytes(auth_bytes)
    auth.chmod(0o600)
    marker = shared / "anthropic_pool_scope.json"
    marker.write_text(json.dumps({"version": 1, "scope": "shared", "epoch": "e"}))
    marker.chmod(0o600)
    bdir = tmp_path / "backup"
    bdir.mkdir()
    bdir.chmod(0o700)

    proc = run_bg(
        "--root",
        str(root),
        "--backup-dir",
        str(bdir),
        "--yes",
    )
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists()
    assert auth.read_bytes() == auth_bytes
    backups = list(bdir.glob("*-anthropic-scope.json"))
    assert len(backups) == 1
    assert backups[0].stat().st_mode & 0o077 == 0


def test_symlink_marker_refused(tmp_path):
    root = tmp_path / "h"
    shared = root / "shared"
    shared.mkdir(parents=True)
    real = tmp_path / "real-marker.json"
    real.write_text("{}")
    marker = shared / "anthropic_pool_scope.json"
    marker.symlink_to(real)
    bdir = tmp_path / "b"
    bdir.mkdir()
    bdir.chmod(0o700)
    proc = run_bg("--root", str(root), "--backup-dir", str(bdir), "--yes")
    assert proc.returncode == 1
    assert marker.is_symlink()  # untouched


def test_relative_paths_rejected(tmp_path):
    proc = run_bg("--root", "relative", "--backup-dir", str(tmp_path), "--yes")
    assert proc.returncode == 2
