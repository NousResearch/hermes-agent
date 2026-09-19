"""Tests for scripts/verify-bootstrap-version-stamp.py.

The bootstrap installers stamp ``.hermes-bootstrap-complete`` with the
commit/branch they pinned; this script reads the stamp back and cross-checks
it against the installed checkout. Tests build a real temp git repo, write
stamps by hand, and verify the honest and lying cases.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _git_exe() -> str:
    """A git CreateProcess can start. conftest blanks SystemRoot/ComSpec and
    this host's PATH can resolve `git` to the MSIX payload copy under
    ``C:\\Program Files\\WindowsApps\\...`` — WinError 5 outside its package
    context. Prefer a conventional install."""
    candidates = [hit for hit in (shutil.which("git"),) if hit]
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git"
        for rel in (("cmd", "git.exe"), ("bin", "git.exe")):
            p = base.joinpath(*rel)
            if p.exists():
                candidates.append(str(p))
    for cand in candidates:
        if "windowsapps" not in cand.lower():
            return cand
    return candidates[0]


_GIT = _git_exe()


def _git(repo: Path, *args: str) -> str:
    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")
    out = subprocess.run(
        [_GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify-bootstrap-version-stamp.py"


def _install_repo(tmp_path: Path) -> Path:
    """A minimal 'installed checkout': git repo + the shipped-version field."""
    repo = tmp_path / "install"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "ci")
    (repo / "hermes_cli").mkdir()
    (repo / "hermes_cli" / "__init__.py").write_text('__version__ = "0.1.2"\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "seed")
    return repo


@pytest.mark.parametrize("changes,expect,error", [
    ({}, [], None),
    ({"pinnedCommit": "a" * 40}, [], "installed HEAD"),
    ({}, ["--expect-commit", "b" * 40], "expected"),
    ({}, ["--expect-branch", "release"], "pinnedBranch"),
    ({"schemaVersion": 2}, [], "schemaVersion"),
    ({"pinnedCommit": "deadbeef"}, [], "40-char"),
    ({"completedAt": "not-a-time"}, [], "ISO-8601"),
    ({"completedAt": "2026-08-30T12:00:00+01:00"}, [], "not UTC"),
    ({"missing": "stamp"}, [], "cannot read stamp"),
    ({"missing": "version"}, [], "no __version__"),
])
def test_verifier_cli(tmp_path, changes, expect, error):
    repo = _install_repo(tmp_path)
    stamp = {"schemaVersion": 1, "pinnedCommit": _git(repo, "rev-parse", "HEAD"),
             "pinnedBranch": "main", "completedAt": "2026-08-30T12:00:00.000Z", **changes}
    path = repo / ".hermes-bootstrap-complete"
    if changes.get("missing") != "stamp":
        path.write_text(json.dumps(stamp), encoding="utf-8")
    if changes.get("missing") == "version":
        (repo / "hermes_cli/__init__.py").unlink()
    if not error:
        expect = ["--expect-commit", stamp["pinnedCommit"], "--expect-branch", "main"]
    env = dict(os.environ)
    if os.name == "nt":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")
    result = subprocess.run([sys.executable, str(_SCRIPT), "--repo", str(repo), "--stamp", str(path), *expect],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (1 if error else 0), result.stderr
    assert error in result.stderr if error else "stamp verified" in result.stdout
