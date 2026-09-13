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
_spec = importlib.util.spec_from_file_location("verify_bootstrap_version_stamp", _SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load verify-bootstrap-version-stamp.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


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


def _stamp(repo: Path, **overrides: object) -> Path:
    stamp = {
        "schemaVersion": 1,
        "pinnedCommit": _git(repo, "rev-parse", "HEAD"),
        "pinnedBranch": "main",
        "completedAt": "2026-08-30T12:00:00.000Z",
    }
    stamp.update(overrides)
    path = repo / ".hermes-bootstrap-complete"
    path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return path


def test_honest_stamp_verifies(tmp_path: Path):
    repo = _install_repo(tmp_path)
    errors = _mod.verify_stamp(_stamp(repo), repo, None, None)
    assert errors == []


def test_pinned_commit_must_match_installed_head(tmp_path: Path):
    repo = _install_repo(tmp_path)
    liar = "a" * 40
    errors = _mod.verify_stamp(_stamp(repo, pinnedCommit=liar), repo, None, None)
    assert any("pinnedCommit" in e and "HEAD" in e for e in errors), errors


def test_expect_commit_and_branch_are_enforced(tmp_path: Path):
    repo = _install_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    # Matching expectations pass.
    assert _mod.verify_stamp(_stamp(repo), repo, head, "main") == []
    # A stale expectation fails.
    errors = _mod.verify_stamp(_stamp(repo), repo, "b" * 40, "release")
    assert len(errors) == 2, errors


def test_bad_shape_and_timestamps_are_refused(tmp_path: Path):
    repo = _install_repo(tmp_path)
    errors = _mod.verify_stamp(
        _stamp(
            repo,
            schemaVersion=2,
            pinnedCommit="deadbeef",
            completedAt="not-a-time",
        ),
        repo,
        None,
        None,
    )
    joined = "\n".join(errors)
    assert "schemaVersion" in joined
    assert "40-char" in joined
    assert "ISO-8601" in joined


def test_missing_stamp_or_version_is_refused(tmp_path: Path):
    repo = _install_repo(tmp_path)
    errors = _mod.verify_stamp(repo / "nope.json", repo, None, None)
    assert errors and "cannot read stamp" in errors[0]

    empty = tmp_path / "empty"
    empty.mkdir()
    stamp_path = _stamp(repo)  # valid stamp...
    (repo / "hermes_cli" / "__init__.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "drop version")
    # The stamp's commit no longer matches HEAD AND no version ships.
    errors = _mod.verify_stamp(stamp_path, repo, None, None)
    joined = "\n".join(errors)
    assert "no __version__" in joined
