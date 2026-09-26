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
    """A minimal 'installed checkout': git repo + the source identity stamp."""
    repo = tmp_path / "install"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "ci")
    (repo / "hermes_cli").mkdir()
    (repo / "hermes_cli" / "__init__.py").write_text(
        '"""Hermes CLI."""\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "seed")
    (repo / "install-stamp.json").write_text(json.dumps({
        "schemaVersion": 2, "commit": _git(repo, "rev-parse", "HEAD"),
        "baseVersion": "0.1.2", "updateMechanism": "self",
    }), encoding="utf-8")
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
    ({"missing": "version"}, [], "no source identity stamp"),
    ({"canonicalCommit": "c" * 40}, [], "canonical"),
    # A checkout with no reachable release tag honestly stamps a null base.
    ({"canonicalBase": None}, [], None),
    # The protocol lane never runs the products stage that writes the stamp.
    ({"missing": "version"}, ["--no-source-stamp"], None),
])
def test_verifier_cli(tmp_path, changes, expect, error):
    repo = _install_repo(tmp_path)
    stamp = {"schemaVersion": 1, "pinnedCommit": _git(repo, "rev-parse", "HEAD"),
             "pinnedBranch": "main", "completedAt": "2026-08-30T12:00:00.000Z", **changes}
    path = repo / ".hermes-bootstrap-complete"
    if changes.get("missing") != "stamp":
        path.write_text(json.dumps(stamp), encoding="utf-8")
    if changes.get("missing") == "version":
        (repo / "install-stamp.json").unlink()
    if "canonicalCommit" in changes or "canonicalBase" in changes:
        canonical = json.loads((repo / "install-stamp.json").read_text(encoding="utf-8"))
        canonical["commit"] = changes.get("canonicalCommit", canonical["commit"])
        canonical["baseVersion"] = changes.get("canonicalBase", canonical["baseVersion"])
        (repo / "install-stamp.json").write_text(json.dumps(canonical), encoding="utf-8")
    if not error:
        expect = ["--expect-commit", stamp["pinnedCommit"], "--expect-branch", "main", *expect]
    env = dict(os.environ)
    if os.name == "nt":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")
    result = subprocess.run([sys.executable, str(_SCRIPT), "--repo", str(repo), "--stamp", str(path), *expect],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (1 if error else 0), result.stderr
    assert error in result.stderr if error else "stamp verified" in result.stdout


def test_git_helper_decodes_output_explicitly_as_utf8(tmp_path, monkeypatch):
    """The verifier's git helper carries the same cp936 hazard as the writer's.

    ``text=True`` without ``encoding=`` decodes git's stdout through the
    platform code page (cp936 on Chinese Windows). The reader thread dies on
    the first byte it cannot map and the caller sees a *successful* command
    with no output — which is exactly what makes a healthy checkout look like
    it has no identity at all, so the decode must be pinned here too.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_bootstrap_version_stamp", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorded: dict = {}

    def fake_run(*args, **kwargs):
        recorded.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert module._git(tmp_path, "rev-parse", "HEAD") == "abc123"
    assert recorded.get("encoding") == "utf-8"
    assert recorded.get("errors") == "replace"
