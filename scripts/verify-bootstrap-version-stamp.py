#!/usr/bin/env python3
"""Verify the version stamp a bootstrap installer ships.

The bootstrap installers (scripts/install.sh, scripts/install.ps1) finish
with a ``complete`` stage that writes ``.hermes-bootstrap-complete`` into the
install dir: ``{"schemaVersion": 1, "pinnedCommit": <40-hex sha>,
"pinnedBranch": <name>, "completedAt": <UTC ISO-8601>}``. That stamp is what
update tooling and support triage read back, so it must tell the truth about
the bytes actually checked out.

This script reads a stamp back and cross-checks it against the installed
checkout it describes:

* ``schemaVersion`` is 1
* ``pinnedCommit`` is a full 40-char lowercase hex sha AND equals
  ``git rev-parse HEAD`` of the install repo (or ``--expect-commit``)
* ``pinnedBranch`` equals the repo's checked-out branch (or ``--expect-branch``)
* ``completedAt`` parses as an ISO-8601 UTC timestamp
* the install carries a shipped version — ``hermes_cli/__init__.py``'s
  ``__version__``, read through the same authority as
  scripts/write_install_stamp.py

Usage:

    python3 scripts/verify-bootstrap-version-stamp.py \
        --stamp <install-dir>/.hermes-bootstrap-complete --repo <install-dir>
    # CI lane additionally pins the expected commit/branch:
    python3 scripts/verify-bootstrap-version-stamp.py --stamp ... --repo ... \
        --expect-commit "$SHA" --expect-branch ci-under-test

Exit 0 = the stamp tells the truth; exit 1 = any check fails (each failure
prints one ``FAIL:`` line).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_exe() -> str:
    """Resolve git to an executable CreateProcess can start.

    On sandboxed hosts PATH can resolve ``git`` to a store-app payload copy
    (e.g. under ``C:\\Program Files\\WindowsApps\\...``) that fails to exec
    outside its package context; prefer a conventional install there.
    """
    candidates: list[str] = []
    hit = shutil.which("git")
    if hit:
        candidates.append(hit)
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git"
        for rel in (("cmd", "git.exe"), ("bin", "git.exe")):
            p = base.joinpath(*rel)
            if p.exists():
                candidates.append(str(p))
    for cand in candidates:
        if "windowsapps" not in cand.lower():
            return cand
    return candidates[0] if candidates else "git"


_GIT = _git_exe()


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            [_GIT, *args], cwd=str(repo), capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return value if result.returncode == 0 and value else None


def verify_stamp(stamp_path: Path, repo: Path, expect_commit: str | None, expect_branch: str | None) -> list[str]:
    errors: list[str] = []

    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8-sig"))
    except OSError as e:
        _fail(errors, f"cannot read stamp {stamp_path}: {e}")
        return errors
    except ValueError as e:
        _fail(errors, f"stamp is not valid JSON: {e}")
        return errors
    if not isinstance(stamp, dict):
        _fail(errors, f"stamp top level is {type(stamp).__name__}, expected object")
        return errors

    if stamp.get("schemaVersion") != 1:
        _fail(errors, f"schemaVersion {stamp.get('schemaVersion')!r}, expected 1")

    commit = stamp.get("pinnedCommit")
    if not isinstance(commit, str) or not SHA_RE.match(commit):
        _fail(errors, f"pinnedCommit {commit!r} is not a full 40-char lowercase hex sha")
        commit = None

    branch = stamp.get("pinnedBranch")
    if not isinstance(branch, str) or not branch:
        _fail(errors, f"pinnedBranch {branch!r} is empty")
        branch = None

    completed = stamp.get("completedAt")
    if not isinstance(completed, str):
        _fail(errors, f"completedAt {completed!r} is missing")
    else:
        try:
            parsed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                _fail(errors, f"completedAt {completed!r} is not UTC")
        except ValueError:
            _fail(errors, f"completedAt {completed!r} does not parse as ISO-8601")

    # Cross-check the stamp against the checkout it claims to describe.
    if commit is not None:
        head = _git(repo, "rev-parse", "HEAD")
        if head is None:
            _fail(errors, f"could not read HEAD of {repo} to compare with pinnedCommit")
        elif head != commit:
            _fail(errors, f"pinnedCommit {commit[:12]} != installed HEAD {head[:12]}")
        if expect_commit and commit != expect_commit:
            _fail(errors, f"pinnedCommit {commit[:12]} != expected {expect_commit[:12]}")

    if branch is not None:
        actual = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if expect_branch:
            if branch != expect_branch:
                _fail(errors, f"pinnedBranch {branch!r} != expected {expect_branch!r}")
        elif actual and actual != "HEAD" and actual != branch:
            _fail(errors, f"pinnedBranch {branch!r} != checked-out branch {actual!r}")

    # The install must carry a shipped version at all.
    version, _ = _read_version(repo)
    if not version:
        _fail(errors, f"no __version__ found in {repo}/hermes_cli/__init__.py — the install carries no shipped version")

    return errors


def _read_version(repo: Path) -> tuple[str | None, str | None]:
    """Read __version__ from the INSTALL repo (not this script's checkout)."""
    init_py = repo / "hermes_cli" / "__init__.py"
    try:
        text = init_py.read_text(encoding="utf-8-sig")
    except OSError:
        return None, None
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return (match.group(1) if match else None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True, help="Path to .hermes-bootstrap-complete")
    parser.add_argument("--repo", required=True, help="The install checkout the stamp describes")
    parser.add_argument("--expect-commit", default=None, help="Fail unless pinnedCommit equals this sha")
    parser.add_argument("--expect-branch", default=None, help="Fail unless pinnedBranch equals this branch")
    args = parser.parse_args()

    errors = verify_stamp(
        Path(args.stamp), Path(args.repo), args.expect_commit, args.expect_branch
    )
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("bootstrap version stamp verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
