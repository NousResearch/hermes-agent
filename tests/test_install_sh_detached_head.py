"""Regression tests for recovering install.sh from a detached checkout."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _extract_branch_recovery_block() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(
        r"(?P<block>git remote set-branches origin \"\$BRANCH\".*?"
        r"git checkout -B \"\$BRANCH\" \"origin/\$BRANCH\"\n\s*fi)",
        text,
        re.DOTALL,
    )
    assert match is not None, "could not find install.sh detached-branch recovery block"
    return match["block"]


def test_install_sh_recreates_missing_local_branch_from_origin(tmp_path: Path) -> None:
    remote = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"

    remote.mkdir()
    _run_git("init", "--bare", cwd=remote)

    seed.mkdir()
    _run_git("init", "-b", "main", cwd=seed)
    _run_git("config", "user.email", "test@example.com", cwd=seed)
    _run_git("config", "user.name", "Test User", cwd=seed)
    (seed / "README.md").write_text("current\n", encoding="utf-8")
    _run_git("add", "README.md", cwd=seed)
    _run_git("commit", "-m", "seed", cwd=seed)
    _run_git("remote", "add", "origin", str(remote), cwd=seed)
    _run_git("push", "-u", "origin", "main", cwd=seed)

    _run_git("clone", "--branch", "main", str(remote), str(checkout), cwd=tmp_path)
    _run_git("checkout", "--detach", cwd=checkout)
    _run_git("branch", "-D", "main", cwd=checkout)
    assert _run_git("branch", "--show-current", cwd=checkout) == ""

    script = f"set -e\ncd {checkout!s}\nBRANCH=main\n{_extract_branch_recovery_block()}\n"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "branch recovery failed:\n"
        f"stdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    assert _run_git("branch", "--show-current", cwd=checkout) == "main"
    assert _run_git("rev-parse", "HEAD", cwd=checkout) == _run_git(
        "rev-parse", "origin/main", cwd=checkout
    )
