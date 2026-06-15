"""Regression for #40998: installer fails on an interrupted prior clone.

A previous clone that died before its first commit leaves ``$INSTALL_DIR/.git``
present but with no resolvable ``HEAD``. ``git rev-parse --is-inside-work-tree``
and ``git status`` both still succeed there, so the installer treated it as a
valid checkout and tried to *update* it -- but ``git stash``/``git checkout``
abort with "You do not have the initial commit yet", failing the install at the
"Cloning Hermes repository" stage.

Both installers must instead treat a commit-less checkout as broken and
re-clone fresh.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _extract_no_commit_guard() -> str:
    """Pull the clone_repo() guard that drops a commit-less checkout."""
    text = INSTALL_SH.read_text()
    m = re.search(
        r'if \[ -d "\$INSTALL_DIR/\.git" \] && ! git -C "\$INSTALL_DIR" '
        r"rev-parse --verify HEAD.*?\n    fi",
        text,
        re.DOTALL,
    )
    assert m is not None, "no-commit guard not found in install.sh clone_repo()"
    return m.group(0)


def _run_guard(install_dir: Path) -> None:
    block = _extract_no_commit_guard()
    script = (
        "log_warn() { echo \"WARN: $*\"; }\n"
        f"INSTALL_DIR={shlex.quote(str(install_dir))}\n"
        f"{block}\n"
    )
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_install_sh_guard_moves_commitless_checkout_aside(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _git(install_dir, "init")
    (install_dir / "leftover.txt").write_text("partial download")  # untracked

    # Sanity: this is exactly the state that breaks `git stash`.
    head = subprocess.run(
        ["git", "-C", str(install_dir), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
    )
    assert head.returncode != 0

    _run_guard(install_dir)
    # The original path is cleared so a fresh clone can proceed, but the
    # content is preserved in a backup (never deleted -- review feedback).
    assert not install_dir.exists(), "commit-less checkout should be moved aside"
    backups = list(install_dir.parent.glob(install_dir.name + ".broken-*"))
    assert len(backups) == 1, "broken checkout should be moved to one backup dir"
    assert (backups[0] / "leftover.txt").read_text() == "partial download"


def test_install_sh_guard_keeps_repo_with_commits(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _git(install_dir, "init")
    (install_dir / "f.txt").write_text("real content")
    _git(install_dir, "add", "f.txt")
    _git(install_dir, "commit", "-m", "init")

    _run_guard(install_dir)
    assert install_dir.exists()
    assert (install_dir / "f.txt").exists(), "a real checkout must be left intact"
    assert not list(install_dir.parent.glob(install_dir.name + ".broken-*")), (
        "a healthy checkout must not be moved aside"
    )


def test_install_sh_guard_ignores_non_repo_dir(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    (install_dir / "f.txt").write_text("not a repo")

    _run_guard(install_dir)
    # No .git → not our concern; the existing "not a git repository" branch
    # still handles it. The guard must leave it untouched.
    assert install_dir.exists()
    assert (install_dir / "f.txt").exists()


def test_install_ps1_validity_requires_initial_commit() -> None:
    """The PowerShell repo-validity gate must also require a resolvable HEAD."""
    text = INSTALL_PS1.read_text()
    assert "rev-parse --verify HEAD" in text, (
        "install.ps1 must probe for an initial commit (#40998)"
    )
    # Contract: $repoValid is only set when the HEAD probe succeeded too.
    assert re.search(
        r"if \(\$revParseOk -and \$statusOk -and \$hasCommit\) \{",
        text,
    ), "repo validity must be gated on $hasCommit, not just rev-parse + status"
    # Cleanup must be non-destructive: move the broken checkout aside, never
    # `Remove-Item -Recurse -Force` it (review feedback on #40998).
    assert "Move-Item -LiteralPath $InstallDir" in text, (
        "install.ps1 must move an invalid checkout aside, not delete it"
    )
    assert "Remove-Item -Recurse -Force $InstallDir -ErrorAction Stop" not in text, (
        "the destructive wipe of an existing install dir must be gone "
        "(transient cleanup of a just-failed clone is fine)"
    )
