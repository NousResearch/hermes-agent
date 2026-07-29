"""Regression: a stale ``--commit`` pin must not roll an install backwards.

``hermes-setup.exe`` bakes its build-time commit into the binary
(``BUILD_PIN_COMMIT``) and passes it as ``-Commit`` / ``--commit`` on every
install-mode run — including the retry the desktop's "Update didn't finish"
screen kicks off. The repository stage used to ``git checkout --detach`` that
SHA unconditionally, so an installer built months earlier rewound a current
managed checkout to its build commit (observed: ~9k commits back), leaving
ancient source against a current venv — npm workspaces and Python deps that no
longer match, and every subsequent update failing against the wrong tree.

The pin is skipped when its target is already an ancestor of HEAD, unless the
caller explicitly passes ``--force-commit`` / ``-ForceCommit``. A fresh clone
has no such ancestry, so reproducible/CI pinning is unaffected.

``install.ps1`` carries the same guard (that is the path the Windows report
hit), but there is no PowerShell host in CI to execute it against a real repo,
and asserting on the script's *source text* would test its shape rather than
its behavior. These run the bash implementation of the same logic for real.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _extract_pin_block() -> str:
    """Pull the commit-pin block out of install.sh's update_repo()."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r'if \[ -n "\$INSTALL_COMMIT" \]; then.*?\n    fi\n',
        text,
        re.DOTALL,
    )
    assert match is not None, "commit-pin block not found in install.sh"
    return match.group(0)


@pytest.fixture
def repo(tmp_path):
    """A checkout with three commits, HEAD at the newest."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    shas = []
    for n in range(3):
        (origin / "f.txt").write_text(f"rev{n}\n")
        _git(origin, "add", "f.txt")
        _git(origin, "commit", "-qm", f"rev{n}")
        shas.append(_git(origin, "rev-parse", "HEAD"))
    return origin, shas


def _run_pin_block(repo_dir: Path, commit: str, *, force: bool = False) -> str:
    """Execute install.sh's pin block standalone against ``repo_dir``."""
    script = "\n".join(
        [
            "set -e",
            "log_info() { echo \"INFO $*\"; }",
            "log_warn() { echo \"WARN $*\"; }",
            f'INSTALL_COMMIT="{commit}"',
            f'FORCE_COMMIT={"true" if force else "false"}',
            f'cd "{repo_dir}"',
            _extract_pin_block(),
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_stale_pin_does_not_rewind_a_newer_checkout(repo):
    """The reported failure: an old baked-in pin downgrading a current tree."""
    repo_dir, shas = repo
    head_before = _git(repo_dir, "rev-parse", "HEAD")

    out = _run_pin_block(repo_dir, shas[0])

    assert _git(repo_dir, "rev-parse", "HEAD") == head_before, (
        "a pin older than HEAD must leave the checkout where it is"
    )
    assert "already newer" in out


def test_force_commit_still_rolls_back(repo):
    """Reproducible/CI installs that genuinely want an older SHA keep working."""
    repo_dir, shas = repo

    _run_pin_block(repo_dir, shas[0], force=True)

    assert _git(repo_dir, "rev-parse", "HEAD") == shas[0]


def test_pin_to_current_head_is_applied(repo):
    """Pinning to HEAD itself is a no-op checkout, not a skipped one."""
    repo_dir, shas = repo

    out = _run_pin_block(repo_dir, shas[2])

    assert _git(repo_dir, "rev-parse", "HEAD") == shas[2]
    assert "already newer" not in out


def test_pin_to_a_newer_commit_is_applied(repo):
    """Rolling FORWARD to a newer pin is the legitimate case — never blocked."""
    repo_dir, shas = repo
    _git(repo_dir, "checkout", "-q", "--detach", shas[0])

    out = _run_pin_block(repo_dir, shas[2])

    assert _git(repo_dir, "rev-parse", "HEAD") == shas[2]
    assert "already newer" not in out
