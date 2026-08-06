"""Regression: installer update should discard pure npm lockfile churn.

Desktop/bootstrap installs update an existing managed checkout in place. Local
build steps often rewrite tracked ``package-lock.json`` without touching the
matching ``package.json``; treating that churn as a real local edit forces an
autostash and can abort the repository stage before the desktop comes back up.

The installer should discard that generated churn before its stash/checkout
logic, while still preserving intentional package edits where ``package.json``
and ``package-lock.json`` changed together.
"""

from __future__ import annotations

import re
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


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _extract_install_sh_function(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(rf"{name}\(\) \{{.*?\n\}}", text, re.DOTALL)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _extract_install_sh_autostash_block() -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        r'local autostash_ref="".*?\n            fi\n',
        text,
        re.DOTALL,
    )
    assert match is not None, "autostash block not found in install.sh"
    return match.group(0)


@pytest.mark.live_system_guard_bypass
def test_install_sh_discards_runtime_lockfile_churn_before_stash(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _git(repo, "init")
    (repo / "package.json").write_text('{"dependencies":{"a":"1"}}\n')
    (repo / "package-lock.json").write_text('{"lock":"old"}\n')
    _git(repo, "add", "package.json", "package-lock.json")
    _git(repo, "commit", "-m", "init")

    (repo / "package-lock.json").write_text('{"lock":"runtime-churn"}\n')

    script = (
        "set -e\n"
        'log_info() { echo "INFO: $*"; }\n'
        'INSTALL_DIR="$PWD"\n'
        f"{_extract_install_sh_function('discard_update_lockfile_churn')}\n"
        "run() {\n"
        f"{_extract_install_sh_autostash_block()}"
        "}\n"
        "run\n"
    )
    res = subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True
    )

    assert res.returncode == 0, res.stderr
    assert "Discarded npm lockfile churn (1 file(s))" in res.stdout
    assert _git(repo, "stash", "list").stdout.strip() == ""
    assert (repo / "package-lock.json").read_text() == '{"lock":"old"}\n'


def test_install_sh_discards_lockfile_churn_before_status_probe() -> None:
    text = INSTALL_SH.read_text()
    idx_cleanup = text.index('discard_update_lockfile_churn "$INSTALL_DIR"')
    idx_status = text.index('if [ -n "$(git status --porcelain)" ]')
    idx_stash = text.index("git stash push --include-untracked")
    assert idx_cleanup < idx_status < idx_stash


def test_install_ps1_discards_lockfile_churn_before_status_probe() -> None:
    text = INSTALL_PS1.read_text()
    assert "function Discard-LockfileChurn" in text
    idx_cleanup = text.index("Discard-LockfileChurn $InstallDir")
    idx_status = text.index(
        "$statusOut = git -c windows.appendAtomically=false status --porcelain"
    )
    idx_stash = text.index("stash push --include-untracked")
    assert idx_cleanup < idx_status < idx_stash
