"""Regression: installer fails when the existing checkout has an unmerged index.

A previously interrupted update can leave ``$INSTALL_DIR`` with unmerged index
entries (files in a conflicted, "needs merge" state). In that state the update
path's ``git stash`` aborts with "could not write index" and the following
``git checkout <branch>`` aborts with "you need to resolve your current index
first" -- surfacing to GUI/bootstrap users as ``git checkout main failed
(exit 1)`` and failing the whole install at the repository stage.

The ``hermes update`` Python path already clears the conflict with ``git reset``
before stashing (#4735); both installer scripts must do the same.
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


def _extract_autostash_block() -> str:
    """Pull the autostash if-block from install.sh's update_repo()."""
    text = INSTALL_SH.read_text()
    m = re.search(
        r'local autostash_ref="".*?\n            fi\n',
        text,
        re.DOTALL,
    )
    assert m is not None, "autostash block not found in install.sh"
    return m.group(0)


def _extract_install_sh_function(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(rf"{name}\(\) \{{.*?\n\}}", text, re.DOTALL)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _make_unmerged_repo(repo: Path) -> None:
    """Leave ``repo`` with a conflicted (unmerged) index, as an interrupted
    update would."""
    _git(repo, "init")
    (repo / "f.txt").write_text("base\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "base")
    # Capture the default branch name only after the first commit exists
    # (rev-parse on an unborn HEAD errors).
    start = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    _git(repo, "checkout", "-b", "feature")
    (repo / "f.txt").write_text("feature side\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "feature")

    _git(repo, "checkout", start)
    (repo / "f.txt").write_text("main side\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "mainside")

    # Conflicting merge — exits non-zero and leaves the index unmerged.
    _git(repo, "merge", "feature", check=False)


@pytest.mark.live_system_guard_bypass  # runs against a dedicated throwaway repo
def test_install_sh_clears_unmerged_index_then_stashes(tmp_path: Path) -> None:
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _make_unmerged_repo(repo)

    # Sanity: this is exactly the state that breaks `git stash` / `git checkout`.
    assert _git(repo, "ls-files", "--unmerged").stdout.strip(), (
        "test setup failed to produce an unmerged index"
    )

    block = _extract_autostash_block()
    script = (
        "set -e\n"
        'log_info() { echo "INFO: $*"; }\n'
        f'INSTALL_DIR="{repo}"\n'
        f"{_extract_install_sh_function('discard_update_lockfile_churn')}\n"
        "run() {\n"
        f"{block}"
        "}\n"
        "run\n"
        "echo BLOCK_OK\n"
    )
    res = subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True
    )

    # The block must complete (previously `git stash` failed with "could not
    # write index" on the unmerged tree).
    assert res.returncode == 0, res.stderr
    assert "BLOCK_OK" in res.stdout
    assert "Clearing unmerged index entries" in res.stdout

    # The conflict state is gone ...
    assert _git(repo, "ls-files", "--unmerged").stdout.strip() == "", (
        "unmerged entries should have been cleared"
    )
    # ... and the local changes were preserved in a stash, not discarded.
    assert _git(repo, "stash", "list").stdout.strip(), (
        "local changes should be preserved in a stash"
    )


def test_install_ps1_clears_unmerged_index_before_stash() -> None:
    """install.ps1 must clear an unmerged index before stash/checkout, and do
    so *before* the stash push (order matters — the fix is a no-op otherwise)."""
    text = INSTALL_PS1.read_text()
    assert "ls-files --unmerged" in text, (
        "install.ps1 must detect an unmerged index before updating"
    )
    idx_unmerged = text.index("ls-files --unmerged")
    idx_reset = text.index("reset -q", idx_unmerged)
    idx_stash = text.index("stash push --include-untracked")
    assert idx_unmerged < idx_stash, (
        "the unmerged-index clear must run before `git stash push`"
    )
    assert idx_reset < idx_stash, "`git reset` must run before `git stash push`"


def test_install_sh_clears_unmerged_index_before_stash_source_order() -> None:
    """Same ordering contract for install.sh's source."""
    text = INSTALL_SH.read_text()
    assert "ls-files --unmerged" in text
    idx_unmerged = text.index("ls-files --unmerged")
    idx_stash = text.index("stash push --include-untracked")
    assert idx_unmerged < idx_stash


def test_install_ps1_stops_venv_resident_processes_before_removing_venv() -> None:
    """The Windows venv-recreate path must stop every process running out of the
    old venv before deleting it.

    A gateway autostarted by a scheduled task runs as
    ``venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run`` — image name
    ``pythonw``, not ``hermes.exe`` — so the ``taskkill /IM hermes.exe`` guard
    misses it, the loaded ``.pyd`` stays locked, and ``Remove-Item venv`` fails
    mid-recursion (issues #47036/#47557/#47910). The recreate branch must also
    sweep by venv path prefix, and that sweep must run before the delete.
    """
    text = INSTALL_PS1.read_text()

    # The hermes.exe tree-kill is preserved (kills spawned child processes too).
    assert 'taskkill /F /T /IM hermes.exe' in text

    # The venv path-prefix sweep exists. It must match by case-insensitive
    # StartsWith, NOT PowerShell -like: a venv path containing wildcard
    # metacharacters ('[', ']') — legal in a Windows user name — silently fails
    # to match under -like, reintroducing the exact miss this fix closes.
    idx_recreate = text.index("Virtual environment already exists, recreating")
    idx_sweep = text.index("StartsWith($venvPrefix", idx_recreate)
    assert "[System.StringComparison]::OrdinalIgnoreCase" in text[idx_sweep:idx_sweep + 200]
    assert 'ExecutablePath -like "$venvRoot' not in text, (
        "the -like wildcard match must not be used for venv path scoping"
    )

    # The process sweep must run before the venv is removed, or it is a no-op.
    idx_remove = text.index('Remove-Item -Recurse -Force "venv"', idx_recreate)
    assert idx_sweep < idx_remove, (
        "venv-resident processes must be stopped before Remove-Item deletes the venv"
    )
