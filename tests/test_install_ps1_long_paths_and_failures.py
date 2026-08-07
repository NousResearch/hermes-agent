"""Regression coverage for Windows installer checkout and exit semantics."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_repository_stage_enables_git_long_paths_before_checkout() -> None:
    text = _install_ps1()

    assert (
        "git -c windows.appendAtomically=false -c core.longpaths=true "
        "clone --depth 1 --branch $Branch $RepoUrlSsh $InstallDir"
    ) in text
    assert (
        "git -c windows.appendAtomically=false -c core.longpaths=true "
        "clone --depth 1 --branch $Branch $RepoUrlHttps $InstallDir"
    ) in text

    # Existing installs, ZIP fallbacks, and fresh clones must all persist the
    # setting before a fetch/checkout can materialize long repository paths.
    assert text.count("config core.longpaths true") >= 3


def test_interactive_install_failure_is_rethrown() -> None:
    text = _install_ps1()
    match = re.search(
        r"} catch \{\s*"
        r"if \(\$Json -or \$Stage\) \{[\s\S]*?"
        r"# Interactive mode:[\s\S]*?"
        r"Write-Host \"\"\s*"
        r"(?P<tail>[\s\S]*?)"
        r"\n}",
        text,
    )
    assert match is not None, "installer top-level catch block not found"
    assert re.search(r"(?m)^\s*throw\s*$", match["tail"]), (
        "interactive installer failures must propagate to the caller instead "
        "of reporting exit code 0"
    )
