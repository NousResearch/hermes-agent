"""Regression for #61828: install.sh --stage protocol masks stage failures.

``run_stage_protocol()`` in ``scripts/install.sh`` runs each desktop-bootstrap
stage body in a subshell:

    set +e
    ( run_stage_body "$stage" )
    local code=$?
    set -e

The outer ``set +e`` is needed so a stage helper that calls ``exit 1`` only
terminates the subshell and the parent still reaches the JSON result frame.
But the subshell *inherits* errexit **off**, and every stage helper
(``setup_venv``, ``install_deps``, ``clone_repo``, …) was written for the
monolithic ``set -e`` flow and relies on errexit to abort on a failed command.
So a hard failure mid-helper (e.g. ``uv venv`` failing on a poisoned
``venv -> ./venv`` self-symlink) is swallowed: the helper runs past the failed
command, prints its success line, and the stage wrongly reports exit 0 /
``{ok:true}``.

The fix re-enables errexit *inside* the subshell (``( set -e; run_stage_body
"$stage" )``) so helpers get the failure semantics they were written for, while
the outer ``set +e`` still lets the parent survive a helper's ``exit 1``.

A secondary defect from the same reproduction: ``setup_venv``'s recreate guard
``[ -d "venv" ]`` misses a dangling/self-referential ``venv`` symlink, so the
stale-venv ``rm -rf`` never runs. Widened to ``[ -e "venv" ] || [ -L "venv" ]``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function_body(name: str) -> str:
    """Return the body of a top-level ``name() { ... }`` block from install.sh."""
    text = INSTALL_SH.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n(?P<body>.*?)\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Could not locate {name}() in scripts/install.sh"
    return match["body"]


def _extract_stage_subshell_region() -> str:
    """Return the ``set +e`` … ``local code=$?`` wrapper from run_stage_protocol."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"^ *set \+e\n"
        r"^ *\([^\n]*run_stage_body \"\$stage\"[^\n]*\)\n"
        r"^ *local code=\$\?",
        text,
        re.MULTILINE,
    )
    assert match is not None, (
        "Could not locate the run_stage_protocol subshell wrapper in "
        "scripts/install.sh"
    )
    return match.group(0)


def test_stage_subshell_reenables_errexit() -> None:
    """Static guard: the stage subshell must re-enable errexit (`set -e`).

    On origin/main the subshell is ``( run_stage_body "$stage" )`` with no inner
    ``set -e``, so this assertion FAILS before the fix and PASSES after.
    """
    body = _extract_function_body("run_stage_protocol")
    assert re.search(r"\(\s*set -e;?\s+run_stage_body", body) is not None, (
        "run_stage_protocol must re-enable errexit inside the stage subshell "
        "(`( set -e; run_stage_body \"$stage\" )`) so a failed command in a "
        "stage helper aborts the helper and yields a {ok:false} frame instead "
        "of a false success. See #61828."
    )


def test_stage_subshell_aborts_on_failing_command() -> None:
    """Behavioral repro: a failing command in the stage body aborts the subshell.

    Drives the ACTUAL wrapper region extracted from run_stage_protocol with a
    stub ``run_stage_body`` whose first command fails. Before the fix the
    subshell inherits errexit off, so ``false`` does not abort, ``LEAK`` prints
    and the subshell reports 0. After the fix the inner ``set -e`` aborts the
    subshell so ``LEAK`` never prints and ``code`` is non-zero.
    """
    region = _extract_stage_subshell_region()
    script = (
        "set -e\n"  # mimic install.sh's top-level errexit
        'run_stage_body() { false; echo LEAK; }\n'
        "stage=venv\n"
        "_probe() {\n"
        f"{region}\n"
        '    echo "code=$code"\n'
        "}\n"
        "_probe\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"probe harness itself failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    assert "code=1" in result.stdout, (
        "a failing command inside the stage body must propagate a non-zero exit "
        f"code (got: {result.stdout!r}). The stage subshell is swallowing the "
        "failure — see #61828."
    )
    assert "LEAK" not in result.stdout, (
        "the stage body kept running after a failed command (LEAK printed) — "
        "errexit is not active inside the subshell. See #61828."
    )


def test_setup_venv_recreate_guard_catches_symlink() -> None:
    """Static guard: the recreate guard must not rely on a bare ``[ -d "venv" ]``.

    A dangling/self-referential ``venv`` symlink is neither a directory (``-d``)
    nor an existing target (``-e`` alone once it dangles), so the stale-venv
    ``rm -rf`` never fires. The guard must also cover the symlink case (``-L``).
    On origin/main two bare ``-d`` guards remain → FAILS; after the fix → PASSES.
    """
    body = _extract_function_body("setup_venv")
    assert re.search(r'\[\s*-d\s+"venv"\s*\]', body) is None, (
        "setup_venv's recreate guard must not rely on a bare `[ -d \"venv\" ]` "
        "check — a dangling/self-referential `venv` symlink slips past `-d`, so "
        "the stale-venv `rm -rf` never runs and `uv venv` then hard-fails. Use "
        "`[ -e \"venv\" ] || [ -L \"venv\" ]`. See #61828."
    )
