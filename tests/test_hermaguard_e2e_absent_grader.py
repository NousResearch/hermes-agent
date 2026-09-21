#!/usr/bin/env python3
"""Regression seam: Hermaguard E2E must SKIP (not error) when the optional
deployed grader is absent.

The grader (``hermaguard_grader.py``) is an OPTIONAL skill-side artifact at
``$HERMES_HOME/skills/software-development/hermaguard/scripts``. It is NOT a
package dependency of KenseiAgent. When it is missing from the environment the
E2E suite (tests/test_hermaguard_e2e.py) must be skipped cleanly during
collection, never raise ``ModuleNotFoundError`` and abort the entire
broad-suite collection. A skipped-at-import module can legitimately return
pytest exit 5 because it collects zero test items; the required invariant is
clean skip/no collection error, not a fabricated exit-0 claim.

This test lives OUTSIDE the E2E module on purpose: if it were inside, it would
be skipped together with the module when the grader is absent and could never
catch a regression. Here it always runs, spawns the E2E module in a subprocess
with a disposable HERMES_HOME (grader guaranteed absent), and asserts the
module skips (pytest exit 0, no ModuleNotFoundError, no collection error).

Run:  pytest tests/test_hermaguard_e2e_absent_grader.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_MODULE = "tests/test_hermaguard_e2e.py"


def _run_e2e_with_absent_grader(tmp_home: Path) -> subprocess.CompletedProcess:
    """Run only the Hermaguard E2E module with NO deployed grader present.

    Uses a disposable HERMES_HOME so the result does not depend on Sahil's live
    ~/.hermes. Asserts the pre-condition (grader genuinely absent) before launch.
    """
    grader_dir = tmp_home / "skills/software-development/hermaguard/scripts"
    grader_script = grader_dir / "hermaguard_grader.py"
    # Pre-condition for the regression: grader must be absent.
    assert not grader_script.exists(), f"grader unexpectedly present at {grader_script}"

    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_home)

    return subprocess.run(
        [
            sys.executable, "-m", "pytest", E2E_MODULE,
            "-p", "no:cacheprovider", "-o", "addopts=", "--no-header", "-v",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True, text=True,
    )


def test_absent_grader_skips_cleanly_not_collection_error(tmp_path):
    """E2E module must skip when grader is absent.

    RED state (bug present): collection raises ModuleNotFoundError -> exit 2.
    GREEN state (fixed): module skipped -> exit 0, no ModuleNotFoundError.
    """
    tmp_home = tmp_path / "empty_hermes"
    tmp_home.mkdir()

    res = _run_e2e_with_absent_grader(tmp_home)
    combined = res.stdout + res.stderr

    # 1. pytest must NOT abort collection. A skipped module returns 0 (tests
    #    collected+skipped) OR 5 (module skipped at import => 0 items
    #    collected). Both are correct "not errored" outcomes. A collection
    #    failure would be exit 2/4; that is what we forbid.
    assert res.returncode in (0, 5), (
        f"E2E module must SKIP (exit 0) when grader absent, got {res.returncode}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )

    # 2. No collection-time import error may leak through.
    assert "ModuleNotFoundError" not in combined, (
        f"grader-absent path must not raise ModuleNotFoundError:\n{combined}"
    )
    assert "ERROR collecting" not in combined, (
        f"grader-absent path must not produce a collection ERROR:\n{combined}"
    )

    # 3. The E2E module must be reported as skipped (not collected/errored).
    #    A module skipped at import via importorskip reports "N skipped" with
    #    0 items collected — the module name is not echoed, so assert on the
    #    skip signal itself rather than the path string.
    assert "1 skipped" in combined, (
        f"expected the E2E module to be reported as skipped:\n{combined}"
    )
