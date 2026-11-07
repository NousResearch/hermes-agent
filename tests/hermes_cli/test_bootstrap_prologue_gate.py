"""Importing an entry point in a test run must not boot the install the suite runs from (PER-590).

``hermes_bootstrap``'s import side effect is a launch prologue: settle a killed update, prepare
(and possibly relaunch into) the launch dependencies, recover, activate. Run from a deployed
checkout -- one a real data root installed -- every one of those probes or mutates the operator's
live install. A test process that imports it transitively must skip the prologue; the tests that
prove launch behaviour call the functions directly, or opt in for this module's own prologue.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_the_prologue_is_off_inside_the_suite(monkeypatch):
    import hermes_bootstrap

    assert hermes_bootstrap._launch_prologue_enabled() is False
    assert hermes_bootstrap._prologue_skipped_for_test is True
    monkeypatch.setenv("HERMES_TEST_ALLOW_LIVE_INSTALL", "1")
    assert hermes_bootstrap._launch_prologue_enabled() is True


def test_a_product_launch_still_runs_the_prologue(monkeypatch):
    import hermes_bootstrap

    for name in hermes_bootstrap._TEST_PROCESS_ENV:
        monkeypatch.setenv(name, "")
    assert hermes_bootstrap._launch_prologue_enabled() is True


def test_an_import_in_a_test_process_skips_the_prologue(tmp_path):
    """The gate is decided before the prologue's first probe, and inheritance (not just
    ``PYTEST_*``, which children rebuild away) is enough to keep a child hermetic."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    env.pop("HERMES_TEST_ALLOW_LIVE_INSTALL", None)
    isolation = tmp_path / "isolation-home"
    env["HERMES_HOME"] = str(isolation)
    env["HERMES_TEST_ISOLATION"] = str(isolation)
    env["PYTHONPATH"] = str(REPO)
    script = "import hermes_bootstrap as b; print(b._prologue_skipped_for_test)"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
