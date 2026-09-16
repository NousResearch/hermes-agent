"""P13 dry-run + no-detach contract tests for the three blog-daily scripts.

pr-to-blog-daily.sh, blog-stream-daily.sh, blog-backlog-pregen.sh all share
the BLOG_DAILY_DRY_RUN=1 short-circuit. The old scripts detached a background
child via backtick-amp; P13 removes the detach and runs synchronously. These
tests verify:
- BLOG_DAILY_DRY_RUN=1 prints the dry-run line and exits 0 before any Python.
- No background ampersand remains in the script source (synchronous under cron).
- HERMES_HOME is respected (no hardcoded ~/.hermes source).
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    REPO_ROOT / "scripts" / "content" / "pr-to-blog-daily.sh",
    REPO_ROOT / "scripts" / "content" / "blog-stream-daily.sh",
    REPO_ROOT / "scripts" / "content" / "blog-backlog-pregen.sh",
]
EXPECTED_DRY_RUN_FRAGMENTS = {
    "pr-to-blog-daily.sh": "dry-run: would launch PR-to-blog pipeline",
    "blog-stream-daily.sh": "dry-run: would launch blog stream pipeline",
    "blog-backlog-pregen.sh": "dry-run: would launch backlog pregen",
}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_dry_run_exits_zero_and_prints_plan(tmp_path, script):
    """BLOG_DAILY_DRY_RUN=1 prints the plan and exits 0."""
    env = {"BLOG_DAILY_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True,
        env=env, cwd=str(REPO_ROOT), timeout=15,
    )
    assert r.returncode == 0, r.stderr
    frag = EXPECTED_DRY_RUN_FRAGMENTS[script.name]
    assert frag in r.stdout, f"{script.name}: expected fragment in stdout"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_dry_run_never_invokes_python(tmp_path, script):
    """Dry-run must exit before the cd content_engine + python3 invocation."""
    env = {"BLOG_DAILY_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True,
        env=env, cwd=str(REPO_ROOT), timeout=15,
    )
    assert r.returncode == 0, r.stderr
    assert "No such file" not in r.stderr
    assert "Traceback" not in r.stderr
    assert "ModuleNotFoundError" not in r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_no_background_detach(script):
    """No background ampersand remains — synchronous under cron."""
    text = script.read_text()
    bad_detach = re.search(r"\)\s*>>.*2>&1\s*<\s*/dev/null\s*&", text)
    assert not bad_detach, (
        f"{script.name}: still has background detach pattern"
    )
    assert ") >>" in text and "2>&1" in text, (
        f"{script.name}: missing synchronous log redirect block"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_respects_hermes_home(script):
    """The .env source must use HERMES_HOME, not hardcoded path."""
    text = script.read_text()
    assert "/home/kensei/.hermes" not in text, (
        f"{script.name}: hardcoded /home/kensei/.hermes path"
    )
    assert "HERMES_HOME" in text, (
        f"{script.name}: does not reference HERMES_HOME for .env source"
    )
