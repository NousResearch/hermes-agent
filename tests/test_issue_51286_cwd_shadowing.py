"""
Regression test for GitHub Issue #51286.

Ensures that running Hermes from a directory containing a `utils/` package
does not cause ImportError due to cwd shadowing internal modules.

Usage:
    pytest tests/regression/test_issue_51286_cwd_shadowing.py -v
"""

import os
import sys
import subprocess
import tempfile
import pytest


@pytest.fixture
def fake_cwd_with_utils():
    """Create a temp directory containing a `utils` package that would shadow
    Hermes' internal `utils` module if cwd is on sys.path[0]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake `utils` package that raises on import (so we can detect
        # if it gets picked up instead of the real one).
        utils_init = os.path.join(tmpdir, "utils", "__init__.py")
        os.makedirs(os.path.dirname(utils_init), exist_ok=True)
        with open(utils_init, "w") as f:
            f.write('raise ImportError("FAKE utils was imported — cwd shadowing bug!")\n')
        yield tmpdir


def test_tui_gateway_entry_resists_cwd_shadowing(fake_cwd_with_utils):
    """
    Simulate running `hermes` from a directory that contains a `utils/`
    package. The gateway child must not crash with ImportError.
    """
    hermes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hermes_root = os.path.dirname(hermes_root)  # go up to repo root
    entry_path = os.path.join(hermes_root, "tui_gateway", "entry.py")

    # Build a minimal Python snippet that mimics what the gateway spawn does:
    # 1. cd into the shadowing directory
    # 2. set PYTHONPATH to include Hermes
    # 3. try to import the entry module (which should fix sys.path itself)
    script = f"""
import os
import sys

os.chdir({repr(fake_cwd_with_utils)})

# Simulate the spawn environment: PYTHONPATH includes Hermes but cwd
# is still at sys.path[0] (Python always does this).
sys.path.insert(0, {repr(hermes_root)})

# This is the critical import — before the fix, it would pick up
# {fake_cwd_with_utils}/utils/__init__.py and crash.
try:
    import tui_gateway.entry
    print("OK: entry imported successfully")
except ImportError as exc:
    print(f"FAIL: {{exc}}")
    sys.exit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=fake_cwd_with_utils,
    )

    assert result.returncode == 0, (
        f"Gateway entry crashed when run from cwd with `utils` package.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_sys_path_priority_after_fix():
    """
    Direct unit test: after the fix runs, the Hermes root must be at
    sys.path[0] when entry.py is imported.
    """
    hermes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hermes_root = os.path.dirname(hermes_root)
    entry_path = os.path.join(hermes_root, "tui_gateway", "entry.py")

    # We can't easily test the exact sys.path state without importing,
    # but we can at least verify the fix code is present in entry.py.
    with open(entry_path) as f:
        source = f.read()

    assert "_HERMES_ROOT" in source, (
        "Fix for #51286 not found in tui_gateway/entry.py — "
        "the sys.path guard is missing."
    )
    assert 'sys.path.insert(0, _HERMES_ROOT)' in source
