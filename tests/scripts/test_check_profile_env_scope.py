"""Behavioral tests for the profile-env-scope CI guard.

The guard must go red when a profile-varying env var is read raw, green on a
clean tree, and honor the # scope-exempt opt-out. It backstops the whole
cross-profile-env-leak fix: without it, the next raw os.getenv reopens the leak.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_profile_env_scope.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True, text=True, check=False,
    )


def _plant(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_clean_tree_passes(tmp_path):
    _plant(tmp_path, "tools/ok.py", "import os\nx = os.getenv('SOME_OTHER_VAR')\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_raw_terminal_env_fails(tmp_path):
    _plant(tmp_path, "tools/bad.py", "import os\nx = os.getenv('TERMINAL_ENV', 'local')\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "TERMINAL_ENV" in result.stdout
    assert "tools/bad.py:2" in result.stdout


def test_raw_write_safe_root_subscript_fails(tmp_path):
    _plant(tmp_path, "agent/bad.py", "import os\nx = os.environ['HERMES_WRITE_SAFE_ROOT']\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "HERMES_WRITE_SAFE_ROOT" in result.stdout


def test_environ_get_policy_var_fails(tmp_path):
    _plant(tmp_path, "agent/bad.py", "import os\nx = os.environ.get('HERMES_ACCEPT_HOOKS', '')\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "HERMES_ACCEPT_HOOKS" in result.stdout


def test_scope_exempt_marker_allows(tmp_path):
    _plant(tmp_path, "tools/exempt.py",
           "import os\nx = os.getenv('TERMINAL_CWD', '')  # scope-exempt: ImportError fallback\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_unrelated_var_ignored(tmp_path):
    _plant(tmp_path, "tools/ok.py", "import os\nx = os.getenv('PATH')\n")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_real_tree_is_clean():
    # The actual repo must always pass - this is the wired CI invocation.
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
