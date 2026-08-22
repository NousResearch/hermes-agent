"""Tests for the portable bash shebang checker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-bash-shebangs.py"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def test_full_repo_scan_is_clean():
    result = run_checker("--all")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_scan_catches_markdown_and_honors_suppression(tmp_path):
    old_shebang = "#!" + "/bin/bash\n"
    markdown = tmp_path / "optional-skills" / "example.md"
    markdown.parent.mkdir()
    markdown.write_text(f"```bash\n{old_shebang}```\n", encoding="utf-8")

    result = run_checker(str(markdown))
    assert result.returncode == 1
    assert "example.md:2" in result.stdout

    suppressed = tmp_path / "suppressed.md"
    suppressed.write_text(
        "#!" + "/bin/bash # shebang: ok fixture-only example\n",
        encoding="utf-8",
    )
    result = run_checker(str(suppressed))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
