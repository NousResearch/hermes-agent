"""is_junction answers for any path, including one that does not exist yet."""
import os
import subprocess
from pathlib import Path

import pytest

from pm.filesystem import is_junction


@pytest.mark.platforms("any")
def test_missing_path_is_not_a_junction(tmp_path: Path) -> None:
    # A first plugin install probes plugins/<name> before anything is there.
    assert is_junction(tmp_path / "absent") is False
    assert is_junction(tmp_path / "absent" / "deeper") is False
    # POSIX stat results have no reparse tag at all.
    assert is_junction(tmp_path) is False


@pytest.mark.platforms("windows")
def test_real_junction_and_plain_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    command = str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe")
    result = subprocess.run(
        [command, "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        assert is_junction(junction) is True
        assert is_junction(target) is False
    finally:
        junction.rmdir()
