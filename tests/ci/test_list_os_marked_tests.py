"""Tests for scripts/ci/list_os_marked_tests.py.

The properties this tool must keep are "finds real gates", "refuses to emit
nothing", and "rejects unknown platforms" — the macOS lane imports exactly
what it emits, so under-selection silently drops coverage while
over-selection is corrected by the per-test host skips.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "list_os_marked_tests.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )


def _write(root: Path, relpath: str, body: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("platform,expected", [
    ("linux", {"linux", "posix", "any", "not_macos", "not_linux", "unknown", "dynamic", "empty"}),
    ("macos", {"module", "single", "posix", "any", "not_macos", "not_linux", "unknown", "dynamic", "empty"}),
    ("windows", {"arch", "multiline", "any", "not_macos", "not_linux", "unknown", "dynamic", "empty"}),
])
def test_selector_corpus(tmp_path, platform, expected):
    # Selection deliberately overselects negated/unknown inputs; the runtime
    # marker, not this file selector, decides whether each test runs.
    calls = {
        "linux": '"linux"', "single": "'macos'", "posix": '"posix"',
        "any": '"any"', "not_macos": '"not macos"', "not_linux": '"not linux"',
        "arch": '"windows", arch="arm64"',
        "multiline": '\n"windows",\narch="arm64",\n',
        "unknown": '"amiga"', "dynamic": 'PLATFORM', "empty": '',
    }
    for name, arguments in calls.items():
        _write(tmp_path, f"test_{name}.py",
               f"@pytest.mark.platforms({arguments})\ndef test_x(): pass\n")
    _write(tmp_path, "nested/test_module.py", 'pytestmark = pytest.mark.platforms("macos")\n')
    _write(tmp_path, "test_plain.py", "def test_x(): pass\n")
    _write(tmp_path, "test_identifier.py", '@pytest.mark.parametrize("kind", ["linux", "macos", "windows"])\ndef test_x(): pass\n')
    result = _run(platform, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert {Path(p).stem.removeprefix("test_") for p in result.stdout.split()} == expected


def test_unknown_platform_is_rejected(tmp_path):
    result = _run("amiga", str(tmp_path))
    assert result.returncode == 2
    assert "unknown platform" in result.stderr


def test_exits_nonzero_when_no_file_gates_on_the_platform(tmp_path):
    """A platform with zero gated files must fail, not emit nothing."""
    _write(
        tmp_path,
        "test_unrelated.py",
        'import pytest\n\n\n@pytest.mark.platforms("linux")\ndef test_x():\n    pass\n',
    )

    result = _run("macos", str(tmp_path))

    # No genuine match: the helper must fail rather than emit nothing.
    assert result.returncode != 0
    assert "no test files" in result.stderr


def test_rejects_missing_root():
    result = _run("macos", "/nonexistent/path/for/this/test")
    assert result.returncode == 2
    assert "no such directory" in result.stderr


def test_real_tree_selects_files_for_every_platform():
    """Against the actual ``tests/`` tree each platform resolves to real files."""
    for platform in ("linux", "macos", "windows"):
        result = _run(platform)
        assert result.returncode == 0, (platform, result.stderr)
        assert result.stdout.split(), f"{platform} selected nothing in the real tree"
        for line in result.stdout.split():
            assert "\\" not in line
            assert not Path(line).is_absolute()
            assert (REPO_ROOT / line).is_file()

def test_unparseable_file_is_still_listed(tmp_path):
    """A file the selector cannot parse is conservatively listed so the lane
    fails visibly there rather than silently dropping that coverage."""
    _write(
        tmp_path,
        "test_gated.py",
        'import pytest\n\n\n@pytest.mark.platforms("macos")\ndef test_x():\n    pass\n',
    )
    _write(tmp_path, "test_broken.py", "def f(:\n")

    result = _run("macos", str(tmp_path))

    assert result.returncode == 0, result.stderr
    listed = result.stdout.split()
    assert any(p.endswith("test_broken.py") for p in listed)
    assert any(p.endswith("test_gated.py") for p in listed)
    # The problem is still reported, not swallowed.
    assert "test_broken.py" in result.stderr
