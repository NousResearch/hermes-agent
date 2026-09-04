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


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_finds_decorator_and_pytestmark_forms(tmp_path, platform):
    """Both the decorator form and module-level ``pytestmark`` are detected."""
    _write(
        tmp_path,
        "test_decorated.py",
        f'import pytest\n\n\n@pytest.mark.platforms("{platform}")\ndef test_x():\n    pass\n',
    )
    _write(
        tmp_path,
        "nested/test_module_level.py",
        f'import pytest\n\npytestmark = pytest.mark.platforms("{platform}")\n\n\ndef test_y():\n    pass\n',
    )
    # A file with no gate at all must not be selected.
    _write(tmp_path, "test_plain.py", "def test_z():\n    pass\n")
    # A platforms() file gating a DIFFERENT platform must not be selected.
    other = "windows" if platform != "windows" else "linux"
    _write(
        tmp_path,
        "test_other_platform.py",
        f'import pytest\n\n\n@pytest.mark.platforms("{other}")\ndef test_w():\n    pass\n',
    )
    # A bare identifier that merely LOOKS like the platform name must not
    # match — the spec has to appear inside the quoted string literal.
    _write(
        tmp_path,
        "test_bare_identifier.py",
        f"import pytest\n\n\n@pytest.mark.parametrize(\"kind\", [\"{platform}\"])\ndef test_v():\n    assert kind\n",
    )

    result = _run(platform, str(tmp_path))

    assert result.returncode == 0, result.stderr
    listed = result.stdout.split()
    assert any(p.endswith("test_decorated.py") for p in listed)
    assert any(p.endswith("test_module_level.py") for p in listed)
    assert not any(p.endswith("test_plain.py") for p in listed)
    assert not any(p.endswith("test_other_platform.py") for p in listed)
    assert not any(p.endswith("test_bare_identifier.py") for p in listed)


def test_negated_spec_lists_for_that_lane(tmp_path):
    """``platforms("not macos")`` lists the file for the macOS lane import.

    The script only decides which files are IMPORTED; the per-test host
    skips remain authoritative. A negated-spec file may still contain other
    tests, so it must be imported on the macOS lane — under-selecting is
    the failure mode this tool exists to prevent.
    """
    _write(
        tmp_path,
        "test_negated.py",
        'import pytest\n\n\n@pytest.mark.platforms("not macos")\ndef test_x():\n    pass\n',
    )

    result = _run("macos", str(tmp_path))

    assert result.returncode == 0, result.stderr
    listed = result.stdout.split()
    assert any(p.endswith("test_negated.py") for p in listed)


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


def test_emits_repo_relative_posix_paths():
    """Output feeds a bash command line on the Windows runner, so separators
    must be POSIX and paths repo-relative.

    Asserted against the real ``tests/`` tree, which is the only case CI
    exercises — an out-of-repo root can't be made repo-relative and is
    emitted absolute instead.
    """
    result = _run("windows")

    assert result.returncode == 0, result.stderr
    listed = result.stdout.split()
    assert listed
    for line in listed:
        assert "\\" not in line
        assert not Path(line).is_absolute()


def test_real_tree_selects_files_for_every_platform():
    """Against the actual ``tests/`` tree each platform resolves to real files."""
    for platform in ("linux", "macos", "windows"):
        result = _run(platform)
        assert result.returncode == 0, (platform, result.stderr)
        assert result.stdout.split(), f"{platform} selected nothing in the real tree"
