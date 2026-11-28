"""A filesystem root passed as a scan target must be refused, not walked.

``check-windows-footguns.py <path>`` documents its positional argument as
"specific files/dirs to scan". A bare anchor (``/``, ``C:\\``) is neither: it
is an entire drive, so ``iter_files()``'s ``os.walk`` would walk the whole
filesystem. The run is refused up front with an error naming the cause and a
distinct exit status, instead of appearing to hang.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# A bare anchor for the platform running this test.
ROOT = Path(Path.cwd().anchor)


@pytest.fixture(scope="module")
def linter():
    path = REPO_ROOT / "scripts/check-windows-footguns.py"
    spec = importlib.util.spec_from_file_location("filesystem_root_footguns", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/check-windows-footguns.py"), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO_ROOT),
        timeout=timeout,
    )


@pytest.fixture
def repo_scratch():
    """A scratch dir inside the checkout.

    Paths *outside* the repo are the subject of a separate PR; these controls
    only care that the anchor guard does not reject an ordinary directory.
    """
    path = Path(tempfile.mkdtemp(dir=REPO_ROOT / "tests", prefix="tmp-footgun-root-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ── is_filesystem_root ──────────────────────────────────────────────────────
def test_anchor_is_a_filesystem_root(linter):
    assert linter.is_filesystem_root(ROOT) is True


def test_ordinary_dirs_and_files_are_not_roots(linter, tmp_path):
    assert linter.is_filesystem_root(tmp_path) is False
    assert linter.is_filesystem_root(REPO_ROOT) is False
    assert linter.is_filesystem_root(tmp_path / "a.py") is False


def test_drive_relative_path_is_not_a_root(linter):
    """``C:foo`` keeps its anchor but names a subdirectory of the drive."""
    assert linter.is_filesystem_root(Path("C:foo")) is False


@pytest.mark.skipif(os.name != "nt", reason="drive anchors only exist on Windows")
def test_windows_drive_anchor_is_a_root(linter):
    assert linter.is_filesystem_root(Path("C:\\")) is True
    assert linter.is_filesystem_root(Path("C:\\Users")) is False


# ── iter_files ──────────────────────────────────────────────────────────────
def test_iter_files_refuses_a_root(linter):
    """The regression: this used to start an os.walk of the whole filesystem."""
    with pytest.raises(linter.FilesystemRootError) as excinfo:
        list(linter.iter_files([ROOT]))
    message = str(excinfo.value)
    assert str(ROOT) in message
    assert "filesystem root" in message
    # Names the cause and the way out, rather than a bare ValueError.
    assert "subdirectory" in message


def test_filesystem_root_error_is_a_value_error(linter):
    assert issubclass(linter.FilesystemRootError, ValueError)


def test_iter_files_still_walks_ordinary_dirs(linter, repo_scratch):
    """Control: the guard must not reject real scan targets."""
    (repo_scratch / "sub").mkdir()
    (repo_scratch / "sub" / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert [p.name for p in linter.iter_files([repo_scratch])] == ["a.py"]


# ── end-to-end through the CLI ──────────────────────────────────────────────
def test_cli_refuses_root_without_crashing():
    result = run_cli(str(ROOT))
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 2, result.stdout + result.stderr
    assert "filesystem root" in result.stderr
    assert "file(s) scanned" not in result.stdout


def test_cli_refuses_root_quickly():
    """A walk of a real root would not finish inside the timeout."""
    try:
        result = run_cli(str(ROOT), timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - only on regression
        pytest.fail("scanning a filesystem root did not return; the guard is gone")
    assert result.returncode == 2


def test_cli_still_scans_an_ordinary_directory(repo_scratch):
    """Control: the refusal is scoped to anchors."""
    (repo_scratch / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = run_cli(str(repo_scratch))
    assert result.returncode == 0, result.stderr
    assert "1 file(s) scanned" in result.stdout
