"""Explicitly-named paths outside the checkout must scan, not crash.

``check-windows-footguns.py <path>`` is documented as "specific files/dirs to
scan", but any path outside ``REPO_ROOT`` used to raise an unhandled
``ValueError`` from ``Path.relative_to()`` — in ``should_scan_file()`` for the
EXCLUDED_FILES lookup, and again in ``main()`` when printing a match.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def linter():
    path = REPO_ROOT / "scripts/check-windows-footguns.py"
    spec = importlib.util.spec_from_file_location("outside_path_footguns", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/check-windows-footguns.py"), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO_ROOT),
    )


# ── should_scan_file ────────────────────────────────────────────────────────
def test_should_scan_outside_repo_python_file(linter, tmp_path):
    """The regression: this raised ValueError instead of returning a bool."""
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    assert linter.should_scan_file(outside) is True


def test_should_scan_outside_repo_non_python_file(linter, tmp_path):
    outside = tmp_path / "notes.txt"
    outside.write_text("hello\n", encoding="utf-8")
    assert linter.should_scan_file(outside) is False


def test_excluded_files_still_skipped_inside_repo(linter):
    """Control: the in-repo exclusion list must keep working."""
    assert linter.should_scan_file(REPO_ROOT / "scripts/check-windows-footguns.py") is False
    assert linter.should_scan_file(REPO_ROOT / "CONTRIBUTING.md") is False


def test_excluded_dirs_and_suffixes_still_apply_outside_repo(linter, tmp_path):
    nested = tmp_path / "node_modules" / "pkg"
    nested.mkdir(parents=True)
    py = nested / "index.py"
    py.write_text("x = 1\n", encoding="utf-8")
    assert linter.should_scan_file(py) is False  # node_modules pruned

    compiled = tmp_path / "compiled.pyc"
    compiled.write_bytes(b"\x00")
    assert linter.should_scan_file(compiled) is False


# ── display_path ────────────────────────────────────────────────────────────
def test_display_path_relative_inside_repo(linter):
    assert linter.display_path(REPO_ROOT / "cli.py") == "cli.py"


def test_display_path_absolute_outside_repo(linter, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    assert linter.display_path(outside) == str(outside)


# ── end-to-end through the CLI ──────────────────────────────────────────────
def test_cli_does_not_crash_on_outside_path(tmp_path):
    outside = tmp_path / "clean.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    result = run_cli(str(outside))
    assert "Traceback" not in result.stderr, result.stderr
    assert "ValueError" not in result.stderr, result.stderr
    assert result.returncode == 0
    assert "1 file(s) scanned" in result.stdout


def test_cli_scans_outside_directory(tmp_path):
    """A directory arg must walk and scan its files, not bail out."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    result = run_cli(str(tmp_path))
    assert result.returncode == 0
    assert "2 file(s) scanned" in result.stdout


def test_cli_reports_violation_in_outside_file(tmp_path):
    """The strongest case: scanning still WORKS, and the match is reported.

    This exercises the second crash site — the ``relative_to`` in main()'s
    match-printing loop — and proves out-of-repo paths are still scanned rather
    than being silently skipped to dodge the exception.
    """
    outside = tmp_path / "bad.py"
    outside.write_text(
        "import os\n"
        "def alive(pid):\n"
        "    os.kill(pid, 0)\n"  # windows-footgun: ok — fixture payload text, never executed
        "    return True\n",
        encoding="utf-8",
    )
    result = run_cli(str(outside))
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 1, "a footgun in an outside file must fail the run"
    assert f"{outside}:3" in result.stdout
    assert "[os.kill(pid, 0)]" in result.stdout  # windows-footgun: ok — asserting on linter output
