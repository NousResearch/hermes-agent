"""Tests for scripts/code-scan/scan_project.py."""
import json
import os
import subprocess
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "scan_project.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"


def run_scan(target_dir, extra_args=None):
    """Run scan_project.py and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(SCAN_SCRIPT), str(target_dir)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


class TestScanSmallProject:
    """Test scanning the small_project fixture."""

    def test_scan_small_project_returns_zero(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        assert rc == 0, f"scan exited {rc}: {stderr}"

    def test_scan_small_project_valid_json(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        assert "project_root" in data
        assert "scanned_at" in data
        assert "total_files" in data
        assert "total_lines" in data
        assert "languages" in data
        assert "categories" in data
        assert "frameworks" in data
        assert "files" in data
        assert "warnings" in data

    def test_scan_small_project_file_count(self):
        """small_project has exactly 4 files: main.py, utils.py, test_main.py, pyproject.toml, README.md = 5."""
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        assert data["total_files"] == 5

    def test_scan_small_project_languages(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        assert "python" in data["languages"]
        assert "toml" in data["languages"]
        assert "markdown" in data["languages"]

    def test_scan_small_project_categories(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        assert "code" in data["categories"]
        assert "config" in data["categories"] or "infra" in data["categories"]
        assert "doc" in data["categories"]

    def test_scan_small_project_relative_paths(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        for f in data["files"]:
            assert f["relative_path"] is not None
            assert not f["relative_path"].startswith("/")
            assert f["path"].endswith(f["relative_path"])

    def test_scan_small_project_file_records_complete(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "small_project")
        data = json.loads(stdout)
        for f in data["files"]:
            assert "path" in f
            assert "relative_path" in f
            assert "language" in f
            assert "category" in f
            assert "lines" in f
            assert "size_bytes" in f
            assert isinstance(f["lines"], int)
            assert isinstance(f["size_bytes"], int)
            assert f["lines"] > 0


class TestScanMixedProject:
    """Test scanning the mixed_project fixture."""

    def test_scan_mixed_project_detects_typescript(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "mixed_project")
        assert rc == 0, f"scan exited {rc}: {stderr}"
        data = json.loads(stdout)
        assert "typescript" in data["languages"]

    def test_scan_mixed_project_detects_javascript(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "mixed_project")
        data = json.loads(stdout)
        assert "javascript" in data["languages"]

    def test_scan_mixed_project_frameworks(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "mixed_project")
        data = json.loads(stdout)
        assert "react" in data["frameworks"]
        assert "nextjs" in data["frameworks"]

    def test_scan_mixed_project_respects_local_hermesignore(self):
        """mixed_project has .hermesignore with vendor/ - vendor dir should not appear in files."""
        # Create a vendor directory to verify it's excluded
        vendor_dir = FIXTURES_DIR / "mixed_project" / "vendor"
        vendor_dir.mkdir(exist_ok=True)
        (vendor_dir / "lib.js").write_text("module.exports = {};")
        try:
            rc, stdout, stderr = run_scan(FIXTURES_DIR / "mixed_project")
            data = json.loads(stdout)
            paths = [f["relative_path"] for f in data["files"]]
            assert not any(p.startswith("vendor") for p in paths)
        finally:
            (vendor_dir / "lib.js").unlink()
            vendor_dir.rmdir()


class TestScanIgnoredProject:
    """Test that excluded directories are not scanned."""

    def test_ignored_project_excludes_node_modules(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "ignored_project")
        assert rc == 0, f"scan exited {rc}: {stderr}"
        data = json.loads(stdout)
        paths = [f["relative_path"] for f in data["files"]]
        assert not any(p.startswith("node_modules") for p in paths)

    def test_ignored_project_includes_src(self):
        rc, stdout, stderr = run_scan(FIXTURES_DIR / "ignored_project")
        data = json.loads(stdout)
        paths = [f["relative_path"] for f in data["files"]]
        assert any(p.startswith("src") for p in paths)


class TestScanOutputModes:
    """Test --output flag and stdout mode."""

    def test_output_flag_writes_file(self, tmp_path):
        output_file = tmp_path / "scan-result.json"
        rc, stdout, stderr = run_scan(
            FIXTURES_DIR / "small_project",
            ["--output", str(output_file)],
        )
        assert rc == 0, f"scan exited {rc}: {stderr}"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["total_files"] > 0

    def test_output_flag_no_stdout(self, tmp_path):
        output_file = tmp_path / "scan-result.json"
        rc, stdout, stderr = run_scan(
            FIXTURES_DIR / "small_project",
            ["--output", str(output_file)],
        )
        # When --output is given, stdout should be empty
        assert stdout.strip() == ""


class TestScanErrors:
    """Test error handling."""

    def test_nonexistent_path(self):
        rc, stdout, stderr = run_scan("/nonexistent/path/xyz123")
        assert rc != 0
        assert stderr.strip() != ""

    def test_path_is_file_not_dir(self, tmp_path):
        test_file = tmp_path / "not_a_dir.txt"
        test_file.write_text("hello")
        rc, stdout, stderr = run_scan(str(test_file))
        assert rc != 0

    def test_verbose_flag(self):
        rc, stdout, stderr = run_scan(
            FIXTURES_DIR / "small_project", ["--verbose"]
        )
        assert rc == 0
