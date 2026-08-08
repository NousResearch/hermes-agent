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


# ── D2: Incremental scan fixtures helper ──────────────────────────────────────

INCR_FIXTURE = FIXTURES_DIR / "incremental"


def _clean_incremental_fp(target_dir: Path):
    """Remove any leftover fingerprints.json from the fixture's .hermes dir."""
    fp_path = target_dir / ".hermes" / "code-state" / "fingerprints.json"
    if fp_path.exists():
        fp_path.unlink()
    # Also clean the .hermes dir if empty
    code_state = target_dir / ".hermes" / "code-state"
    hermes_dir = target_dir / ".hermes"
    if code_state.exists():
        code_state.rmdir()
    if hermes_dir.exists():
        hermes_dir.rmdir()


class TestIncrementalNoPriorFingerprints:
    """--incremental with no existing fingerprints behaves like a full scan."""

    def setup_method(self):
        _clean_incremental_fp(INCR_FIXTURE)

    def teardown_method(self):
        _clean_incremental_fp(INCR_FIXTURE)

    def test_incremental_no_fp_returns_zero(self):
        """--incremental should succeed even without prior fingerprints."""
        rc, stdout, stderr = run_scan(INCR_FIXTURE, ["--incremental"])
        assert rc == 0, f"scan exited {rc}: {stderr}"

    def test_incremental_no_fp_valid_json(self):
        """Output should be valid JSON with all required keys."""
        rc, stdout, stderr = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)
        for key in ("project_root", "total_files", "total_lines",
                     "languages", "categories", "frameworks", "files",
                     "warnings"):
            assert key in data, f"Missing key: {key}"

    def test_incremental_no_fp_creates_fingerprint_file(self):
        """A fingerprints.json file should be created after the scan."""
        run_scan(INCR_FIXTURE, ["--incremental"])
        fp_path = INCR_FIXTURE / ".hermes" / "code-state" / "fingerprints.json"
        assert fp_path.exists(), "fingerprints.json was not created"

    def test_incremental_no_fp_includes_warning(self):
        """Warnings array should mention incremental_scan when no prior fps."""
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)
        warnings_str = " ".join(data.get("warnings", []))
        assert "incremental_scan" in warnings_str, (
            f"Expected 'incremental_scan' in warnings, got: {data['warnings']}"
        )

    def test_incremental_no_fp_same_file_count_as_full(self):
        """File count must match an equivalent full scan."""
        rc_full, stdout_full, _ = run_scan(INCR_FIXTURE)
        rc_incr, stdout_incr, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        assert rc_full == rc_incr == 0
        full_data = json.loads(stdout_full)
        incr_data = json.loads(stdout_incr)
        assert incr_data["total_files"] == full_data["total_files"]
        assert incr_data["total_lines"] == full_data["total_lines"]


class TestIncrementalWithFingerprints:
    """--incremental with existing fingerprints file classifies files."""

    def setup_method(self):
        _clean_incremental_fp(INCR_FIXTURE)
        # Run an initial incremental scan to create fingerprints
        run_scan(INCR_FIXTURE, ["--incremental"])

    def teardown_method(self):
        _clean_incremental_fp(INCR_FIXTURE)

    def test_incremental_unchanged_returns_zero(self):
        """Second run with no changes should succeed."""
        rc, stdout, stderr = run_scan(INCR_FIXTURE, ["--incremental"])
        assert rc == 0, f"scan exited {rc}: {stderr}"

    def test_incremental_unchanged_has_warning(self):
        """Warnings should mention incremental_scan with counts."""
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)
        warnings_str = " ".join(data.get("warnings", []))
        assert "incremental_scan" in warnings_str

    def test_incremental_unchanged_all_files_present(self):
        """All current files must still appear in output."""
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)
        paths = {f["relative_path"] for f in data["files"]}
        assert "main.py" in paths
        assert "utils.py" in paths
        assert "config.json" in paths

    def test_incremental_excludes_generated_hermes_state(self):
        """Generated .hermes/code-state/fingerprints.json must NOT appear in scan output.

        This prevents the fingerprint file from polluting subsequent scans,
        which would violate fresh scan equivalence and make incremental unstable.
        """
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)
        paths = [f["relative_path"] for f in data["files"]]
        assert not any(p.startswith(".hermes/") for p in paths), (
            f".hermes/ files found in scan output: "
            f"{[p for p in paths if p.startswith('.hermes/')]}"
        )

    def test_incremental_stable_repeated_total_files(self):
        """Second incremental scan must match the initial full-scan file count.

        Bug repro: first incremental creates .hermes/code-state/fingerprints.json,
        which then gets scanned on the second incremental, inflating total_files.

        Strategy: clean state, get baseline full count, run two incrementals,
        both must match the baseline.
        """
        _clean_incremental_fp(INCR_FIXTURE)
        try:
            # Baseline full scan (no .hermes dir yet)
            rc_full, stdout_full, _ = run_scan(INCR_FIXTURE)
            full_total = json.loads(stdout_full)["total_files"]

            # First incremental — should match baseline
            rc1, stdout1, _ = run_scan(INCR_FIXTURE, ["--incremental"])
            assert json.loads(stdout1)["total_files"] == full_total

            # Second incremental — must STILL match baseline (this is the bug)
            rc2, stdout2, _ = run_scan(INCR_FIXTURE, ["--incremental"])
            incr_total = json.loads(stdout2)["total_files"]
            assert incr_total == full_total, (
                f"Second incremental total_files ({incr_total}) != "
                f"baseline full scan ({full_total}) — "
                f"generated .hermes/ state is leaking into scan"
            )
        finally:
            _clean_incremental_fp(INCR_FIXTURE)

    def test_incremental_output_equivalent_to_full(self):
        """Incremental output must be equivalent to full scan (same files/lines)."""
        rc_full, stdout_full, _ = run_scan(INCR_FIXTURE)
        rc_incr, stdout_incr, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        assert rc_full == rc_incr == 0
        full_data = json.loads(stdout_full)
        incr_data = json.loads(stdout_incr)
        assert incr_data["total_files"] == full_data["total_files"]
        assert incr_data["total_lines"] == full_data["total_lines"]
        full_paths = {f["relative_path"] for f in full_data["files"]}
        incr_paths = {f["relative_path"] for f in incr_data["files"]}
        assert full_paths == incr_paths

    def test_incremental_fingerprint_file_updated(self):
        """Fingerprints file should be updated after incremental scan."""
        fp_path = INCR_FIXTURE / ".hermes" / "code-state" / "fingerprints.json"
        mtime_before = fp_path.stat().st_mtime
        import time
        time.sleep(0.1)
        run_scan(INCR_FIXTURE, ["--incremental"])
        mtime_after = fp_path.stat().st_mtime
        assert mtime_after > mtime_before, "Fingerprint file not updated"

    def test_incremental_no_change_level_in_fingerprint(self):
        """Persisted fingerprints must NOT have change_level key."""
        run_scan(INCR_FIXTURE, ["--incremental"])
        fp_path = INCR_FIXTURE / ".hermes" / "code-state" / "fingerprints.json"
        data = json.loads(fp_path.read_text())
        for path, fp in data.get("files", {}).items():
            assert "change_level" not in fp, (
                f"change_level found in fingerprint for {path}"
            )


class TestIncrementalStructuralChange:
    """--incremental detects structural changes when files are modified."""

    def setup_method(self):
        _clean_incremental_fp(INCR_FIXTURE)
        run_scan(INCR_FIXTURE, ["--incremental"])

    def teardown_method(self):
        _clean_incremental_fp(INCR_FIXTURE)

    def test_incremental_structural_change_detected(self):
        """Adding a function to main.py should yield STRUCTURAL classification."""
        main_path = INCR_FIXTURE / "main.py"
        original = main_path.read_text()
        try:
            # Make a structural change - add a new function
            with open(main_path, "a") as f:
                f.write("\n\ndef subtract(a: int, b: int) -> int:\n"
                         "    return a - b\n")
            rc, stdout, stderr = run_scan(INCR_FIXTURE, ["--incremental"])
            assert rc == 0, f"scan exited {rc}: {stderr}"
            data = json.loads(stdout)
            warnings_str = " ".join(data.get("warnings", []))
            assert "STRUCTURAL" in warnings_str, (
                f"Expected STRUCTURAL in warnings, got: {warnings_str}"
            )
        finally:
            main_path.write_text(original)

    def test_incremental_structural_all_files_fresh(self):
        """Even after structural change, all current files must have fresh data."""
        main_path = INCR_FIXTURE / "main.py"
        original = main_path.read_text()
        try:
            with open(main_path, "a") as f:
                f.write("\n\ndef subtract(a: int, b: int) -> int:\n"
                         "    return a - b\n")
            rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
            data = json.loads(stdout)
            for f_record in data["files"]:
                assert "path" in f_record
                assert "relative_path" in f_record
                assert "language" in f_record
                assert "lines" in f_record
                assert f_record["lines"] > 0
        finally:
            main_path.write_text(original)

    def test_incremental_structural_has_metadata_with_paths(self):
        """After structural change, incremental_scan metadata must expose structured paths."""
        main_path = INCR_FIXTURE / "main.py"
        original = main_path.read_text()
        try:
            with open(main_path, "a") as f:
                f.write("\n\ndef subtract(a: int, b: int) -> int:\n"
                         "    return a - b\n")
            rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
            data = json.loads(stdout)

            meta = data.get("incremental_scan")
            assert meta is not None, "incremental_scan metadata missing from output"
            assert meta["mode"] == "incremental"
            assert "counts" in meta
            assert "paths" in meta

            # STRUCTURAL must include main.py
            assert meta["counts"]["STRUCTURAL"] >= 1
            assert "main.py" in meta["paths"]["STRUCTURAL"]

            # Unchanged files should appear in UNCHANGED paths
            assert meta["counts"]["UNCHANGED"] >= 2  # utils.py + config.json
        finally:
            main_path.write_text(original)

    def test_incremental_unchanged_has_metadata(self):
        """Incremental with no changes must expose incremental_scan metadata."""
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
        data = json.loads(stdout)

        meta = data.get("incremental_scan")
        assert meta is not None, "incremental_scan metadata missing"
        assert meta["mode"] == "incremental"
        assert "counts" in meta
        assert "paths" in meta
        # All files should be UNCHANGED
        assert meta["counts"]["UNCHANGED"] == data["total_files"]
        assert meta["counts"]["COSMETIC"] == 0
        assert meta["counts"]["STRUCTURAL"] == 0

    def test_incremental_no_fp_has_metadata(self):
        """Incremental with no prior fps must expose structured metadata."""
        _clean_incremental_fp(INCR_FIXTURE)
        try:
            rc, stdout, _ = run_scan(INCR_FIXTURE, ["--incremental"])
            data = json.loads(stdout)

            meta = data.get("incremental_scan")
            assert meta is not None, "incremental_scan metadata missing"
            assert meta["mode"] == "no_prior_fingerprints"
            assert "counts" in meta
            assert "paths" in meta
            # All files should be STRUCTURAL (scanned) for no_prior
            assert meta["counts"]["STRUCTURAL"] == data["total_files"]
            assert len(meta["paths"]["STRUCTURAL"]) == data["total_files"]
        finally:
            _clean_incremental_fp(INCR_FIXTURE)

    def test_incremental_full_scan_has_no_metadata(self):
        """Full scan must NOT contain incremental_scan metadata."""
        rc, stdout, _ = run_scan(INCR_FIXTURE)
        data = json.loads(stdout)
        assert "incremental_scan" not in data


class TestFullOverride:
    """--full forces full scan behavior regardless of fingerprints."""

    def setup_method(self):
        _clean_incremental_fp(INCR_FIXTURE)
        # Create fingerprints first
        run_scan(INCR_FIXTURE, ["--incremental"])

    def teardown_method(self):
        _clean_incremental_fp(INCR_FIXTURE)

    def test_full_with_incremental_ignores_fps(self):
        """--full --incremental together should behave like a full scan."""
        rc, stdout, _ = run_scan(
            INCR_FIXTURE, ["--full", "--incremental"]
        )
        assert rc == 0
        data = json.loads(stdout)
        warnings_str = " ".join(data.get("warnings", []))
        # Should NOT have incremental_scan warning when --full used
        assert "incremental_scan" not in warnings_str

    def test_full_only_forces_full_scan(self):
        """--full alone should behave identically to a normal full scan."""
        rc_full, stdout_full, _ = run_scan(INCR_FIXTURE)
        rc_full_flag, stdout_ff, _ = run_scan(INCR_FIXTURE, ["--full"])
        assert rc_full == rc_full_flag == 0
        full_data = json.loads(stdout_full)
        ff_data = json.loads(stdout_ff)
        assert ff_data["total_files"] == full_data["total_files"]
        assert ff_data["total_lines"] == full_data["total_lines"]

    def test_full_no_incremental_metadata(self):
        """--full output should not contain incremental_scan metadata."""
        rc, stdout, _ = run_scan(INCR_FIXTURE, ["--full"])
        data = json.loads(stdout)
        for w in data.get("warnings", []):
            assert "incremental_scan" not in w
