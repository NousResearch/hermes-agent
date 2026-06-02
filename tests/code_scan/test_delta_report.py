"""Tests for scripts/code-scan/delta_report.py.

Covers file set comparison, language/category/framework deltas,
fingerprint change classification, schema-version mismatch,
CLI argument parsing, deterministic/sorted output, and warning propagation.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/code-scan is importable
_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "code-scan"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import delta_report

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "delta_report"


def _load_json(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


# ── Unit tests for delta_report.compute_delta ─────────────────────────────

class TestComputeDeltaFileSets:
    """File set comparison: added, removed, common_count."""

    def test_identical_scans(self):
        old = _load_json("identical_scan_a.json")
        new = _load_json("identical_scan_b.json")
        result = delta_report.compute_delta(old, new)
        assert result["files"]["added"] == []
        assert result["files"]["removed"] == []
        assert result["files"]["common_count"] == 5

    def test_added_files(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        assert "src/new_module.py" in result["files"]["added"]
        assert "src/routes.ts" in result["files"]["added"]
        # helpers.py was in old but not in new (removed)
        assert "src/helpers.py" not in result["files"]["added"]

    def test_removed_files(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        assert "src/helpers.py" in result["files"]["removed"]
        assert len(result["files"]["removed"]) == 1

    def test_common_count(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        # 4 files are in both: main.py, utils.py, config.js, README.md
        assert result["files"]["common_count"] == 4


class TestComputeDeltaLanguages:
    """Language count deltas."""

    def test_language_delta(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        langs = result["languages"]
        # python: old=3, new=4, delta=1
        assert langs["python"]["old"] == 3
        assert langs["python"]["new"] == 4
        assert langs["python"]["delta"] == 1
        # typescript: not in old, new=1, delta=1
        assert langs["typescript"]["old"] == 0
        assert langs["typescript"]["new"] == 1
        assert langs["typescript"]["delta"] == 1
        # javascript: old=1, new=1, delta=0
        assert langs["javascript"]["delta"] == 0

    def test_category_delta(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        cats = result["categories"]
        # code: old=4, new=5, delta=1
        assert cats["code"]["old"] == 4
        assert cats["code"]["new"] == 5
        assert cats["code"]["delta"] == 1


class TestComputeDeltaFrameworks:
    """Framework comparison."""

    def test_framework_delta(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        fw = result["frameworks"]
        assert sorted(fw["added"]) == ["express"]
        assert fw["removed"] == []

    def test_framework_removed(self):
        old = _load_json("new_scan.json")
        new = _load_json("old_scan.json")
        result = delta_report.compute_delta(old, new)
        fw = result["frameworks"]
        assert sorted(fw["removed"]) == ["express"]
        assert fw["added"] == []


class TestComputeDeltaFingerprints:
    """Fingerprint change classification."""

    def test_fingerprint_classification(self):
        old_fp = _load_json("old_fingerprints.json")
        new_fp = _load_json("new_fingerprints.json")
        result = delta_report.compute_delta(
            _load_json("old_scan.json"),
            _load_json("new_scan.json"),
            old_fingerprints=old_fp,
            new_fingerprints=new_fp,
        )
        fp = result["fingerprints"]
        # src/utils.py: same hash -> UNCHANGED
        assert fp["UNCHANGED"] >= 1
        # src/main.py: different hash, same structural lists -> COSMETIC
        assert fp["COSMETIC"] >= 1
        # src/config.js: structural list changed + src/new_module.py added -> STRUCTURAL
        assert fp["STRUCTURAL"] >= 1

    def test_fingerprint_unchanged_only(self):
        old_fp = _load_json("old_fingerprints.json")
        # Same fingerprints everywhere
        result = delta_report.compute_delta(
            _load_json("old_scan.json"),
            _load_json("old_scan.json"),
            old_fingerprints=old_fp,
            new_fingerprints=old_fp,
        )
        fp = result["fingerprints"]
        assert fp["UNCHANGED"] == 3  # 3 files in old_fingerprints
        assert fp["COSMETIC"] == 0
        assert fp["STRUCTURAL"] == 0

    def test_no_fingerprints_provided(self):
        result = delta_report.compute_delta(
            _load_json("old_scan.json"),
            _load_json("new_scan.json"),
        )
        assert result["fingerprints"] is None


class TestComputeDeltaWarnings:
    """Warning propagation and generation."""

    def test_warnings_from_new_scan_propagated(self):
        new = _load_json("new_scan.json")
        old = _load_json("old_scan.json")
        result = delta_report.compute_delta(old, new)
        assert "Detected new framework: express" in result["warnings"]

    def test_identical_no_extra_warnings(self):
        old = _load_json("identical_scan_a.json")
        new = _load_json("identical_scan_b.json")
        result = delta_report.compute_delta(old, new)
        assert result["warnings"] == []


class TestSchemaVersionMismatch:
    """Schema version validation and warning."""

    def test_old_schema_mismatch_warning(self):
        old = _load_json("old_scan.json")
        old["schema_version"] = "0.9.0"
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        assert any("schema" in w.lower() and "old" in w.lower() for w in result["warnings"])

    def test_new_schema_mismatch_warning(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        new["schema_version"] = "2.0.0"
        result = delta_report.compute_delta(old, new)
        assert any("schema" in w.lower() and "new" in w.lower() for w in result["warnings"])


class TestDeterministicOutput:
    """Output stability: sorted keys, consistent results."""

    def test_stable_file_lists(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        r1 = delta_report.compute_delta(old, new)
        r2 = delta_report.compute_delta(old, new)
        assert r1 == r2

    def test_language_keys_sorted(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        lang_keys = list(result["languages"].keys())
        assert lang_keys == sorted(lang_keys)

    def test_category_keys_sorted(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        cat_keys = list(result["categories"].keys())
        assert cat_keys == sorted(cat_keys)


class TestSchemaVersionField:
    """Output schema_version is always present."""

    def test_output_has_schema_version(self):
        old = _load_json("old_scan.json")
        new = _load_json("new_scan.json")
        result = delta_report.compute_delta(old, new)
        assert result["schema_version"] == "1.0.0"


# ── CLI integration tests ─────────────────────────────────────────────────

class TestCLI:
    """End-to-end CLI invocation tests."""

    SCRIPT = str(_SCRIPT_DIR / "delta_report.py")

    def _run_cli(self, args: list[str], stdin_str: str | None = None) -> subprocess.CompletedProcess:
        cmd = [sys.executable, self.SCRIPT] + args
        return subprocess.run(cmd, capture_output=True, text=True, input=stdin_str)

    def test_cli_basic_comparison(self):
        old = str(_FIXTURE_DIR / "old_scan.json")
        new = str(_FIXTURE_DIR / "new_scan.json")
        proc = self._run_cli([old, new])
        assert proc.returncode == 0
        result = json.loads(proc.stdout)
        assert result["schema_version"] == "1.0.0"
        assert "files" in result
        assert "languages" in result
        assert "fingerprints" in result

    def test_cli_with_fingerprints(self):
        old_scan = str(_FIXTURE_DIR / "old_scan.json")
        new_scan = str(_FIXTURE_DIR / "new_scan.json")
        old_fp = str(_FIXTURE_DIR / "old_fingerprints.json")
        new_fp = str(_FIXTURE_DIR / "new_fingerprints.json")
        proc = self._run_cli([old_scan, new_scan, "--old-fingerprints", old_fp, "--new-fingerprints", new_fp])
        assert proc.returncode == 0
        result = json.loads(proc.stdout)
        assert result["fingerprints"] is not None
        assert "UNCHANGED" in result["fingerprints"]

    def test_cli_missing_file_error(self):
        proc = self._run_cli(["nonexistent_old.json", "nonexistent_new.json"])
        assert proc.returncode != 0
        assert proc.stderr.strip() != ""

    def test_cli_json_on_stdout_errors_on_stderr(self):
        old = str(_FIXTURE_DIR / "old_scan.json")
        new = str(_FIXTURE_DIR / "new_scan.json")
        proc = self._run_cli([old, new])
        # stdout must parse as valid JSON
        json.loads(proc.stdout)
        # stderr should be clean (no errors)
        assert proc.stderr.strip() == ""

    def test_cli_no_args_shows_usage(self):
        proc = self._run_cli([])
        assert proc.returncode != 0
        # Should print usage/help to stderr
        assert proc.stderr.strip() != ""
