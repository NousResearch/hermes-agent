"""Tests for the UA-001 canonical run bundle and read-only target cache.

Strict TDD: tests written first (RED), then implementation (GREEN).
"""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "run_bundle.py"
SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "scan_project.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"


def _script_hash(script_path: Path) -> str:
    """Compute sha256 hex digest of a script file."""
    h = hashlib.sha256()
    with open(script_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run_bundle(target_dir: Path, bundle_dir: Path, extra_args=None):
    """Run run_bundle.py and return (returncode, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(BUNDLE_SCRIPT),
        str(target_dir),
        str(bundle_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _make_temp_target(tmp_path: Path) -> Path:
    """Create a minimal target project for scanning."""
    target = tmp_path / "target_project"
    target.mkdir()
    (target / "main.py").write_text(
        textwrap.dedent("""\
            import os
            import sys

            def main():
                print("hello")

            if __name__ == "__main__":
                main()
        """)
    )
    (target / "utils.py").write_text(
        textwrap.dedent("""\
            def helper():
                pass
        """)
    )
    # Initialise a git repo so we can check HEAD
    subprocess.run(
        ["git", "init"], cwd=target,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=target, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=target, capture_output=True,
    )
    subprocess.run(
        ["git", "add", "."], cwd=target, capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=target,
        capture_output=True,
    )
    return target


# ── Module import test ────────────────────────────────────────────────

class TestRunBundleImport:
    """Verify the run_bundle module can be imported."""

    def test_imports(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "code-scan"))
        from run_bundle import (
            RunBundle,
            run_bundle_pipeline,
            main,
        )


# ── Canonical bundle artifacts ────────────────────────────────────────

class TestCanonicalBundleArtifacts:
    """A run bundle must produce all required artifacts."""

    REQUIRED_FILES = [
        "scan.json",
        "imports.json",
        "graph.json",
        "validation.json",
        "summary.json",
        "manifest.json",
        "REPORT.md",
    ]

    def test_bundle_creates_all_artifacts(self, tmp_path: Path):
        """Default run must create scan.json, imports.json, graph.json,
        validation.json, summary.json, manifest.json, and REPORT.md."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"

        rc, stdout, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        assert bundle_dir.exists()

        for fname in self.REQUIRED_FILES:
            fpath = bundle_dir / fname
            assert fpath.exists(), f"Missing required artifact: {fname}"

    def test_scan_json_valid(self, tmp_path: Path):
        """scan.json must contain valid scan output."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        data = json.loads((bundle_dir / "scan.json").read_text())
        assert "files" in data
        assert "total_files" in data
        assert data["total_files"] > 0

    def test_imports_json_valid(self, tmp_path: Path):
        """imports.json must contain valid import map."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        data = json.loads((bundle_dir / "imports.json").read_text())
        assert "files" in data
        assert "schema_version" in data

    def test_graph_json_valid(self, tmp_path: Path):
        """graph.json must contain valid graph with nodes and edges."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        data = json.loads((bundle_dir / "graph.json").read_text())
        assert "nodes" in data
        assert "edges" in data
        assert "summary" in data

    def test_validation_json_valid(self, tmp_path: Path):
        """validation.json must contain validation results."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        data = json.loads((bundle_dir / "validation.json").read_text())
        assert "issues" in data
        assert "warnings" in data

    def test_summary_json_valid(self, tmp_path: Path):
        """summary.json must contain a summary dict."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        data = json.loads((bundle_dir / "summary.json").read_text())
        assert isinstance(data, dict)
        assert "target" in data

    def test_report_md_exists_and_nonempty(self, tmp_path: Path):
        """REPORT.md must exist and have content."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)
        report = bundle_dir / "REPORT.md"
        assert report.exists()
        assert report.stat().st_size > 0


# ── Manifest shape ────────────────────────────────────────────────────

class TestManifestShape:
    """manifest.json must record required fields."""

    def _load_manifest(self, tmp_path: Path, extra_args=None) -> dict:
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, stdout, stderr = run_bundle(target, bundle_dir, extra_args=extra_args)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        return json.loads((bundle_dir / "manifest.json").read_text())

    def test_manifest_has_run_id(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "run_id" in data
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) > 0

    def test_manifest_has_timestamp(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_manifest_has_target_path(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "target_path" in data
        assert isinstance(data["target_path"], str)
        assert os.path.isabs(data["target_path"])

    def test_manifest_has_git_head(self, tmp_path: Path):
        """When target has a git repo, manifest must record HEAD."""
        data = self._load_manifest(tmp_path)
        # _make_temp_target creates a git repo with a commit
        assert "target_git_head" in data
        assert data["target_git_head"] is not None
        assert len(data["target_git_head"]) == 40  # SHA-1 hex

    def test_manifest_has_command_flags(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "command_flags" in data
        assert isinstance(data["command_flags"], dict)

    def test_manifest_has_artifact_paths(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "artifact_paths" in data
        assert isinstance(data["artifact_paths"], dict)
        assert "scan.json" in data["artifact_paths"]

    def test_manifest_artifact_paths_complete_default_mode(self, tmp_path: Path):
        """manifest['artifact_paths'] must include ALL required artifacts
        in default (graph) mode, including manifest.json and REPORT.md."""
        data = self._load_manifest(tmp_path)
        ap = data["artifact_paths"]
        required = [
            "scan.json",
            "imports.json",
            "graph.json",
            "validation.json",
            "summary.json",
            "manifest.json",
            "REPORT.md",
        ]
        for name in required:
            assert name in ap, (
                f"artifact_paths missing required entry '{name}'; "
                f"found keys: {sorted(ap.keys())}"
            )

    def test_manifest_has_script_versions(self, tmp_path: Path):
        data = self._load_manifest(tmp_path)
        assert "script_versions" in data
        assert isinstance(data["script_versions"], dict)
        # Should have hashes for the pipeline scripts
        assert "scan_project.py" in data["script_versions"]

    def test_manifest_has_target_mutation_allowed(self, tmp_path: Path):
        """Manifest must record whether target mutation was allowed."""
        data = self._load_manifest(tmp_path)
        assert "target_mutation_allowed" in data
        assert isinstance(data["target_mutation_allowed"], bool)
        # Default must be False (non-mutating)
        assert data["target_mutation_allowed"] is False

    def test_manifest_mutation_allowed_true_with_flag(self, tmp_path: Path):
        """--in-repo-cache sets target_mutation_allowed to True."""
        data = self._load_manifest(tmp_path, extra_args=["--in-repo-cache"])
        assert data["target_mutation_allowed"] is True


# ── Read-only target cache (UA-001 core invariant) ───────────────────

class TestReadOnlyTargetCache:
    """Default external assessment must be non-mutating."""

    def test_default_no_files_created_in_target(self, tmp_path: Path):
        """Default run must NOT create any new files in the target directory."""
        target = _make_temp_target(tmp_path)

        # Snapshot target files before
        before_files = set()
        for root, dirs, files in os.walk(target):
            for f in files:
                abs_f = os.path.join(root, f)
                before_files.add(os.path.relpath(abs_f, target))
            # Also capture dirs to detect new directories
            for d in dirs:
                abs_d = os.path.join(root, d)
                before_files.add(os.path.relpath(abs_d, target) + "/")

        bundle_dir = tmp_path / "bundle"
        rc, stdout, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"

        # Snapshot after
        after_files = set()
        for root, dirs, files in os.walk(target):
            for f in files:
                abs_f = os.path.join(root, f)
                after_files.add(os.path.relpath(abs_f, target))
            for d in dirs:
                abs_d = os.path.join(root, d)
                after_files.add(os.path.relpath(abs_d, target) + "/")

        new_files = after_files - before_files
        assert new_files == set(), (
            f"Default run created files in target: {new_files}"
        )

    def test_default_no_hermes_dir_in_target(self, tmp_path: Path):
        """Default run must NOT create .hermes/ directory in target."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir)

        hermes_dir = target / ".hermes"
        assert not hermes_dir.exists(), (
            "Default run created .hermes/ in target repo"
        )

    def test_in_repo_cache_creates_fingerprints_in_target(self, tmp_path: Path):
        """--in-repo-cache flag should allow writing fingerprints to target."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, stdout, stderr = run_bundle(
            target, bundle_dir, extra_args=["--in-repo-cache"]
        )
        assert rc == 0, f"bundle exited {rc}: {stderr}"

        fp_path = target / ".hermes" / "code-state" / "fingerprints.json"
        assert fp_path.exists(), (
            "--in-repo-cache did not create fingerprints in target"
        )


# ── Graph mode vs non-graph ───────────────────────────────────────────

class TestBundleModes:
    """Testing --no-graph and other mode flags."""

    def test_no_graph_skips_graph_artifacts(self, tmp_path: Path):
        """--no-graph must skip graph.json and validation.json generation."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, _, _ = run_bundle(target, bundle_dir, extra_args=["--no-graph"])
        assert rc == 0

        # Core artifacts still present
        assert (bundle_dir / "scan.json").exists()
        assert (bundle_dir / "imports.json").exists()
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "REPORT.md").exists()

        # Graph artifacts not created
        assert not (bundle_dir / "graph.json").exists()
        assert not (bundle_dir / "validation.json").exists()

    def test_no_graph_manifest_reflects_flag(self, tmp_path: Path):
        """manifest command_flags must record --no-graph."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        run_bundle(target, bundle_dir, extra_args=["--no-graph"])
        data = json.loads((bundle_dir / "manifest.json").read_text())
        assert data["command_flags"].get("no_graph") is True


# ── Non-git target ────────────────────────────────────────────────────

class TestNonGitTarget:
    """Bundles work even when target has no git repository."""

    def test_bundle_without_git_target(self, tmp_path: Path):
        """Target without git repo should still produce valid bundle."""
        target = tmp_path / "no_git_target"
        target.mkdir()
        (target / "main.py").write_text("x = 1\n")

        bundle_dir = tmp_path / "bundle"
        rc, stdout, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        assert (bundle_dir / "manifest.json").exists()

        data = json.loads((bundle_dir / "manifest.json").read_text())
        # git_head should be None for non-git targets
        assert data["target_git_head"] is None


# ── UA-P1-002: Target Cleanliness Hardening ───────────────────────────

CLEANLINESS_FIELDS = [
    "target_dirty_before",
    "target_dirty_after",
    "target_dirty_files_before",
    "target_dirty_files_after",
    "unexpected_target_changes",
    "target_cleanliness_status",
]


class TestManifestCleanlinessFields:
    """UA-P1-002: Manifest must include target cleanliness fields."""

    def _load_manifest(self, tmp_path: Path, extra_args=None) -> dict:
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, stdout, stderr = run_bundle(target, bundle_dir, extra_args=extra_args)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        return json.loads((bundle_dir / "manifest.json").read_text())

    @pytest.mark.parametrize("field", CLEANLINESS_FIELDS)
    def test_manifest_has_cleanliness_field(self, tmp_path: Path, field: str):
        """Manifest must include each required cleanliness field."""
        data = self._load_manifest(tmp_path)
        assert field in data, f"Missing required cleanliness field: {field}"

    def test_clean_target_status(self, tmp_path: Path):
        """A clean target (all committed) should report 'clean' status."""
        data = self._load_manifest(tmp_path)
        assert data["target_dirty_before"] is False
        assert data["target_dirty_after"] is False
        assert data["target_dirty_files_before"] == []
        assert data["target_dirty_files_after"] == []
        assert data["target_cleanliness_status"] == "clean"
        assert data["unexpected_target_changes"] == []

    def test_dirty_target_status(self, tmp_path: Path):
        """A target with uncommitted changes should report 'preexisting_dirty'."""
        target = _make_temp_target(tmp_path)
        # Introduce a dirty change
        (target / "main.py").write_text("import os\nprint('dirty')\n")
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        data = json.loads((bundle_dir / "manifest.json").read_text())
        assert data["target_dirty_before"] is True, "Should detect pre-existing dirty files"
        assert data["target_dirty_files_before"] != [], "Should list dirty files"
        assert data["target_cleanliness_status"] == "preexisting_dirty"

    def test_dirty_files_are_strings(self, tmp_path: Path):
        """Dirty file lists should contain strings."""
        target = _make_temp_target(tmp_path)
        (target / "main.py").write_text("import os\nprint('dirty')\n")
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        data = json.loads((bundle_dir / "manifest.json").read_text())
        for f in data["target_dirty_files_before"]:
            assert isinstance(f, str), f"Dirty file entry must be string: {f}"


class TestSuccessfulManifestStatus:
    """UA-P1-002: Successful runs must include status: complete."""

    def test_successful_run_has_status_complete(self, tmp_path: Path):
        """A successful run bundle must have status=complete in manifest."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        data = json.loads((bundle_dir / "manifest.json").read_text())
        assert "status" in data, "Manifest must include 'status' field"
        assert data["status"] == "complete"


class TestNoTargetLocalCacheDirs:
    """UA-P1-002: Default external-cache scans must not create target-local dirs."""

    def test_no_hermes_code_state_in_target(self, tmp_path: Path):
        """Default run must NOT create .hermes/code-state in target."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        assert not (target / ".hermes" / "code-state").exists(), (
            "Default run created .hermes/code-state in target"
        )

    def test_no_hermes_code_scan_cache_in_target(self, tmp_path: Path):
        """Default run must NOT create .hermes/code-scan-cache in target."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        assert not (target / ".hermes" / "code-scan-cache").exists(), (
            "Default run created .hermes/code-scan-cache in target"
        )

    def test_no_ua_dir_in_target(self, tmp_path: Path):
        """Default run must NOT create .ua in target."""
        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"
        rc, _, stderr = run_bundle(target, bundle_dir)
        assert rc == 0, f"bundle exited {rc}: {stderr}"
        assert not (target / ".ua").exists(), (
            "Default run created .ua in target"
        )


class TestPartialFailureManifest:
    """UA-P1-002: Partial pipeline failures must still write a useful manifest."""

    def test_failure_manifest_has_status_failed(self, tmp_path: Path):
        """When a pipeline stage fails, manifest must include status: failed."""
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root / "scripts" / "code-scan"))
        from run_bundle import RunBundle

        target = _make_temp_target(tmp_path)
        bundle_dir = tmp_path / "bundle"

        bundle = RunBundle(
            str(target), str(bundle_dir),
            no_graph=False, in_repo_cache=False,
        )

        # Simulate a scan failure by patching _scan
        def _failing_scan():
            raise RuntimeError("simulated scan failure")
        bundle._scan = _failing_scan

        with pytest.raises(RuntimeError, match="simulated scan failure"):
            bundle.run()

        manifest_path = bundle_dir / "manifest.json"
        assert manifest_path.exists(), "Failure manifest was not written"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["status"] == "failed"
        assert data["failure_stage"] == "scan"
        assert "simulated scan failure" in data["error_message"]
        # Cleanliness fields should still be present
        assert "target_dirty_before" in data
        assert "target_dirty_after" in data
        assert "target_cleanliness_status" in data
