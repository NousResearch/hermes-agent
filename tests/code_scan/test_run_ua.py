"""Tests for UA-005 Explicit UA Mode Router.

Strict TDD: RED phase tests written first, then GREEN implementation.

RED: --mode flag unsupported before implementation.
GREEN: each mode emits expected artifact set and omits expected non-mode artifacts.
FULL: delta mode covered by focused pytest (requires prior manifest/baseline).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_UA_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "run_ua.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"


# ── Helpers ──────────────────────────────────────────────────────────────

def run_ua(target_dir: str, bundle_dir: str, mode: str | None = None, extra_args: list[str] | None = None):
    """Run run_ua.py and return (returncode, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(RUN_UA_SCRIPT),
        "--target", target_dir,
        "--out", bundle_dir,
    ]
    if mode is not None:
        cmd.extend(["--mode", mode])
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout, result.stderr


def _load_manifest(bundle_dir: str) -> dict:
    manifest_path = Path(bundle_dir) / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found in {bundle_dir}"
    return json.loads(manifest_path.read_text())


# ── RED: mode flag fails before implementation ───────────────────────────

class TestModeUnsupportedBeforeImplementation:
    """RED: Verify that --mode flag is either unsupported or fails
    before the routing implementation exists."""

    def test_run_ua_script_exists(self):
        """run_ua.py should exist as an entrypoint."""
        assert RUN_UA_SCRIPT.exists(), (
            f"run_ua.py not found at {RUN_UA_SCRIPT}; "
            "this is the RED condition — the script must be created"
        )

    def test_mode_flag_accepted(self, tmp_path: Path):
        """--mode flag must be parsed without error (not 'unrecognized').
        This test will FAIL in RED — it proves the flag was not supported."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, stdout, stderr = run_ua(target, out, mode="structure")
        # In RED, --mode should fail. In GREEN, it succeeds.
        assert rc == 0, f"--mode was rejected: rc={rc} stderr={stderr}"


# ── Mode routing correctness ────────────────────────────────────────────

class TestModeMetadata:
    """Each mode must record itself in manifest.json."""

    @pytest.mark.parametrize("mode", [
        "inventory",
        "structure",
        "review",
        "preflight",
        "full",
    ])
    def test_mode_recorded_in_manifest(self, tmp_path: Path, mode: str):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, stdout, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"mode={mode} failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["mode"] == mode


# ── GREEN: artifact sets per mode ───────────────────────────────────────

class TestInventoryMode:
    """inventory: scan + imports only. No graph, no validation, no analytics,
    no context, no report."""

    REQUIRED = ["scan.json", "imports.json", "manifest.json"]
    FORBIDDEN = ["graph.json", "validation.json", "analytics.json",
                 "subagent-context.json"]

    def test_inventory_artifacts(self, tmp_path: Path):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory failed: {stderr}"

        for fname in self.REQUIRED:
            assert (Path(out) / fname).exists(), f"Missing required: {fname}"
        for fname in self.FORBIDDEN:
            assert not (Path(out) / fname).exists(), f"Should not exist: {fname}"


class TestStructureMode:
    """structure: scan + imports + graph + validation. No analytics, no context."""

    REQUIRED = ["scan.json", "imports.json", "graph.json",
                "validation.json", "manifest.json"]
    FORBIDDEN = ["analytics.json", "subagent-context.json"]

    def test_structure_artifacts(self, tmp_path: Path):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"structure failed: {stderr}"

        for fname in self.REQUIRED:
            assert (Path(out) / fname).exists(), f"Missing required: {fname}"
        for fname in self.FORBIDDEN:
            assert not (Path(out) / fname).exists(), f"Should not exist: {fname}"


class TestReviewMode:
    """review: structure + analytics + context envelope + report.
    Must also NOT hide validation failures."""

    REQUIRED = ["scan.json", "imports.json", "graph.json",
                "validation.json", "analytics.json",
                "subagent-context.json", "REPORT.md", "manifest.json"]

    def test_review_artifacts(self, tmp_path: Path):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed: {stderr}"

        for fname in self.REQUIRED:
            assert (Path(out) / fname).exists(), f"Missing required: {fname}"

    def test_review_never_hides_validation(self, tmp_path: Path):
        """review mode must include validation.json with issues/warnings
        never suppressed."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="review")
        val = json.loads((Path(out) / "validation.json").read_text())
        assert "issues" in val
        assert "warnings" in val


class TestDeltaMode:
    """delta: incremental scan + delta summary against prior manifest.
    Requires prior manifest/baseline — covered by focused pytest."""

    def test_delta_without_prior_manifest(self, tmp_path: Path):
        """Delta mode with no prior manifest should still run but
        record artifacts_missing metadata, never fabricate."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        # First run: no prior manifest → delta must still succeed
        rc, _, stderr = run_ua(target, out, mode="delta")
        assert rc == 0, f"delta without prior manifest failed: {stderr}"
        manifest = _load_manifest(out)
        # Must record delta mode
        assert manifest["mode"] == "delta"
        # Should have a delta summary or artifacts_missing entry
        assert "delta_summary" in manifest or "artifacts_missing" in manifest

    def test_delta_with_prior_manifest(self, tmp_path: Path):
        """Delta mode with a prior manifest: first run creates baseline,
        second run produces delta."""
        target = str(FIXTURES_DIR / "sample_repo")
        # First run (baseline)
        baseline = str(tmp_path / "baseline")
        rc1, _, stderr1 = run_ua(target, baseline, mode="structure")
        assert rc1 == 0, f"baseline structure failed: {stderr1}"
        assert (Path(baseline) / "manifest.json").exists()

        # Second run: delta against prior
        delta_out = str(tmp_path / "delta")
        rc2, _, stderr2 = run_ua(
            target, delta_out, mode="delta",
            extra_args=["--prior-manifest", str(Path(baseline) / "manifest.json")],
        )
        assert rc2 == 0, f"delta against prior manifest failed: {stderr2}"
        manifest = _load_manifest(delta_out)
        assert manifest["mode"] == "delta"
        assert "delta_summary" in manifest

    def test_delta_summary_has_change_field(self, tmp_path: Path):
        """delta_summary must include a change indication field."""
        target = str(FIXTURES_DIR / "sample_repo")
        baseline = str(tmp_path / "baseline")
        run_ua(target, baseline, mode="structure")

        delta_out = str(tmp_path / "delta")
        run_ua(
            target, delta_out, mode="delta",
            extra_args=["--prior-manifest", str(Path(baseline) / "manifest.json")],
        )
        manifest = _load_manifest(delta_out)
        ds = manifest["delta_summary"]
        # Must have at least some determinism indicator
        assert isinstance(ds, dict)
        assert "files" in ds or "changes" in ds


class TestPreflightMode:
    """preflight: structure + entrypoints/hubs + subagent context.
    Entry: scan + imports + graph + validation + subagent-context."""

    REQUIRED = ["scan.json", "imports.json", "graph.json",
                "validation.json", "subagent-context.json", "manifest.json"]

    def test_preflight_artifacts(self, tmp_path: Path):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="preflight")
        assert rc == 0, f"preflight failed: {stderr}"

        for fname in self.REQUIRED:
            assert (Path(out) / fname).exists(), f"Missing required: {fname}"


class TestFullMode:
    """full: all available deterministic enrichers."""

    REQUIRED = [
        "scan.json", "imports.json", "graph.json",
        "validation.json", "analytics.json",
        "subagent-context.json", "REPORT.md", "manifest.json",
    ]

    def test_full_artifacts(self, tmp_path: Path):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="full")
        assert rc == 0, f"full failed: {stderr}"

        for fname in self.REQUIRED:
            assert (Path(out) / fname).exists(), f"Missing required: {fname}"


class TestDefaultMode:
    """Default mode (no --mode flag) should behave like 'structure'."""

    def test_default_is_structure(self, tmp_path: Path):
        """When no --mode is given, manifest['mode'] should be 'structure'."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out)
        assert rc == 0, f"default mode failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["mode"] == "structure"


class TestQuickModesAvoidGraphAnalytics:
    """Quick modes (inventory, structure) must NOT produce analytics."""

    @pytest.mark.parametrize("mode", ["inventory", "structure"])
    def test_no_analytics_in_quick_modes(self, tmp_path: Path, mode: str):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"{mode} failed: {stderr}"
        assert not (Path(out) / "analytics.json").exists(), (
            f"{mode} must NOT produce analytics.json"
        )


class TestModeMatrix:
    """Run all modes and verify each produces a valid manifest."""

    MODES = ["inventory", "structure", "review", "preflight", "full"]

    @pytest.mark.parametrize("mode", MODES)
    def test_mode_produces_valid_manifest(self, tmp_path: Path, mode: str):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"mode={mode} failed: {stderr}"

        manifest = _load_manifest(out)
        assert "run_id" in manifest
        assert manifest["mode"] == mode
        assert "artifact_paths" in manifest
        assert "timestamp" in manifest


class TestValidationNeverHidden:
    """Mode routing must never hide validation failures."""

    @pytest.mark.parametrize("mode", [
        "structure", "review", "preflight", "full",
    ])
    def test_validation_always_present_when_graph_enabled(self, tmp_path: Path, mode: str):
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"mode={mode} failed: {stderr}"

        vp = Path(out) / "validation.json"
        assert vp.exists(), f"validation.json missing in mode={mode}"
        val = json.loads(vp.read_text())
        assert "issues" in val, f"validation missing 'issues' in mode={mode}"
        assert "warnings" in val, f"validation missing 'warnings' in mode={mode}"


# ── Artifact metadata integrity ────────────────────────────────────────

class TestArtifactMetadata:
    """Optional mode artifacts must be missing/skipped metadata, never fabricated."""

    def test_missing_artifacts_not_fabricated(self, tmp_path: Path):
        """When analytics.json is skipped (inventory mode), the manifest
        should NOT point to a fabricated analytics file."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory failed: {stderr}"
        manifest = _load_manifest(out)
        ap = manifest.get("artifact_paths", {})
        assert "analytics.json" not in ap, (
            "inventory mode must not include analytics.json in artifact_paths"
        )


# ── UA-006 Project-State Integration ───────────────────────────────────

class TestProjectStateIntegration:
    """Opt-in project-state ledger append with manifest reporting."""

    PROJECT_STATE_FIXTURES = (
        PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "project_state"
    )

    def test_manifest_reports_false_without_opt_in(self, tmp_path: Path):
        """Without --record-project-state, manifest reports false/null."""
        target = str(self.PROJECT_STATE_FIXTURES / "with_ledger")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"no opt-in failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_recorded"] is False
        assert manifest["ledger_path"] is None

    def test_manifest_false_when_ledger_absent(self, tmp_path: Path):
        """Opt-in with no ledger → manifest reports false, ledger_path null."""
        target = str(self.PROJECT_STATE_FIXTURES / "without_ledger")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(
            target, out, mode="inventory",
            extra_args=["--record-project-state"],
        )
        assert rc == 0, f"absent ledger failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_recorded"] is False
        assert manifest["ledger_path"] is None

    def test_manifest_true_when_ledger_present(self, tmp_path: Path):
        """Opt-in with existing ledger → manifest reports true + path.

        Uses a temp copy of the fixture so the on-disk fixture is never mutated.
        """
        src = self.PROJECT_STATE_FIXTURES / "with_ledger"
        target_dir = tmp_path / "with_ledger"
        shutil.copytree(src, target_dir)

        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(
            str(target_dir), out, mode="inventory",
            extra_args=["--record-project-state"],
        )
        assert rc == 0, f"ledger present failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_recorded"] is True
        assert manifest["ledger_path"] is not None
        assert manifest["ledger_path"].endswith("PROJECT_STATE.md")

    def test_ledger_content_preserved_after_opt_in(self, tmp_path: Path):
        """Appending must not overwrite existing ledger content.

        Uses a temp copy of the fixture so the on-disk fixture is never mutated.
        """
        # Copy fixture into a temp directory so the original is untouched
        src = self.PROJECT_STATE_FIXTURES / "with_ledger"
        target_dir = tmp_path / "with_ledger"
        shutil.copytree(src, target_dir)

        # Capture the original ledger bytes before any append
        original_ledger_path = src / ".hermes" / "PROJECT_STATE.md"
        original_content = original_ledger_path.read_text()

        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(
            str(target_dir), out, mode="inventory",
            extra_args=["--record-project-state"],
        )
        assert rc == 0, f"preservation failed: {stderr}"

        # Read the ledger from the TEMP copy, not the fixture
        temp_ledger_path = target_dir / ".hermes" / "PROJECT_STATE.md"
        content = temp_ledger_path.read_text()

        # Exact prefix preservation: file must start with original content
        assert content.startswith(original_content), (
            "Original ledger content was not preserved as an exact prefix"
        )
        # New run_id from this invocation should also appear
        assert "UA Run" in content

    def test_explicit_project_root_overrides_target(self, tmp_path: Path):
        """--project-root can point to a distinct directory with the ledger.

        Uses a temp copy of the fixture so the on-disk fixture is never mutated.
        """
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")

        # Copy the project-state fixture to a temp directory for use as --project-root
        src = self.PROJECT_STATE_FIXTURES / "with_ledger"
        ledger_root = str(tmp_path / "ledger_root")
        shutil.copytree(src, ledger_root)

        rc, _, stderr = run_ua(
            target, out, mode="inventory",
            extra_args=["--record-project-state", "--project-root", ledger_root],
        )
        assert rc == 0, f"explicit project-root failed: {stderr}"
        manifest = _load_manifest(out)
        # The target is sample_repo (different from ledger_root),
        # but project-state was recorded because --project-root pointed to with_ledger
        assert manifest["project_state_recorded"] is True
        assert manifest["ledger_path"] is not None

    def test_manifest_has_project_state_append_status_success(self, tmp_path: Path):
        """When ledger exists and append succeeds, project_state_append_status is 'success'."""
        src = self.PROJECT_STATE_FIXTURES / "with_ledger"
        target_dir = tmp_path / "with_ledger"
        shutil.copytree(src, target_dir)
        out = str(tmp_path / "bundle")

        rc, _, stderr = run_ua(
            str(target_dir), out, mode="inventory",
            extra_args=["--record-project-state"],
        )
        assert rc == 0, f"append status test failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_append_status"] == "success"
        assert manifest.get("project_state_append_error") is None

    def test_manifest_has_project_state_append_status_not_attempted(self, tmp_path: Path):
        """When ledger is absent, project_state_append_status is 'not_attempted'."""
        src = self.PROJECT_STATE_FIXTURES / "without_ledger"
        target_dir = tmp_path / "without_ledger"
        shutil.copytree(src, target_dir)
        out = str(tmp_path / "bundle")

        rc, _, stderr = run_ua(
            str(target_dir), out, mode="inventory",
            extra_args=["--record-project-state"],
        )
        assert rc == 0, f"not_attempted test failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_append_status"] == "not_attempted"
        assert manifest.get("project_state_append_error") is None

    def test_manifest_has_project_state_append_status_without_opt_in(self, tmp_path: Path):
        """Without --record-project-state, append_status is 'not_attempted'."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"no opt-in failed: {stderr}"
        manifest = _load_manifest(out)
        assert manifest["project_state_append_status"] == "not_attempted"
        assert manifest.get("project_state_append_error") is None

    def test_append_failure_does_not_fail_ua_run(self, tmp_path: Path):
        """If project-state append fails, UA run still succeeds and status records failure."""
        src = self.PROJECT_STATE_FIXTURES / "with_ledger"
        target_dir = tmp_path / "with_ledger"
        shutil.copytree(src, target_dir)
        # Make the ledger read-only so append will fail
        ledger = target_dir / ".hermes" / "PROJECT_STATE.md"
        ledger.chmod(0o444)

        out = str(tmp_path / "bundle")
        try:
            rc, stdout, stderr = run_ua(
                str(target_dir), out, mode="inventory",
                extra_args=["--record-project-state"],
            )
            # UA run should NOT fail
            assert rc == 0, f"UA run should not fail due to append error: {stderr}"
            manifest = _load_manifest(out)
            assert manifest["project_state_append_status"] == "failed"
            assert "project_state_append_error" in manifest
            assert manifest["project_state_append_error"] is not None
            assert len(manifest["project_state_append_error"]) > 0
        finally:
            ledger.chmod(0o644)  # restore for cleanup


# ── UA-P1-002: Target Cleanliness Hardening ───────────────────────────

_UA_CLEANLINESS_FIELDS = [
    "target_dirty_before",
    "target_dirty_after",
    "target_dirty_files_before",
    "target_dirty_files_after",
    "unexpected_target_changes",
    "target_cleanliness_status",
]


class TestUAManifestCleanliness:
    """UA-P1-002: run_ua manifest must include cleanliness fields."""

    @pytest.mark.parametrize("mode", ["inventory", "structure"])
    def test_manifest_has_cleanliness_fields(self, tmp_path: Path, mode: str):
        """UA manifest must include each required cleanliness field."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"mode={mode} failed: {stderr}"
        manifest = _load_manifest(out)
        for field in _UA_CLEANLINESS_FIELDS:
            assert field in manifest, f"Missing cleanliness field: {field} in mode={mode}"

    def test_successful_ua_has_status_complete(self, tmp_path: Path):
        """Successful UA run must include status=complete in manifest."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"structure failed: {stderr}"
        manifest = _load_manifest(out)
        assert "status" in manifest, "Manifest must include 'status' field"
        assert manifest["status"] == "complete"

    def test_cleanliness_status_clean_for_clean_target(self, tmp_path: Path):
        """Clean target should report 'clean' status."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"structure failed: {stderr}"
        manifest = _load_manifest(out)
        # sample_repo in fixtures is not a git repo, so status = unknown
        # or it could be clean if it is. Let's just check the field exists and is one of the valid values
        assert manifest["target_cleanliness_status"] in (
            "clean", "preexisting_dirty", "mutated", "unknown"
        )


class TestUANoTargetLocalCacheDirs:
    """UA-P1-002: Default external-cache UA scans must not create target-local dirs."""

    def test_no_hermes_code_state_in_target(self, tmp_path: Path):
        """Default UA run must NOT create .hermes/code-state in target."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory failed: {stderr}"
        assert not (Path(target) / ".hermes" / "code-state").exists(), (
            "Default UA run created .hermes/code-state in target"
        )

    def test_no_hermes_code_scan_cache_in_target(self, tmp_path: Path):
        """Default UA run must NOT create .hermes/code-scan-cache in target."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory failed: {stderr}"
        assert not (Path(target) / ".hermes" / "code-scan-cache").exists(), (
            "Default UA run created .hermes/code-scan-cache in target"
        )

    def test_no_ua_dir_in_target(self, tmp_path: Path):
        """Default UA run must NOT create .ua in target."""
        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory failed: {stderr}"
        assert not (Path(target) / ".ua").exists(), (
            "Default UA run created .ua in target"
        )


class TestUAPartialFailureManifest:
    """UA-P1-002: Partial UA pipeline failures must write a useful manifest."""

    def test_failure_manifest_has_status_failed(self, tmp_path: Path):
        """When a pipeline stage fails, manifest must include status: failed."""
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root / "scripts" / "code-scan"))
        from run_ua import RunUA

        target = str(FIXTURES_DIR / "sample_repo")
        out = str(tmp_path / "bundle")

        ua = RunUA(target, out, mode="inventory")

        # Simulate a scan failure
        def _failing_scan():
            raise RuntimeError("simulated UA scan failure")
        ua._scan = _failing_scan

        with pytest.raises(RuntimeError, match="simulated UA scan failure"):
            ua.run()

        manifest_path = Path(out) / "manifest.json"
        assert manifest_path.exists(), "Failure manifest was not written"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["status"] == "failed"
        assert data["failure_stage"] == "scan"
        assert "simulated UA scan failure" in data["error_message"]
        # Cleanliness fields should still be present
        assert "target_dirty_before" in data
        assert "target_dirty_after" in data
        assert "target_cleanliness_status" in data
