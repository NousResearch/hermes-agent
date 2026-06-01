"""Tests for UA-005 Explicit UA Mode Router.

Strict TDD: RED phase tests written first, then GREEN implementation.

RED: --mode flag unsupported before implementation.
GREEN: each mode emits expected artifact set and omits expected non-mode artifacts.
FULL: delta mode covered by focused pytest (requires prior manifest/baseline).
"""
import json
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
