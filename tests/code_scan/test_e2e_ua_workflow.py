"""UA-008 Golden Fixtures and End-to-End UA Workflow Gate.

Strict TDD: RED phase tests verify golden fixtures are exercised through
the full UA pipeline.  GREEN phase: all tests pass against the fixtures.
FULL phase: complete code-scan test suite + real-repo smoke run.

Golden fixture classes:
    - tiny_py_package:  tiny Python package with imports and one entrypoint
    - mixed_docs_assets:  mixed docs/assets repo with expected orphans
    - ts_react_light:  lightweight TypeScript/React-style fixture
    - suspicious_isolated:  fixture with suspicious isolated source file
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_UA_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "run_ua.py"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "golden"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"

FIXTURE_NAMES = [
    "tiny_py_package",
    "mixed_docs_assets",
    "ts_react_light",
    "suspicious_isolated",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_ua(target_dir: str, bundle_dir: str, mode: str = "review",
           extra_args: list[str] | None = None) -> tuple[int, str, str]:
    """Run run_ua.py and return (rc, stdout, stderr)."""
    cmd = [
        sys.executable, str(RUN_UA_SCRIPT),
        "--target", target_dir,
        "--out", bundle_dir,
        "--mode", mode,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout, result.stderr


def load_json(path: Path) -> dict:
    assert path.exists(), f"{path} does not exist"
    return json.loads(path.read_text())


def fixture_abs(name: str) -> str:
    return str(GOLDEN_DIR / name)


# ── RED: Golden fixture existence ──────────────────────────────────────────────

class TestGoldenFixtureExistence:
    """RED phase: verify all required golden fixture directories exist
    with at least one source file."""

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_directory_exists(self, name: str):
        gd = GOLDEN_DIR / name
        assert gd.is_dir(), f"Golden fixture missing: {gd}"

    def test_tiny_py_package_has_source(self):
        """tiny_py_package must have at least one .py file with imports."""
        py_files = list((GOLDEN_DIR / "tiny_py_package").rglob("*.py"))
        assert len(py_files) >= 2, "tiny_py_package needs multiple .py files"
        # At least one file must have an import
        has_import = False
        for pf in py_files:
            if "import " in pf.read_text():
                has_import = True
                break
        assert has_import, "tiny_py_package must have at least one import"

    def test_mixed_docs_assets_has_orphan(self):
        """mixed_docs_assets must contain a docs dir, assets dir, and
        a Python file in an orphan subdirectory."""
        base = GOLDEN_DIR / "mixed_docs_assets"
        assert (base / "docs").is_dir()
        assert (base / "assets").is_dir()
        orphan_files = list((base / "orphan").rglob("*.py"))
        assert len(orphan_files) >= 1, "orphan dir must have a .py file"
        # The orphan file should import stdlib but not project modules
        for of in orphan_files:
            content = of.read_text()
            assert "import " in content, "orphan file should have imports"
            # Verify it doesn't import from the project (orphan)
            assert "from src" not in content, "orphan should not import project"

    def test_ts_react_light_has_typescript(self):
        """ts_react_light must contain .tsx/.ts files and package.json."""
        base = GOLDEN_DIR / "ts_react_light"
        assert (base / "package.json").is_file()
        ts_files = list(base.rglob("*.ts")) + list(base.rglob("*.tsx"))
        assert len(ts_files) >= 2, "TS fixture needs multiple TS/TSX files"

    def test_suspicious_isolated_has_single_source(self):
        """suspicious_isolated: exactly one .py source file, orphan status."""
        base = GOLDEN_DIR / "suspicious_isolated"
        py_files = list(base.rglob("*.py"))
        assert len(py_files) == 1, "suspicious_isolated must have exactly 1 .py"
        # File must have imports (stdlib) but no project imports
        content = py_files[0].read_text()
        assert "import " in content, "isolated file must have stdlib imports"


# ── RED: Scan succeeds on each golden fixture ─────────────────────────────────

class TestGoldenFixtureScan:
    """Each golden fixture must be scannable by run_ua.py inventory mode."""

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_inventory_scan_succeeds(self, name: str, tmp_path: Path):
        target = fixture_abs(name)
        out = str(tmp_path / "bundle")
        rc, stdout, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0, f"inventory scan failed for {name}: {stderr}"
        assert (Path(out) / "scan.json").exists()
        assert (Path(out) / "imports.json").exists()
        assert (Path(out) / "manifest.json").exists()


# ── RED: Review mode produces all expected artifacts on fixtures ───────────────

class TestGoldenFixtureReview:
    """Review mode must produce complete artifact set on each golden fixture.

    RED: analytics.json may be missing if analyser not present — this is the
    initial RED state. GREEN: all artifacts present.
    """

    REVIEW_ARTIFACTS = [
        "scan.json", "imports.json", "graph.json",
        "validation.json", "manifest.json", "summary.json",
    ]
    # These are optional — depend on upstream enrichers
    OPTIONAL_ARTIFACTS = [
        "analytics.json", "subagent-context.json", "REPORT.md",
    ]

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_review_required_artifacts(self, name: str, tmp_path: Path):
        target = fixture_abs(name)
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed for {name}: {stderr}"

        for fname in self.REVIEW_ARTIFACTS:
            p = Path(out) / fname
            assert p.exists(), f"Missing required artifact in {name}: {fname}"

    def test_review_optional_artifacts_present(self, tmp_path: Path):
        """When all enrichers are available, review must write optional artifacts.

        This test is RED until analytics and context bundler are both importable.
        """
        target = fixture_abs("tiny_py_package")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed: {stderr}"

        for fname in self.OPTIONAL_ARTIFACTS:
            p = Path(out) / fname
            assert p.exists(), (
                f"Optional artifact missing: {fname} — "
                "this is the RED condition; check enricher imports"
            )


# ── Manifest integrity ─────────────────────────────────────────────────────────

class TestManifestIntegrity:
    """Manifest must have required fields with correct types and values."""

    REQUIRED_FIELDS = [
        "run_id", "mode", "timestamp", "target_path",
        "bundle_dir", "artifact_paths", "script_versions",
        "target_mutation_allowed",
    ]

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_manifest_has_required_fields(self, name: str, tmp_path: Path):
        target = fixture_abs(name)
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed for {name}: {stderr}"

        manifest = load_json(Path(out) / "manifest.json")
        for field in self.REQUIRED_FIELDS:
            assert field in manifest, f"Manifest missing field: {field} in {name}"

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_manifest_run_id_present(self, name: str, tmp_path: Path):
        target = fixture_abs(name)
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="review")
        manifest = load_json(Path(out) / "manifest.json")
        assert isinstance(manifest["run_id"], str)
        assert len(manifest["run_id"]) > 0

    def test_manifest_artifacts_not_fabricated(self, tmp_path: Path):
        """Artifact paths in manifest must point to real files."""
        target = fixture_abs("tiny_py_package")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="inventory")
        assert rc == 0
        manifest = load_json(Path(out) / "manifest.json")
        for fname, fpath in manifest.get("artifact_paths", {}).items():
            assert Path(fpath).exists(), f"Artifact {fname} path does not exist: {fpath}"


# ── Validation severity summary ────────────────────────────────────────────────

class TestValidationSeveritySummary:
    """validation.json must contain issues and warnings arrays."""

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_validation_has_issues_and_warnings(self, name: str, tmp_path: Path):
        target = fixture_abs(name)
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed for {name}: {stderr}"

        val = load_json(Path(out) / "validation.json")
        assert "issues" in val, f"validation missing 'issues' in {name}"
        assert "warnings" in val, f"validation missing 'warnings' in {name}"
        assert isinstance(val["issues"], list)
        assert isinstance(val["warnings"], list)


# ── Context envelope ───────────────────────────────────────────────────────────

class TestContextEnvelope:
    """subagent-context.json must exist in review mode when enricher available."""

    def test_context_envelope_structure(self, tmp_path: Path):
        """Context envelope must have expected structural keys."""
        target = fixture_abs("tiny_py_package")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review")
        assert rc == 0, f"review failed: {stderr}"

        ctx_path = Path(out) / "subagent-context.json"
        if not ctx_path.exists():
            pytest.skip("Context binder not available (enricher missing)")

        ctx = load_json(ctx_path)
        assert "artifacts_included" in ctx or "artifacts" in ctx, (
            "context envelope must list included artifacts"
        )


# ── Read-only mode: no target mutation ─────────────────────────────────────────

class TestReadOnlyNoMutation:
    """Read-only mode must not create cache artifacts in the target."""

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_read_only_no_target_mutation(self, name: str, tmp_path: Path):
        """Running with --read-only-target must not write into target."""
        # Copy fixture to a temp dir so we can check for mutation
        src = fixture_abs(name)
        target = str(tmp_path / "target")
        shutil.copytree(src, target)

        # Snapshot target state before
        before_files = set()
        for root, _dirs, files in os.walk(target):
            for f in files:
                before_files.add(os.path.relpath(os.path.join(root, f), target))

        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="review",
                               extra_args=["--read-only-target"])
        assert rc == 0, f"read-only review failed for {name}: {stderr}"

        # Verify manifest reports target_mutation_allowed = False
        manifest = load_json(Path(out) / "manifest.json")
        assert manifest["target_mutation_allowed"] is False, (
            f"read-only mode: target_mutation_allowed should be False for {name}"
        )

        # Snapshot after
        after_files = set()
        for root, _dirs, files in os.walk(target):
            for f in files:
                after_files.add(os.path.relpath(os.path.join(root, f), target))

        # No new files in target
        new_files = after_files - before_files
        assert not new_files, (
            f"read-only mode created files in target for {name}: {new_files}"
        )


# ── Real-repo smoke: Hermes checkout read-only ─────────────────────────────────

class TestHermesCheckoutSmoke:
    """Smoke run against the Hermes repo itself in read-only mode."""

    def test_hermes_readonly_bundle(self, tmp_path: Path):
        """Run review mode against the Hermes checkout in read-only mode.

        Must produce manifest.json, validation.json, and NOT create
        .hermes/code-scan-cache in the target.
        """
        hermes_root = str(PROJECT_ROOT)
        out = str(tmp_path / "hermes-bundle")

        rc, _, stderr = run_ua(hermes_root, out, mode="review",
                               extra_args=["--read-only-target"])
        assert rc == 0, f"Hermes review smoke failed: {stderr}"

        # Artifact validation
        manifest = load_json(Path(out) / "manifest.json")
        assert (Path(out) / "validation.json").exists(), "validation.json missing"
        assert manifest.get("target_mutation_allowed") is False, manifest

        # No target-local UA cache
        target_cache = PROJECT_ROOT / ".hermes" / "code-scan-cache"
        assert not target_cache.exists(), "target-local UA cache created in Hermes root"

    def test_hermes_readonly_manifest_target_path(self, tmp_path: Path):
        """Manifest target_path must point to the Hermes checkout."""
        hermes_root = str(PROJECT_ROOT)
        out = str(tmp_path / "hermes-bundle2")
        run_ua(hermes_root, out, mode="inventory",
               extra_args=["--read-only-target"])
        manifest = load_json(Path(out) / "manifest.json")
        assert hermes_root in manifest.get("target_path", "")


# ── Full pipeline: all modes on tiny_py_package ───────────────────────────────

class TestFullModeMatrix:
    """Run all UA modes on the tiny_py_package fixture and verify artifacts."""

    MODES = ["inventory", "structure", "review", "preflight", "full"]

    @pytest.mark.parametrize("mode", MODES)
    def test_all_modes_succeed(self, mode: str, tmp_path: Path):
        target = fixture_abs("tiny_py_package")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"mode={mode} failed: {stderr}"

        manifest = load_json(Path(out) / "manifest.json")
        assert manifest["mode"] == mode
        assert "run_id" in manifest
        assert "artifact_paths" in manifest

    def test_delta_mode_with_prior(self, tmp_path: Path):
        """Delta mode should compare against a prior manifest."""
        target = fixture_abs("tiny_py_package")
        baseline = str(tmp_path / "baseline")
        rc1, _, stderr1 = run_ua(target, baseline, mode="structure")
        assert rc1 == 0, f"baseline failed: {stderr1}"

        delta_out = str(tmp_path / "delta")
        rc2, _, stderr2 = run_ua(
            target, delta_out, mode="delta",
            extra_args=["--prior-manifest", str(Path(baseline) / "manifest.json")],
        )
        assert rc2 == 0, f"delta failed: {stderr2}"

        manifest = load_json(Path(delta_out) / "manifest.json")
        assert manifest["mode"] == "delta"
        assert "delta_summary" in manifest
