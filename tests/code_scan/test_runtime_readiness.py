"""Tests for UA-P1-003 Runtime Readiness Artifact.

runtime_readiness module is implemented and importable.
runtime-readiness.json is produced for Go, Python, and unknown fixtures
with correct shape, verification_status, and blockers.
run_ua.py and run_bundle.py both emit runtime readiness artifacts.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"
RUNTIME_FIXTURES = FIXTURES_DIR / "runtime_readiness"
RUN_UA_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "run_ua.py"
RUN_BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "run_bundle.py"


# ── Helpers ──────────────────────────────────────────────────────────────

def run_ua(target_dir: str, bundle_dir: str, mode: str = "structure",
           extra_args: list[str] | None = None):
    """Run run_ua.py and return (returncode, stdout, stderr)."""
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


def run_bundle(target_dir: str, bundle_dir: str,
               extra_args: list[str] | None = None):
    """Run run_bundle.py and return (returncode, stdout, stderr)."""
    cmd = [
        sys.executable, str(RUN_BUNDLE_SCRIPT),
        target_dir, bundle_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout, result.stderr


def _load_readiness_json(bundle_dir: str) -> dict:
    """Load runtime-readiness.json from the bundle directory."""
    path = Path(bundle_dir) / "runtime-readiness.json"
    assert path.exists(), f"runtime-readiness.json not found in {bundle_dir}"
    return json.loads(path.read_text())


def _load_readiness_md(bundle_dir: str) -> str:
    """Load runtime-readiness.md from the bundle directory."""
    path = Path(bundle_dir) / "runtime-readiness.md"
    assert path.exists(), f"runtime-readiness.md not found in {bundle_dir}"
    return path.read_text()


def _load_manifest(bundle_dir: str) -> dict:
    """Load manifest.json from the bundle directory."""
    path = Path(bundle_dir) / "manifest.json"
    assert path.exists(), f"manifest.json not found in {bundle_dir}"
    return json.loads(path.read_text())


# ── Module is importable (GREEN) ──────────────────────────────────────────

class TestReadinessModuleImportable:
    """runtime_readiness.py exists and is importable."""

    def test_runtime_readiness_module_exists(self):
        """After implementation (GREEN), scripts/code-scan/runtime_readiness.py
        exists and is importable."""
        mod_path = PROJECT_ROOT / "scripts" / "code-scan" / "runtime_readiness.py"
        assert mod_path.exists(), (
            "runtime_readiness.py is missing — implementation should exist"
        )


# ── GREEN: Go fixture ────────────────────────────────────────────────────

class TestGoFixtureReadiness:
    """Go fixture: go.mod is present, go binary likely absent."""

    def test_go_readiness_json_exists(self, tmp_path: Path):
        """run_ua structure mode must emit runtime-readiness.json for Go fixture."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"run_ua failed on go fixture: {stderr}"
        assert (Path(out) / "runtime-readiness.json").exists()

    def test_go_readiness_json_shape(self, tmp_path: Path):
        """runtime-readiness.json for Go fixture must have minimal shape."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        # Required top-level keys
        assert "detected_stacks" in data
        assert "required_commands" in data
        assert "suggested_verification" in data
        assert "verification_status" in data
        assert "blockers" in data
        # Stack detection
        assert "go" in data["detected_stacks"]
        # required_commands shape
        assert len(data["required_commands"]) > 0
        cmd_entry = data["required_commands"][0]
        assert "command" in cmd_entry
        assert "available" in cmd_entry
        assert "version" in cmd_entry
        assert "reason" in cmd_entry

    def test_go_verification_blocked_when_go_missing(self, tmp_path: Path):
        """If go binary is not available, verification_status = verification_blocked."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        # Go binary is unlikely to be installed on the test machine
        go_cmd = next(
            (c for c in data["required_commands"] if c["command"] == "go"),
            None,
        )
        if go_cmd and not go_cmd["available"]:
            assert data["verification_status"] == "verification_blocked"
            assert len(data["blockers"]) > 0
            assert any("go" in str(b).lower() for b in data["blockers"])

    def test_go_suggests_go_test(self, tmp_path: Path):
        """Go readiness must suggest `go test -short ./...`."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        assert any("go test" in s for s in data["suggested_verification"])

    def test_go_readiness_md_exists(self, tmp_path: Path):
        """run_ua must also emit runtime-readiness.md for Go fixture."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        md = _load_readiness_md(out)
        assert len(md) > 0
        assert "go" in md.lower() or "Go" in md


# ── GREEN: Python fixture ────────────────────────────────────────────────

class TestPythonFixtureReadiness:
    """Python fixture: pyproject.toml is present, python binary is available."""

    def test_python_readiness_json_exists(self, tmp_path: Path):
        """run_ua structure mode must emit runtime-readiness.json for Python fixture."""
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"run_ua failed on python fixture: {stderr}"
        assert (Path(out) / "runtime-readiness.json").exists()

    def test_python_readiness_json_shape(self, tmp_path: Path):
        """runtime-readiness.json for Python fixture must have minimal shape."""
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        assert "detected_stacks" in data
        assert "python" in data["detected_stacks"]
        assert "required_commands" in data
        assert "suggested_verification" in data
        assert "verification_status" in data
        assert "blockers" in data

    def test_python_ready_when_python_available(self, tmp_path: Path):
        """Python is almost certainly available → verification_ready."""
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        py_cmd = next(
            (c for c in data["required_commands"] if c["command"] == "python"),
            None,
        )
        if py_cmd and py_cmd["available"]:
            assert data["verification_status"] == "verification_ready"

    def test_python_suggests_pytest(self, tmp_path: Path):
        """Python readiness must suggest `python -m pytest` when tests exist."""
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        assert any("pytest" in s.lower() for s in data["suggested_verification"])


# ── GREEN: Unknown/minimal fixture ───────────────────────────────────────

class TestUnknownFixtureReadiness:
    """Unknown fixture: no recognized stack markers."""

    def test_unknown_readiness_json_shape(self, tmp_path: Path):
        """run_ua must still emit a valid runtime-readiness.json for unknown projects."""
        target = str(RUNTIME_FIXTURES / "unknown_project")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode="structure")
        assert rc == 0, f"run_ua failed on unknown fixture: {stderr}"
        data = _load_readiness_json(out)
        assert "detected_stacks" in data
        assert isinstance(data["detected_stacks"], list)
        assert "required_commands" in data
        assert "verification_status" in data
        # No stacks detected → unknown status
        if not data["detected_stacks"]:
            assert data["verification_status"] == "unknown"


# ── GREEN: all run_ua modes emit readiness ───────────────────────────────

class TestAllModesEmitReadiness:
    """All run_ua modes (structure, review, preflight, full) must emit readiness."""

    @pytest.mark.parametrize("mode", [
        "structure", "review", "preflight", "full",
    ])
    def test_mode_emits_readiness_json(self, tmp_path: Path, mode: str):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(target, out, mode=mode)
        assert rc == 0, f"run_ua mode={mode} failed: {stderr}"
        assert (Path(out) / "runtime-readiness.json").exists(), (
            f"runtime-readiness.json missing in mode={mode}"
        )
        assert (Path(out) / "runtime-readiness.md").exists(), (
            f"runtime-readiness.md missing in mode={mode}"
        )


# ── GREEN: manifest artifact_paths registration ──────────────────────────

class TestManifestArtifactPathsReadiness:
    """manifest.json artifact_paths must include runtime-readiness.json/.md."""

    def test_ua_manifest_includes_readiness_files(self, tmp_path: Path):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        manifest = _load_manifest(out)
        ap = manifest.get("artifact_paths", {})
        assert "runtime-readiness.json" in ap, (
            f"artifact_paths missing runtime-readiness.json; keys: {sorted(ap.keys())}"
        )
        assert "runtime-readiness.md" in ap, (
            f"artifact_paths missing runtime-readiness.md; keys: {sorted(ap.keys())}"
        )


# ── GREEN: run_bundle emits readiness ────────────────────────────────────

class TestBundleReadiness:
    """run_bundle.py must emit the same runtime readiness artifacts."""

    def test_bundle_emits_readiness_json(self, tmp_path: Path):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_bundle(target, out)
        assert rc == 0, f"run_bundle failed: {stderr}"
        assert (Path(out) / "runtime-readiness.json").exists()

    def test_bundle_emits_readiness_md(self, tmp_path: Path):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_bundle(target, out)
        assert (Path(out) / "runtime-readiness.md").exists()

    def test_bundle_readiness_json_shape(self, tmp_path: Path):
        """runtime-readiness.json from run_bundle must have correct shape."""
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_bundle(target, out)
        data = _load_readiness_json(out)
        assert "detected_stacks" in data
        assert "required_commands" in data
        assert "suggested_verification" in data
        assert "verification_status" in data
        assert "blockers" in data

    def test_bundle_manifest_includes_readiness_files(self, tmp_path: Path):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_bundle(target, out)
        manifest = _load_manifest(out)
        ap = manifest.get("artifact_paths", {})
        assert "runtime-readiness.json" in ap, (
            f"bundle manifest artifact_paths missing runtime-readiness.json; "
            f"keys: {sorted(ap.keys())}"
        )
        assert "runtime-readiness.md" in ap


# ── GREEN: Safety — readiness does NOT run build/test commands ───────────

class TestReadinessSafety:
    """Readiness must only run version commands, never build/test."""

    def test_readiness_does_not_run_go_test(self, tmp_path: Path):
        """The runtime-readiness.json output must not claim tests passed."""
        target = str(RUNTIME_FIXTURES / "go_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        status = data.get("verification_status", "")
        assert status != "verification_passed", (
            "verification_status must NOT be 'verification_passed' — "
            "readiness does not run tests"
        )
        # Must not claim anything about test results
        for entry in data.get("required_commands", []):
            reason = entry.get("reason", "")
            assert "test" not in reason.lower() or "suggest" in reason.lower() or "go test" in reason.lower(), (
                f"reason field should not claim test results: {reason}"
            )

    def test_readiness_does_not_run_pytest(self, tmp_path: Path):
        target = str(RUNTIME_FIXTURES / "python_project")
        out = str(tmp_path / "bundle")
        run_ua(target, out, mode="structure")
        data = _load_readiness_json(out)
        status = data.get("verification_status", "")
        assert status not in ("verification_passed", "tests_passed"), (
            "verification_status must NOT claim tests/builds passed"
        )


# ── GREEN: Stack detection for Rust and Docker ───────────────────────────

class TestRustFixtureReadiness:
    """Rust fixture: Cargo.toml is present."""

    def test_rust_detection(self, tmp_path: Path):
        target_dir = tmp_path / "rust_project"
        target_dir.mkdir()
        (target_dir / "Cargo.toml").write_text("[package]\nname = 'test'\nversion = '0.1.0'\n")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(str(target_dir), out, mode="structure")
        assert rc == 0, f"run_ua failed on rust fixture: {stderr}"
        data = _load_readiness_json(out)
        assert "rust" in data["detected_stacks"]
        assert any("cargo" in s.lower() for s in data["suggested_verification"])


class TestDockerFixtureReadiness:
    """Docker fixture: Dockerfile is present."""

    def test_docker_detection(self, tmp_path: Path):
        target_dir = tmp_path / "docker_project"
        target_dir.mkdir()
        (target_dir / "Dockerfile").write_text("FROM alpine:latest\n")
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(str(target_dir), out, mode="structure")
        assert rc == 0, f"run_ua failed on docker fixture: {stderr}"
        data = _load_readiness_json(out)
        assert "docker" in data["detected_stacks"]
        assert any("docker" in s.lower() for s in data["suggested_verification"])


class TestNodeFixtureReadiness:
    """Node fixture: package.json is present."""

    def test_node_detection(self, tmp_path: Path):
        target_dir = tmp_path / "node_project"
        target_dir.mkdir()
        (target_dir / "package.json").write_text(
            '{"name":"test","scripts":{"test":"echo ok"}}'
        )
        out = str(tmp_path / "bundle")
        rc, _, stderr = run_ua(str(target_dir), out, mode="structure")
        assert rc == 0, f"run_ua failed on node fixture: {stderr}"
        data = _load_readiness_json(out)
        assert "node" in data["detected_stacks"]
        assert any("node" in s.lower() for s in data["suggested_verification"])


# ── GREEN: runtime_readiness module is importable ────────────────────────

class TestRuntimeReadinessModule:
    """Direct import and unit tests for runtime_readiness functions."""

    def _ensure_sys_path(self):
        scan_dir = str(PROJECT_ROOT / "scripts" / "code-scan")
        if scan_dir not in sys.path:
            sys.path.insert(0, scan_dir)
        import importlib
        import runtime_readiness
        importlib.reload(runtime_readiness)
        return __import__("runtime_readiness")

    def test_detect_stacks_go(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        target = str(RUNTIME_FIXTURES / "go_project")
        stacks = rr.detect_stacks(target)
        assert "go" in stacks

    def test_detect_stacks_python(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        target = str(RUNTIME_FIXTURES / "python_project")
        stacks = rr.detect_stacks(target)
        assert "python" in stacks

    def test_detect_stacks_empty(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        target = str(RUNTIME_FIXTURES / "unknown_project")
        stacks = rr.detect_stacks(target)
        assert stacks == []

    def test_check_command_available(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        # python should be available
        result = rr.check_command("python")
        assert result["command"] == "python"
        assert isinstance(result["available"], bool)

    def test_check_command_not_found(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        result = rr.check_command("definitely_not_a_command_12345")
        assert result["available"] is False
        assert result["version"] is None

    def test_build_readiness_artifact(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        target = str(RUNTIME_FIXTURES / "go_project")
        artifact = rr.build_readiness_artifact(target)
        assert "detected_stacks" in artifact
        assert "required_commands" in artifact
        assert "suggested_verification" in artifact
        assert "verification_status" in artifact
        assert "blockers" in artifact

    def test_readiness_to_markdown(self, tmp_path: Path):
        rr = self._ensure_sys_path()
        target = str(RUNTIME_FIXTURES / "go_project")
        artifact = rr.build_readiness_artifact(target)
        md = rr.readiness_to_markdown(artifact)
        assert isinstance(md, str)
        assert len(md) > 0
