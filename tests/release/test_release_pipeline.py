"""Focused tests for release/ pipeline contracts and determinism."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ["01_audit", "02_plan", "03_validate", "04_communicate", "05_ship"]


def _run_stage_script(name: str) -> subprocess.CompletedProcess:
    script = REPO_ROOT / "release" / name / "run.sh"
    return subprocess.run([str(script)], cwd=REPO_ROOT, capture_output=True, text=True)


def _ensure_upstream_artifacts() -> None:
    needed = [
        REPO_ROOT / "release" / "01_audit" / "audit.json",
        REPO_ROOT / "release" / "02_plan" / "plan.json",
    ]
    for path in needed:
        if not path.exists():
            proc = _run_stage_script(path.parent.name)
            if proc.returncode != 0:
                raise RuntimeError(f"failed to seed upstream artifact: {path}\n{proc.stderr}")


def test_stage_directories_exist() -> None:
    missing = [name for name in STAGES if not (REPO_ROOT / "release" / name).is_dir()]
    assert not missing, f"missing release stages: {missing}"


def test_stage_context_files_exist() -> None:
    missing = []
    for name in STAGES:
        path = REPO_ROOT / "release" / name / "CONTEXT.md"
        if not path.exists():
            missing.append(name)
    assert not missing, f"missing CONTEXT.md in stages: {missing}"


def test_stage_run_scripts_executable() -> None:
    missing = []
    for name in STAGES:
        path = REPO_ROOT / "release" / name / "run.sh"
        if not path.exists() or not os.access(path, os.X_OK):
            missing.append(name)
    assert not missing, f"missing or non-executable run.sh in stages: {missing}"


def test_audit_produces_artifact() -> None:
    proc = _run_stage_script("01_audit")
    assert proc.returncode == 0, proc.stderr
    artifact = REPO_ROOT / "release" / "01_audit" / "audit.json"
    assert artifact.exists(), "audit.json was not created"
    payload = json.loads(artifact.read_text())
    assert payload.get("stage") == "01_audit"
    assert "ok" in payload


def test_plan_requires_upstream_artifact() -> None:
    upstream = REPO_ROOT / "release" / "01_audit" / "audit.json"
    backup = upstream.read_bytes() if upstream.exists() else b""
    if upstream.exists():
        upstream.unlink()
    try:
        proc = _run_stage_script("02_plan")
        assert proc.returncode != 0, "02_plan should fail without upstream artifact"
        artifact = REPO_ROOT / "release" / "02_plan" / "plan.json"
        assert artifact.exists()
        payload = json.loads(artifact.read_text())
        assert payload.get("ok") is False
    finally:
        if backup:
            upstream.write_bytes(backup)
        else:
            upstream.unlink(missing_ok=True)
        error_artifact = REPO_ROOT / "release" / "02_plan" / "plan.json"
        error_artifact.unlink(missing_ok=True)


def test_validate_produces_contract_report() -> None:
    _ensure_upstream_artifacts()
    proc = _run_stage_script("03_validate")
    assert proc.returncode == 0, proc.stderr
    artifact = REPO_ROOT / "release" / "03_validate" / "validation.json"
    assert artifact.exists(), "validation.json was not created"
    payload = json.loads(artifact.read_text())
    assert payload.get("stage") == "03_validate"
    checks = payload.get("checks", {})
    assert checks.get("stage_contracts_present") is True
    assert checks.get("deterministic_scripts_executable") is True
    assert checks.get("upstream_artifacts_ok") is True


def test_validate_fails_without_upstream() -> None:
    upstream = REPO_ROOT / "release" / "02_plan" / "plan.json"
    backup = upstream.read_bytes() if upstream.exists() else b""
    if upstream.exists():
        upstream.unlink()
    try:
        proc = _run_stage_script("03_validate")
        assert proc.returncode != 0, "03_validate should fail without upstream artifact"
    finally:
        if backup:
            upstream.write_bytes(backup)
        else:
            upstream.unlink(missing_ok=True)
        validation = REPO_ROOT / "release" / "03_validate" / "validation.json"
        validation.unlink(missing_ok=True)


def _seed_all_upstream_artifacts() -> None:
    needed = [
        REPO_ROOT / "release" / "01_audit" / "audit.json",
        REPO_ROOT / "release" / "02_plan" / "plan.json",
        REPO_ROOT / "release" / "03_validate" / "validation.json",
    ]
    for path in needed:
        if not path.exists():
            stage_name = path.parent.name
            proc = _run_stage_script(stage_name)
            if proc.returncode != 0:
                raise RuntimeError(f"failed to seed upstream artifact: {path}\n{proc.stderr}")


def test_communicate_produces_artifacts() -> None:
    _seed_all_upstream_artifacts()
    proc = _run_stage_script("04_communicate")
    assert proc.returncode == 0, proc.stderr
    plan = json.loads((REPO_ROOT / "release" / "02_plan" / "plan.json").read_text())
    version = str(plan.get("version_candidate", "unknown"))
    changelog = REPO_ROOT / "release" / "04_communicate" / "changelog.md"
    notes = REPO_ROOT / "release" / "04_communicate" / "notes.md"
    assert changelog.exists() and notes.exists()
    assert changelog.read_text().splitlines()[0] == f"# Release {version}"
    assert notes.read_text().splitlines()[0] == f"# Release Notes {version}"


def test_ship_produces_manifest() -> None:
    _seed_all_upstream_artifacts()
    proc = _run_stage_script("05_ship")
    assert proc.returncode == 0, proc.stderr
    artifact = REPO_ROOT / "release" / "05_ship" / "ship_manifest.json"
    assert artifact.exists(), "ship_manifest.json was not created"
    payload = json.loads(artifact.read_text())
    assert payload.get("stage") == "05_ship"
    assert payload.get("ok") is True


def test_communicate_missing_upstream_fails() -> None:
    audit = REPO_ROOT / "release" / "01_audit" / "audit.json"
    plan = REPO_ROOT / "release" / "02_plan" / "plan.json"
    audit_backup = audit.read_bytes() if audit.exists() else b""
    plan_backup = plan.read_bytes() if plan.exists() else b""
    for path in [audit, plan]:
        if path.exists():
            path.unlink()
    try:
        proc = _run_stage_script("04_communicate")
        assert proc.returncode != 0, "04_communicate should fail without upstream artifacts"
    finally:
        if audit_backup:
            audit.write_bytes(audit_backup)
        else:
            audit.unlink(missing_ok=True)
        if plan_backup:
            plan.write_bytes(plan_backup)
        else:
            plan.unlink(missing_ok=True)


def test_ship_missing_upstream_fails() -> None:
    _ensure_upstream_artifacts()
    validation = REPO_ROOT / "release" / "03_validate" / "validation.json"
    backup = validation.read_bytes() if validation.exists() else b""
    if validation.exists():
        validation.unlink()
    try:
        proc = _run_stage_script("05_ship")
        assert proc.returncode != 0, "05_ship should fail without validation artifact"
    finally:
        if backup:
            validation.write_bytes(backup)
        else:
            validation.unlink(missing_ok=True)
