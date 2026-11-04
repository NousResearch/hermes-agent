"""Contract tests for the Core TestFlight provenance gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[2]
    / "optional-skills/devops/core-testflight/scripts/verify-release-provenance.py"
)
_SPEC = importlib.util.spec_from_file_location("verify_release_provenance", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


COMMIT = "a" * 40


def _report() -> dict:
    return {
        "target": {
            "bundle_id": "com.accessiblecore.capture",
            "marketing_version": "1.0",
            "build_number": "91",
            "device_id": "PHN-01-UDID",
        },
        "github": {
            "commit": COMMIT,
            "ref": "native-auth-client",
            "pushed": True,
            "untracked_files": [
                {"path": "docs/old-receipt.json", "classification": "pre-existing"}
            ],
        },
        "xcode_cloud": {
            "workflow_id": "workflow-1",
            "run_id": "run-1",
            "source_commit": COMMIT,
            "archive_status": "SUCCEEDED",
            "build_number": "91",
        },
        "testflight": {
            "source_commit": COMMIT,
            "available": True,
            "tester_visible": True,
            "build_number": "91",
        },
        "installation": {
            "method": "testflight",
            "device_id": "PHN-01-UDID",
            "bundle_id": "com.accessiblecore.capture",
            "bundle_version": "91",
            "readback_status": "verified",
            "evidence": "fresh structured device read",
        },
    }


def test_accepts_matching_current_release_provenance() -> None:
    _MODULE.validate(_report())


def test_rejects_historical_or_wrong_installed_build() -> None:
    report = _report()
    report["installation"]["bundle_version"] = "84"
    with pytest.raises(_MODULE.ProvenanceError, match="installed CFBundleVersion"):
        _MODULE.validate(report)


def test_blocks_when_device_version_cannot_be_read() -> None:
    report = _report()
    report["installation"]["readback_status"] = "unavailable"
    with pytest.raises(_MODULE.ProvenanceError, match="BLOCKED"):
        _MODULE.validate(report)


def test_rejects_xcode_cloud_commit_mismatch() -> None:
    report = _report()
    report["xcode_cloud"]["source_commit"] = "b" * 40
    with pytest.raises(_MODULE.ProvenanceError, match="source commit"):
        _MODULE.validate(report)
