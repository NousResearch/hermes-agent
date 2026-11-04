#!/usr/bin/env python3
"""Validate the current release provenance and installation evidence.

This checker deliberately validates a current release report, not a historical
receipt. It refuses to infer the installed build from App Store Connect or from
an old receipt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_UNTRACKED = {"pre-existing", "intended", "generated-and-reviewed"}


class ProvenanceError(ValueError):
    """A release report does not satisfy the release gates."""


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    value = mapping.get(key)
    if value in (None, "", []):
        raise ProvenanceError(f"missing {label}.{key}")
    return value


def _require_true(mapping: dict[str, Any], key: str, label: str) -> None:
    if mapping.get(key) is not True:
        raise ProvenanceError(f"{label}.{key} must be true")


def validate(report: dict[str, Any]) -> None:
    target = _required(report, "target", "report")
    github = _required(report, "github", "report")
    xcode = _required(report, "xcode_cloud", "report")
    testflight = _required(report, "testflight", "report")
    installation = _required(report, "installation", "report")

    commit = _required(github, "commit", "github")
    if not isinstance(commit, str) or not _SHA_RE.fullmatch(commit):
        raise ProvenanceError("github.commit must be a 40-character lowercase SHA")
    _required(github, "ref", "github")
    _require_true(github, "pushed", "github")

    if "untracked_files" not in github:
        raise ProvenanceError("missing github.untracked_files")
    untracked = github["untracked_files"]
    if not isinstance(untracked, list):
        raise ProvenanceError("github.untracked_files must be a list")
    for item in untracked:
        if not isinstance(item, dict):
            raise ProvenanceError("each untracked file must be classified")
        if item.get("classification") not in _ALLOWED_UNTRACKED:
            raise ProvenanceError(
                f"untracked file {item.get('path', '[unknown]')} has no approved classification"
            )

    _required(xcode, "workflow_id", "xcode_cloud")
    _required(xcode, "run_id", "xcode_cloud")
    if xcode.get("source_commit") != commit:
        raise ProvenanceError("Xcode Cloud source commit does not equal GitHub commit")
    if xcode.get("archive_status") != "SUCCEEDED":
        raise ProvenanceError("Xcode Cloud archive was not successful")

    target_build = str(_required(target, "build_number", "target"))
    if str(_required(xcode, "build_number", "xcode_cloud")) != target_build:
        raise ProvenanceError("Xcode Cloud build number does not equal target build")
    if str(_required(testflight, "build_number", "testflight")) != target_build:
        raise ProvenanceError("TestFlight build number does not equal target build")
    if testflight.get("source_commit") != commit:
        raise ProvenanceError("TestFlight source commit does not equal GitHub commit")
    _require_true(testflight, "available", "testflight")
    _require_true(testflight, "tester_visible", "testflight")

    if str(_required(installation, "device_id", "installation")) != str(
        _required(target, "device_id", "target")
    ):
        raise ProvenanceError("installation device does not equal target device")
    if installation.get("readback_status") != "verified":
        raise ProvenanceError(
            "BLOCKED: installed CFBundleVersion was not read reliably from the device"
        )
    if str(_required(installation, "bundle_id", "installation")) != str(
        _required(target, "bundle_id", "target")
    ):
        raise ProvenanceError("installed bundle ID does not equal target bundle ID")
    if str(_required(installation, "bundle_version", "installation")) != target_build:
        raise ProvenanceError("installed CFBundleVersion does not equal target build")
    if not _required(installation, "evidence", "installation"):
        raise ProvenanceError("installation.evidence is required")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        validate(report)
    except (OSError, json.JSONDecodeError, ProvenanceError) as exc:
        print(f"RELEASE_PROVENANCE_BLOCKED {exc}")
        return 2
    print(
        "RELEASE_PROVENANCE_PASSED "
        f"commit={report['github']['commit']} "
        f"build={report['target']['build_number']} "
        f"device={report['installation']['device_id']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
