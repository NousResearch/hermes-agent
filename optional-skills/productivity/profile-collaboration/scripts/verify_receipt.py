#!/usr/bin/env python3
"""Verify a profile-panel receipt without rereading profile report contents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from run_panel import MAX_CAPTURE_BYTES, PROFILES, redact_text, runtime_revision, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(checks: list[dict[str, object]], name: str, detail: str) -> None:
    checks.append({"check": name, "status": "failed", "detail": detail})


def pass_check(checks: list[dict[str, object]], name: str, detail: str) -> None:
    checks.append({"check": name, "status": "passed", "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Ares profile-panel receipt artifacts")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path.home() / ".ares" / "runtime" / "current",
    )
    args = parser.parse_args()
    receipt_dir = args.receipt.expanduser().resolve()
    panel_path = receipt_dir / "panel.json"
    checks: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read panel.json: {exc}")
        panel = {}

    if panel.get("schema") != "AresProfilePanelReceiptV1":
        errors.append("unexpected receipt schema")
    required_profiles = panel.get("required_profiles")
    if not isinstance(required_profiles, list) or not required_profiles:
        errors.append("required profile set is empty or invalid")
        required_profiles = []
    else:
        required_set = set(required_profiles)
        canonical_subset = [profile for profile in PROFILES if profile in required_set]
        if (
            len(required_profiles) != len(required_set)
            or not required_set.issubset(PROFILES)
            or required_profiles != canonical_subset
        ):
            errors.append("required profiles are duplicated, unknown, or out of canonical order")
    results = panel.get("results")
    if not isinstance(results, list):
        errors.append("results is not a list")
        results = []
    result_profiles = [item.get("profile") for item in results if isinstance(item, dict)]
    if result_profiles != required_profiles:
        errors.append("results are missing, duplicated, or out of admitted order")

    if panel.get("dry_run") is not False:
        errors.append("receipt is not an executed (non-dry-run) panel")
    if panel.get("execution_complete") is not True:
        errors.append("panel execution is not complete")

    output_policy = panel.get("output_policy")
    if not isinstance(output_policy, dict) or output_policy.get("mode") != "redacted_utf8_bounded_v1":
        errors.append("receipt does not declare the redacted bounded output policy")
    elif output_policy.get("max_bytes_per_stream") != MAX_CAPTURE_BYTES:
        errors.append("receipt output limit does not match the runner")

    expected_revision = runtime_revision(args.runtime.expanduser().resolve())
    recorded_revision = panel.get("runtime_revision")
    if expected_revision and recorded_revision == expected_revision:
        pass_check(checks, "runtime_identity", expected_revision)
    else:
        errors.append(
            f"runtime identity mismatch: recorded={recorded_revision!r}, current={expected_revision!r}"
        )

    for result in results:
        if not isinstance(result, dict):
            errors.append("non-object profile result")
            continue
        profile = result.get("profile")
        if result.get("outcome") != "returned" or result.get("exit_code") != 0:
            errors.append(f"{profile}: profile did not return successfully")
        stdout_rel = result.get("stdout_path")
        stderr_rel = result.get("stderr_path")
        if not isinstance(stdout_rel, str) or not isinstance(stderr_rel, str):
            errors.append(f"{profile}: missing artifact paths")
            continue
        for label, relative, digest_key, bytes_key in (
            ("stdout", stdout_rel, "stdout_sha256", "stdout_bytes"),
            ("stderr", stderr_rel, "stderr_sha256", "stderr_bytes"),
        ):
            path = (receipt_dir / relative).resolve()
            try:
                path.relative_to(receipt_dir)
                actual_size = path.stat().st_size
                actual_digest = sha256_file(path)
            except (OSError, ValueError) as exc:
                errors.append(f"{profile}: invalid {label} artifact: {exc}")
                continue
            if actual_size != result.get(bytes_key):
                errors.append(f"{profile}: {label} byte count changed")
            if actual_digest != result.get(digest_key):
                errors.append(f"{profile}: {label} digest changed")
            if label == "stdout" and actual_size == 0:
                errors.append(f"{profile}: empty report")

    if not errors:
        pass_check(checks, "profile_artifacts", f"{len(required_profiles)} selected profile reports and diagnostics verified")
    else:
        for error in errors:
            fail(checks, "receipt_integrity", error)

    verification = {
        "schema": "AresProfilePanelVerificationV1",
        "verified_at": utc_now(),
        "receipt": str(panel_path),
        "runtime": str(args.runtime.expanduser().resolve()),
        "runtime_revision": expected_revision,
        "checks": checks,
        "controller_verified": not errors,
        "semantic_review_required": True,
        "evidence_state": "execution_artifacts_verified" if not errors else "blocked_unknown",
        "remaining_delta": [
            "Controller must review every selected domain report for evidence, dissent, and relevance.",
            "Desktop recovery remains a separate acceptance gate.",
        ],
    }
    output_path = receipt_dir / "verification.json"
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
