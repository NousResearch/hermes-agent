#!/usr/bin/env python3
"""Non-agent cron reconciliation wrapper for github-pr-feedback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_EXPECTED_MERGE_BLOCKERS = frozenset(
    {"ci_receipt_missing", "ci_receipt_not_passing"}
)
_HARD_FAILURE_REASONS = frozenset(
    {
        "admission_cap",
        "base_state_unavailable",
        "dispatch_failed",
        "exact_head_unavailable",
        "github_ci_state_unavailable",
        "github_error",
        "github_state_unavailable",
    }
)
_SOFT_SCAN_DEGRADATION_REASONS = frozenset({"github_ci_state_unavailable"})


def _has_reported_failure(section: object) -> bool:
    if not isinstance(section, dict):
        return False
    skipped = section.get("skipped")
    if not isinstance(skipped, dict):
        return False
    for reason, count in skipped.items():
        positive = (
            isinstance(count, int) and not isinstance(count, bool) and count > 0
        )
        if positive and (
            reason in _HARD_FAILURE_REASONS or section.get("status") == "degraded"
        ):
            return True
    return False


def _has_soft_scan_degradation(section: object) -> bool:
    """Whether only the known read-only GitHub CI metadata gap was reported."""

    if not isinstance(section, dict) or section.get("status") != "degraded":
        return False
    skipped = section.get("skipped")
    if not isinstance(skipped, dict):
        return False
    return any(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and reason in _SOFT_SCAN_DEGRADATION_REASONS
        for reason, count in skipped.items()
    )


def _has_hard_failure_reason(section: object) -> bool:
    if not isinstance(section, dict):
        return False
    skipped = section.get("skipped")
    if not isinstance(skipped, dict):
        return False
    return any(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and reason in _HARD_FAILURE_REASONS
        and reason not in _SOFT_SCAN_DEGRADATION_REASONS
        for reason, count in skipped.items()
    )


def _has_only_expected_merge_blockers(section: object) -> bool:
    if not isinstance(section, dict) or section.get("status") != "degraded":
        return False
    blocked = section.get("blocked")
    if not isinstance(blocked, dict) or not blocked:
        return False
    for reasons in blocked.values():
        if not isinstance(reasons, list) or not reasons:
            return False
        if any(reason not in _EXPECTED_MERGE_BLOCKERS for reason in reasons):
            return False
    return True


def _has_successful_merge(section: object) -> bool:
    if not isinstance(section, dict):
        return False
    merged = section.get("merged")
    return bool(
        isinstance(merged, list)
        and merged
        and all(
            isinstance(item, dict)
            and isinstance(item.get("pr_number"), int)
            and not isinstance(item.get("pr_number"), bool)
            and item["pr_number"] > 0
            for item in merged
        )
    )


def _cron_exit_code(stdout: str, process_returncode: int, stderr: str = "") -> int:
    """Classify durable progress without discarding degraded scan telemetry."""

    if process_returncode == 0:
        return 0
    if "Traceback (most recent call last)" in stderr:
        return process_returncode
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return process_returncode
    try:
        payload = json.loads(lines[-1])
    except (json.JSONDecodeError, TypeError):
        return process_returncode
    if not isinstance(payload, dict):
        return process_returncode
    if payload.get("status") not in {"ok", "degraded"}:
        return process_returncode
    merge = payload.get("merge")
    partial_merge_success = _has_successful_merge(merge)
    soft_scan_degradation = _has_soft_scan_degradation(payload)
    if payload.get("status") == "degraded" and not (
        partial_merge_success or soft_scan_degradation
    ):
        return process_returncode
    repair = payload.get("repair")
    top_level_failure = (
        _has_hard_failure_reason(payload)
        if soft_scan_degradation
        else _has_reported_failure(payload)
    )
    if top_level_failure or _has_reported_failure(repair):
        return process_returncode
    if isinstance(merge, dict) and merge.get("status") == "degraded":
        if not _has_only_expected_merge_blockers(merge):
            return process_returncode
    maintenance = payload.get("release_maintenance")
    if (
        isinstance(maintenance, dict)
        and maintenance.get("status") == "degraded"
        and not partial_merge_success
    ):
        return process_returncode
    if not isinstance(repair, dict):
        return process_returncode
    return 0


def main() -> int:
    configured = os.environ.get("HERMES_EXECUTABLE", "").strip()
    executable = Path(configured) if configured else None
    if (
        executable is None
        or not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        print(
            "HERMES_EXECUTABLE must name an executable absolute path",
            file=sys.stderr,
        )
        return 127
    completed = subprocess.run(
        [str(executable), "github-pr-feedback", "scan"],
        check=False,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return _cron_exit_code(completed.stdout, completed.returncode, completed.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
