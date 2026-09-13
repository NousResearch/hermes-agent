"""Validate provenance and shape of tool-performance A/B reports."""
from __future__ import annotations

from collections.abc import Mapping
import re

REQUIRED = {"baseline_sha", "fixes_sha", "model", "concurrency", "metrics", "status"}
_SHA = re.compile(r"^[0-9a-f]{40}$")


def validate_toolperf_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED - report.keys())]
    if report.get("status") not in {"pass", "fail", "partial", "unavailable"}:
        errors.append("invalid_status")
    if not isinstance(report.get("metrics"), Mapping):
        errors.append("metrics_must_be_mapping")
    concurrency = report.get("concurrency")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        errors.append("concurrency_must_be_positive_integer")
    for key in ("baseline_sha", "fixes_sha"):
        value = report.get(key)
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            errors.append(f"{key}_must_be_git_sha")
    if report.get("baseline_sha") == report.get("fixes_sha"):
        errors.append("arms_must_use_distinct_shas")
    return errors
