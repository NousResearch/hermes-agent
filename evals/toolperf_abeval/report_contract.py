"""Validate provenance and shape of tool-performance A/B reports."""
from __future__ import annotations

from collections.abc import Mapping
import re

REQUIRED = {
    "baseline_sha", "fixes_sha", "model", "concurrency", "metrics", "status",
    "model_provenance", "arm_model_provenance", "evaluator_provenance",
}
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def validate_toolperf_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED - report.keys())]
    if report.get("status") not in {"pass", "fail", "partial", "unavailable"}:
        errors.append("invalid_status")
    if not isinstance(report.get("metrics"), Mapping):
        errors.append("metrics_must_be_mapping")
    provenance = report.get("model_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("model_provenance_must_be_mapping")
    else:
        for key in ("model", "provider"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                errors.append(f"model_provenance.{key}_must_be_nonempty_string")
        if not isinstance(provenance.get("config_digest"), str) or not _DIGEST.fullmatch(
            provenance.get("config_digest", "")
        ):
            errors.append("model_provenance.config_digest_must_be_sha256")
    arm_provenance = report.get("arm_model_provenance")
    if not isinstance(arm_provenance, Mapping):
        errors.append("arm_model_provenance_must_be_mapping")
    else:
        for arm in ("baseline", "fixes"):
            value = arm_provenance.get(arm)
            if not isinstance(value, Mapping):
                errors.append(f"arm_model_provenance.{arm}_must_be_mapping")
            else:
                if value != provenance:
                    errors.append(f"arm_model_provenance.{arm}_differs_from_report")
                for key in ("model", "provider"):
                    if not isinstance(value.get(key), str) or not value[key].strip():
                        errors.append(f"arm_model_provenance.{arm}.{key}_must_be_nonempty_string")
                if not isinstance(value.get("config_digest"), str) or not _DIGEST.fullmatch(
                    value.get("config_digest", "")
                ):
                    errors.append(f"arm_model_provenance.{arm}.config_digest_must_be_sha256")
    evaluator = report.get("evaluator_provenance")
    if not isinstance(evaluator, Mapping):
        errors.append("evaluator_provenance_must_be_mapping")
    else:
        for key in ("evaluator_digest", "battery_digest"):
            if not isinstance(evaluator.get(key), str) or not _DIGEST.fullmatch(
                evaluator.get(key, "")
            ):
                errors.append(f"evaluator_provenance.{key}_must_be_sha256")
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
