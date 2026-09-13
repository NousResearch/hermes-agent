"""Validate scrubbed, reproducible context-compression reports."""
from __future__ import annotations

from collections.abc import Mapping

REQUIRED_KEYS = {"schema_version", "source_sha", "fixture_digest", "compressed_tokens", "baseline_tokens", "probe_scores", "artifact_trail_preserved", "continuity_preserved", "status"}
FORBIDDEN_MARKERS = ("OPENAI_API_KEY=", "ANTHROPIC_API_KEY=", "/Users/", "/home/")


def validate_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED_KEYS - report.keys())]
    if report.get("schema_version") != 1:
        errors.append("schema_version_must_be_1")
    for key in ("compressed_tokens", "baseline_tokens"):
        if not isinstance(report.get(key), int) or isinstance(report.get(key), bool):
            errors.append(f"{key}_must_be_integer")
    if not isinstance(report.get("probe_scores"), Mapping):
        errors.append("probe_scores_must_be_mapping")
    if report.get("status") not in {"pass", "fail", "unavailable"}:
        errors.append("invalid_status")
    text = repr(dict(report))
    errors.extend(f"forbidden_marker:{marker}" for marker in FORBIDDEN_MARKERS if marker in text)
    return errors
