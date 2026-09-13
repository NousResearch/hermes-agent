"""Validate scrubbed, reproducible context-compression reports."""
from __future__ import annotations

from collections.abc import Mapping
import re

REQUIRED_KEYS = {
    "schema_version", "source_sha", "fixture_digest", "compressed_tokens",
    "baseline_tokens", "probe_scores", "artifact_trail_preserved",
    "continuity_preserved", "model_provenance", "status",
}
FORBIDDEN_MARKERS = ("/Users/", "/home/")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[a-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret(?:[_-]access[_-]?key)?|password|authorization|credential|token)"
    r"\s*=\s*[^\s,;}\]]+"
)
_CREDENTIAL_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret|password|authorization|credential|token)$"
)
_PROVENANCE_KEYS = {"compression_model", "evaluator_model", "provider", "model_config"}


def _credential_errors(value: object, path: str = "report") -> list[str]:
    if isinstance(value, Mapping):
        errors = []
        for key, child in value.items():
            name = str(key)
            if _CREDENTIAL_KEY.search(name):
                errors.append(f"forbidden_key:{path}.{name}")
            errors.extend(_credential_errors(child, f"{path}.{name}"))
        return errors
    if isinstance(value, (list, tuple)):
        return [error for index, child in enumerate(value)
                for error in _credential_errors(child, f"{path}[{index}]")]
    return []


def validate_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED_KEYS - report.keys())]
    if report.get("schema_version") != 1:
        errors.append("schema_version_must_be_1")
    for key in ("compressed_tokens", "baseline_tokens"):
        if not isinstance(report.get(key), int) or isinstance(report.get(key), bool):
            errors.append(f"{key}_must_be_integer")
        elif report[key] < 0:
            errors.append(f"{key}_must_be_nonnegative")
    if isinstance(report.get("baseline_tokens"), int) and not isinstance(report.get("baseline_tokens"), bool):
        if report["baseline_tokens"] <= 0:
            errors.append("baseline_tokens_must_be_positive")
    if not isinstance(report.get("probe_scores"), Mapping):
        errors.append("probe_scores_must_be_mapping")
    if not isinstance(report.get("artifact_trail_preserved"), bool):
        errors.append("artifact_trail_preserved_must_be_boolean")
    if not isinstance(report.get("continuity_preserved"), bool):
        errors.append("continuity_preserved_must_be_boolean")
    elif report.get("status") == "pass" and not report["continuity_preserved"]:
        errors.append("continuity_preserved_must_be_true_for_pass")
    if report.get("status") == "pass" and not report.get("artifact_trail_preserved"):
        errors.append("artifact_trail_preserved_must_be_true_for_pass")
    provenance = report.get("model_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("model_provenance_must_be_mapping")
    else:
        errors.extend(f"missing:model_provenance.{key}"
                      for key in sorted(_PROVENANCE_KEYS - provenance.keys()))
        for key in _PROVENANCE_KEYS:
            value = provenance.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"model_provenance.{key}_must_be_nonempty_string")
    if report.get("status") not in {"pass", "fail", "unavailable"}:
        errors.append("invalid_status")
    errors.extend(_credential_errors(report))
    text = repr(dict(report))
    if _CREDENTIAL_ASSIGNMENT.search(text):
        errors.append("forbidden_credential_assignment")
    errors.extend(f"forbidden_marker:{marker}" for marker in FORBIDDEN_MARKERS if marker in text)
    return errors
