"""Validate scrubbed, reproducible context-compression reports."""
from __future__ import annotations

from collections.abc import Mapping
import math
import json
import re

REQUIRED_KEYS = {
    "schema_version", "source_sha", "fixture_digest", "compressed_tokens",
    "baseline_tokens", "probe_scores", "artifact_trail_preserved",
    "continuity_preserved", "probe_manifest", "model_provenance", "status",
    "evaluator_digest", "battery_digest",
}
_FIXTURE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_HOME_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]+(?:users|documents and settings)[\\/]+|"
    r"/(?:users|home|root|private/var|var/folders)/)"
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[a-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret(?:[_-]access[_-]?key)?|password|authorization|credential|token)"
    r"\s*(?:=|:)\s*(?:bearer\s+)?[^\s,;}\]]+"
)
_CREDENTIAL_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret|password|authorization|credential|token)(?:$|[_-])"
)
_PROVENANCE_KEYS = {"compression_model", "evaluator_model", "provider", "model_config"}
_BARE_CREDENTIAL = re.compile(r"(?i)^(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9_]{20,}|xox[baprs]-[a-z0-9-]{12,}|AIza[0-9A-Za-z_-]{20,})$")


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
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            return _credential_errors(decoded, path)
        if _CREDENTIAL_ASSIGNMENT.search(value):
            return ["forbidden_credential_assignment"]
        if _BARE_CREDENTIAL.fullmatch(value.strip()):
            return ["forbidden_bare_credential"]
    return []


def _nonfinite_errors(value: object, path: str) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"nonfinite_number:{path}"]
    if isinstance(value, Mapping):
        return [
            error
            for key, child in value.items()
            for error in _nonfinite_errors(child, f"{path}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            error
            for index, child in enumerate(value)
            for error in _nonfinite_errors(child, f"{path}[{index}]")
        ]
    return []


def validate_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED_KEYS - report.keys())]
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1:
        errors.append("schema_version_must_be_1")
    fixture_digest = report.get("fixture_digest")
    if not isinstance(fixture_digest, str) or not _FIXTURE_DIGEST.fullmatch(fixture_digest):
        errors.append("fixture_digest_must_be_sha256")
    for key in ("evaluator_digest", "battery_digest"):
        if not isinstance(report.get(key), str) or not _FIXTURE_DIGEST.fullmatch(report[key]):
            errors.append(f"{key}_must_be_sha256")
    for key in ("compressed_tokens", "baseline_tokens"):
        if not isinstance(report.get(key), int) or isinstance(report.get(key), bool):
            errors.append(f"{key}_must_be_integer")
        elif report[key] < 0:
            errors.append(f"{key}_must_be_nonnegative")
    if isinstance(report.get("baseline_tokens"), int) and not isinstance(report.get("baseline_tokens"), bool):
        if report["baseline_tokens"] <= 0:
            errors.append("baseline_tokens_must_be_positive")
    if report.get("status") == "pass" and all(isinstance(report.get(key), int) and not isinstance(report.get(key), bool) for key in ("compressed_tokens", "baseline_tokens")) and report["compressed_tokens"] >= report["baseline_tokens"]:
        errors.append("compressed_tokens_must_be_less_than_baseline_tokens_for_pass")
    if not isinstance(report.get("probe_scores"), Mapping):
        errors.append("probe_scores_must_be_mapping")
    else:
        errors.extend(_nonfinite_errors(report["probe_scores"], "probe_scores"))
        for name, score in report["probe_scores"].items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                errors.append(f"probe_scores.{name}_must_be_numeric")
    probe_manifest = report.get("probe_manifest")
    if not isinstance(probe_manifest, (list, tuple)) or not probe_manifest:
        errors.append("probe_manifest_must_be_nonempty_list")
    elif any(not isinstance(name, str) or not name.strip() for name in probe_manifest):
        errors.append("probe_manifest_entries_must_be_nonempty_strings")
    elif len(set(probe_manifest)) != len(probe_manifest):
        errors.append("probe_manifest_entries_must_be_unique")
    elif report.get("status") == "pass" and isinstance(report.get("probe_scores"), Mapping):
        missing = sorted(set(probe_manifest) - set(report["probe_scores"]))
        if missing:
            errors.append("missing_probe_scores:" + ",".join(missing))
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
    if _FORBIDDEN_HOME_PATH.search(text):
        errors.append("forbidden_home_path")
    return errors
