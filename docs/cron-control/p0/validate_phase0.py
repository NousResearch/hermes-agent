#!/usr/bin/env python3
"""Validate the phase-0 cron contract pack."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - surfaced by CLI
        raise ValidationError(f"{path.name}: invalid JSON ({exc})") from exc


def _datetime_ok(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return False
    return True


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _check_schema(schema: dict[str, Any], instance: Any, path: str = "$") -> None:
    if "anyOf" in schema:
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                _check_schema(option, instance, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        raise ValidationError(f"{path}: none of anyOf matched ({'; '.join(errors)})")

    if "oneOf" in schema:
        matches = 0
        last_error: ValidationError | None = None
        for option in schema["oneOf"]:
            try:
                _check_schema(option, instance, path)
                matches += 1
            except ValidationError as exc:
                last_error = exc
        if matches < 1:
            raise ValidationError(f"{path}: no oneOf branch matched ({last_error})")
        return

    if "const" in schema and instance != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(f"{path}: value {instance!r} not in enum {schema['enum']!r}")

    if "type" in schema and not _matches_type(instance, schema["type"]):
        raise ValidationError(f"{path}: expected type {schema['type']!r}, got {type(instance).__name__}")

    if "pattern" in schema:
        if not isinstance(instance, str) or re.fullmatch(schema["pattern"], instance) is None:
            raise ValidationError(f"{path}: pattern mismatch {schema['pattern']!r}")

    if "minLength" in schema and isinstance(instance, str) and len(instance) < schema["minLength"]:
        raise ValidationError(f"{path}: string shorter than minLength {schema['minLength']}")

    if "minimum" in schema and isinstance(instance, (int, float)) and instance < schema["minimum"]:
        raise ValidationError(f"{path}: number smaller than minimum {schema['minimum']}")

    if schema.get("format") == "date-time":
        if not isinstance(instance, str) or not _datetime_ok(instance):
            raise ValidationError(f"{path}: invalid date-time {instance!r}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise ValidationError(f"{path}: missing required key {key!r}")
        for key, value in instance.items():
            if key in properties:
                _check_schema(properties[key], value, f"{path}.{key}")
            elif not schema.get("additionalProperties", True):
                raise ValidationError(f"{path}: unexpected key {key!r}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ValidationError(f"{path}: array shorter than minItems {schema['minItems']}")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in instance}) != len(instance):
            raise ValidationError(f"{path}: array items are not unique")
        if "items" in schema:
            for index, value in enumerate(instance):
                _check_schema(schema["items"], value, f"{path}[{index}]")


def validate_examples(job_schema: dict[str, Any], evidence_schema: dict[str, Any], verdict_schema: dict[str, Any], audit_schema: dict[str, Any]) -> None:
    example_dir = ROOT / "examples"
    job_examples = [
        example_dir / "job-metadata.agent.example.json",
        example_dir / "job-metadata.no-agent.example.json",
    ]
    for path in job_examples:
        _check_schema(job_schema, load_json(path), path.name)

    registry = load_json(example_dir / "jobs-registry.example.json")
    if not isinstance(registry, dict):
        raise ValidationError("jobs-registry.example.json: expected object")
    if "jobs" not in registry or "updated_at" not in registry:
        raise ValidationError("jobs-registry.example.json: missing required keys")
    _check_schema(job_schema, registry["jobs"][0], "jobs-registry.example.json.jobs[0]")
    _check_schema(job_schema, registry["jobs"][1], "jobs-registry.example.json.jobs[1]")
    _check_schema({"type": "string", "format": "date-time"}, registry["updated_at"], "jobs-registry.example.json.updated_at")

    _check_schema(evidence_schema, load_json(example_dir / "evidence.example.json"), "evidence.example.json")
    _check_schema(verdict_schema, load_json(example_dir / "verdict.example.json"), "verdict.example.json")
    _check_schema(audit_schema, load_json(example_dir / "audit.example.json"), "audit.example.json")


def validate_state_machine() -> dict[str, Any]:
    state_machine = load_json(ROOT / "state-machine.json")
    if not isinstance(state_machine, dict):
        raise ValidationError("state-machine.json: expected object")
    states = state_machine.get("states")
    transitions = state_machine.get("transitions")
    rule_ids = state_machine.get("rule_ids")
    if not isinstance(states, list) or not all(isinstance(item, str) for item in states):
        raise ValidationError("state-machine.json: states must be a list of strings")
    if len(set(states)) != len(states):
        raise ValidationError("state-machine.json: states are not unique")
    if not isinstance(transitions, list) or not transitions:
        raise ValidationError("state-machine.json: transitions must be a non-empty list")
    if not isinstance(rule_ids, list) or len(rule_ids) != len(set(rule_ids)):
        raise ValidationError("state-machine.json: rule_ids must be unique")
    known_rules = set(rule_ids)
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValidationError(f"state-machine.json.transitions[{index}]: expected object")
        for field in ("from", "to", "rule_id", "condition", "action", "automatic"):
            if field not in transition:
                raise ValidationError(f"state-machine.json.transitions[{index}]: missing {field!r}")
        if transition["rule_id"] not in known_rules:
            raise ValidationError(f"state-machine.json.transitions[{index}]: unknown rule_id {transition['rule_id']!r}")
    return state_machine


def validate_fixtures(known_rules: set[str]) -> None:
    fixture_dir = ROOT / "fixtures"
    required = {
        "hmm-policy-block.json",
        "cc-audit-stale-running.json",
        "receipt-conflict-429.json",
        "state-store-unavailable.json",
        "suppressed-receipt-success.json",
        "unknown-nonidempotent.json",
        "action-readback-failed.json",
    }
    found = {path.name for path in fixture_dir.glob("*.json")}
    missing = required - found
    if missing:
        raise ValidationError(f"fixtures: missing files {sorted(missing)!r}")

    for path in sorted(fixture_dir.glob("*.json")):
        fixture = load_json(path)
        if not isinstance(fixture, dict):
            raise ValidationError(f"{path.name}: expected object")
        for field in ("fixture_id", "scenario", "description", "job", "evidence", "expected"):
            if field not in fixture:
                raise ValidationError(f"{path.name}: missing {field!r}")
        if not isinstance(fixture["evidence"], list) or not fixture["evidence"]:
            raise ValidationError(f"{path.name}: evidence must be a non-empty list")
        if not isinstance(fixture["expected"], dict):
            raise ValidationError(f"{path.name}: expected must be an object")
        if fixture["expected"].get("rule_id") not in known_rules:
            raise ValidationError(f"{path.name}: unknown expected.rule_id {fixture['expected'].get('rule_id')!r}")
        if fixture["expected"].get("state") not in {"healthy", "suspect", "transient_failure", "systemic_failure", "stale_running", "recoverable", "repair_in_progress", "recovered", "quarantined", "human_required"}:
            raise ValidationError(f"{path.name}: unsupported expected.state {fixture['expected'].get('state')!r}")
        for evidence in fixture["evidence"]:
            _check_schema(load_json(ROOT / "evidence.schema.json"), evidence, f"{path.name}.evidence")
        fixture_text = json.dumps(fixture, ensure_ascii=False)
        if any(token in fixture_text for token in ("BEGIN PRIVATE KEY", "sk-", "Bearer ", "api_key", "token=")):
            raise ValidationError(f"{path.name}: suspicious secret-like content")


def validate_manifest() -> None:
    manifest = ROOT / "artifact-manifest.sha256"
    if not manifest.exists():
        raise ValidationError("artifact-manifest.sha256 is missing")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        if rel.startswith("*"):
            rel = rel[1:]
        path = ROOT / rel
        if not path.exists():
            raise ValidationError(f"manifest references missing file: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValidationError(f"hash mismatch for {rel}: expected {digest}, got {actual}")


def main() -> int:
    job_schema = load_json(ROOT / "job-metadata.schema.json")
    evidence_schema = load_json(ROOT / "evidence.schema.json")
    verdict_schema = load_json(ROOT / "verdict.schema.json")
    audit_schema = load_json(ROOT / "audit.schema.json")
    state_machine = validate_state_machine()

    validate_examples(job_schema, evidence_schema, verdict_schema, audit_schema)
    validate_fixtures(set(state_machine["rule_ids"]))
    validate_manifest()

    print("phase-0 contract pack validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
