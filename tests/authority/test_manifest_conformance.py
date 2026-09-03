from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Mapping

import pytest

from hermes_cli.authority.manifest import (
    CANONICAL_AUTHORITY_MANIFEST,
    AdmissionRequestValidationError,
    ManifestValidationError,
    admission_request_from_mapping,
    compile_authority_manifest,
    evaluate_authority_operation,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "authority" / "manifest.v1.json"
SCHEMA_PATH = ROOT / "authority" / "manifest.schema.json"
MUTATIONS_PATH = ROOT / "authority" / "manifest-mutations.v1.json"
VECTORS_PATH = ROOT / "authority" / "conformance.v1.json"

CANONICAL_GITHUB_OPERATIONS = frozenset(
    {
        "github.issue.metadata.write",
        "github.comment.write",
        "github.contents.write",
        "github.gitdata.write",
        "github.pull_request.create",
        "github.actions.dispatch",
    }
)
CANONICAL_GITHUB_SINKS = frozenset(CANONICAL_GITHUB_OPERATIONS)

_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "additionalProperties",
        "required",
        "properties",
        "const",
        "minLength",
        "minProperties",
        "minItems",
        "items",
        "uniqueItems",
        "enum",
        "not",
        "pattern",
    }
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise AssertionError(f"unsupported external schema reference: {ref}")
    value: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        value = value[part]
    if not isinstance(value, Mapping):
        raise AssertionError(f"schema reference does not resolve to an object: {ref}")
    return value


def _schema_errors(
    instance: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str = "$",
) -> list[str]:
    unsupported = set(schema) - _SCHEMA_KEYWORDS
    if unsupported:
        raise AssertionError(
            f"schema gate does not implement keywords: {sorted(unsupported)}"
        )
    if "$ref" in schema:
        return _schema_errors(instance, _resolve_ref(root, schema["$ref"]), root, path)

    errors: list[str] = []
    if "const" in schema:
        expected = schema["const"]
        if instance != expected or type(instance) is not type(expected):
            errors.append(f"{path}: expected const {expected!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")

    expected_type = schema.get("type")
    type_matches = (
        expected_type is None
        or (expected_type == "object" and isinstance(instance, Mapping))
        or (expected_type == "array" and isinstance(instance, list))
        or (expected_type == "string" and isinstance(instance, str))
    )
    if not type_matches:
        errors.append(f"{path}: expected {expected_type}")
        return errors

    if isinstance(instance, Mapping):
        if len(instance) < schema.get("minProperties", 0):
            errors.append(f"{path}: too few properties")
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required property {key}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_path = f"{path}.{key}"
            if key in properties:
                errors.extend(_schema_errors(value, properties[key], root, child_path))
            elif additional is False:
                errors.append(f"{child_path}: additional property is forbidden")
            elif isinstance(additional, Mapping):
                errors.extend(_schema_errors(value, additional, root, child_path))

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in instance]
            if len(set(encoded)) != len(encoded):
                errors.append(f"{path}: items are not unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, value in enumerate(instance):
                errors.extend(
                    _schema_errors(value, item_schema, root, f"{path}[{index}]")
                )

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance):
            pass

    negated = schema.get("not")
    if isinstance(negated, Mapping) and not _schema_errors(instance, negated, root, path):
        errors.append(f"{path}: matched forbidden schema")

    if isinstance(instance, str) and "pattern" in schema:
        if re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: pattern did not match")

    return errors


def _apply_mutations(raw: Any, mutations: list[Mapping[str, Any]]) -> Any:
    result = deepcopy(raw)
    for mutation in mutations:
        path = mutation["path"]
        parent = result
        for part in path[:-1]:
            parent = parent[part]
        key = path[-1]
        operation = mutation["op"]
        if operation == "set":
            parent[key] = deepcopy(mutation["value"])
        elif operation == "delete":
            del parent[key]
        elif operation == "append":
            parent[key].append(deepcopy(mutation["value"]))
        else:
            raise AssertionError(f"unknown mutation operation: {operation}")
    return result


def test_canonical_runtime_artifact_matches_exact_source_bytes_and_hash() -> None:
    source_bytes = MANIFEST_PATH.read_bytes()
    artifact = CANONICAL_AUTHORITY_MANIFEST

    assert artifact.manifest_bytes == source_bytes
    assert artifact.manifest_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert artifact.manifest.policy_version == "2026.08.25-v2"

    with pytest.raises(FrozenInstanceError):
        artifact.manifest_sha256 = "mutable"  # type: ignore[misc]


def test_shared_conformance_vectors_match_python_evaluator() -> None:
    manifest = CANONICAL_AUTHORITY_MANIFEST.manifest
    vectors = _read_json(VECTORS_PATH)
    assert vectors["schema_version"] == 1

    for vector in vectors["vectors"]:
        request = admission_request_from_mapping(vector["request"])
        decision = evaluate_authority_operation(manifest, request)
        assert decision.as_dict() == vector["expected"], vector["name"]


def test_manifest_uses_the_canonical_github_vertical_slice() -> None:
    manifest = CANONICAL_AUTHORITY_MANIFEST.manifest
    domain = manifest.domains["github.operation"]

    assert frozenset(domain.operations) == CANONICAL_GITHUB_OPERATIONS
    assert frozenset(domain.sinks) == CANONICAL_GITHUB_SINKS

    for operation_name, operation in domain.operations.items():
        assert operation.sink_class == operation_name

    assert domain.operations["github.contents.write"].required_capabilities == (
        "contents:write",
    )
    assert domain.operations["github.gitdata.write"].required_capabilities == (
        "git_objects:write",
        "refs:write",
    )
    assert domain.operations["github.pull_request.create"].required_capabilities == (
        "pull_requests:create",
    )
    assert domain.operations["github.actions.dispatch"].required_capabilities == (
        "actions:dispatch",
    )


def test_schema_and_compiler_share_the_mutation_corpus() -> None:
    canonical = _read_json(MANIFEST_PATH)
    schema = _read_json(SCHEMA_PATH)
    corpus = _read_json(MUTATIONS_PATH)
    assert corpus["schema_version"] == 1

    for case in corpus["cases"]:
        candidate = _apply_mutations(canonical, case["mutations"])
        schema_valid = not _schema_errors(candidate, schema, schema)
        assert schema_valid is case["expected"]["schema_valid"], case["name"]

        try:
            compile_authority_manifest(candidate)
        except ManifestValidationError:
            compiler_valid = False
        else:
            compiler_valid = True
        assert compiler_valid is case["expected"]["compiler_valid"], case["name"]


def test_compiler_preserves_immutable_sink_ownership_metadata() -> None:
    manifest = CANONICAL_AUTHORITY_MANIFEST.manifest
    domain = manifest.domains["github.operation"]
    sink = domain.sinks["github.comment.write"]

    assert sink.broker == "githubMutationBroker"
    assert sink.direct_symbols == (
        "issues.createComment",
        "pulls.createReview",
        "pulls.createReviewComment",
    )
    for operation in domain.operations.values():
        assert operation.sink_class in domain.sinks

    with pytest.raises(TypeError):
        manifest.domains["github.operation"] = domain  # type: ignore[index]
    with pytest.raises(TypeError):
        domain.sinks["github.comment.write"] = sink  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        sink.broker = "ambientBroker"  # type: ignore[misc]


def test_admission_request_parser_rejects_open_or_coerced_shapes() -> None:
    raw = {
        "domain": "github.operation",
        "operation_class": "github.comment.write",
        "actor_class": "human",
        "resource_state": "open",
        "capabilities": [
            {
                "capability": "comments:write",
                "granted": True,
                "source": "user_api_token",
                "generation": "cred-1",
            }
        ],
    }

    with pytest.raises(AdmissionRequestValidationError, match="unknown"):
        admission_request_from_mapping({**raw, "ambient_fallback": True})

    invalid_grant = deepcopy(raw)
    invalid_grant["capabilities"][0]["granted"] = "yes"
    with pytest.raises(AdmissionRequestValidationError, match="boolean"):
        admission_request_from_mapping(invalid_grant)
