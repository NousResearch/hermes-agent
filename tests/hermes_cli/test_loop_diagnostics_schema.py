"""Schema conformance for the loop-diagnostics contract.

Validates ``hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json``
using only the standard library — the repo pins exact deps and ``jsonschema`` is
not a declared dependency, so importing it here would break CI on a clean
checkout.

The checks mirror the JSON Schema surface the schema actually declares:
union-branch shape, required fields, enum membership, ``additionalProperties:
false`` strictness, and the field-level type constraints that the diagnosis
engine and graph recorder rely on. See ``docs/loop-diagnostics-design.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "hermes_cli"
    / "observability"
    / "schemas"
    / "hermes.loop_diagnostics.v1.schema.json"
)

KIND_TO_BRANCH = {
    "run_header": "RunHeader",
    "action_start": "ActionStart",
    "action_end": "ActionEnd",
    "edge": "DependencyEdge",
    "run_footer": "RunFooter",
    "diagnosis_result": "DiagnosisResult",
}

VALID_TRACE_RECORDS = [
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "run_header",
        "task_id": "t_example",
        "run_id": 41,
        "attempt": 1,
        "profile": "default",
        "goal_mode": False,
        "ts": 1785739000,
    },
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "action_start",
        "task_id": "t_example",
        "run_id": 41,
        "action_id": "41:1",
        "parent_action_id": None,
        "loop_id": None,
        "iteration": None,
        "ts": 1785739001,
        "action_kind": "tool_call",
        "tool_name": "terminal",
        "summary": "clone repo",
    },
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "action_end",
        "task_id": "t_example",
        "run_id": 41,
        "action_id": "41:1",
        "ts": 1785739004,
        "status": "ok",
        "duration_ms": 3000,
        "summary": "repo cloned",
        "result_hash": "a1b2c3d4e5f6",
    },
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "edge",
        "task_id": "t_example",
        "run_id": 41,
        "from_action_id": "41:1",
        "to_action_id": "41:2",
        "edge_kind": "data",
        "ts": 1785739004,
    },
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "run_footer",
        "task_id": "t_example",
        "run_id": 41,
        "ts": 1785739008,
        "outcome": "completed",
        "event_count": 7,
    },
    {
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "diagnosis_result",
        "diagnosis_version": "hermes.loop_diagnostics.diagnosis.v1",
        "task_id": "t_example",
        "run_id": 42,
        "status": "root_cause_found",
        "root_cause_action_ids": ["42:1"],
        "confidence": 0.9,
        "category": "input_invalid",
        "propagation_path": ["42:2", "42:1"],
        "explanation": "root cause is the clone failure",
        "interventions": [
            {
                "kind": "retry_from_checkpoint",
                "action_id": "42:1",
                "rationale": "retry after correcting remote URL",
                "payload": {"checkpoint_action_id": "42:0"},
            }
        ],
        "evidence": {
            "failed_action_id": "42:2",
            "trace_event_count": 7,
            "missing_action_ends": [],
            "malformed_lines": [],
            "duplicate_hashes": [],
        },
    },
]

INVALID_TRACE_RECORDS = [
    {"kind": "bogus"},
    {  # action_start missing required action_id
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "action_start",
        "task_id": "t_x",
        "run_id": 1,
        "ts": 1,
        "action_kind": "tool_call",
    },
    {  # action_end with invalid status enum
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "action_end",
        "task_id": "t_x",
        "run_id": 1,
        "action_id": "1:1",
        "ts": 1,
        "status": "nope",
        "duration_ms": 5,
    },
    {  # edge with invalid edge_kind
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "edge",
        "task_id": "t_x",
        "run_id": 1,
        "from_action_id": "1:1",
        "to_action_id": "1:2",
        "edge_kind": "sideways",
        "ts": 1,
    },
    {  # action_start with stray unknown field (additionalProperties=false)
        "schema_version": "hermes.loop_diagnostics.v1",
        "kind": "action_start",
        "task_id": "t_x",
        "run_id": 1,
        "action_id": "1:1",
        "ts": 1,
        "action_kind": "tool_call",
        "stray_raw_secret": "should never be stored",
    },
]


def _load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _branch_defs(schema: dict) -> dict:
    """Map branch names to their $defs objects from the union root."""
    defs = schema["$defs"]
    out = {}
    for branch in schema.get("oneOf", []):
        ref = branch.get("$ref", "")
        name = ref.rsplit("/", 1)[-1]
        if name in defs:
            out[name] = defs[name]
    return out


def _validate_record(record: dict, schema: dict) -> None:
    """Minimal validator for the loop-diagnostics union contract."""
    assert isinstance(record, dict), "record must be an object"

    kind = record.get("kind") or ""
    branch_name = KIND_TO_BRANCH.get(kind)
    assert branch_name is not None, f"unknown kind {kind!r}"

    branch = _branch_defs(schema).get(branch_name)
    assert branch is not None, f"no branch def for {branch_name}"

    # schema_version const — every branch carries it.
    if branch_name != "DiagnosisResult":
        assert (
            record.get("schema_version") == "hermes.loop_diagnostics.v1"
        ), "schema_version const mismatch"

    # required fields
    for field in branch.get("required", []):
        assert field in record, f"{branch_name} missing required {field!r}"

    # additionalProperties: false
    if branch.get("additionalProperties") is False:
        allowed = set(branch.get("properties", {}).keys())
        stray = set(record.keys()) - allowed
        assert not stray, f"{branch_name} has stray fields: {sorted(stray)}"

    # enum fields (resolve $ref first — ActionStatus, ActionKind,
    # FailureCategory etc. are declared as $defs and referenced)
    for field, spec in branch.get("properties", {}).items():
        if field not in record or record[field] is None:
            continue
        ref = spec.get("$ref")
        if ref:
            ref_name = ref.rsplit("/", 1)[-1]
            resolved = schema.get("$defs", {}).get(ref_name, {})
            if resolved:
                spec = resolved
        enum = spec.get("enum")
        if enum is not None:
            assert record[field] in enum, (
                f"{branch_name}.{field} {record[field]!r} not in {enum}"
            )
        # type check where cheap and not null-union
        typ = spec.get("type")
        if isinstance(typ, str) and typ != "object" and not isinstance(
            spec.get("const"), str
        ):
            # allow nullable fields: type can be [type, "null"] — skip those
            if isinstance(typ, str) and record[field] is not None:
                assert isinstance(record[field], _TYPE_FN[typ]), (
                    f"{branch_name}.{field} wrong type {type(record[field]).__name__}"
                )


_TYPE_FN = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def test_schema_file_parses():
    schema = _load_schema()
    assert schema["$id"] == "urn:hermes-agent:schema:loop-diagnostics:v1"
    assert schema["title"] == "Hermes Loop Diagnostics v1"
    assert len(schema["oneOf"]) == 6
    for branch in schema["oneOf"]:
        assert branch["$ref"].startswith("#/$defs/")


def test_every_union_kind_has_branch():
    schema = _load_schema()
    branches = _branch_defs(schema)
    assert set(KIND_TO_BRANCH.values()) <= set(branches.keys())


@pytest.mark.parametrize("record", VALID_TRACE_RECORDS, ids=lambda r: r["kind"])
def test_valid_records_conform(record):
    schema = _load_schema()
    _validate_record(record, schema)


@pytest.mark.parametrize("record", INVALID_TRACE_RECORDS, ids=lambda r: r.get("kind", "bogus"))
def test_invalid_records_rejected(record):
    schema = _load_schema()
    with pytest.raises(AssertionError):
        _validate_record(record, schema)
