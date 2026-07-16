"""Closed JSON schemas for the three planner-facing tools."""

from __future__ import annotations

from .approval import ApprovalScope
from .models import (
    ConflictRecord,
    DedupCluster,
    EvidenceSnapshot,
    FigureSpec,
    ManifestRecord,
    PlanV1,
    RunConfig,
    SourceCandidate,
    SourceQuality,
    VerdictRecord,
)

from .storage.postgres import CONTEXT_QUERY_IDS

_ARTIFACT_MODELS = {
    "plan": PlanV1,
    "run_config": RunConfig,
    "manifest": ManifestRecord,
    "source_candidate": SourceCandidate,
    "source_quality": SourceQuality,
    "evidence_snapshot": EvidenceSnapshot,
    "conflict_record": ConflictRecord,
    "dedup_cluster": DedupCluster,
    "verdict_record": VerdictRecord,
    "figure_spec": FigureSpec,
}
ARTIFACT_TYPES = tuple(_ARTIFACT_MODELS)


def _tool_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def _close_pattern_mappings(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "patternProperties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _close_pattern_mappings(value)
    elif isinstance(node, list):
        for value in node:
            _close_pattern_mappings(value)


def _closed_model_schema(model) -> dict:
    schema = model.model_json_schema(mode="validation")
    _close_pattern_mappings(schema)
    return schema


PLAN_CONTEXT_READ_SCHEMA = _tool_schema(
    "plan_context_read",
    "Read bounded planner context through a versioned query identifier; never accepts SQL or a DSN.",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_id": {
                "type": "string",
                "enum": list(CONTEXT_QUERY_IDS),
            },
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "approval_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                    },
                    "run_id": {
                        "type": "string",
                        "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                        "maxLength": 128,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
        },
        "required": ["query_id", "parameters"],
    },
)

PLAN_ARTIFACT_WRITE_SCHEMA = _tool_schema(
    "plan_artifact_write",
    "Validate and atomically persist one versioned research artifact below the configured root.",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "artifact_type": {
                "type": "string",
                "enum": list(ARTIFACT_TYPES),
            },
            "artifact_id": {
                "type": "string",
                "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                "maxLength": 128,
            },
            "payload": {
                "oneOf": [
                    _closed_model_schema(model) for model in _ARTIFACT_MODELS.values()
                ],
            },
        },
        "required": ["artifact_type", "artifact_id", "payload"],
    },
)

PLAN_APPROVAL_REQUEST_SCHEMA = _tool_schema(
    "plan_approval_request",
    "Request native approval for an existing plan whose bytes match the expected hash and exact scope.",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "artifact_id": {
                "type": "string",
                "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                "maxLength": 128,
            },
            "expected_sha256": {
                "type": "string",
                "pattern": r"^[0-9a-f]{64}$",
            },
            "scope": _closed_model_schema(ApprovalScope),
        },
        "required": ["artifact_id", "expected_sha256", "scope"],
    },
)

__all__ = [
    "ARTIFACT_TYPES",
    "CONTEXT_QUERY_IDS",
    "PLAN_APPROVAL_REQUEST_SCHEMA",
    "PLAN_ARTIFACT_WRITE_SCHEMA",
    "PLAN_CONTEXT_READ_SCHEMA",
]
