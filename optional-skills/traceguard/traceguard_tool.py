"""TraceGuard tool for RLM-style evidence-gated synthesis."""

from __future__ import annotations

import json
from typing import Any

from traceguard import normalize_allowed_evidence_manifest
from traceguard import validate_parent_synthesis
from tools.registry import registry


TRACEGUARD_VALIDATE_SCHEMA = {
    "name": "traceguard_validate",
    "description": (
        "Validate structured parent-synthesis claims against accepted child "
        "evidence handles. Use this for RLM-style synthesis where every claimed "
        "fact must cite a fact_id plus matching evidence_chunk_id/chunk_id from "
        "the evidence manifest."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "evidence_manifest": {
                "type": "array",
                "description": (
                    "Accepted child evidence handles. Each item should include "
                    "fact_id, chunk_id or evidence_chunk_id, text or quoted_evidence, "
                    "and optionally child_call_id."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "evidence_chunk_id": {"type": "string"},
                        "text": {"type": "string"},
                        "quoted_evidence": {"type": "string"},
                        "child_call_id": {"type": "string"},
                    },
                },
            },
            "parent_synthesis": {
                "type": "object",
                "description": (
                    "Structured parent answer to validate. Claim-bearing fields "
                    "may appear in result.retained_facts, result.observed_facts, "
                    "result.facts, result.retained_evidence, result.observed_evidence, "
                    "or evidence_references."
                ),
            },
        },
        "required": ["evidence_manifest", "parent_synthesis"],
    },
}


def traceguard_validate(
    evidence_manifest: Any,
    parent_synthesis: Any,
) -> str:
    """Validate a parent synthesis and return a JSON string result."""
    if not isinstance(evidence_manifest, list):
        return json.dumps(
            {
                "success": False,
                "error": "evidence_manifest must be a list of evidence handle objects",
            },
            ensure_ascii=False,
        )
    if not isinstance(parent_synthesis, dict):
        return json.dumps(
            {
                "success": False,
                "error": "parent_synthesis must be a structured JSON object",
            },
            ensure_ascii=False,
        )

    result = validate_parent_synthesis(
        evidence_manifest=evidence_manifest,
        parent_synthesis=parent_synthesis,
    )
    return json.dumps(
        {
            "success": True,
            "traceguard": result.to_dict(),
            "normalized_evidence_manifest": list(
                normalize_allowed_evidence_manifest(evidence_manifest)
            ),
        },
        ensure_ascii=False,
    )


def _handle_traceguard_validate(args: dict, **kwargs: Any) -> str:
    return traceguard_validate(
        evidence_manifest=args.get("evidence_manifest"),
        parent_synthesis=args.get("parent_synthesis"),
    )


registry.register(
    name="traceguard_validate",
    toolset="traceguard",
    schema=TRACEGUARD_VALIDATE_SCHEMA,
    handler=_handle_traceguard_validate,
)
