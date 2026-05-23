from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from agent.memory_compiler import compile_memory_patterns


SUPPORTED_MEMORY_BLOCK_TYPES = (
    "human_profile",
    "persona",
    "collaboration_style",
    "project_context",
    "procedural_rules",
    "safety_policy",
    "methodology",
    "current_task_state",
)

MEMORY_BLOCK_VERSION = "0.1"
MEMORY_BLOCK_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_review_candidates_only": True,
}
MEMORY_BLOCK_MUTATION_POLICY = {
    "mutation_policy": "proposal_only",
    "direct_write_allowed": False,
}


def normalize_memory_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic memory-block candidate envelope without side effects."""
    normalized = deepcopy(dict(block))
    metadata = _as_dict(normalized.get("metadata"))
    source_pattern_ids = _sorted_strings(normalized.get("source_pattern_ids", []))
    source_fact_ids = _sorted_strings(normalized.get("source_fact_ids", []))
    policy = {**MEMORY_BLOCK_POLICY, **_as_dict(normalized.get("policy"))}
    mutation = {
        **MEMORY_BLOCK_MUTATION_POLICY,
        "mutation_policy": normalized.get("mutation_policy", MEMORY_BLOCK_MUTATION_POLICY["mutation_policy"]),
        "direct_write_allowed": normalized.get(
            "direct_write_allowed",
            MEMORY_BLOCK_MUTATION_POLICY["direct_write_allowed"],
        ),
    }

    candidate = {
        "block_id": normalized.get("block_id"),
        "block_type": normalized.get("block_type"),
        "status": normalized.get("status", "review_required"),
        "project_scope": normalized.get("project_scope"),
        "content": deepcopy(normalized.get("content")),
        "source_pattern_ids": source_pattern_ids,
        "source_fact_ids": source_fact_ids,
        "metadata": metadata,
        "version": str(normalized.get("version", MEMORY_BLOCK_VERSION)),
        "source_event_id": normalized.get("source_event_id", metadata.get("source_event_id")),
        "confidence": _confidence(normalized.get("confidence", metadata.get("confidence", 0.0))),
        "last_reviewed_at": normalized.get("last_reviewed_at"),
        "validity": deepcopy(normalized.get("validity", {"valid": None, "errors": []})),
        "mutation_policy": mutation["mutation_policy"],
        "direct_write_allowed": mutation["direct_write_allowed"],
        "policy": policy,
    }
    candidate["block_id"] = candidate["block_id"] or _block_id(candidate)
    candidate["validity"] = validate_memory_block_candidate(candidate)
    return candidate


def create_memory_block_candidate(
    block_type: str,
    content: Any,
    project_scope: str | None = None,
    source_pattern_ids: list[str] | tuple[str, ...] | None = None,
    source_fact_ids: list[str] | tuple[str, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a review-only memory block candidate; never apply or persist it."""
    return normalize_memory_block(
        {
            "block_type": block_type,
            "status": "review_required",
            "project_scope": project_scope,
            "content": deepcopy(content),
            "source_pattern_ids": list(source_pattern_ids or []),
            "source_fact_ids": list(source_fact_ids or []),
            "metadata": deepcopy(dict(metadata or {})),
            "version": MEMORY_BLOCK_VERSION,
            "policy": dict(MEMORY_BLOCK_POLICY),
            "mutation_policy": MEMORY_BLOCK_MUTATION_POLICY["mutation_policy"],
            "direct_write_allowed": MEMORY_BLOCK_MUTATION_POLICY["direct_write_allowed"],
        }
    )


def compile_blocks_from_compiler_result(
    compiler_result: Mapping[str, Any],
    project_scope: str | None = None,
) -> list[dict[str, Any]]:
    """Convert Memory Compiler candidates into review-only Memory Blocks."""
    source = deepcopy(dict(compiler_result))
    if "procedure_block_candidate" not in source and "facts" in source:
        source = compile_memory_patterns(
            source.get("facts", []),
            project_scope=project_scope or source.get("project_scope"),
        )

    procedure = source.get("procedure_block_candidate")
    if not isinstance(procedure, Mapping):
        return []

    scope = project_scope if project_scope is not None else procedure.get("project_scope", source.get("project_scope"))
    source_pattern_ids = list(procedure.get("source_pattern_ids", []))
    source_fact_ids = _sorted_strings(procedure.get("source_fact_ids", [])) or _fact_ids_from_compiler_result(source)
    content = {
        "rules": deepcopy(list(procedure.get("rules", []))),
        "safety_notes": deepcopy(list(procedure.get("safety_notes", []))),
        "compiler_candidate_type": procedure.get("block_type"),
        "applied": False,
    }
    return [
        create_memory_block_candidate(
            "procedural_rules",
            content,
            project_scope=scope,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
            metadata={
                "source": "memory_compiler",
                "source_event_id": source.get("source_event_id"),
                "compiler_policy": deepcopy(source.get("policy", {})),
            },
        )
    ]


def validate_memory_block_candidate(block: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    block_type = block.get("block_type")
    policy = _as_dict(block.get("policy"))

    if block_type not in SUPPORTED_MEMORY_BLOCK_TYPES:
        errors.append(f"unsupported_block_type:{block_type}")
    if block.get("status") != "review_required":
        errors.append("status_must_be_review_required")
    if block.get("mutation_policy") != "proposal_only":
        errors.append("mutation_policy_must_be_proposal_only")
    if block.get("direct_write_allowed") is not False:
        errors.append("direct_write_allowed_must_be_false")
    for key, expected in MEMORY_BLOCK_POLICY.items():
        if policy.get(key) is not expected:
            errors.append(f"policy_{key}_must_be_{str(expected).lower()}")
    if block.get("version") != MEMORY_BLOCK_VERSION:
        errors.append("unsupported_version")
    if not block.get("block_id"):
        errors.append("missing_block_id")
    if "content" not in block:
        errors.append("missing_content")

    return {"valid": not errors, "errors": errors}


def explain_memory_block_candidate(block: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_memory_block(block)
    return {
        "block_id": normalized["block_id"],
        "block_type": normalized["block_type"],
        "status": normalized["status"],
        "project_scope": normalized["project_scope"],
        "source_pattern_count": len(normalized["source_pattern_ids"]),
        "source_fact_count": len(normalized["source_fact_ids"]),
        "review_required": True,
        "applied": False,
        "policy": dict(normalized["policy"]),
        "validity": deepcopy(normalized["validity"]),
    }


def recommend_memory_block_action(block: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_memory_block(block)
    validity = normalized["validity"]
    if not validity["valid"]:
        action = "reject_candidate"
        reason = "The memory block candidate violates the v0.1 validation contract."
    else:
        action = "review_candidate"
        reason = "The memory block candidate is valid but requires review before any governed proposal path."
    return {
        "action": action,
        "reason": reason,
        "creates_review_candidates_only": True,
        "policy": dict(MEMORY_BLOCK_POLICY),
    }


def _block_id(block: Mapping[str, Any]) -> str:
    identity = {
        "block_type": block.get("block_type"),
        "project_scope": block.get("project_scope"),
        "content": block.get("content"),
        "source_pattern_ids": list(block.get("source_pattern_ids", [])),
        "source_fact_ids": list(block.get("source_fact_ids", [])),
        "metadata": block.get("metadata", {}),
        "version": block.get("version", MEMORY_BLOCK_VERSION),
        "source_event_id": block.get("source_event_id"),
        "confidence": block.get("confidence", 0.0),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return f"memory-block:v0.1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _fact_ids_from_compiler_result(result: Mapping[str, Any]) -> list[str]:
    fact_ids: list[str] = []
    for pattern in result.get("patterns", []):
        if isinstance(pattern, Mapping):
            fact_ids.extend(_sorted_strings(pattern.get("fact_ids", [])))
    return _sorted_strings(fact_ids)


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _sorted_strings(values: Any) -> list[str]:
    if values is None:
        return []
    return sorted({str(value) for value in values})


def _confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
