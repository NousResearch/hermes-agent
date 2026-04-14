#!/usr/bin/env python3
"""Deterministic route-policy session topology router (v1).

Note: this is the model-route selector. Transport/topic routing uses
`ops/openclaw/continuity/session_topology_router.py`.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

from model_pool_policy_contract import load_pool_policy, policy_allowed_models, policy_route_entry
from session_topology_routing_policy_contract import (
    routing_policy_telegram_direct_worker_target_disallowed_lane_tokens,
    load_routing_policy,
    routing_policy_coding_allowed_readiness,
    routing_policy_coding_min_score,
    routing_policy_coding_require_qualification_signal,
    routing_policy_coding_strict_readiness_allowed_readiness,
    routing_policy_coding_strict_readiness_trigger_complexity_tiers,
    routing_policy_coding_strict_readiness_trigger_verification_classes,
    routing_policy_known_task_classes,
    routing_policy_task_family,
    routing_policy_qualification_signal_max_age_seconds_by_risk_tier,
    routing_policy_provider_evidence_max_age_seconds_by_risk_tier,
    routing_policy_legacy_missing_timestamp_grace_status_by_risk_tier,
)
from context_delta_transport import evaluate_anchor_preserving_summary_compaction, evaluate_context_delta_transport
from hybrid_retrieval_efficiency import evaluate_hybrid_retrieval_efficiency, validate_hybrid_retrieval_request


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_TOPOLOGY_SCHEMA = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "session_topology_contract.schema.json"
DEFAULT_POOL_POLICY_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "model_pool_policy_v1.json"
DEFAULT_POOL_POLICY_SCHEMA = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "model_pool_policy.schema.json"
DEFAULT_ROUTING_POLICY_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "session_topology_routing_policy_v1.json"
DEFAULT_ROUTING_POLICY_SCHEMA = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "session_topology_routing_policy.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "decisions.jsonl"
DEFAULT_TOKEN_VIOLATION_LEDGER = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "token_violations.jsonl"
)
DEFAULT_PROMPT_TOOL_CACHE_PATH = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "prompt_tool_cache_v1.json"
)
DEFAULT_PROMPT_TOOL_CACHE_TTL_SEC = 900
DEFAULT_CONTEXT_DELTA_CACHE_PATH = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "context_delta_cache_v1.json"
)
DEFAULT_CONTEXT_COMPACTION_CACHE_PATH = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "context_compaction_cache_v1.json"
)
DEFAULT_EVENT_BACKBONE_TYPED_LOG = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "typed_events.jsonl"
)
DEFAULT_EVENT_BACKBONE_DB = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "event_backbone.sqlite"
)
DEFAULT_EVENT_BACKBONE_DLQ = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "event_backbone_dlq.jsonl"
)
DEFAULT_EVENT_BACKBONE_METRICS = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "event_backbone_metrics.json"
)
DEFAULT_WORKFLOW_STATE_JOURNAL = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "workflow_state_machine_journal.jsonl"
)
DEFAULT_WORKFLOW_STATE_LATEST = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "session_topology_router" / "workflow_state_machine_latest.json"
)
DEFAULT_EVENT_BACKBONE_MAX_ATTEMPTS = 3
DEFAULT_EVENT_BACKBONE_BASE_BACKOFF_MS = 25
EVENT_BACKBONE_EVENT_SCHEMA = "clawd.orchestration.event.v1"
EVENT_BACKBONE_DLQ_SCHEMA = "clawd.orchestration.event_dlq.v1"
WORKFLOW_STATE_TRANSITION_SCHEMA = "clawd.session_topology.workflow_state_transition.v1"
WORKFLOW_STATE_LATEST_SCHEMA = "clawd.session_topology.workflow_state_snapshot.v1"
WORKFLOW_DAG_ORCHESTRATION_SCHEMA = "clawd.session_topology.workflow_dag_orchestration.v1"
WORKFLOW_ALLOWED_STATES = {"INIT", "ROUTE_BLOCKED", "ACTIVE", "RECOVERY_REQUIRED"}
WORKFLOW_HEALTHY_STATES = {"ACTIVE"}
WORKFLOW_ALLOWED_TRANSITIONS = {
    "INIT": {"ROUTE_BLOCKED", "ACTIVE", "RECOVERY_REQUIRED"},
    "ROUTE_BLOCKED": {"ROUTE_BLOCKED", "ACTIVE", "RECOVERY_REQUIRED"},
    "ACTIVE": {"ACTIVE", "ROUTE_BLOCKED", "RECOVERY_REQUIRED"},
    "RECOVERY_REQUIRED": {"ACTIVE", "ROUTE_BLOCKED", "RECOVERY_REQUIRED"},
}

ALLOWED_ROUTE_CLASSES = {"NO_LLM", "SPARK", "HEAVY"}
ALLOWED_RISK_TIERS = {"low", "medium", "high", "critical"}
RISK_TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
ALLOWED_SCOPE_SHAPES = {"single_surface", "multi_surface_disjoint", "multi_surface_coupled"}
ALLOWED_VERIFICATION_CLASSES = {"self_check", "validator_required", "validator_plus_human"}
ALLOWED_WORKER_TOPOLOGIES = {"single", "parallel_fanout", "staged_serial"}
ALLOWED_ROLLOUT_STAGES = {"canary", "active"}
ROUTER_LAYER = "route_policy"
TRANSPORT_DECISION_SCHEMA = "clawd.session_topology_transport_routing.decision.v1"
TRANSPORT_ROUTE_SCHEMA = "session.topology.routing_decision.v1"
ROUTE_CLASS_ORDER = {"NO_LLM": 0, "SPARK": 1, "HEAVY": 2}
PROMPT_GUARDRAIL_SCHEMA = "clawd.session_topology.prompt_guardrail_eval.v1"
PROMPT_TOKEN_VIOLATION_SCHEMA = "clawd.session_topology.prompt_token_violation.v1"

PROMPT_TOKEN_BUDGETS: Dict[str, int] = {
    "NO_LLM": 0,
    "SPARK": 1200,
    "HEAVY": 2400,
}

TOTAL_TOKEN_BUDGETS: Dict[str, int] = {
    "NO_LLM": 0,
    "SPARK": 2000,
    "HEAVY": 4000,
}

DEFAULT_OUTPUT_TOKEN_BUDGETS: Dict[str, int] = {
    "NO_LLM": 0,
    "SPARK": 600,
    "HEAVY": 1400,
}

TASK_CLASS_MODEL_FAMILY_DEFAULTS: Dict[str, str] = {
    "reading": "DeepSeek",
    "triage": "DeepSeek",
    "audit_compression": "DeepSeek",
    "research": "Gemini",
    "planning": "Gemini",
    "comparison": "Kimi",
    "implementation": "Codex",
    "code:generate": "Codex",
    "code:edit": "Codex",
    "code:review": "DeepSeek",
    "code:test": "Codex",
    "code:docs": "DeepSeek",
}

TASK_CLASS_MODEL_FAMILY_FALLBACKS: Dict[str, List[str]] = {
    "reading": ["Gemini", "Kimi", "Codex"],
    "triage": ["Gemini", "Codex"],
    "audit_compression": ["Gemini", "Kimi", "Codex"],
    "research": ["Kimi", "Codex"],
    "planning": ["Kimi", "Codex"],
    "comparison": ["Gemini", "Codex"],
    "implementation": ["Gemini", "DeepSeek"],
    "code:generate": ["Gemini", "Kimi", "DeepSeek"],
    "code:edit": ["Gemini", "DeepSeek", "Kimi"],
    "code:review": ["Gemini", "Codex", "Kimi"],
    "code:test": ["Gemini", "DeepSeek", "Kimi"],
    "code:docs": ["Gemini", "Codex", "Kimi"],
}

SYNTHESIS_TASK_CLASSES = {"reading", "triage", "audit_compression", "research", "planning", "comparison"}
CODING_TASK_CLASS_ROUTING_ALIASES: Dict[str, str] = {
    "code:generate": "implementation",
    "code:edit": "implementation",
    "code:review": "implementation",
    "code:test": "implementation",
    "code:docs": "implementation",
}
CODING_TASK_CLASSES = {"implementation", *CODING_TASK_CLASS_ROUTING_ALIASES.keys()}
CODING_MUTATION_TASK_CLASSES = {"implementation", "code:generate", "code:edit", "code:test", "code:docs"}
CODING_COMPLEXITY_TIERS = {"low", "moderate", "high"}
CODING_COMPLEXITY_TASK_CLASS_DEFAULTS: Dict[str, str] = {
    "implementation": "moderate",
    "code:generate": "high",
    "code:edit": "moderate",
    "code:review": "low",
    "code:test": "moderate",
    "code:docs": "low",
}
WORKER_LANE_ALLOWLIST = {
    "subagent_default",
    "main_session_tiny_exception",
}
WORKER_LANE_ALIASES = {
    "subagent": "subagent_default",
    "subagent-default": "subagent_default",
    "main-session-tiny-exception": "main_session_tiny_exception",
    "tiny_exception": "main_session_tiny_exception",
    "tiny-exception": "main_session_tiny_exception",
}
TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS: Tuple[str, ...] = (
    "codex-worker-plus-*",
    "codex-worker-pro",
    "codex-executioner",
)
QUALIFICATION_SIGNAL_READINESS_ORDER = {
    "qualified": 0,
    "provisional": 1,
    "unknown": 2,
    "hold": 3,
    "stale": 4,
}
CODING_HIGH_RISK_BAKEOFF_MIN_SCORE_0_100 = {
    "high": 85.0,
    "critical": 90.0,
}
PROPOSAL_PACKET_SCHEMA_VERSION = "proposal_packet.v1"
PROPOSAL_FIRST_DELTA_SPEC_SCHEMA = "clawd.session_topology.proposal_first_delta_spec.v1"
PROPOSAL_FLOW_STATE_SCHEMA = "clawd.session_topology.proposal_apply_archive_state.v1"
PROPOSAL_ARCHIVE_PACKET_SCHEMA_VERSION = "proposal_archive_packet.v1"
REGRESSION_RISK_PACKET_SCHEMA = "clawd.session_topology.regression_risk_packet.v1"
REGRESSION_RISK_PACKET_SCHEMA_BY_VERSION: Dict[str, str] = {
    "1.0": "clawd.session_topology.regression_risk_packet.v1",
    "2.0": "clawd.session_topology.regression_risk_packet.v2",
}
REGRESSION_RISK_PACKET_SUPPORTED_VERSIONS = set(REGRESSION_RISK_PACKET_SCHEMA_BY_VERSION.keys())
REGRESSION_RISK_PACKET_DEFAULT_VERSION = "1.0"
REGRESSION_RISK_SCORE_DIMENSIONS = [
    "blast_radius",
    "code_churn",
    "dependency_impact",
    "test_coverage_delta",
    "historical_instability",
    "critical_path_impact",
]
REGRESSION_RISK_REQUIRED_APPROVALS_BY_TIER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}
REGRESSION_RISK_BLOCKING_CLASSIFICATIONS = {"blocking", "non_blocking"}
REGRESSION_RISK_REPLAY_EVIDENCE_GATE_STATUSES = {"passed", "failed", "skipped"}
REFACTOR_RISK_PACKET_SCHEMA = "clawd.session_topology.refactor_risk_packet.v1"
REFACTOR_RISK_PACKET_VERSION = "1.0"
REFACTOR_RISK_DIMENSIONS = [
    "complexity",
    "blast_radius",
    "conceptual_clarity",
]
REFACTOR_RISK_REQUIRED_APPROVALS_BY_TIER = {
    "low": 0,
    "medium": 0,
    "high": 1,
    "critical": 2,
}
REFACTOR_RISK_ROLLBACK_STRATEGIES = {
    "revert_commit",
    "feature_flag_revert",
    "artifact_restore",
    "state_restore",
}
REFACTOR_DECOMPOSITION_MAX_FILES_PER_CHUNK = 3
PROPOSAL_PACKET_TASK_CLASSES = {
    "code:generate",
    "code:edit",
    "code:review",
    "code:test",
    "code:docs",
}
PROPOSAL_FLOW_PHASES = ["proposal", "apply", "archive"]
PROPOSAL_FLOW_PHASE_ALLOWLIST = set(PROPOSAL_FLOW_PHASES)
REFACTOR_DECOMPOSITION_TASK_CLASSES = set(PROPOSAL_PACKET_TASK_CLASSES)
PROPOSAL_ARCHIVE_REQUIRED_ARTIFACTS = [
    "proposal_packet",
    "approval_records",
    "delta_spec",
    "code_delta",
    "validation_results",
    "routing_audit",
]
PROPOSAL_APPROVAL_POLICY_BY_RISK: Dict[str, Dict[str, Any]] = {
    "low": {
        "proposal_checkpoint": "automatic",
        "proposal_min_approvers": 0,
        "post_implementation_approval": "automatic_merge",
        "post_implementation_min_approvers": 0,
    },
    "medium": {
        "proposal_checkpoint": "optional_operator_review",
        "proposal_min_approvers": 0,
        "post_implementation_approval": "optional_human_review",
        "post_implementation_min_approvers": 0,
    },
    "high": {
        "proposal_checkpoint": "mandatory_operator_approval",
        "proposal_min_approvers": 1,
        "post_implementation_approval": "mandatory_human_review",
        "post_implementation_min_approvers": 1,
    },
    "critical": {
        "proposal_checkpoint": "mandatory_multi_operator_approval",
        "proposal_min_approvers": 2,
        "post_implementation_approval": "mandatory_multi_person_review",
        "post_implementation_min_approvers": 2,
    },
}

TASK_TAXONOMY_PROFILE: Dict[str, Dict[str, Any]] = {
    "watchdog": {
        "taxonomy_tier": "T0",
        "default_route_class": "NO_LLM",
        "default_required_rollout_stage": "canary",
        "allow_escalation": False,
    },
    "worker_slice": {
        "implementation": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "escalation_route_class": "HEAVY",
            "escalation_required_rollout_stage": "active",
            "allow_escalation": True,
        },
        "code:generate": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "escalation_route_class": "HEAVY",
            "escalation_required_rollout_stage": "active",
            "allow_escalation": True,
        },
        "code:edit": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "escalation_route_class": "HEAVY",
            "escalation_required_rollout_stage": "active",
            "allow_escalation": True,
        },
        "code:review": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "code:test": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "escalation_route_class": "HEAVY",
            "escalation_required_rollout_stage": "active",
            "allow_escalation": True,
        },
        "code:docs": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "research": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "reading": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "triage": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "planning": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "comparison": {
            "taxonomy_tier": "T1",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
        "_default": {
            "taxonomy_tier": "T2",
            "default_route_class": "SPARK",
            "default_required_rollout_stage": "canary",
            "allow_escalation": True,
        },
    },
    "_default": {
        "taxonomy_tier": "T2",
        "default_route_class": "SPARK",
        "default_required_rollout_stage": "canary",
        "allow_escalation": True,
    },
}

ESCALATION_SIGNAL_FIELDS = {
    "quality_gate_failed": "quality_gate_failed",
    "unresolved_blocker": "unresolved_blocker",
    "explicit_criticality": "explicit_criticality",
    "previous_tier_failed": "previous_tier_failed",
}

FOLD_IN_TARGET_ALLOWLIST = {"canonical_doctrine", "roadmap_pair", "queue_continuity", "support_only"}


def _normalize_fold_in_target(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    if text not in FOLD_IN_TARGET_ALLOWLIST:
        return text
    return text


def _normalize_worker_lane(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None, None, None

    token = raw.lower().replace(" ", "_")
    if token in WORKER_LANE_ALLOWLIST:
        return token, raw, token

    aliased = WORKER_LANE_ALIASES.get(token)
    if aliased:
        return aliased, raw, token

    return None, raw, token


def _telegram_direct_worker_target_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _telegram_direct_worker_target_pattern_match(value: Any) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return any(fnmatch.fnmatchcase(token, pattern) for pattern in TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS)


def _session_key_agent_id(session_key: Any) -> Optional[str]:
    token = str(session_key or "").strip()
    if not token:
        return None
    parts = token.split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    agent_id = str(parts[1] or "").strip()
    return agent_id or None


def _is_support_only_helper_request(request: Mapping[str, Any]) -> bool:
    if request.get("support_only") is True:
        return True

    fold_in_target = _normalize_fold_in_target(request.get("fold_in_target"))
    return bool(fold_in_target == "support_only")


def _request_dispatch_contract(request: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = request.get("dispatch_contract")
    if isinstance(raw, Mapping):
        return raw
    return {}


def _dispatch_contract_field(request: Mapping[str, Any], key: str) -> Optional[str]:
    direct = request.get(key)
    if isinstance(direct, str) and direct.strip():
        return str(direct).strip()

    nested = _request_dispatch_contract(request).get(key)
    if isinstance(nested, str) and nested.strip():
        return str(nested).strip()
    return None


def _worker_allocation_contract_gate(
    *,
    request: Mapping[str, Any],
    session_kind: str,
    risk_tier: str,
    fold_in_target: Optional[str],
    require_worker_allocation_contract: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    details: Dict[str, Any] = {
        "schema": "clawd.session_topology.worker_allocation_contract.v1",
        "enforced": bool(require_worker_allocation_contract),
        "required_for_request": bool(require_worker_allocation_contract and session_kind == "worker_slice"),
        "session_kind": session_kind,
    }

    dispatch_contract_raw = request.get("dispatch_contract")
    if dispatch_contract_raw is not None and not isinstance(dispatch_contract_raw, Mapping):
        details["error"] = "dispatch_contract_invalid"
        details["detail"] = "expected_object"
        return False, "worker_allocation_contract_violation", details

    scope_shape = _dispatch_contract_field(request, "scope_shape")
    verification_class = _dispatch_contract_field(request, "verification_class")
    worker_topology = _dispatch_contract_field(request, "worker_topology")

    details.update(
        {
            "scope_shape": scope_shape,
            "verification_class": verification_class,
            "worker_topology": worker_topology,
            "fold_in_target": fold_in_target,
        }
    )

    for key, value, allowlist in (
        ("scope_shape", scope_shape, ALLOWED_SCOPE_SHAPES),
        ("verification_class", verification_class, ALLOWED_VERIFICATION_CLASSES),
        ("worker_topology", worker_topology, ALLOWED_WORKER_TOPOLOGIES),
    ):
        if value is None:
            continue
        if value not in allowlist:
            details["error"] = f"{key}_invalid"
            details[key] = value
            details["allowed"] = sorted(allowlist)
            return False, "worker_allocation_contract_violation", details

    if session_kind != "worker_slice":
        details["status"] = "not_required_non_worker_slice"
        return True, None, details

    missing_fields: List[str] = []
    if require_worker_allocation_contract:
        if scope_shape is None:
            missing_fields.append("scope_shape")
        if verification_class is None:
            missing_fields.append("verification_class")
        if worker_topology is None:
            missing_fields.append("worker_topology")
        if fold_in_target is None:
            missing_fields.append("fold_in_target")

    if missing_fields:
        details["error"] = "worker_allocation_fields_missing"
        details["missing_fields"] = missing_fields
        return False, "worker_allocation_contract_violation", details

    if scope_shape == "multi_surface_coupled" and worker_topology == "parallel_fanout":
        details["error"] = "coupled_scope_forbids_parallel_fanout"
        return False, "worker_allocation_contract_violation", details

    if risk_tier in {"high", "critical"} and verification_class == "self_check":
        details["error"] = "high_risk_requires_validator"
        return False, "worker_allocation_contract_violation", details

    if risk_tier in {"high", "critical"} and worker_topology == "parallel_fanout":
        details["error"] = "high_risk_forbids_parallel_fanout"
        return False, "worker_allocation_contract_violation", details

    details["status"] = "pass"
    return True, None, details


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_clone(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _cache_now_epoch() -> int:
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def _normalize_prompt_semantic(prompt: Any) -> str:
    raw = str(prompt or "")
    if not raw:
        return ""
    lint = _lint_invocation_prompt(raw)
    return str(lint.get("normalized_prompt") or "")


def _cache_exact_and_semantic_keys(
    *,
    topology: Any,
    request: Mapping[str, Any],
    gate_decisions: List[Mapping[str, Any]],
    transport_decision: Optional[Mapping[str, Any]],
    pool_policy: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    require_transport_decision: bool,
    require_worker_allocation_contract: bool,
    require_telegram_direct_heavy_offload: bool,
    require_proposal_packet_for_coding: bool,
    require_regression_risk_packet_for_coding: bool,
    require_refactor_risk_packet_for_coding: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    request_obj = dict(request)
    semantic_request_obj = dict(request_obj)
    semantic_request_obj["invocation_prompt"] = _normalize_prompt_semantic(request_obj.get("invocation_prompt"))

    base_payload = {
        "schema": "clawd.session_topology.prompt_tool_cache_input.v1",
        "router_layer": ROUTER_LAYER,
        "topology": topology,
        "request": request_obj,
        "gate_decisions": list(gate_decisions or []),
        "transport_decision": transport_decision,
        "pool_policy_id": pool_policy.get("policy_id") if isinstance(pool_policy, Mapping) else None,
        "pool_policy": pool_policy,
        "routing_policy_id": routing_policy.get("policy_id") if isinstance(routing_policy, Mapping) else None,
        "routing_policy": routing_policy,
        "require_transport_decision": bool(require_transport_decision),
        "require_worker_allocation_contract": bool(require_worker_allocation_contract),
        "require_telegram_direct_heavy_offload": bool(require_telegram_direct_heavy_offload),
        "require_proposal_packet_for_coding": bool(require_proposal_packet_for_coding),
        "require_regression_risk_packet_for_coding": bool(require_regression_risk_packet_for_coding),
        "require_refactor_risk_packet_for_coding": bool(require_refactor_risk_packet_for_coding),
    }
    semantic_payload = dict(base_payload)
    semantic_payload["request"] = semantic_request_obj

    exact_blob = stable_json_dumps(base_payload)
    semantic_blob = stable_json_dumps(semantic_payload)
    exact_key = hashlib.sha256(exact_blob.encode("utf-8")).hexdigest()
    semantic_key = hashlib.sha256(semantic_blob.encode("utf-8")).hexdigest()

    fingerprint = {
        "exact_key": exact_key,
        "semantic_key": semantic_key,
        "request_sha256": hashlib.sha256(stable_json_dumps(request_obj).encode("utf-8")).hexdigest(),
        "semantic_request_sha256": hashlib.sha256(stable_json_dumps(semantic_request_obj).encode("utf-8")).hexdigest(),
        "topology_sha256": hashlib.sha256(stable_json_dumps(topology if isinstance(topology, Mapping) else {}).encode("utf-8")).hexdigest(),
        "gate_decisions_sha256": hashlib.sha256(stable_json_dumps({"gate_decisions": list(gate_decisions or [])}).encode("utf-8")).hexdigest(),
        "transport_decision_sha256": hashlib.sha256(stable_json_dumps(transport_decision if isinstance(transport_decision, Mapping) else {}).encode("utf-8")).hexdigest(),
    }
    return exact_key, semantic_key, fingerprint


def _load_prompt_tool_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": "clawd.session_topology.prompt_tool_cache.v1",
            "updated_at": now_iso(),
            "entries": {},
        }

    payload = load_json_file(path)
    if not isinstance(payload, Mapping):
        raise ValueError("cache_store_not_object")

    entries = payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {}
    return {
        "schema": str(payload.get("schema") or "clawd.session_topology.prompt_tool_cache.v1"),
        "updated_at": str(payload.get("updated_at") or now_iso()),
        "entries": dict(entries),
    }


def _save_prompt_tool_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "clawd.session_topology.prompt_tool_cache.v1",
        "updated_at": now_iso(),
        "entries": payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {},
    }
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cache_lookup(
    *,
    store: Dict[str, Any],
    exact_key: str,
    semantic_key: str,
    now_epoch: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    entries = store.get("entries") if isinstance(store.get("entries"), Mapping) else {}
    entries_obj: Dict[str, Any] = dict(entries)

    expired_keys: List[str] = []
    for key, entry in entries_obj.items():
        if not isinstance(entry, Mapping):
            expired_keys.append(str(key))
            continue
        expires_at_epoch = int(entry.get("expires_at_epoch") or 0)
        if expires_at_epoch <= now_epoch:
            expired_keys.append(str(key))

    for key in expired_keys:
        entries_obj.pop(key, None)

    store["entries"] = entries_obj

    hit_entry: Optional[Dict[str, Any]] = None
    hit_type: Optional[str] = None
    if exact_key in entries_obj:
        entry = entries_obj.get(exact_key)
        if isinstance(entry, Mapping):
            hit_entry = dict(entry)
            hit_type = "exact"
    elif semantic_key in entries_obj:
        entry = entries_obj.get(semantic_key)
        if isinstance(entry, Mapping):
            hit_entry = dict(entry)
            hit_type = "semantic"

    return hit_entry, {
        "expired_entry_count": len(expired_keys),
        "entry_count": len(entries_obj),
        "hit_type": hit_type,
    }


def _cache_store_result(
    *,
    store: Dict[str, Any],
    key: str,
    now_epoch: int,
    ttl_sec: int,
    result: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> None:
    entries = store.get("entries") if isinstance(store.get("entries"), Mapping) else {}
    entries_obj: Dict[str, Any] = dict(entries)
    entries_obj[key] = {
        "created_at": now_iso(),
        "created_at_epoch": now_epoch,
        "expires_at_epoch": now_epoch + max(1, int(ttl_sec)),
        "fingerprint": dict(fingerprint),
        "result": _json_clone(dict(result)),
    }
    store["entries"] = entries_obj


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_tokens(text: str) -> int:
    compact = str(text or "")
    if not compact:
        return 0
    return max(1, int(math.ceil(len(compact) / 4.0)))


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _truncate_preview(text: str, *, limit: int = 280) -> str:
    token = str(text or "")
    if len(token) <= limit:
        return token
    return token[: limit - 1] + "…"


def _lint_invocation_prompt(prompt: str) -> Dict[str, Any]:
    original = str(prompt or "")
    original_lines = original.splitlines()

    trimmed_lines: List[str] = []
    removed_consecutive_duplicates = 0
    removed_extra_blank_lines = 0
    blank_run = 0
    previous_nonempty: Optional[str] = None

    for raw in original_lines:
        line = raw.rstrip()
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                removed_extra_blank_lines += 1
                continue
            trimmed_lines.append("")
            previous_nonempty = None
            continue

        blank_run = 0
        normalized = line.strip()
        if previous_nonempty is not None and normalized == previous_nonempty:
            removed_consecutive_duplicates += 1
            continue

        trimmed_lines.append(normalized)
        previous_nonempty = normalized

    trimmed_text = "\n".join(trimmed_lines).strip()

    return {
        "status": "linted",
        "original": {
            "chars": len(original),
            "lines": len(original_lines),
            "estimated_tokens": _estimate_tokens(original),
            "sha256": _prompt_hash(original),
            "preview": _truncate_preview(original),
        },
        "trimmed": {
            "chars": len(trimmed_text),
            "lines": len(trimmed_text.splitlines()) if trimmed_text else 0,
            "estimated_tokens": _estimate_tokens(trimmed_text),
            "sha256": _prompt_hash(trimmed_text),
            "preview": _truncate_preview(trimmed_text),
        },
        "mutations": {
            "removed_consecutive_duplicates": removed_consecutive_duplicates,
            "removed_extra_blank_lines": removed_extra_blank_lines,
            "changed": trimmed_text != original,
        },
        "normalized_prompt": trimmed_text,
    }


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def gate_schema(topology: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(topology),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        },
    )


def _match_selector(selector: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    def task_class_variants(task_class: str) -> set[str]:
        variants = {task_class}
        alias = CODING_TASK_CLASS_ROUTING_ALIASES.get(task_class)
        if alias:
            variants.add(alias)
        return variants

    for key in ("session_kind", "task_class", "risk_tier"):
        expected = str(selector.get(key) or "").strip()
        actual = str(request.get(key) or "").strip()
        if not expected:
            return False
        if expected == "*":
            continue
        if key == "task_class":
            if expected not in task_class_variants(actual):
                return False
            continue
        if expected != actual:
            return False
    return True


def _resolve_rule(topology: Mapping[str, Any], request: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    rules = topology.get("rules") if isinstance(topology.get("rules"), list) else []
    matches: List[Dict[str, Any]] = []
    for idx, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, dict):
            continue
        selector = raw_rule.get("selector") if isinstance(raw_rule.get("selector"), dict) else {}
        if _match_selector(selector, request):
            candidate = dict(raw_rule)
            candidate["_index"] = idx
            matches.append(candidate)

    if not matches:
        return None, []

    matches.sort(key=lambda row: (int(row.get("priority") or 0), str(row.get("rule_id") or ""), int(row.get("_index") or 0)))
    return matches[0], matches


def _request_route_lock(request: Mapping[str, Any]) -> Dict[str, str]:
    lock_raw = request.get("route_lock") if isinstance(request.get("route_lock"), Mapping) else {}

    def pick(*keys: str) -> str:
        for key in keys:
            value = request.get(key)
            if isinstance(lock_raw, Mapping) and key in lock_raw:
                value = lock_raw.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return ""

    out: Dict[str, str] = {}
    route_class = pick("requested_route_class", "route_class")
    required_stage = pick("requested_required_rollout_stage", "required_rollout_stage")
    model_key = pick("requested_model_key", "requested_model", "model_key", "selected_model")
    rule_id = pick("requested_rule_id", "rule_id")

    if route_class:
        out["route_class"] = route_class
    if required_stage:
        out["required_rollout_stage"] = required_stage
    if model_key:
        out["model_key"] = model_key
    if rule_id:
        out["rule_id"] = rule_id
    return out


def _request_transport_binding(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("transport_route") if isinstance(request.get("transport_route"), Mapping) else {}

    def pick_str(*keys: str) -> str:
        for key in keys:
            value = raw.get(key) if key in raw else request.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return ""

    out: Dict[str, Any] = {}
    transport_key = pick_str("transport_key")
    lane_name = pick_str("lane_name")
    agent_id = pick_str("agent_id")
    session_key = pick_str("session_key")

    if transport_key:
        out["transport_key"] = transport_key
    if lane_name:
        out["lane_name"] = lane_name
    if agent_id:
        out["agent_id"] = agent_id
    if session_key:
        out["session_key"] = session_key

    thread_raw = raw.get("message_thread_id") if isinstance(raw, Mapping) and "message_thread_id" in raw else request.get("message_thread_id")
    if thread_raw is not None:
        if isinstance(thread_raw, bool):
            out["message_thread_id"] = thread_raw
        elif isinstance(thread_raw, int):
            out["message_thread_id"] = thread_raw
        elif isinstance(thread_raw, str):
            token = thread_raw.strip().lower()
            if token in {"", "none", "null", "main"}:
                out["message_thread_id"] = None
            elif token.isdigit():
                out["message_thread_id"] = int(token)
            else:
                out["message_thread_id"] = thread_raw
        else:
            out["message_thread_id"] = None

    return out


def _request_agent_binding(request: Mapping[str, Any]) -> Dict[str, str]:
    lock_raw = request.get("route_lock") if isinstance(request.get("route_lock"), Mapping) else {}

    def pick(*keys: str) -> str:
        for key in keys:
            value = lock_raw.get(key) if key in lock_raw else request.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return ""

    out: Dict[str, str] = {}
    agent_id = pick("requested_agent_id", "agent_id")
    session_key = pick("requested_session_key", "session_key")
    lane_name = pick("requested_lane_name", "lane_name")

    if agent_id:
        out["agent_id"] = agent_id
    if session_key:
        out["session_key"] = session_key
    if lane_name:
        out["lane_name"] = lane_name
    return out


def _task_taxonomy_for_request(
    request: Mapping[str, Any],
    topology: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    session_kind = str(request.get("session_kind") or "").strip()
    task_class = str(request.get("task_class") or "").strip()

    session_profile = TASK_TAXONOMY_PROFILE.get(session_kind)
    if isinstance(session_profile, Mapping):
        if "default_route_class" in session_profile:
            profile = dict(session_profile)
        else:
            profile = dict(
                session_profile.get(task_class)
                if isinstance(session_profile.get(task_class), Mapping)
                else session_profile.get("_default")
                if isinstance(session_profile.get("_default"), Mapping)
                else TASK_TAXONOMY_PROFILE.get("_default", {})
            )
    else:
        profile = dict(TASK_TAXONOMY_PROFILE.get("_default", {}))

    default_route_class = str(profile.get("default_route_class") or "").strip()
    if default_route_class not in ALLOWED_ROUTE_CLASSES:
        default_route_class = str(topology.get("default_route_class") or "SPARK").strip() or "SPARK"

    default_required_rollout_stage = str(profile.get("default_required_rollout_stage") or "").strip()
    if default_required_rollout_stage not in ALLOWED_ROLLOUT_STAGES:
        default_required_rollout_stage = str(topology.get("default_required_rollout_stage") or "canary").strip() or "canary"

    default_model_family, fallback_model_families = routing_policy_task_family(routing_policy, task_class)
    if default_model_family in fallback_model_families:
        fallback_model_families = [fam for fam in fallback_model_families if fam != default_model_family]

    profile["session_kind"] = session_kind
    profile["task_class"] = task_class
    profile["default_route_class"] = default_route_class
    profile["default_required_rollout_stage"] = default_required_rollout_stage
    profile["taxonomy_tier"] = str(profile.get("taxonomy_tier") or "T2")
    profile["allow_escalation"] = bool(profile.get("allow_escalation") is True)
    profile["default_model_family"] = default_model_family
    profile["fallback_model_families"] = fallback_model_families
    profile["routing_policy_id"] = routing_policy.get("policy_id") if isinstance(routing_policy, Mapping) else None
    return profile


def _normalize_escalation_evidence(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("escalation_evidence") if isinstance(request.get("escalation_evidence"), Mapping) else {}
    out: Dict[str, Any] = {}
    for key in ESCALATION_SIGNAL_FIELDS:
        out[key] = bool(raw.get(key) is True)

    refs: List[str] = []
    for item in (raw.get("artifact_refs") if isinstance(raw.get("artifact_refs"), list) else []):
        text = str(item or "").strip()
        if text:
            refs.append(text)
    out["artifact_refs"] = refs

    notes = str(raw.get("notes") or "").strip()
    if notes:
        out["notes"] = notes

    return out


def _normalize_prompt_guardrail_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("prompt_guardrails") if isinstance(request.get("prompt_guardrails"), Mapping) else {}
    out: Dict[str, Any] = {}

    for key in ("requested_output_tokens", "max_prompt_tokens", "max_total_tokens"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except Exception:
            out[key] = None
            continue
        out[key] = parsed

    return out


def _evaluate_prompt_guardrails(
    *,
    request: Mapping[str, Any],
    route_class: Optional[str],
    required_rollout_stage: Optional[str],
) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    prompt_raw = request.get("invocation_prompt")
    route = str(route_class or "").strip() or "SPARK"
    hard_prompt_budget = int(PROMPT_TOKEN_BUDGETS.get(route, PROMPT_TOKEN_BUDGETS["SPARK"]))
    hard_total_budget = int(TOTAL_TOKEN_BUDGETS.get(route, TOTAL_TOKEN_BUDGETS["SPARK"]))
    hard_output_budget = int(DEFAULT_OUTPUT_TOKEN_BUDGETS.get(route, DEFAULT_OUTPUT_TOKEN_BUDGETS["SPARK"]))

    guardrail_request = _normalize_prompt_guardrail_request(request)

    if prompt_raw is None:
        details = {
            "schema": PROMPT_GUARDRAIL_SCHEMA,
            "status": "not_provided",
            "route_class": route,
            "required_rollout_stage": required_rollout_stage,
            "hard_budgets": {
                "max_prompt_tokens": hard_prompt_budget,
                "max_total_tokens": hard_total_budget,
                "default_output_tokens": hard_output_budget,
            },
            "requested_guardrails": guardrail_request,
        }
        return True, None, details, None

    prompt_text = str(prompt_raw)
    lint = _lint_invocation_prompt(prompt_text)
    trimmed_prompt = str(lint.get("normalized_prompt") or "")
    prompt_tokens = int(((lint.get("trimmed") or {}).get("estimated_tokens") or 0))

    req_prompt_budget = guardrail_request.get("max_prompt_tokens")
    req_total_budget = guardrail_request.get("max_total_tokens")
    req_output_tokens = guardrail_request.get("requested_output_tokens")

    max_prompt_tokens = hard_prompt_budget
    if isinstance(req_prompt_budget, int) and req_prompt_budget > 0:
        max_prompt_tokens = min(max_prompt_tokens, req_prompt_budget)

    max_total_tokens = hard_total_budget
    if isinstance(req_total_budget, int) and req_total_budget > 0:
        max_total_tokens = min(max_total_tokens, req_total_budget)

    requested_output_tokens = hard_output_budget
    if isinstance(req_output_tokens, int) and req_output_tokens >= 0:
        requested_output_tokens = min(hard_output_budget, req_output_tokens)

    violations: List[str] = []
    if route == "NO_LLM" and prompt_tokens > 0:
        violations.append("prompt_not_allowed_for_no_llm_route")
    if prompt_tokens > max_prompt_tokens:
        violations.append("prompt_token_budget_exceeded")
    if (prompt_tokens + requested_output_tokens) > max_total_tokens:
        violations.append("total_token_budget_exceeded")

    details = {
        "schema": PROMPT_GUARDRAIL_SCHEMA,
        "status": "pass" if not violations else "blocked",
        "route_class": route,
        "required_rollout_stage": required_rollout_stage,
        "prompt_lint": {
            "status": lint.get("status"),
            "original": lint.get("original"),
            "trimmed": lint.get("trimmed"),
            "mutations": lint.get("mutations"),
        },
        "normalized_prompt": trimmed_prompt,
        "budgets": {
            "max_prompt_tokens": max_prompt_tokens,
            "max_total_tokens": max_total_tokens,
            "requested_output_tokens": requested_output_tokens,
            "hard_max_prompt_tokens": hard_prompt_budget,
            "hard_max_total_tokens": hard_total_budget,
            "hard_default_output_tokens": hard_output_budget,
        },
        "requested_guardrails": guardrail_request,
        "effective_prompt_tokens": prompt_tokens,
        "effective_total_tokens": prompt_tokens + requested_output_tokens,
        "violations": violations,
    }

    violation_row: Optional[Dict[str, Any]] = None
    if violations:
        violation_row = {
            "schema": PROMPT_TOKEN_VIOLATION_SCHEMA,
            "recorded_at": now_iso(),
            "route_class": route,
            "required_rollout_stage": required_rollout_stage,
            "session_kind": request.get("session_kind"),
            "task_class": request.get("task_class"),
            "risk_tier": request.get("risk_tier"),
            "violations": violations,
            "prompt_lint": details.get("prompt_lint"),
            "budgets": details.get("budgets"),
            "effective_prompt_tokens": details.get("effective_prompt_tokens"),
            "effective_total_tokens": details.get("effective_total_tokens"),
        }

    if violations:
        return False, "prompt_token_budget_exceeded", details, violation_row
    return True, None, details, violation_row


def _route_rank(route_class: Optional[str]) -> int:
    if not route_class:
        return -1
    return int(ROUTE_CLASS_ORDER.get(str(route_class), -1))


def _model_family_from_model_key(model_key: Optional[str]) -> Optional[str]:
    token = str(model_key or "").strip().lower()
    if not token:
        return None
    if token.startswith("openai-codex/") or "codex" in token:
        return "Codex"
    if token.startswith("deepseek/") or "deepseek" in token:
        return "DeepSeek"
    if (token.startswith("google/") and "gemini" in token) or "gemini" in token:
        return "Gemini"
    if token.startswith("moonshot/") or "kimi" in token:
        return "Kimi"
    return "Other"


def _coerce_float(raw: Any) -> Optional[float]:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return None
        try:
            return float(token)
        except Exception:
            return None
    return None


def _normalize_rate(raw: Any) -> Optional[float]:
    value = _coerce_float(raw)
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 100.0:
        return value / 100.0
    return None


def _parse_iso_timestamp(raw: Any) -> Optional[dt.datetime]:
    """Parse ISO 8601 timestamp string to datetime object."""
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _normalize_qualification_signal(
    raw: Any,
    max_age_seconds: Optional[int] = None,
    provider_max_age_seconds: Optional[int] = None,
    legacy_grace_period_seconds: Optional[int] = None,
    legacy_grace_status: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    signal = raw if isinstance(raw, Mapping) else {}
    grace_status = dict(legacy_grace_status) if isinstance(legacy_grace_status, Mapping) else {}

    # Parse timestamps
    evaluated_at = _parse_iso_timestamp(signal.get("evaluated_at"))
    scored_at = _parse_iso_timestamp(signal.get("scored_at"))
    provider_evidence_updated_at = _parse_iso_timestamp(signal.get("provider_evidence_updated_at"))
    
    # Detect legacy packets (missing timestamps)
    is_legacy_packet = False
    missing_fields = []
    if evaluated_at is None:
        missing_fields.append("evaluated_at")
    if scored_at is None:
        missing_fields.append("scored_at")
    if provider_evidence_updated_at is None:
        missing_fields.append("provider_evidence_updated_at")
    
    is_legacy_packet = len(missing_fields) > 0
    
    # Use the most recent timestamp for freshness check
    latest_timestamp = evaluated_at or scored_at
    is_stale = False
    freshness_reason = None
    grace_period_active = False
    
    if max_age_seconds is not None and max_age_seconds > 0:
        if latest_timestamp is None:
            # Missing timestamp = conservative fail-closed, unless grace period
            if is_legacy_packet and legacy_grace_period_seconds is not None:
                # Temporary acceptance during grace period
                grace_period_active = True
                freshness_reason = "legacy_grace_period_active"
            else:
                is_stale = True
                inactive_reason = str(grace_status.get("inactive_reason") or "").strip()
                freshness_reason = (
                    f"legacy_grace_inactive:{inactive_reason}"
                    if is_legacy_packet and inactive_reason
                    else "missing_timestamp"
                )
        else:
            now = dt.datetime.now(dt.timezone.utc)
            age_seconds = int((now - latest_timestamp).total_seconds())
            if age_seconds > max_age_seconds:
                is_stale = True
                freshness_reason = f"age_exceeds_max_{age_seconds}s>{max_age_seconds}s"

    # Check provider evidence freshness
    provider_evidence_stale = False
    provider_freshness_reason = None
    provider_grace_period_active = False
    
    if provider_max_age_seconds is not None and provider_max_age_seconds > 0:
        if provider_evidence_updated_at is None:
            # Missing provider timestamp = conservative fail-closed, unless grace period
            if is_legacy_packet and legacy_grace_period_seconds is not None:
                # Temporary acceptance during grace period
                provider_grace_period_active = True
                provider_freshness_reason = "legacy_grace_period_active"
            else:
                provider_evidence_stale = True
                inactive_reason = str(grace_status.get("inactive_reason") or "").strip()
                provider_freshness_reason = (
                    f"legacy_grace_inactive:{inactive_reason}"
                    if is_legacy_packet and inactive_reason
                    else "missing_provider_timestamp"
                )
        else:
            now = dt.datetime.now(dt.timezone.utc)
            provider_age_seconds = int((now - provider_evidence_updated_at).total_seconds())
            if provider_age_seconds > provider_max_age_seconds:
                provider_evidence_stale = True
                provider_freshness_reason = f"provider_age_exceeds_max_{provider_age_seconds}s>{provider_max_age_seconds}s"

    readiness = str(signal.get("readiness_state") or "").strip().lower()
    if readiness not in QUALIFICATION_SIGNAL_READINESS_ORDER:
        readiness = "unknown"
    
    # Override readiness if stale (model or provider) and not in grace period
    if (is_stale and not grace_period_active) or (provider_evidence_stale and not provider_grace_period_active):
        readiness = "stale"

    weighted_score = _coerce_float(signal.get("weighted_score_0_100"))
    if weighted_score is not None and not (0.0 <= weighted_score <= 100.0):
        weighted_score = None

    benchmark_score = _coerce_float(signal.get("benchmark_composite_0_100"))
    if benchmark_score is not None and not (0.0 <= benchmark_score <= 100.0):
        benchmark_score = None

    effective_score = _coerce_float(signal.get("effective_score_0_100"))
    if effective_score is not None and not (0.0 <= effective_score <= 100.0):
        effective_score = None
    if effective_score is None:
        effective_score = weighted_score if weighted_score is not None else benchmark_score

    provider_cost_coverage = _normalize_rate(signal.get("provider_cost_coverage_rate"))

    guardrail_violation_count_raw = signal.get("guardrail_violation_count")
    if isinstance(guardrail_violation_count_raw, int) and guardrail_violation_count_raw >= 0:
        guardrail_violation_count = guardrail_violation_count_raw
    elif isinstance(signal.get("guardrail_violations"), list):
        guardrail_violation_count = len(signal.get("guardrail_violations") or [])
    else:
        guardrail_violation_count = None

    eval_sample_size_raw = signal.get("eval_sample_size")
    eval_sample_size = eval_sample_size_raw if isinstance(eval_sample_size_raw, int) and eval_sample_size_raw > 0 else None

    promotion_raw = signal.get("promotion_recommendation")
    promotion_recommendation = (
        str(promotion_raw).strip().lower() if isinstance(promotion_raw, str) and str(promotion_raw).strip() else None
    )

    score_source_raw = signal.get("score_source")
    score_source = str(score_source_raw).strip() if isinstance(score_source_raw, str) and str(score_source_raw).strip() else None

    result = {
        "readiness_state": readiness,
        "weighted_score_0_100": weighted_score,
        "benchmark_composite_0_100": benchmark_score,
        "effective_score_0_100": effective_score,
        "provider_cost_coverage_rate": provider_cost_coverage,
        "guardrail_violation_count": guardrail_violation_count,
        "eval_sample_size": eval_sample_size,
        "promotion_recommendation": promotion_recommendation,
        "score_source": score_source,
    }
    
    # Add timestamp fields if present
    if evaluated_at is not None:
        result["evaluated_at"] = evaluated_at.isoformat().replace("+00:00", "Z")
        result["evaluated_at_age_seconds"] = int((dt.datetime.now(dt.timezone.utc) - evaluated_at).total_seconds())
    
    if scored_at is not None:
        result["scored_at"] = scored_at.isoformat().replace("+00:00", "Z")
        result["scored_at_age_seconds"] = int((dt.datetime.now(dt.timezone.utc) - scored_at).total_seconds())
    
    # Add provider evidence timestamp if present
    if provider_evidence_updated_at is not None:
        result["provider_evidence_updated_at"] = provider_evidence_updated_at.isoformat().replace("+00:00", "Z")
        result["provider_evidence_age_seconds"] = int((dt.datetime.now(dt.timezone.utc) - provider_evidence_updated_at).total_seconds())
    
    # Add freshness information
    if is_stale:
        result["is_stale"] = True
        result["freshness_reason"] = freshness_reason
        if latest_timestamp is not None:
            result["latest_timestamp"] = latest_timestamp.isoformat().replace("+00:00", "Z")
            result["age_seconds"] = int((dt.datetime.now(dt.timezone.utc) - latest_timestamp).total_seconds())
    else:
        result["is_stale"] = False
        if latest_timestamp is not None:
            result["latest_timestamp"] = latest_timestamp.isoformat().replace("+00:00", "Z")
            result["age_seconds"] = int((dt.datetime.now(dt.timezone.utc) - latest_timestamp).total_seconds())
    
    # Add provider evidence freshness information
    if provider_evidence_stale:
        result["provider_evidence_stale"] = True
        result["provider_freshness_reason"] = provider_freshness_reason
    else:
        result["provider_evidence_stale"] = False

    # Add legacy packet diagnostics
    if is_legacy_packet:
        result["is_legacy_packet"] = True
        result["legacy_missing_fields"] = missing_fields
        result["legacy_grace_period_active"] = grace_period_active or provider_grace_period_active
        
        # Add migration guidance
        migration_actions = []
        if "evaluated_at" in missing_fields:
            migration_actions.append("Add evaluated_at field (ISO 8601)")
        if "scored_at" in missing_fields:
            migration_actions.append("Add scored_at field (ISO 8601)")
        if "provider_evidence_updated_at" in missing_fields:
            migration_actions.append("Add provider_evidence_updated_at field (ISO 8601)")
        
        result["legacy_migration_guidance"] = migration_actions
        
        # Add grace period information
        if legacy_grace_period_seconds is not None:
            result["legacy_grace_period_seconds"] = legacy_grace_period_seconds
            if grace_period_active or provider_grace_period_active:
                result["legacy_grace_period_warning"] = "Packet accepted during grace period - migrate before expiration"
        if grace_status:
            result["legacy_grace_window_active"] = bool(grace_status.get("grace_window_active") is True)
            if grace_status.get("inactive_reason") is not None:
                result["legacy_grace_window_inactive_reason"] = grace_status.get("inactive_reason")
            if grace_status.get("grace_window_expires_at"):
                result["legacy_grace_window_expires_at"] = grace_status.get("grace_window_expires_at")
            if isinstance(grace_status.get("grace_window_remaining_seconds"), int):
                result["legacy_grace_window_remaining_seconds"] = int(grace_status.get("grace_window_remaining_seconds"))
            if grace_status.get("policy_generated_at"):
                result["legacy_grace_window_policy_generated_at"] = grace_status.get("policy_generated_at")
    else:
        result["is_legacy_packet"] = False

    return result


def _normalize_string_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = raw
    else:
        values = []
    out: List[str] = []
    for item in values:
        token = str(item or "").strip()
        if token:
            out.append(token)
    return out


def _proposal_policy_for_risk(risk_tier: str) -> Dict[str, Any]:
    base = PROPOSAL_APPROVAL_POLICY_BY_RISK.get(risk_tier)
    if isinstance(base, Mapping):
        return dict(base)
    return {
        "proposal_checkpoint": "unknown",
        "proposal_min_approvers": 0,
        "post_implementation_approval": "unknown",
        "post_implementation_min_approvers": 0,
    }


def _proposal_first_delta_spec_gate(
    *,
    request: Mapping[str, Any],
    require_proposal_packet_for_coding: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()
    coding_task = _is_coding_task_class(task_class)

    proposal_raw = request.get("proposal_packet") if isinstance(request.get("proposal_packet"), Mapping) else None
    proposal = dict(proposal_raw) if isinstance(proposal_raw, Mapping) else {}
    proposal_present = bool(proposal)

    proposal_errors: List[str] = []
    proposal_task_id = str(proposal.get("task_id") or "").strip() if proposal_present else ""
    proposal_task_class = str(proposal.get("task_class") or "").strip() if proposal_present else ""
    proposal_title = str(proposal.get("title") or "").strip() if proposal_present else ""
    proposal_description = str(proposal.get("description") or "").strip() if proposal_present else ""
    risk_assessment = proposal.get("risk_assessment") if isinstance(proposal.get("risk_assessment"), Mapping) else {}
    proposal_risk_tier = str(risk_assessment.get("initial_tier") or "").strip()
    proposal_risk_justification = str(risk_assessment.get("justification") or "").strip()

    if proposal_present:
        if str(proposal.get("schema_version") or "").strip() != PROPOSAL_PACKET_SCHEMA_VERSION:
            proposal_errors.append("schema_version_invalid")
        if not proposal_task_id:
            proposal_errors.append("task_id_missing")
        if proposal_task_class not in PROPOSAL_PACKET_TASK_CLASSES:
            proposal_errors.append("task_class_invalid")
        if not proposal_title:
            proposal_errors.append("title_missing")
        if not proposal_description:
            proposal_errors.append("description_missing")
        if proposal_risk_tier not in ALLOWED_RISK_TIERS:
            proposal_errors.append("risk_assessment_initial_tier_invalid")
        if not proposal_risk_justification:
            proposal_errors.append("risk_assessment_justification_missing")
        if task_class and proposal_task_class and proposal_task_class != task_class:
            proposal_errors.append("task_class_mismatch")
        if risk_tier and proposal_risk_tier and proposal_risk_tier != risk_tier:
            proposal_errors.append("risk_tier_mismatch")

    approval_raw = request.get("proposal_approval") if isinstance(request.get("proposal_approval"), Mapping) else {}
    approval_decision = str(approval_raw.get("decision") or "").strip().lower()
    approved = bool(approval_raw.get("approved") is True or approval_decision == "approved")
    approver_ids = _normalize_string_list(approval_raw.get("approver_ids"))
    if not approver_ids:
        approver_ids = _normalize_string_list(approval_raw.get("approvers"))
    approver_count = len(set(approver_ids))

    policy = _proposal_policy_for_risk(risk_tier)
    proposal_min_approvers = int(policy.get("proposal_min_approvers") or 0)
    proposal_approval_required = proposal_min_approvers > 0
    proposal_approval_satisfied = (not proposal_approval_required) or (
        approved and approver_count >= proposal_min_approvers
    )

    delta_spec_raw = request.get("delta_spec") if isinstance(request.get("delta_spec"), Mapping) else None
    delta_spec_present = isinstance(delta_spec_raw, Mapping)
    delta_spec_obj = dict(delta_spec_raw) if isinstance(delta_spec_raw, Mapping) else {}
    delta_spec_instruction_surface = None
    delta_spec_valid = True
    if delta_spec_present:
        instructions = delta_spec_obj.get("instructions")
        prompt = delta_spec_obj.get("prompt")
        if isinstance(instructions, str) and str(instructions).strip():
            delta_spec_instruction_surface = "instructions"
            delta_spec_valid = True
        elif isinstance(prompt, str) and str(prompt).strip():
            delta_spec_instruction_surface = "prompt"
            delta_spec_valid = True
        else:
            delta_spec_instruction_surface = "invalid"
            delta_spec_valid = False

    archive_raw = request.get("proposal_archive") if isinstance(request.get("proposal_archive"), Mapping) else None
    archive_present = isinstance(archive_raw, Mapping)
    archive_obj = dict(archive_raw) if isinstance(archive_raw, Mapping) else {}
    archive_errors: List[str] = []
    archive_schema_version = str(archive_obj.get("schema_version") or "").strip() if archive_present else ""
    archive_task_id = str(archive_obj.get("task_id") or "").strip() if archive_present else ""
    archive_artifacts = _normalize_string_list(archive_obj.get("artifacts")) if archive_present else []
    archive_artifact_set = set(archive_artifacts)
    missing_required_artifacts = (
        sorted(artifact for artifact in PROPOSAL_ARCHIVE_REQUIRED_ARTIFACTS if artifact not in archive_artifact_set)
        if archive_present
        else []
    )

    if archive_present:
        if archive_schema_version != PROPOSAL_ARCHIVE_PACKET_SCHEMA_VERSION:
            archive_errors.append("schema_version_invalid")
        if not archive_task_id:
            archive_errors.append("task_id_missing")
        elif proposal_task_id and archive_task_id != proposal_task_id:
            archive_errors.append("task_id_mismatch")
        if not archive_artifacts:
            archive_errors.append("artifacts_missing")
        elif missing_required_artifacts:
            archive_errors.append("required_artifacts_missing")

    proposal_phase_raw = request.get("proposal_phase")
    proposal_phase = str(proposal_phase_raw or "").strip().lower() if proposal_phase_raw is not None else ""
    proposal_phase_valid = (not proposal_phase) or (proposal_phase in PROPOSAL_FLOW_PHASE_ALLOWLIST)

    if proposal_phase in PROPOSAL_FLOW_PHASE_ALLOWLIST:
        effective_phase = proposal_phase
        phase_source = "declared"
    else:
        phase_source = "inferred"
        if proposal_present:
            if archive_present:
                effective_phase = "archive"
            elif delta_spec_present:
                effective_phase = "apply"
            else:
                effective_phase = "proposal"
        else:
            effective_phase = None

    proposal_contract_satisfied = bool(proposal_present and not proposal_errors and proposal_approval_satisfied)
    apply_ready = bool(coding_task and proposal_contract_satisfied and delta_spec_present and delta_spec_valid)
    archive_ready = bool(coding_task and apply_ready and archive_present and not archive_errors)

    proposal_packet_details = {
        "present": proposal_present,
        "valid": bool(proposal_present and not proposal_errors),
        "schema_version": str(proposal.get("schema_version") or "").strip() if proposal_present else None,
        "task_id": proposal_task_id or None,
        "task_class": proposal_task_class or None,
        "risk_tier": proposal_risk_tier or None,
        "title": proposal_title or None,
        "affected_files_count": len(_normalize_string_list(proposal.get("affected_files"))) if proposal_present else 0,
        "acceptance_criteria_count": len(_normalize_string_list(proposal.get("acceptance_criteria"))) if proposal_present else 0,
    }
    if proposal_errors:
        proposal_packet_details["errors"] = proposal_errors

    details: Dict[str, Any] = {
        "schema": PROPOSAL_FIRST_DELTA_SPEC_SCHEMA,
        "status": "not_applicable",
        "task_class": task_class or None,
        "risk_tier": risk_tier or None,
        "coding_task": coding_task,
        "require_proposal_packet_for_coding": bool(require_proposal_packet_for_coding),
        "proposal_packet": proposal_packet_details,
        "approval_hooks": {
            "proposal_checkpoint": {
                "mode": policy.get("proposal_checkpoint"),
                "approval_required": proposal_approval_required,
                "minimum_approvers": proposal_min_approvers,
                "approved": approved,
                "approver_count": approver_count,
                "satisfied": proposal_approval_satisfied,
            },
            "post_implementation": {
                "mode": policy.get("post_implementation_approval"),
                "minimum_approvers": int(policy.get("post_implementation_min_approvers") or 0),
            },
        },
        "delta_spec": {
            "present": delta_spec_present,
            "valid": delta_spec_valid,
            "instruction_surface": delta_spec_instruction_surface,
        },
        "state_flow": {
            "schema": PROPOSAL_FLOW_STATE_SCHEMA,
            "declared_phase": proposal_phase or None,
            "effective_phase": effective_phase,
            "phase_source": phase_source,
            "phase_valid": proposal_phase_valid,
            "proposal_ready": proposal_contract_satisfied,
            "apply_ready": apply_ready,
            "archive_ready": archive_ready,
        },
        "apply_contract": {
            "dispatch_mode": "delta_spec_worker_execution",
            "ready_for_dispatch": apply_ready,
        },
        "archive_contract": {
            "required_artifacts": list(PROPOSAL_ARCHIVE_REQUIRED_ARTIFACTS),
            "status": "declared",
            "ready_for_archive": archive_ready,
            "archive_packet": {
                "present": archive_present,
                "valid": bool(archive_present and not archive_errors),
                "schema_version": archive_schema_version or None,
                "task_id": archive_task_id or None,
                "artifacts_count": len(archive_artifacts),
                "missing_required_artifacts": missing_required_artifacts,
            },
        },
    }

    if archive_errors:
        details["archive_contract"]["archive_packet"]["errors"] = archive_errors

    if not coding_task:
        details["status"] = "not_applicable"
        details["reason"] = "non_coding_task"
        return True, None, details

    if not proposal_phase_valid:
        details["status"] = "fail"
        details["reason"] = "proposal_phase_invalid"
        return False, "proposal_phase_invalid", details

    if not proposal_present and require_proposal_packet_for_coding:
        details["status"] = "fail"
        details["reason"] = "proposal_packet_required_for_coding"
        return False, "proposal_packet_required_for_coding", details

    if not proposal_present:
        details["status"] = "legacy_passthrough"
        details["reason"] = "proposal_packet_missing_legacy_mode"
        return True, None, details

    if proposal_errors:
        details["status"] = "fail"
        details["reason"] = "proposal_packet_invalid"
        return False, "proposal_packet_invalid", details

    if proposal_phase in {"apply", "archive"} and not delta_spec_present:
        details["status"] = "fail"
        details["reason"] = "proposal_apply_missing_delta_spec"
        return False, "proposal_apply_missing_delta_spec", details

    if not delta_spec_valid:
        details["status"] = "fail"
        details["reason"] = "delta_spec_invalid"
        return False, "delta_spec_invalid", details

    if not proposal_approval_satisfied:
        details["status"] = "fail"
        details["reason"] = "proposal_approval_missing"
        return False, "proposal_approval_missing", details

    if proposal_phase == "archive" and not archive_present:
        details["status"] = "fail"
        details["reason"] = "proposal_archive_missing"
        return False, "proposal_archive_missing", details

    if archive_present and archive_errors:
        details["status"] = "fail"
        details["reason"] = "proposal_archive_invalid"
        return False, "proposal_archive_invalid", details

    details["status"] = "pass"
    details["reason"] = "proposal_first_contract_satisfied"
    return True, None, details


def _effective_risk_tier(request_tier: Any, packet_tier: Any) -> str:
    request_token = str(request_tier or "").strip().lower()
    packet_token = str(packet_tier or "").strip().lower()
    if request_token not in RISK_TIER_ORDER:
        return packet_token if packet_token in RISK_TIER_ORDER else ""
    if packet_token not in RISK_TIER_ORDER:
        return request_token
    return request_token if RISK_TIER_ORDER[request_token] >= RISK_TIER_ORDER[packet_token] else packet_token


def _regression_risk_schema_for_version(packet_version: Any) -> str:
    token = str(packet_version or "").strip()
    return REGRESSION_RISK_PACKET_SCHEMA_BY_VERSION.get(token, REGRESSION_RISK_PACKET_SCHEMA)


def _is_sha256_checksum_token(token: Any) -> bool:
    text = str(token or "").strip().lower()
    if not text.startswith("sha256:"):
        return False
    digest = text[len("sha256:") :]
    if len(digest) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in digest)


def _looks_like_datetime_token(token: Any) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    try:
        dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def _compute_regression_risk_blocking_classification(
    *,
    overall_tier: str,
    critical_path_impact: Optional[int],
    validation_status: str,
    manual_attestations: List[str],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if overall_tier == "critical":
        reasons.append("overall_tier_critical")
    if overall_tier == "high" and isinstance(critical_path_impact, int) and critical_path_impact >= 4:
        reasons.append("high_tier_critical_path_impact")
    if validation_status == "rejected":
        reasons.append("validation_rejected")

    for token in manual_attestations:
        normalized = str(token or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            continue
        security_regression = "security" in normalized and "regression" in normalized
        data_integrity_regression = "data_integrity" in normalized and "regression" in normalized
        if security_regression or data_integrity_regression:
            reasons.append("security_or_data_integrity_regression")
            break

    classification = "blocking" if reasons else "non_blocking"
    return classification, reasons


def _normalize_regression_replay_evidence(raw: Any) -> Dict[str, Any]:
    present = isinstance(raw, Mapping)
    payload = raw if isinstance(raw, Mapping) else {}

    artifact_paths: List[str] = []
    gate_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    artifact_paths_raw = payload.get("artifact_paths")
    if present:
        if not isinstance(artifact_paths_raw, list):
            errors.append("replay_evidence_artifact_paths_invalid")
        else:
            for idx, row in enumerate(artifact_paths_raw, 1):
                token = str(row).strip() if isinstance(row, str) else ""
                if not token:
                    errors.append(f"replay_evidence_artifact_path_{idx}_invalid")
                    continue
                artifact_paths.append(token)
            if not artifact_paths:
                errors.append("replay_evidence_artifact_paths_empty")

    gate_log_raw = payload.get("validation_gate_log")
    failed_gate_count = 0
    if present:
        if not isinstance(gate_log_raw, list):
            errors.append("replay_evidence_validation_gate_log_invalid")
        else:
            for idx, row in enumerate(gate_log_raw, 1):
                if not isinstance(row, Mapping):
                    errors.append(f"replay_evidence_validation_gate_{idx}_invalid")
                    continue
                gate = str(row.get("gate") or "").strip()
                status = str(row.get("status") or "").strip().lower()
                timestamp = str(row.get("timestamp") or "").strip()
                if not gate:
                    errors.append(f"replay_evidence_validation_gate_{idx}_gate_missing")
                if status not in REGRESSION_RISK_REPLAY_EVIDENCE_GATE_STATUSES:
                    errors.append(f"replay_evidence_validation_gate_{idx}_status_invalid")
                if not _looks_like_datetime_token(timestamp):
                    errors.append(f"replay_evidence_validation_gate_{idx}_timestamp_invalid")
                if status == "failed":
                    failed_gate_count += 1
                gate_rows.append(
                    {
                        "gate": gate or None,
                        "status": status or None,
                        "timestamp": timestamp or None,
                    }
                )
            if not gate_rows:
                errors.append("replay_evidence_validation_gate_log_empty")

    replay_checksum = str(payload.get("replay_checksum") or "").strip()
    checksum_valid = _is_sha256_checksum_token(replay_checksum)
    if present:
        if not replay_checksum:
            errors.append("replay_evidence_checksum_missing")
        elif not checksum_valid:
            errors.append("replay_evidence_checksum_invalid")

    return {
        "present": present,
        "artifact_paths": artifact_paths,
        "artifact_path_count": len(artifact_paths),
        "validation_gate_log": gate_rows,
        "gate_count": len(gate_rows),
        "failed_gate_count": failed_gate_count,
        "replay_checksum": replay_checksum or None,
        "checksum_valid": checksum_valid,
        "errors": errors,
    }


def _normalize_refactor_decomposition_plan(raw: Any) -> Dict[str, Any]:
    present = isinstance(raw, list)
    chunks_raw = raw if isinstance(raw, list) else []

    chunk_ids: set[str] = set()
    task_classes: List[str] = []
    task_class_set: set[str] = set()
    chunks: List[Dict[str, Any]] = []
    errors: List[str] = []
    total_file_refs = 0
    bounded_scope_ok = True
    review_chunk_present = False

    for index, row in enumerate(chunks_raw, 1):
        if not isinstance(row, Mapping):
            errors.append(f"chunk_{index}_invalid")
            continue

        chunk_id = str(row.get("chunk_id") or "").strip()
        description = str(row.get("description") or "").strip()
        task_class = str(row.get("task_class") or "").strip()
        scope = row.get("scope") if isinstance(row.get("scope"), Mapping) else None
        files = _normalize_string_list((scope or {}).get("files"))
        lines_raw = (scope or {}).get("lines")
        lines = str(lines_raw).strip() if isinstance(lines_raw, str) and str(lines_raw).strip() else None

        if not chunk_id:
            errors.append(f"chunk_{index}_chunk_id_missing")
        elif chunk_id in chunk_ids:
            errors.append(f"chunk_{index}_chunk_id_duplicate")
        else:
            chunk_ids.add(chunk_id)

        if not description:
            errors.append(f"chunk_{index}_description_missing")

        if task_class not in REFACTOR_DECOMPOSITION_TASK_CLASSES:
            errors.append(f"chunk_{index}_task_class_invalid")
        else:
            if task_class == "code:review":
                review_chunk_present = True
            if task_class not in task_class_set:
                task_class_set.add(task_class)
                task_classes.append(task_class)

        if scope is None:
            errors.append(f"chunk_{index}_scope_missing")

        total_file_refs += len(files)
        if len(files) > REFACTOR_DECOMPOSITION_MAX_FILES_PER_CHUNK:
            bounded_scope_ok = False
            errors.append(f"chunk_{index}_scope_files_exceed_{REFACTOR_DECOMPOSITION_MAX_FILES_PER_CHUNK}")

        chunks.append(
            {
                "index": index,
                "chunk_id": chunk_id or None,
                "description": description or None,
                "task_class": task_class or None,
                "scope": {
                    "files": files,
                    "lines": lines,
                    "file_count": len(files),
                },
            }
        )

    return {
        "present": present,
        "chunk_count": len(chunks_raw),
        "chunks": chunks,
        "errors": errors,
        "task_classes": task_classes,
        "review_chunk_present": review_chunk_present,
        "bounded_scope_ok": bounded_scope_ok,
        "total_file_refs": total_file_refs,
    }


def _refactor_risk_packet_gate(
    *,
    request: Mapping[str, Any],
    require_refactor_risk_packet_for_coding: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    task_class = str(request.get("task_class") or "").strip()
    request_risk_tier = str(request.get("risk_tier") or "").strip().lower()
    coding_task = _is_coding_task_class(task_class)

    packet_raw = request.get("refactor_risk_packet") if isinstance(request.get("refactor_risk_packet"), Mapping) else {}
    packet = dict(packet_raw or {})
    packet_present = bool(packet)

    packet_errors: List[str] = []
    packet_id = str(packet.get("packet_id") or "").strip()
    packet_version = str(packet.get("version") or "").strip()
    packet_schema = _regression_risk_schema_for_version(packet_version)
    packet_version_supported = packet_version in REGRESSION_RISK_PACKET_SUPPORTED_VERSIONS
    proposal_id = str(packet.get("proposal_id") or "").strip()

    risk_assessment = packet.get("risk_assessment") if isinstance(packet.get("risk_assessment"), Mapping) else {}
    overall_tier = str((risk_assessment or {}).get("overall_tier") or "").strip().lower()
    dimensions = (risk_assessment or {}).get("dimensions") if isinstance((risk_assessment or {}).get("dimensions"), Mapping) else {}
    blast_radius_evidence = (
        (risk_assessment or {}).get("blast_radius_evidence")
        if isinstance((risk_assessment or {}).get("blast_radius_evidence"), Mapping)
        else {}
    )
    blast_radius_components = _normalize_string_list((blast_radius_evidence or {}).get("impacted_components"))
    blast_radius_interfaces = _normalize_string_list((blast_radius_evidence or {}).get("impacted_interfaces"))
    blast_radius_evidence_refs = _normalize_string_list((blast_radius_evidence or {}).get("evidence_refs"))
    rationale = str((risk_assessment or {}).get("rationale") or "").strip()

    rollback_plan = packet.get("rollback_plan") if isinstance(packet.get("rollback_plan"), Mapping) else {}
    rollback_strategy = str((rollback_plan or {}).get("strategy") or "").strip().lower()
    rollback_trigger_conditions = _normalize_string_list((rollback_plan or {}).get("trigger_conditions"))
    rollback_verification_refs = _normalize_string_list((rollback_plan or {}).get("verification_evidence_refs"))

    decomposition_eval = _normalize_refactor_decomposition_plan(packet.get("decomposition_plan"))
    decomposition_chunk_count = int(decomposition_eval.get("chunk_count") or 0)
    review_chunk_present = bool(decomposition_eval.get("review_chunk_present") is True)
    total_file_refs = int(decomposition_eval.get("total_file_refs") or 0)
    non_trivial_refactor = bool(
        overall_tier in {"high", "critical"}
        or decomposition_chunk_count > 1
        or total_file_refs > REFACTOR_DECOMPOSITION_MAX_FILES_PER_CHUNK
    )
    requires_review_chunk = non_trivial_refactor

    validation = packet.get("validation") if isinstance(packet.get("validation"), Mapping) else {}
    validation_status = str((validation or {}).get("status") or "").strip().lower()
    required_approvals_raw = (validation or {}).get("required_approvals")
    required_approvals_declared: Optional[int]
    if isinstance(required_approvals_raw, bool):
        required_approvals_declared = None
    elif isinstance(required_approvals_raw, int):
        required_approvals_declared = int(required_approvals_raw)
    elif isinstance(required_approvals_raw, float) and float(required_approvals_raw).is_integer():
        required_approvals_declared = int(required_approvals_raw)
    else:
        required_approvals_declared = None

    approvers_from_list = _normalize_string_list((validation or {}).get("approvers"))
    approvals_raw = (validation or {}).get("approvals") if isinstance((validation or {}).get("approvals"), list) else []
    approvers_from_records = [
        str(row.get("approver_id") or "").strip()
        for row in approvals_raw
        if isinstance(row, Mapping) and str(row.get("approver_id") or "").strip()
    ]
    approver_count = len(set([*approvers_from_list, *approvers_from_records]))

    expected_required_approvals = REFACTOR_RISK_REQUIRED_APPROVALS_BY_TIER.get(overall_tier)
    approval_threshold = expected_required_approvals
    if approval_threshold is None:
        approval_threshold = required_approvals_declared if required_approvals_declared is not None and required_approvals_declared >= 0 else 0

    approval_satisfied = approver_count >= int(approval_threshold or 0)
    effective_risk_tier = _effective_risk_tier(request_risk_tier, overall_tier)

    if packet_present:
        if not packet_id:
            packet_errors.append("packet_id_missing")
        else:
            try:
                uuid.UUID(packet_id)
            except Exception:
                packet_errors.append("packet_id_invalid_uuid")

        if packet_version != REFACTOR_RISK_PACKET_VERSION:
            packet_errors.append("version_invalid")
        if not proposal_id:
            packet_errors.append("proposal_id_missing")

        if overall_tier not in ALLOWED_RISK_TIERS:
            packet_errors.append("overall_tier_invalid")

        if not isinstance(dimensions, Mapping) or not dimensions:
            packet_errors.append("risk_dimensions_missing")
        else:
            for dimension in REFACTOR_RISK_DIMENSIONS:
                value = dimensions.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int):
                    packet_errors.append(f"dimension_{dimension}_invalid")
                    continue
                if value < 1 or value > 5:
                    packet_errors.append(f"dimension_{dimension}_out_of_range")

        if not rationale:
            packet_errors.append("risk_rationale_missing")

        if not isinstance(blast_radius_evidence, Mapping) or not blast_radius_evidence:
            packet_errors.append("blast_radius_evidence_missing")
        else:
            if not blast_radius_components:
                packet_errors.append("blast_radius_impacted_components_missing")
            if not blast_radius_interfaces:
                packet_errors.append("blast_radius_impacted_interfaces_missing")
            if not blast_radius_evidence_refs:
                packet_errors.append("blast_radius_evidence_refs_missing")

        if not isinstance(rollback_plan, Mapping) or not rollback_plan:
            packet_errors.append("rollback_plan_missing")
        else:
            if rollback_strategy not in REFACTOR_RISK_ROLLBACK_STRATEGIES:
                packet_errors.append("rollback_strategy_invalid")
            if not rollback_trigger_conditions:
                packet_errors.append("rollback_trigger_conditions_missing")
            if not rollback_verification_refs:
                packet_errors.append("rollback_verification_evidence_refs_missing")

        if not decomposition_eval.get("present"):
            packet_errors.append("decomposition_plan_missing")
        elif decomposition_chunk_count <= 0:
            packet_errors.append("decomposition_plan_empty")

        packet_errors.extend([str(err) for err in decomposition_eval.get("errors") or []])

        if requires_review_chunk and not review_chunk_present:
            packet_errors.append("decomposition_review_chunk_missing")

        if not isinstance(validation, Mapping) or not validation:
            packet_errors.append("validation_missing")
        else:
            if validation_status not in {"pending", "approved", "rejected"}:
                packet_errors.append("validation_status_invalid")
            if required_approvals_declared is None or required_approvals_declared < 0:
                packet_errors.append("required_approvals_invalid")
            if expected_required_approvals is not None and required_approvals_declared is not None:
                if required_approvals_declared != expected_required_approvals:
                    packet_errors.append("required_approvals_mismatch")
            if validation_status == "approved" and not approval_satisfied:
                packet_errors.append("approved_status_missing_required_approvals")

    details: Dict[str, Any] = {
        "schema": REFACTOR_RISK_PACKET_SCHEMA,
        "status": "not_evaluated",
        "task_class": task_class,
        "coding_task": coding_task,
        "require_refactor_risk_packet_for_coding": bool(require_refactor_risk_packet_for_coding),
        "packet": {
            "present": packet_present,
            "valid": not packet_errors if packet_present else False,
            "errors": packet_errors,
            "packet_id": packet_id or None,
            "proposal_id": proposal_id or None,
            "version": packet_version or None,
        },
        "risk_assessment": {
            "overall_tier": overall_tier or None,
            "dimensions": dict(dimensions) if isinstance(dimensions, Mapping) else {},
            "blast_radius_evidence": {
                "impacted_components": blast_radius_components,
                "impacted_interfaces": blast_radius_interfaces,
                "evidence_refs": blast_radius_evidence_refs,
            },
            "rationale": rationale or None,
        },
        "rollback": {
            "strategy": rollback_strategy or None,
            "trigger_conditions": rollback_trigger_conditions,
            "verification_evidence_refs": rollback_verification_refs,
        },
        "decomposition": {
            "present": bool(decomposition_eval.get("present") is True),
            "chunk_count": decomposition_chunk_count,
            "task_classes": list(decomposition_eval.get("task_classes") or []),
            "review_chunk_present": review_chunk_present,
            "requires_review_chunk": requires_review_chunk,
            "non_trivial_refactor": non_trivial_refactor,
            "bounded_scope_ok": bool(decomposition_eval.get("bounded_scope_ok") is True),
            "max_files_per_chunk": REFACTOR_DECOMPOSITION_MAX_FILES_PER_CHUNK,
            "total_file_refs": total_file_refs,
            "chunks": list(decomposition_eval.get("chunks") or []),
        },
        "validation": {
            "status": validation_status or None,
            "required_approvals_declared": required_approvals_declared,
            "required_approvals_expected": expected_required_approvals,
            "approver_count": approver_count,
            "approval_satisfied": approval_satisfied,
        },
        "routing": {
            "request_risk_tier": request_risk_tier or None,
            "effective_risk_tier": effective_risk_tier or (request_risk_tier or None),
            "tier_elevated": bool(
                request_risk_tier in RISK_TIER_ORDER
                and effective_risk_tier in RISK_TIER_ORDER
                and RISK_TIER_ORDER[effective_risk_tier] > RISK_TIER_ORDER[request_risk_tier]
            ),
        },
    }

    if not coding_task:
        details["status"] = "not_applicable"
        details["reason"] = "non_coding_task"
        return True, None, details

    if not packet_present and require_refactor_risk_packet_for_coding:
        details["status"] = "fail"
        details["reason"] = "refactor_risk_packet_required_for_coding"
        return False, "refactor_risk_packet_required_for_coding", details

    if not packet_present:
        details["status"] = "legacy_passthrough"
        details["reason"] = "refactor_risk_packet_missing_legacy_mode"
        return True, None, details

    if packet_errors:
        details["status"] = "fail"
        details["reason"] = "refactor_risk_packet_invalid"
        return False, "refactor_risk_packet_invalid", details

    if require_refactor_risk_packet_for_coding and validation_status != "approved":
        details["status"] = "fail"
        details["reason"] = "refactor_risk_not_approved"
        return False, "refactor_risk_not_approved", details

    if require_refactor_risk_packet_for_coding and not approval_satisfied:
        details["status"] = "fail"
        details["reason"] = "refactor_risk_approval_missing"
        return False, "refactor_risk_approval_missing", details

    details["status"] = "pass"
    details["reason"] = "refactor_risk_contract_satisfied"
    return True, None, details


def _regression_risk_packet_gate(
    *,
    request: Mapping[str, Any],
    require_regression_risk_packet_for_coding: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    task_class = str(request.get("task_class") or "").strip()
    request_risk_tier = str(request.get("risk_tier") or "").strip().lower()
    coding_task = _is_coding_task_class(task_class)

    packet_raw = request.get("regression_risk_packet") if isinstance(request.get("regression_risk_packet"), Mapping) else {}
    packet = dict(packet_raw or {})
    packet_present = bool(packet)

    packet_errors: List[str] = []
    packet_id = str(packet.get("packet_id") or "").strip()
    packet_version = str(packet.get("version") or "").strip()
    packet_schema = _regression_risk_schema_for_version(packet_version)
    packet_version_supported = packet_version in REGRESSION_RISK_PACKET_SUPPORTED_VERSIONS
    proposal_id = str(packet.get("proposal_id") or "").strip()

    risk_assessment = packet.get("risk_assessment") if isinstance(packet.get("risk_assessment"), Mapping) else {}
    overall_tier = str((risk_assessment or {}).get("overall_tier") or "").strip().lower()
    scores = (risk_assessment or {}).get("scores") if isinstance((risk_assessment or {}).get("scores"), Mapping) else {}

    evidence = packet.get("evidence") if isinstance(packet.get("evidence"), Mapping) else {}
    replay_evidence = _normalize_regression_replay_evidence(packet.get("replay_evidence"))
    blocking_classification_declared = str(packet.get("blocking_classification") or "").strip().lower()
    validation = packet.get("validation") if isinstance(packet.get("validation"), Mapping) else {}

    validation_status = str((validation or {}).get("status") or "").strip().lower()
    required_approvals_raw = (validation or {}).get("required_approvals")
    required_approvals_declared: Optional[int]
    if isinstance(required_approvals_raw, bool):
        required_approvals_declared = None
    elif isinstance(required_approvals_raw, int):
        required_approvals_declared = int(required_approvals_raw)
    elif isinstance(required_approvals_raw, float) and float(required_approvals_raw).is_integer():
        required_approvals_declared = int(required_approvals_raw)
    else:
        required_approvals_declared = None

    approvals_raw = (validation or {}).get("approvals") if isinstance((validation or {}).get("approvals"), list) else []
    approval_records = [row for row in approvals_raw if isinstance(row, Mapping)]
    approval_count = len(
        [row for row in approval_records if str(row.get("approver_id") or "").strip()]
    )

    expected_required_approvals = REGRESSION_RISK_REQUIRED_APPROVALS_BY_TIER.get(overall_tier)
    approval_threshold = expected_required_approvals
    if approval_threshold is None:
        approval_threshold = required_approvals_declared if required_approvals_declared is not None and required_approvals_declared >= 0 else 0

    approval_satisfied = approval_count >= int(approval_threshold or 0)

    manual_attestations = _normalize_string_list((evidence or {}).get("manual_attestations"))
    critical_path_impact_raw = (scores or {}).get("critical_path_impact") if isinstance(scores, Mapping) else None
    critical_path_impact = (
        int(critical_path_impact_raw)
        if isinstance(critical_path_impact_raw, int) and not isinstance(critical_path_impact_raw, bool)
        else None
    )
    computed_blocking_classification, blocking_reasons = _compute_regression_risk_blocking_classification(
        overall_tier=overall_tier,
        critical_path_impact=critical_path_impact,
        validation_status=validation_status,
        manual_attestations=manual_attestations,
    )
    blocking_classification_matches = (
        blocking_classification_declared == computed_blocking_classification
        if blocking_classification_declared in REGRESSION_RISK_BLOCKING_CLASSIFICATIONS
        else False
    )
    effective_risk_tier = _effective_risk_tier(request_risk_tier, overall_tier)

    if packet_present:
        if not packet_id:
            packet_errors.append("packet_id_missing")
        else:
            try:
                uuid.UUID(packet_id)
            except Exception:
                packet_errors.append("packet_id_invalid_uuid")

        if not packet_version_supported:
            packet_errors.append("version_invalid")
        if not proposal_id:
            packet_errors.append("proposal_id_missing")
        if overall_tier not in ALLOWED_RISK_TIERS:
            packet_errors.append("overall_tier_invalid")
        if not isinstance(scores, Mapping) or not scores:
            packet_errors.append("risk_scores_missing")
        else:
            for dimension in REGRESSION_RISK_SCORE_DIMENSIONS:
                value = scores.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int):
                    packet_errors.append(f"score_{dimension}_invalid")
                    continue
                if value < 1 or value > 5:
                    packet_errors.append(f"score_{dimension}_out_of_range")

        if not isinstance(evidence, Mapping) or not evidence:
            packet_errors.append("evidence_missing")

        if packet_version == "2.0":
            if blocking_classification_declared not in REGRESSION_RISK_BLOCKING_CLASSIFICATIONS:
                packet_errors.append("blocking_classification_invalid")
            elif not blocking_classification_matches:
                packet_errors.append("blocking_classification_mismatch")

            if not replay_evidence.get("present"):
                packet_errors.append("replay_evidence_missing")
            packet_errors.extend([str(err) for err in replay_evidence.get("errors") or []])

        if not isinstance(validation, Mapping) or not validation:
            packet_errors.append("validation_missing")
        else:
            if validation_status not in {"pending", "approved", "rejected"}:
                packet_errors.append("validation_status_invalid")
            if required_approvals_declared is None or required_approvals_declared < 0:
                packet_errors.append("required_approvals_invalid")
            if expected_required_approvals is not None and required_approvals_declared is not None:
                if required_approvals_declared != expected_required_approvals:
                    packet_errors.append("required_approvals_mismatch")
            if validation_status == "approved" and not approval_satisfied:
                packet_errors.append("approved_status_missing_required_approvals")

    details: Dict[str, Any] = {
        "schema": packet_schema,
        "status": "not_evaluated",
        "task_class": task_class,
        "coding_task": coding_task,
        "require_regression_risk_packet_for_coding": bool(require_regression_risk_packet_for_coding),
        "packet": {
            "present": packet_present,
            "valid": not packet_errors if packet_present else False,
            "errors": packet_errors,
            "packet_id": packet_id or None,
            "proposal_id": proposal_id or None,
            "version": packet_version or None,
            "version_supported": packet_version_supported,
        },
        "risk_assessment": {
            "overall_tier": overall_tier or None,
            "scores": dict(scores) if isinstance(scores, Mapping) else {},
            "critical_path_impact": critical_path_impact,
        },
        "evidence": {
            "manual_attestations": manual_attestations,
        },
        "replay_evidence": {
            "present": bool(replay_evidence.get("present") is True),
            "artifact_path_count": int(replay_evidence.get("artifact_path_count") or 0),
            "gate_count": int(replay_evidence.get("gate_count") or 0),
            "failed_gate_count": int(replay_evidence.get("failed_gate_count") or 0),
            "checksum_valid": bool(replay_evidence.get("checksum_valid") is True),
            "errors": list(replay_evidence.get("errors") or []),
        },
        "blocking_classification": {
            "declared": blocking_classification_declared or None,
            "computed": computed_blocking_classification,
            "matches": bool(blocking_classification_matches),
            "reasons": blocking_reasons,
        },
        "validation": {
            "status": validation_status or None,
            "required_approvals_declared": required_approvals_declared,
            "required_approvals_expected": expected_required_approvals,
            "approval_count": approval_count,
            "approval_satisfied": approval_satisfied,
        },
        "routing": {
            "request_risk_tier": request_risk_tier or None,
            "effective_risk_tier": effective_risk_tier or (request_risk_tier or None),
            "tier_elevated": bool(
                request_risk_tier in RISK_TIER_ORDER
                and effective_risk_tier in RISK_TIER_ORDER
                and RISK_TIER_ORDER[effective_risk_tier] > RISK_TIER_ORDER[request_risk_tier]
            ),
        },
    }

    if not coding_task:
        details["status"] = "not_applicable"
        details["reason"] = "non_coding_task"
        return True, None, details

    if not packet_present and require_regression_risk_packet_for_coding:
        details["status"] = "fail"
        details["reason"] = "regression_risk_packet_required_for_coding"
        return False, "regression_risk_packet_required_for_coding", details

    if not packet_present:
        details["status"] = "legacy_passthrough"
        details["reason"] = "regression_risk_packet_missing_legacy_mode"
        return True, None, details

    if packet_errors:
        details["status"] = "fail"
        details["reason"] = "regression_risk_packet_invalid"
        return False, "regression_risk_packet_invalid", details

    if require_regression_risk_packet_for_coding and validation_status != "approved":
        details["status"] = "fail"
        details["reason"] = "regression_risk_not_approved"
        return False, "regression_risk_not_approved", details

    if require_regression_risk_packet_for_coding and not approval_satisfied:
        details["status"] = "fail"
        details["reason"] = "regression_risk_approval_missing"
        return False, "regression_risk_approval_missing", details

    details["status"] = "pass"
    details["reason"] = "regression_risk_contract_satisfied"
    return True, None, details


def _is_coding_task_class(task_class: Any) -> bool:
    token = str(task_class or "").strip()
    return token in CODING_TASK_CLASSES


def _request_coding_complexity_tier(request: Mapping[str, Any]) -> str:
    direct = str(request.get("complexity_tier") or "").strip().lower()
    if direct in CODING_COMPLEXITY_TIERS:
        return direct

    nested = str(_request_dispatch_contract(request).get("complexity_tier") or "").strip().lower()
    if nested in CODING_COMPLEXITY_TIERS:
        return nested

    refactor_packet = request.get("refactor_risk_packet") if isinstance(request.get("refactor_risk_packet"), Mapping) else {}
    risk_assessment = refactor_packet.get("risk_assessment") if isinstance(refactor_packet.get("risk_assessment"), Mapping) else {}
    scores = risk_assessment.get("scores") if isinstance(risk_assessment.get("scores"), Mapping) else {}
    complexity_score = scores.get("complexity")
    if isinstance(complexity_score, (int, float)):
        score = float(complexity_score)
        if score >= 4.0:
            return "high"
        if score <= 2.0:
            return "low"
        return "moderate"

    task_class = str(request.get("task_class") or "").strip()
    return CODING_COMPLEXITY_TASK_CLASS_DEFAULTS.get(task_class, "moderate")


def _coding_verification_profile(request: Mapping[str, Any], complexity_tier: str) -> str:
    verification_class = str(_dispatch_contract_field(request, "verification_class") or "").strip()
    if verification_class == "validator_plus_human":
        return "human_gate_required"
    if complexity_tier == "high":
        return "high_complexity"
    if verification_class == "validator_required":
        return "validator_required"
    if verification_class == "self_check":
        return "self_check"
    return "unspecified"


def _coding_strict_readiness_required(
    *,
    request: Mapping[str, Any],
    complexity_tier: str,
    routing_policy: Mapping[str, Any],
) -> bool:
    verification_class = str(_dispatch_contract_field(request, "verification_class") or "").strip()
    strict_verification_classes = routing_policy_coding_strict_readiness_trigger_verification_classes(routing_policy)
    strict_complexity_tiers = routing_policy_coding_strict_readiness_trigger_complexity_tiers(routing_policy)
    return bool(verification_class in strict_verification_classes or complexity_tier in strict_complexity_tiers)


def _selection_rubric_rule_id(*, task_class: Any, risk_tier: Any) -> str:
    task_class_token = str(task_class or "").strip() or "unknown"
    risk_tier_token = str(risk_tier or "").strip() or "unknown"
    rubric_family = "coding" if _is_coding_task_class(task_class_token) else "general"
    return f"{rubric_family}:{task_class_token}:{risk_tier_token}"


def _selection_rubric_rule_id_v2(
    *,
    task_class: Any,
    risk_tier: Any,
    complexity_tier: str,
    verification_profile: str,
) -> str:
    task_class_token = str(task_class or "").strip() or "unknown"
    risk_tier_token = str(risk_tier or "").strip() or "unknown"
    complexity_token = str(complexity_tier or "").strip() or "unknown"
    verification_token = str(verification_profile or "").strip() or "unknown"
    rubric_family = "coding" if _is_coding_task_class(task_class_token) else "general"
    return f"{rubric_family}:{task_class_token}:{risk_tier_token}:{complexity_token}:{verification_token}"


def _build_selection_rubric(
    *,
    request: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    status: str,
    coding_signal_required: Optional[bool] = None,
    candidate_count: Optional[int] = None,
    eligible_candidate_count: Optional[int] = None,
    disqualified_candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()
    verification_class = str(_dispatch_contract_field(request, "verification_class") or "").strip()
    complexity_tier = _request_coding_complexity_tier(request)
    verification_profile = _coding_verification_profile(request, complexity_tier)
    strict_readiness_required = _coding_strict_readiness_required(
        request=request,
        complexity_tier=complexity_tier,
        routing_policy=routing_policy,
    )
    strict_readiness_allowed_states = sorted(
        routing_policy_coding_strict_readiness_allowed_readiness(routing_policy)
    )
    alias = CODING_TASK_CLASS_ROUTING_ALIASES.get(task_class)
    out: Dict[str, Any] = {
        "version": "clawd.session_topology.provider_selection_rubric.v1",
        "status": status,
        "routing_policy_id": routing_policy.get("policy_id") if isinstance(routing_policy, Mapping) else None,
        "task_class": task_class or None,
        "risk_tier": risk_tier or None,
        "verification_class": verification_class or None,
        "complexity_tier": complexity_tier,
        "verification_profile": verification_profile,
        "strict_readiness_required": strict_readiness_required,
        "strict_readiness_allowed_states": strict_readiness_allowed_states,
        "rubric_dimensions": {
            "risk_tier": risk_tier or None,
            "complexity_tier": complexity_tier,
            "verification_class": verification_class or None,
            "verification_profile": verification_profile,
            "strict_readiness_allowed_states": strict_readiness_allowed_states,
        },
        "rubric_rule_id": _selection_rubric_rule_id(task_class=task_class, risk_tier=risk_tier),
        "rubric_rule_id_v2": _selection_rubric_rule_id_v2(
            task_class=task_class,
            risk_tier=risk_tier,
            complexity_tier=complexity_tier,
            verification_profile=verification_profile,
        ),
        "rubric_family": "coding_qualification_first" if _is_coding_task_class(task_class) else "family_preference_first",
    }
    if alias:
        out["selector_alias_task_class"] = alias
    if coding_signal_required is not None:
        out["coding_signal_required"] = bool(coding_signal_required)
    if candidate_count is not None:
        out["candidate_count"] = int(candidate_count)
    if eligible_candidate_count is not None:
        out["eligible_candidate_count"] = int(eligible_candidate_count)
    if disqualified_candidate_count is not None:
        out["disqualified_candidate_count"] = int(disqualified_candidate_count)
    return out


def _provider_route_candidate_summary(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in candidates[:12]:
        if not isinstance(raw, Mapping):
            continue
        model_key = str(raw.get("model_key") or "").strip() or None
        signal = raw.get("qualification_signal") if isinstance(raw.get("qualification_signal"), Mapping) else {}
        coding_gate = raw.get("coding_signal_gate") if isinstance(raw.get("coding_signal_gate"), Mapping) else {}
        rows.append(
            {
                "model_key": model_key,
                "model_family": _model_family_from_model_key(model_key),
                "coding_signal_gate_reason": str(coding_gate.get("reason") or "").strip() or None,
                "readiness_state": str(signal.get("readiness_state") or "").strip() or None,
                "effective_score_0_100": signal.get("effective_score_0_100"),
                "provider_evidence_stale": bool(signal.get("provider_evidence_stale") is True),
            }
        )
    return rows


def _build_provider_route_decision(
    *,
    request: Mapping[str, Any],
    route_class: Optional[str],
    required_stage: Optional[str],
    selected_model: Optional[str],
    selected_qualification_signal: Mapping[str, Any],
    selection_rubric: Mapping[str, Any],
    ranked_candidates: List[Dict[str, Any]],
    eligible_candidates: List[Dict[str, Any]],
    disqualified_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()
    selected_family = _model_family_from_model_key(selected_model)
    disqualified = _provider_route_candidate_summary(disqualified_candidates)
    stale_disqualification = any(
        str(row.get("coding_signal_gate_reason") or "") in {"qualification_signal_stale", "provider_evidence_stale"}
        for row in disqualified
    )

    return {
        "schema": "clawd.session_topology.provider_route_decision.v1",
        "doctrine_id": "ex10_provider_doctrine_v1",
        "status": str(selection_rubric.get("status") or "not_evaluated").strip() or "not_evaluated",
        "task_class": task_class or None,
        "risk_tier": risk_tier or None,
        "route_class": str(route_class or "").strip() or None,
        "required_rollout_stage": str(required_stage or "").strip() or None,
        "selected_model": str(selected_model or "").strip() or None,
        "selected_model_family": selected_family,
        "selected_qualification_signal": dict(selected_qualification_signal or {}),
        "selection_rubric": dict(selection_rubric or {}),
        "rubric_rule_id": str(selection_rubric.get("rubric_rule_id") or "").strip() or None,
        "rubric_rule_id_v2": str(selection_rubric.get("rubric_rule_id_v2") or "").strip() or None,
        "candidate_count": len(ranked_candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "disqualified_candidate_count": len(disqualified_candidates),
        "disqualified_candidates": disqualified,
        "stale_decision_signal": stale_disqualification,
        "doctrine_assertions": {
            "coding_task": _is_coding_task_class(task_class),
            "codex_selected_for_coding": bool(_is_coding_task_class(task_class) and selected_family == "Codex"),
            "codex_avoided_for_synthesis": bool(task_class in SYNTHESIS_TASK_CLASSES and selected_family != "Codex"),
        },
    }


def _coding_signal_gate(
    *,
    request: Mapping[str, Any],
    qualification_signal: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    require_signal_for_coding: bool,
    max_age_seconds: Optional[int] = None,
    provider_max_age_seconds: Optional[int] = None,
    legacy_grace_period_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()
    verification_class = str(_dispatch_contract_field(request, "verification_class") or "").strip()
    complexity_tier = _request_coding_complexity_tier(request)
    verification_profile = _coding_verification_profile(request, complexity_tier)
    strict_readiness_required = _coding_strict_readiness_required(
        request=request,
        complexity_tier=complexity_tier,
        routing_policy=routing_policy,
    )
    strict_readiness_allowed_states = routing_policy_coding_strict_readiness_allowed_readiness(routing_policy)
    min_score = routing_policy_coding_min_score(routing_policy, risk_tier)
    gate_required = _is_coding_task_class(task_class) and (
        bool(require_signal_for_coding) or isinstance(min_score, (int, float))
    )

    readiness_state = str(qualification_signal.get("readiness_state") or "unknown")
    effective_score = qualification_signal.get("effective_score_0_100")
    base_fields = {
        "task_class": task_class,
        "risk_tier": risk_tier,
        "verification_class": verification_class or None,
        "complexity_tier": complexity_tier,
        "verification_profile": verification_profile,
        "strict_readiness_required": strict_readiness_required,
        "strict_readiness_allowed_states": sorted(strict_readiness_allowed_states),
    }

    if not gate_required:
        return {
            "required": False,
            "allowed_for_selection": True,
            "reason": "not_required",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": effective_score,
            "readiness_state": readiness_state,
        }

    # Check for stale qualification signal
    if readiness_state == "stale":
        result = {
            "required": True,
            "allowed_for_selection": False,
            "reason": "qualification_signal_stale",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": effective_score,
            "readiness_state": readiness_state,
            "freshness_reason": qualification_signal.get("freshness_reason"),
            "age_seconds": qualification_signal.get("age_seconds"),
        }
        
        # Add legacy diagnostics if applicable
        if qualification_signal.get("is_legacy_packet", False):
            result["legacy_diagnostics"] = {
                "is_legacy_packet": True,
                "missing_fields": qualification_signal.get("legacy_missing_fields", []),
                "migration_guidance": qualification_signal.get("legacy_migration_guidance", []),
                "grace_period_active": qualification_signal.get("legacy_grace_period_active", False),
                "grace_period_seconds": qualification_signal.get("legacy_grace_period_seconds"),
                "grace_window_active": qualification_signal.get("legacy_grace_window_active"),
                "grace_window_inactive_reason": qualification_signal.get("legacy_grace_window_inactive_reason"),
                "grace_window_expires_at": qualification_signal.get("legacy_grace_window_expires_at"),
                "grace_window_remaining_seconds": qualification_signal.get("legacy_grace_window_remaining_seconds"),
            }
        
        return result
    
    # Check for stale provider evidence
    provider_evidence_stale = qualification_signal.get("provider_evidence_stale", False)
    if provider_evidence_stale:
        result = {
            "required": True,
            "allowed_for_selection": False,
            "reason": "provider_evidence_stale",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": effective_score,
            "readiness_state": readiness_state,
            "provider_freshness_reason": qualification_signal.get("provider_freshness_reason"),
            "provider_evidence_age_seconds": qualification_signal.get("provider_evidence_age_seconds"),
        }
        
        # Add legacy diagnostics if applicable
        if qualification_signal.get("is_legacy_packet", False):
            result["legacy_diagnostics"] = {
                "is_legacy_packet": True,
                "missing_fields": qualification_signal.get("legacy_missing_fields", []),
                "migration_guidance": qualification_signal.get("legacy_migration_guidance", []),
                "grace_period_active": qualification_signal.get("legacy_grace_period_active", False),
                "grace_period_seconds": qualification_signal.get("legacy_grace_period_seconds"),
                "grace_window_active": qualification_signal.get("legacy_grace_window_active"),
                "grace_window_inactive_reason": qualification_signal.get("legacy_grace_window_inactive_reason"),
                "grace_window_expires_at": qualification_signal.get("legacy_grace_window_expires_at"),
                "grace_window_remaining_seconds": qualification_signal.get("legacy_grace_window_remaining_seconds"),
            }
        
        return result

    if not isinstance(effective_score, (int, float)):
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "missing_effective_score",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": effective_score,
            "readiness_state": readiness_state,
        }

    if readiness_state == "unknown":
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "missing_readiness_state",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": effective_score,
            "readiness_state": readiness_state,
        }

    score_value = float(effective_score)
    if isinstance(min_score, (int, float)) and score_value < float(min_score):
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "score_below_threshold",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": score_value,
            "readiness_state": readiness_state,
        }

    allowed_readiness = routing_policy_coding_allowed_readiness(routing_policy, risk_tier)

    if risk_tier == "critical" and readiness_state != "qualified":
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "critical_requires_qualified_readiness",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": score_value,
            "readiness_state": readiness_state,
        }

    if risk_tier == "high" and readiness_state not in {"qualified", "provisional"}:
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "high_requires_provisional_or_better_readiness",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": score_value,
            "readiness_state": readiness_state,
        }

    if strict_readiness_required and readiness_state not in strict_readiness_allowed_states:
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "strict_readiness_required_for_verification_profile",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": score_value,
            "readiness_state": readiness_state,
            "allowed_readiness_states": sorted(strict_readiness_allowed_states),
        }

    if allowed_readiness and readiness_state not in allowed_readiness:
        return {
            "required": True,
            "allowed_for_selection": False,
            "reason": "readiness_not_allowed_for_risk_tier",
            **base_fields,
            "min_score_0_100": min_score,
            "effective_score_0_100": score_value,
            "readiness_state": readiness_state,
            "allowed_readiness_states": sorted(allowed_readiness),
        }

    return {
        "required": True,
        "allowed_for_selection": True,
        "reason": "meets_high_risk_coding_threshold",
        **base_fields,
        "min_score_0_100": min_score,
        "effective_score_0_100": score_value,
        "readiness_state": readiness_state,
    }


def _prioritize_candidates_by_model_family(
    *,
    request: Mapping[str, Any],
    task_taxonomy: Mapping[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    default_family = str(task_taxonomy.get("default_model_family") or "").strip()
    fallback_families = [
        str(item or "").strip()
        for item in (task_taxonomy.get("fallback_model_families") or [])
        if str(item or "").strip()
    ]

    family_priority: Dict[str, int] = {}
    ordered_families: List[str] = []
    for family in [default_family, *fallback_families]:
        if family and family not in family_priority:
            family_priority[family] = len(family_priority)
            ordered_families.append(family)

    task_class = str(request.get("task_class") or "").strip()
    coding_task = _is_coding_task_class(task_class)

    def rank(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
        model_key = str(candidate.get("model_key") or "")
        family = _model_family_from_model_key(model_key)
        signal = _normalize_qualification_signal(candidate.get("qualification_signal"), max_age_seconds=None, provider_max_age_seconds=None)

        readiness_rank = QUALIFICATION_SIGNAL_READINESS_ORDER.get(str(signal.get("readiness_state") or "unknown"), QUALIFICATION_SIGNAL_READINESS_ORDER["unknown"])
        effective_score = signal.get("effective_score_0_100")
        score_rank = -float(effective_score) if isinstance(effective_score, (int, float)) else float("inf")
        provider_coverage = signal.get("provider_cost_coverage_rate")
        coverage_rank = -float(provider_coverage) if isinstance(provider_coverage, (int, float)) else float("inf")

        if family and family in family_priority:
            family_rank = family_priority[family]
            outside_priority_penalty = 0
        else:
            family_rank = len(family_priority)
            outside_priority_penalty = 1

        if coding_task:
            return readiness_rank, score_rank, coverage_rank, family_rank, outside_priority_penalty, model_key

        # Route-class compatible model in pool but outside preferred family set.
        return family_rank, outside_priority_penalty, readiness_rank, score_rank, coverage_rank, model_key

    prioritized = sorted(list(candidates), key=rank)
    return prioritized


def _task_class_guard(*, request: Mapping[str, Any], routing_policy: Mapping[str, Any]) -> Dict[str, Any]:
    session_kind = str(request.get("session_kind") or "").strip()
    task_class = str(request.get("task_class") or "").strip()

    tag_present = bool(task_class)
    known_in_family_matrix = task_class in routing_policy_known_task_classes(routing_policy)

    known_in_taxonomy_profile = False
    session_profile = TASK_TAXONOMY_PROFILE.get(session_kind)
    if isinstance(session_profile, Mapping):
        if "default_route_class" in session_profile:
            known_in_taxonomy_profile = True
        else:
            known_in_taxonomy_profile = task_class in session_profile and task_class != "_default"

    warnings: List[str] = []
    if not tag_present:
        warnings.append("task_class_missing")
    if session_kind == "worker_slice" and not known_in_taxonomy_profile:
        warnings.append("task_class_unmapped_taxonomy_profile")
    if session_kind == "worker_slice" and not known_in_family_matrix:
        warnings.append("task_class_unmapped_family_matrix")

    if not tag_present:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "session_kind": session_kind,
        "task_class": task_class,
        "tag_present": tag_present,
        "known_in_taxonomy_profile": known_in_taxonomy_profile,
        "known_in_family_matrix": known_in_family_matrix,
        "warnings": warnings,
    }


def _routing_telemetry_summary(
    *,
    request: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    task_taxonomy: Mapping[str, Any],
    route_class: Optional[str],
    selected_model: Optional[str],
    escalation_gate_details: Mapping[str, Any],
    telegram_direct_offload: Mapping[str, Any],
) -> Dict[str, Any]:
    task_class = str(request.get("task_class") or "").strip()
    selected_family = _model_family_from_model_key(selected_model)
    default_family = str(task_taxonomy.get("default_model_family") or "").strip() or None
    fallback_families = [
        str(x) for x in (task_taxonomy.get("fallback_model_families") or []) if str(x or "").strip()
    ]
    task_class_guard = _task_class_guard(request=request, routing_policy=routing_policy)

    misrouting_signals: List[str] = []
    if selected_family == "Codex" and task_class in SYNTHESIS_TASK_CLASSES:
        misrouting_signals.append("codex_used_for_synthesis_only")

    escalation_required = bool((escalation_gate_details or {}).get("escalation_required") is True)
    support_only_helper = bool((escalation_gate_details or {}).get("support_only_helper") is True)
    evidence_refs = list((escalation_gate_details or {}).get("artifact_refs") or [])
    if str(route_class or "") == "HEAVY" and not escalation_required:
        misrouting_signals.append("heavy_without_taxonomy_escalation")
    if escalation_required and not evidence_refs:
        misrouting_signals.append("escalated_without_artifact_refs")
    if support_only_helper and str(route_class or "") == "HEAVY":
        misrouting_signals.append("support_only_heavy_escalation")
    if "task_class_unmapped_taxonomy_profile" in (task_class_guard.get("warnings") or []):
        misrouting_signals.append("task_class_unmapped_taxonomy_profile")
    if "task_class_unmapped_family_matrix" in (task_class_guard.get("warnings") or []):
        misrouting_signals.append("task_class_unmapped_family_matrix")

    if bool((telegram_direct_offload or {}).get("handoff_recommended") is True):
        misrouting_signals.append("telegram_direct_worker_handoff_recommended")

    return {
        "task_class": task_class,
        "task_class_guard": task_class_guard,
        "selected_model_family": selected_family,
        "default_model_family": default_family,
        "fallback_model_families": fallback_families,
        "support_only_helper": support_only_helper,
        "codex_premium_mutation_lane": bool(selected_family == "Codex" and task_class in CODING_MUTATION_TASK_CLASSES),
        "misrouting_signals": misrouting_signals,
        "telegram_direct_offload": dict(telegram_direct_offload or {}),
    }


def _evaluate_escalation_gate(
    *,
    request: Mapping[str, Any],
    topology: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    selected_route_class: Optional[str],
    selected_required_stage: Optional[str],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    taxonomy = _task_taxonomy_for_request(request, topology, routing_policy)
    evidence = _normalize_escalation_evidence(request)

    baseline_route = str(taxonomy.get("default_route_class") or "").strip()
    baseline_stage = str(taxonomy.get("default_required_rollout_stage") or "").strip()
    selected_route = str(selected_route_class or "").strip()
    selected_stage = str(selected_required_stage or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()

    escalation_required = _route_rank(selected_route) > _route_rank(baseline_route)
    support_only_helper = _is_support_only_helper_request(request)
    support_only_heavy_path = bool(support_only_helper and selected_route == "HEAVY")

    signals: List[str] = []
    if risk_tier in {"high", "critical"}:
        signals.append("risk_tier_high_or_critical")
    for key, label in ESCALATION_SIGNAL_FIELDS.items():
        if evidence.get(key) is True:
            signals.append(label)

    non_risk_signals = [sig for sig in signals if sig != "risk_tier_high_or_critical"]
    evidence_refs = list(evidence.get("artifact_refs") or [])

    details = {
        "taxonomy": taxonomy,
        "baseline_route": {
            "route_class": baseline_route or None,
            "required_rollout_stage": baseline_stage or None,
        },
        "selected_route": {
            "route_class": selected_route or None,
            "required_rollout_stage": selected_stage or None,
        },
        "escalation_required": escalation_required,
        "support_only_helper": support_only_helper,
        "support_only_heavy_path": support_only_heavy_path,
        "signals": signals,
        "artifact_ref_count": len(evidence_refs),
        "artifact_refs": evidence_refs,
        "evidence": evidence,
    }

    if not escalation_required:
        details["status"] = "not_escalated"
        return True, None, details

    if not bool(taxonomy.get("allow_escalation") is True):
        details["status"] = "blocked"
        details["error"] = "taxonomy_escalation_forbidden"
        return False, "escalation_evidence_missing", details

    if support_only_heavy_path and not non_risk_signals:
        details["status"] = "blocked"
        details["error"] = "support_only_heavy_requires_non_risk_signal"
        return False, "escalation_evidence_missing", details

    if support_only_heavy_path and not evidence_refs:
        details["status"] = "blocked"
        details["error"] = "support_only_heavy_requires_artifact_refs"
        return False, "escalation_evidence_missing", details

    if not signals:
        details["status"] = "blocked"
        details["error"] = "escalation_signals_missing"
        return False, "escalation_evidence_missing", details

    if non_risk_signals and not evidence_refs:
        details["status"] = "blocked"
        details["error"] = "escalation_artifact_refs_missing"
        return False, "escalation_evidence_missing", details

    details["status"] = "escalated_with_evidence"
    return True, None, details


def _transport_binding_from_decision(decision: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    payload = decision
    if str(payload.get("schema") or "") == TRANSPORT_DECISION_SCHEMA:
        wrapper_decision = str(payload.get("decision") or "").strip().upper()
        if wrapper_decision == "BLOCK":
            return (
                False,
                "transport_route_conformance_failed",
                {
                    "error": "transport_decision_blocked",
                    "block_gate": payload.get("block_gate"),
                    "block_reason": payload.get("block_reason"),
                },
            )
        route_payload = payload.get("route")
        if isinstance(route_payload, Mapping):
            payload = route_payload

    if str(payload.get("schema_version") or "") != TRANSPORT_ROUTE_SCHEMA:
        return False, "transport_route_conformance_failed", {"error": "transport_decision_schema_invalid"}

    decision_token = str(payload.get("decision") or "PASS").strip().upper()
    if decision_token == "BLOCK":
        return (
            False,
            "transport_route_conformance_failed",
            {
                "error": "transport_decision_blocked",
                "block_gate": payload.get("block_gate"),
                "block_reason": payload.get("block_reason"),
            },
        )

    routing_basis = payload.get("routing_basis") if isinstance(payload.get("routing_basis"), Mapping) else {}
    lane = payload.get("lane") if isinstance(payload.get("lane"), Mapping) else {}
    session = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}
    transport = payload.get("transport") if isinstance(payload.get("transport"), Mapping) else {}

    transport_key = str(routing_basis.get("transport_key") or "").strip()
    lane_name = str(lane.get("name") or "").strip()
    agent_id = str(lane.get("agent_id") or "").strip()
    session_key = str(session.get("session_key") or "").strip()

    if not transport_key or not lane_name or not agent_id or not session_key:
        return (
            False,
            "transport_route_conformance_failed",
            {
                "error": "transport_decision_missing_required_fields",
                "transport_key": transport_key or None,
                "lane_name": lane_name or None,
                "agent_id": agent_id or None,
                "session_key": session_key or None,
            },
        )

    binding: Dict[str, Any] = {
        "transport_key": transport_key,
        "lane_name": lane_name,
        "agent_id": agent_id,
        "session_key": session_key,
    }
    if "message_thread_id" in transport:
        binding["message_thread_id"] = transport.get("message_thread_id")
    return True, None, binding


def _transport_route_conformance_gate(
    *,
    request_transport_binding: Mapping[str, Any],
    request_agent_binding: Mapping[str, str],
    transport_decision: Optional[Mapping[str, Any]],
    require_transport_decision: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if transport_decision is None:
        if require_transport_decision:
            return (
                False,
                "transport_route_conformance_failed",
                {
                    "error": "transport_decision_required",
                },
            )
        return (
            True,
            None,
            {
                "mode": "not_required",
                "request_transport_binding": dict(request_transport_binding),
                "request_agent_binding": dict(request_agent_binding),
            },
        )

    ok, reason, decision_binding = _transport_binding_from_decision(transport_decision)
    if not ok:
        return False, reason, decision_binding

    mismatches: List[Dict[str, Any]] = []
    for field in ("transport_key", "lane_name", "agent_id", "session_key", "message_thread_id"):
        if field not in request_transport_binding:
            continue
        expected = request_transport_binding.get(field)
        actual = decision_binding.get(field)
        if expected != actual:
            mismatches.append({"field": field, "expected": expected, "actual": actual})

    for field in ("agent_id", "session_key", "lane_name"):
        if field not in request_agent_binding:
            continue
        expected = request_agent_binding.get(field)
        actual = decision_binding.get(field)
        if expected != actual:
            mismatches.append({"field": field, "expected": expected, "actual": actual})

    if mismatches:
        return (
            False,
            "transport_route_mismatch",
            {
                "error": "transport_binding_mismatch",
                "request_transport_binding": dict(request_transport_binding),
                "request_agent_binding": dict(request_agent_binding),
                "transport_decision_binding": decision_binding,
                "mismatches": mismatches,
            },
        )

    return (
        True,
        None,
        {
            "mode": "transport_decision_bound",
            "binding": decision_binding,
            "request_transport_binding": dict(request_transport_binding),
            "request_agent_binding": dict(request_agent_binding),
        },
    )


def _transport_scope_from_key(transport_key: Any) -> Dict[str, Optional[str]]:
    token = str(transport_key or "").strip()
    parts = token.split("|")
    if len(parts) < 4:
        return {
            "transport": None,
            "scope": None,
            "chat_id": None,
            "thread_key": None,
        }
    return {
        "transport": str(parts[0] or "").strip() or None,
        "scope": str(parts[1] or "").strip() or None,
        "chat_id": str(parts[2] or "").strip() or None,
        "thread_key": str(parts[3] or "").strip() or None,
    }


def _telegram_direct_offload_gate(
    *,
    request: Mapping[str, Any],
    route_class: Optional[str],
    resolved_transport_binding: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    enforce: bool,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    def _request_text_field(key: str) -> str:
        direct = request.get(key)
        if isinstance(direct, str) and direct.strip():
            return str(direct).strip()
        nested = _request_dispatch_contract(request).get(key)
        if isinstance(nested, str) and nested.strip():
            return str(nested).strip()
        return ""

    details: Dict[str, Any] = {
        "schema": "clawd.session_topology.telegram_direct_offload_gate.v1",
        "policy_id": "telegram_dm_heavy_offload_v1",
        "enforced": bool(enforce),
        "legacy_heavy_bypass_enabled": bool(not enforce),
        "status": "not_evaluated",
        "reason": None,
        "requires_worker_handoff": False,
        "handoff_recommended": False,
        "target_worker_pattern": TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS[0],
        "target_worker_patterns": list(TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS),
        "route_class": str(route_class or "").strip() or None,
        "session_kind": str(request.get("session_kind") or "").strip() or None,
        "task_class": str(request.get("task_class") or "").strip() or None,
        "risk_tier": str(request.get("risk_tier") or "").strip() or None,
    }

    binding = dict(resolved_transport_binding or {})
    transport_key = str(binding.get("transport_key") or "").strip()
    binding_lane_name = str(binding.get("lane_name") or "").strip()
    binding_agent_id = str(binding.get("agent_id") or "").strip()
    binding_session_key = str(binding.get("session_key") or "").strip()
    scope = _transport_scope_from_key(transport_key)
    details["transport_binding"] = {
        "transport_key": transport_key or None,
        "lane_name": binding_lane_name or None,
        "agent_id": binding_agent_id or None,
        "session_key": binding_session_key or None,
        "message_thread_id": binding.get("message_thread_id"),
        "transport": scope.get("transport"),
        "scope": scope.get("scope"),
        "chat_id": scope.get("chat_id"),
        "thread_key": scope.get("thread_key"),
    }

    if scope.get("transport") != "telegram" or scope.get("scope") != "direct":
        details["status"] = "not_applicable"
        details["reason"] = "non_telegram_direct_transport"
        return True, None, details

    session_kind = str(request.get("session_kind") or "").strip()
    if session_kind != "worker_slice":
        details["status"] = "blocked"
        details["reason"] = "telegram_direct_session_kind_disallowed"
        details["allowed_session_kinds"] = ["worker_slice"]
        details["requires_worker_handoff"] = False
        return False, "telegram_direct_session_kind_invalid", details

    scope_shape = str(_dispatch_contract_field(request, "scope_shape") or "").strip()
    worker_topology = str(_dispatch_contract_field(request, "worker_topology") or "").strip()
    verification_class = str(_dispatch_contract_field(request, "verification_class") or "").strip()
    worker_lane_input = _request_text_field("worker_lane")
    worker_lane, worker_lane_raw, worker_lane_token = _normalize_worker_lane(worker_lane_input)
    worker_lane_valid: Optional[bool]
    if worker_lane_raw is None:
        worker_lane_valid = None
    else:
        worker_lane_valid = worker_lane is not None
    delegation_basis = _request_text_field("delegation_basis")

    heavy_shape = bool(
        scope_shape in {"multi_surface_disjoint", "multi_surface_coupled"}
        or worker_topology in {"parallel_fanout", "staged_serial"}
    )

    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()

    is_worker_slice = bool(session_kind == "worker_slice")
    is_coding_task = _is_coding_task_class(task_class)

    tiny_exception_requested = bool(worker_lane == "main_session_tiny_exception")
    tiny_exception_contract = {
        "worker_lane_main_session_tiny_exception": tiny_exception_requested,
        "risk_tier_low": risk_tier == "low",
        "scope_shape_single_surface": scope_shape == "single_surface",
        "verification_class_self_check": verification_class == "self_check",
        "worker_topology_single": worker_topology == "single",
        "delegation_basis_present": bool(delegation_basis),
    }
    tiny_exception_missing: List[str] = []
    if tiny_exception_requested:
        tiny_exception_missing = [key for key, ok in tiny_exception_contract.items() if not bool(ok)]
    tiny_exception_allowed = bool(tiny_exception_requested and not tiny_exception_missing)

    worker_offload_declared = bool(worker_lane == "subagent_default")
    risk_tier_non_trivial = bool(risk_tier in {"medium", "high", "critical"})
    verification_non_trivial = bool(
        verification_class in {"validator_required", "validator_plus_human"}
    )
    non_trivial_worker_shape = bool(verification_non_trivial or risk_tier_non_trivial)

    requires_heavy_handoff = bool(str(route_class or "").strip() == "HEAVY" or heavy_shape)
    requires_non_trivial_handoff = bool(
        is_worker_slice
        and non_trivial_worker_shape
        and not worker_offload_declared
        and not tiny_exception_allowed
    )
    requires_coding_handoff = bool(
        is_worker_slice
        and is_coding_task
        and not worker_offload_declared
        and not tiny_exception_allowed
    )
    requires_handoff = bool(requires_heavy_handoff or requires_non_trivial_handoff or requires_coding_handoff)
    details["scope_shape"] = scope_shape or None
    details["worker_topology"] = worker_topology or None
    details["verification_class"] = verification_class or None
    details["worker_lane"] = worker_lane or None
    details["worker_lane_input"] = worker_lane_raw
    details["worker_lane_token"] = worker_lane_token
    details["worker_lane_valid"] = worker_lane_valid
    details["delegation_basis"] = delegation_basis or None
    details["tiny_exception_requested"] = tiny_exception_requested
    details["tiny_exception_allowed"] = tiny_exception_allowed
    details["tiny_exception_contract"] = tiny_exception_contract
    details["tiny_exception_missing"] = tiny_exception_missing
    details["worker_offload_declared"] = worker_offload_declared
    details["is_coding_task"] = is_coding_task
    details["risk_tier_non_trivial"] = risk_tier_non_trivial
    details["verification_non_trivial"] = verification_non_trivial
    details["heavy_orchestration_shape"] = heavy_shape
    details["non_trivial_worker_shape"] = non_trivial_worker_shape
    details["requires_non_trivial_worker_handoff"] = requires_non_trivial_handoff
    details["requires_coding_worker_handoff"] = requires_coding_handoff
    details["requires_worker_handoff"] = requires_handoff
    details["target_worker_lane"] = "subagent_default"
    disallowed_lane_tokens = routing_policy_telegram_direct_worker_target_disallowed_lane_tokens(routing_policy)

    worker_target_topology_evidence_missing_fields: List[str] = []
    if not binding_lane_name:
        worker_target_topology_evidence_missing_fields.append("lane_name")
    if not binding_agent_id:
        worker_target_topology_evidence_missing_fields.append("agent_id")
    if not binding_session_key:
        worker_target_topology_evidence_missing_fields.append("session_key")

    worker_target_topology_evidence_present = len(worker_target_topology_evidence_missing_fields) == 0
    worker_target_conformance_errors: List[str] = []
    lane_name_pattern_match = _telegram_direct_worker_target_pattern_match(binding_lane_name)
    agent_id_pattern_match = _telegram_direct_worker_target_pattern_match(binding_agent_id)
    lane_token = _telegram_direct_worker_target_token(binding_lane_name)
    if lane_token and lane_token in disallowed_lane_tokens:
        worker_target_conformance_errors.append("lane_name_cockpit_lane_disallowed")
    if binding_lane_name and not lane_name_pattern_match:
        worker_target_conformance_errors.append("lane_name_pattern_mismatch")
    if binding_agent_id and not agent_id_pattern_match:
        worker_target_conformance_errors.append("agent_id_pattern_mismatch")
    if binding_lane_name and binding_agent_id and binding_lane_name != binding_agent_id:
        worker_target_conformance_errors.append("lane_agent_mismatch")

    session_key_agent_id = _session_key_agent_id(binding_session_key)
    if binding_session_key and not session_key_agent_id:
        worker_target_conformance_errors.append("session_key_format_invalid")
    if session_key_agent_id and binding_agent_id and session_key_agent_id != binding_agent_id:
        worker_target_conformance_errors.append("session_key_agent_mismatch")

    worker_target_topology_conformance_ok = len(worker_target_conformance_errors) == 0
    details["worker_target_topology_evidence"] = {
        "required_for_declared_offload": True,
        "present": worker_target_topology_evidence_present,
        "missing_fields": worker_target_topology_evidence_missing_fields,
        "lane_name": binding_lane_name or None,
        "agent_id": binding_agent_id or None,
        "session_key": binding_session_key or None,
        "allowed_patterns": list(TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS),
        "disallowed_lane_tokens": sorted(disallowed_lane_tokens),
        "lane_name_pattern_match": lane_name_pattern_match,
        "agent_id_pattern_match": agent_id_pattern_match,
        "session_key_agent_id": session_key_agent_id,
        "conformance_ok": worker_target_topology_conformance_ok,
        "conformance_errors": worker_target_conformance_errors,
    }

    if requires_heavy_handoff:
        if not enforce:
            details["status"] = "legacy_bypass"
            details["reason"] = "legacy_allow_telegram_direct_heavy"
            details["legacy_bypass_scope"] = "heavy_only"
            return True, None, details
        details["status"] = "blocked"
        details["reason"] = "telegram_direct_heavy_not_offloaded"
        return False, "telegram_direct_heavy_offload_required", details

    if requires_non_trivial_handoff:
        details["status"] = "blocked"
        if verification_non_trivial:
            details["reason"] = "telegram_direct_nontrivial_not_offloaded"
        else:
            details["reason"] = "telegram_direct_mediumplus_not_offloaded"
        return False, "telegram_direct_worker_offload_required", details

    if requires_coding_handoff:
        details["status"] = "blocked"
        if worker_lane_valid is False:
            details["reason"] = "telegram_direct_coding_worker_lane_invalid"
        else:
            details["reason"] = "telegram_direct_coding_not_offloaded"
        return False, "telegram_direct_worker_offload_required", details

    if worker_offload_declared and is_worker_slice and not worker_target_topology_evidence_present:
        details["status"] = "blocked"
        details["reason"] = "telegram_direct_worker_target_evidence_missing"
        return False, "telegram_direct_worker_target_evidence_missing", details

    if (
        worker_offload_declared
        and is_worker_slice
        and worker_target_topology_evidence_present
        and not worker_target_topology_conformance_ok
    ):
        details["status"] = "blocked"
        details["reason"] = "telegram_direct_worker_target_evidence_invalid"
        return False, "telegram_direct_worker_target_evidence_invalid", details

    if worker_offload_declared and is_worker_slice:
        details["status"] = "worker_offload_declared"
        details["reason"] = "telegram_direct_worker_lane_offload_declared"
        return True, None, details

    if tiny_exception_allowed and is_worker_slice:
        details["status"] = "tiny_exception_allowed"
        details["reason"] = "telegram_direct_tiny_exception_allowed"
        return True, None, details

    if is_worker_slice:
        details["status"] = "handoff_recommended"
        details["reason"] = "telegram_direct_worker_slice_handoff_recommended"
        details["handoff_recommended"] = True
    else:
        details["status"] = "pass"
        details["reason"] = "no_direct_offload_requirement"

    return True, None, details


def _request_contract_gate(
    request: Mapping[str, Any],
    *,
    routing_policy: Mapping[str, Any],
    require_worker_allocation_contract: bool = False,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    session_kind = str(request.get("session_kind") or "").strip()
    task_class = str(request.get("task_class") or "").strip()
    risk_tier = str(request.get("risk_tier") or "").strip()
    route_lock = _request_route_lock(request)
    transport_binding = _request_transport_binding(request)
    agent_binding = _request_agent_binding(request)

    if not session_kind:
        return False, "routing_request_invalid", {"error": "session_kind_missing"}
    if not task_class:
        return False, "routing_request_invalid", {"error": "task_class_missing"}
    if risk_tier not in ALLOWED_RISK_TIERS:
        return False, "routing_request_invalid", {"error": "risk_tier_invalid", "risk_tier": risk_tier or None}

    support_only_raw = request.get("support_only")
    if support_only_raw is not None and not isinstance(support_only_raw, bool):
        return (
            False,
            "routing_request_invalid",
            {
                "error": "support_only_invalid",
                "detail": "expected_boolean",
            },
        )

    fold_in_target = _normalize_fold_in_target(request.get("fold_in_target"))
    if fold_in_target is not None and fold_in_target not in FOLD_IN_TARGET_ALLOWLIST:
        return (
            False,
            "routing_request_invalid",
            {
                "error": "fold_in_target_invalid",
                "fold_in_target": fold_in_target,
                "allowed": sorted(FOLD_IN_TARGET_ALLOWLIST),
            },
        )

    worker_allocation_ok, worker_allocation_reason, worker_allocation_details = _worker_allocation_contract_gate(
        request=request,
        session_kind=session_kind,
        risk_tier=risk_tier,
        fold_in_target=fold_in_target,
        require_worker_allocation_contract=require_worker_allocation_contract,
    )
    if not worker_allocation_ok:
        return False, worker_allocation_reason, worker_allocation_details

    route_class = route_lock.get("route_class")
    if route_class and route_class not in ALLOWED_ROUTE_CLASSES:
        return False, "routing_request_invalid", {"error": "requested_route_class_invalid", "requested_route_class": route_class}

    required_stage = route_lock.get("required_rollout_stage")
    if required_stage and required_stage not in ALLOWED_ROLLOUT_STAGES:
        return (
            False,
            "routing_request_invalid",
            {
                "error": "requested_required_rollout_stage_invalid",
                "requested_required_rollout_stage": required_stage,
            },
        )

    thread_id = transport_binding.get("message_thread_id") if "message_thread_id" in transport_binding else None
    if "message_thread_id" in transport_binding:
        if thread_id is not None and (not isinstance(thread_id, int) or thread_id <= 0):
            return (
                False,
                "routing_request_invalid",
                {
                    "error": "transport_binding_message_thread_id_invalid",
                    "message_thread_id": thread_id,
                },
            )

    prompt_raw = request.get("invocation_prompt")
    if prompt_raw is not None and not isinstance(prompt_raw, str):
        return (
            False,
            "routing_request_invalid",
            {
                "error": "invocation_prompt_invalid",
                "detail": "expected_string",
            },
        )

    prompt_guardrails_raw = request.get("prompt_guardrails")
    if prompt_guardrails_raw is not None and not isinstance(prompt_guardrails_raw, Mapping):
        return (
            False,
            "routing_request_invalid",
            {
                "error": "prompt_guardrails_invalid",
                "detail": "expected_object",
            },
        )

    if isinstance(prompt_guardrails_raw, Mapping):
        for key in ("requested_output_tokens", "max_prompt_tokens", "max_total_tokens"):
            if key not in prompt_guardrails_raw:
                continue
            value = prompt_guardrails_raw.get(key)
            try:
                parsed = int(value)
            except Exception:
                return (
                    False,
                    "routing_request_invalid",
                    {
                        "error": "prompt_guardrails_invalid",
                        "detail": f"{key}_must_be_int",
                    },
                )
            if parsed < 0:
                return (
                    False,
                    "routing_request_invalid",
                    {
                        "error": "prompt_guardrails_invalid",
                        "detail": f"{key}_must_be_non_negative",
                    },
                )

    escalation_evidence_raw = request.get("escalation_evidence")
    if escalation_evidence_raw is not None and not isinstance(escalation_evidence_raw, Mapping):
        return (
            False,
            "routing_request_invalid",
            {
                "error": "escalation_evidence_invalid",
                "detail": "expected_object",
            },
        )
    if isinstance(escalation_evidence_raw, Mapping):
        artifact_refs = escalation_evidence_raw.get("artifact_refs")
        if artifact_refs is not None and not isinstance(artifact_refs, list):
            return (
                False,
                "routing_request_invalid",
                {
                    "error": "escalation_evidence_invalid",
                    "detail": "artifact_refs_must_be_array",
                },
            )

    retrieval_ok, retrieval_reason, retrieval_details = validate_hybrid_retrieval_request(request)
    if not retrieval_ok:
        return False, retrieval_reason, retrieval_details

    workflow_dag_packet = _workflow_dag_orchestration_request(request)
    if workflow_dag_packet.get("status") == "fail":
        return (
            False,
            "routing_request_invalid",
            {
                "error": "workflow_dag_invalid",
                "workflow_dag": workflow_dag_packet,
            },
        )

    return (
        True,
        None,
        {
            "session_kind": session_kind,
            "task_class": task_class,
            "risk_tier": risk_tier,
            "fold_in_target": fold_in_target,
            "support_only_helper": _is_support_only_helper_request(request),
            "route_lock": route_lock,
            "transport_binding": transport_binding,
            "agent_binding": agent_binding,
            "escalation_evidence": _normalize_escalation_evidence(request),
            "task_taxonomy": _task_taxonomy_for_request(request, {}, routing_policy),
            "prompt_guardrails": _normalize_prompt_guardrail_request(request),
            "invocation_prompt_present": isinstance(request.get("invocation_prompt"), str),
            "proposal_packet_present": isinstance(request.get("proposal_packet"), Mapping),
            "proposal_phase": str(request.get("proposal_phase") or "").strip().lower() or None,
            "regression_risk_packet_present": isinstance(request.get("regression_risk_packet"), Mapping),
            "refactor_risk_packet_present": isinstance(request.get("refactor_risk_packet"), Mapping),
            "knowledge_retrieval": retrieval_details,
            "workflow_dag": workflow_dag_packet,
            "worker_allocation_contract": worker_allocation_details,
        },
    )


def _workflow_dag_orchestration_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("workflow_dag") if isinstance(request, Mapping) else None

    if raw is None:
        return {
            "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
            "status": "not_requested",
            "enabled": False,
            "requested": False,
            "reason": "workflow_dag_not_present",
            "node_count": 0,
            "edge_count": 0,
            "execution_order": [],
            "execution_levels": [],
        }

    if not isinstance(raw, Mapping):
        return {
            "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
            "status": "fail",
            "enabled": True,
            "requested": True,
            "reason": "workflow_dag_must_be_object",
            "detail": {"error": "workflow_dag_invalid_type"},
            "node_count": 0,
            "edge_count": 0,
            "execution_order": [],
            "execution_levels": [],
        }

    nodes_raw = raw.get("nodes") if isinstance(raw.get("nodes"), list) else None
    if nodes_raw is None or not nodes_raw:
        return {
            "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
            "status": "fail",
            "enabled": True,
            "requested": True,
            "reason": "workflow_dag_nodes_missing_or_empty",
            "detail": {"error": "nodes_required_and_non_empty"},
            "node_count": 0,
            "edge_count": 0,
            "execution_order": [],
            "execution_levels": [],
        }

    node_set: set[str] = set()
    normalized_nodes: List[str] = []
    for index, raw_node in enumerate(nodes_raw, 1):
        if isinstance(raw_node, str):
            node_id = str(raw_node).strip()
        elif isinstance(raw_node, Mapping):
            node_id = str(raw_node.get("id") or "").strip()
        else:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_node_invalid",
                "detail": {"error": "all_nodes_must_be_string_or_object", "index": index},
                "node_count": 0,
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if not node_id:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_node_id_missing",
                "detail": {"error": "node_id_must_be_non_empty", "index": index},
                "node_count": 0,
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if node_id in node_set:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_duplicate_node",
                "detail": {"error": "duplicate_node_id", "node_id": node_id},
                "node_count": 0,
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        node_set.add(node_id)
        normalized_nodes.append(node_id)

    edges_raw = raw.get("edges") if raw.get("edges") is not None else []
    if not isinstance(edges_raw, list):
        return {
            "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
            "status": "fail",
            "enabled": True,
            "requested": True,
            "reason": "workflow_dag_edges_invalid",
            "detail": {"error": "edges_must_be_array"},
            "node_count": len(normalized_nodes),
            "edge_count": 0,
            "execution_order": [],
            "execution_levels": [],
        }

    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in normalized_nodes}
    indegree: Dict[str, int] = {node_id: 0 for node_id in normalized_nodes}
    seen_edges: set[tuple[str, str]] = set()
    normalized_edges: List[Dict[str, str]] = []

    for index, raw_edge in enumerate(edges_raw, 1):
        if isinstance(raw_edge, Mapping):
            source = str(raw_edge.get("from") or "").strip()
            target = str(raw_edge.get("to") or "").strip()
        elif isinstance(raw_edge, (list, tuple)) and len(raw_edge) == 2:
            source = str(raw_edge[0]).strip()
            target = str(raw_edge[1]).strip()
        else:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_edge_invalid",
                "detail": {"error": "edges_must_be_object_or_pair", "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if not source or not target:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_edge_endpoint_missing",
                "detail": {"error": "edge_source_and_target_required", "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if source not in node_set:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_unknown_source",
                "detail": {"error": "edge_source_unknown", "source": source, "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if target not in node_set:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_unknown_target",
                "detail": {"error": "edge_target_unknown", "target": target, "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        if source == target:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_self_cycle",
                "detail": {"error": "self_cycle_not_allowed", "node_id": source, "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        key = (source, target)
        if key in seen_edges:
            return {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "fail",
                "enabled": True,
                "requested": True,
                "reason": "workflow_dag_duplicate_edge",
                "detail": {"error": "duplicate_edge", "from": source, "to": target, "index": index},
                "node_count": len(normalized_nodes),
                "edge_count": 0,
                "execution_order": [],
                "execution_levels": [],
            }

        seen_edges.add(key)
        adjacency[source].append(target)
        indegree[target] += 1
        normalized_edges.append({"from": source, "to": target})

    ready: deque[str] = deque([node for node, degree in indegree.items() if degree == 0])
    if ready:
        ready = deque(sorted(ready))

    execution_order: List[str] = []
    execution_levels: List[List[str]] = []
    longest_path: Dict[str, int] = {node: 1 for node in normalized_nodes}

    while ready:
        current_level = list(ready)
        ready = deque()
        execution_levels.append(current_level)
        for node in current_level:
            ready_nodes: List[str] = []
            execution_order.append(node)
            for target in sorted(adjacency.get(node, [])):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready_nodes.append(target)
                longest_path[target] = max(longest_path[target], longest_path[node] + 1)
            for ready_node in ready_nodes:
                if ready_node not in ready:
                    ready.append(ready_node)
        ready = deque(sorted(ready))

    if len(execution_order) != len(normalized_nodes):
        return {
            "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
            "status": "fail",
            "enabled": True,
            "requested": True,
            "reason": "workflow_dag_cycle_detected",
            "detail": {"error": "topological_sort_failed", "remaining_nodes": sorted([node for node, degree in indegree.items() if degree > 0])},
            "node_count": len(normalized_nodes),
            "edge_count": len(normalized_edges),
            "execution_order": [],
            "execution_levels": [],
        }

    return {
        "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
        "status": "pass",
        "enabled": True,
        "requested": True,
        "node_count": len(normalized_nodes),
        "edge_count": len(normalized_edges),
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "execution_order": execution_order,
        "execution_levels": execution_levels,
        "critical_path_length": max(longest_path.values()) if longest_path else 0,
        "max_parallelism": max((len(level) for level in execution_levels), default=0),
        "parallel_stages": len(execution_levels),
        "canary_evaluation_time_ms": 0,
        "detail": {"source": str(raw.get("source") or "user"), "name": str(raw.get("name") or "workflow")},
    }



def _normalize_allowed_stages(row: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    stages = row.get("allowed_stages") if isinstance(row.get("allowed_stages"), list) else []
    for stage in stages:
        text = str(stage or "").strip()
        if text in ALLOWED_ROLLOUT_STAGES:
            out.add(text)

    approved_stage = str(row.get("approved_stage") or row.get("target_stage") or "").strip()
    if approved_stage in ALLOWED_ROLLOUT_STAGES:
        out.add(approved_stage)

    return out


def _collect_model_gate_index(gate_decisions: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in gate_decisions:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("schema") or "") != "clawd.model_rollout_gate.decision.v1":
            continue
        if str(row.get("decision") or "") != "PASS":
            continue

        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        model_key = str(model.get("model_key") or model.get("model_ref") or "").strip()
        route_class = str(model.get("route_class") or model.get("model_family") or "").strip()
        provider = str(model.get("provider") or "").strip() or None
        if not model_key or route_class not in ALLOWED_ROUTE_CLASSES:
            continue

        rollout = row.get("rollout") if isinstance(row.get("rollout"), Mapping) else {}
        allowed_stages = _normalize_allowed_stages(rollout)
        if not allowed_stages:
            continue

        index[model_key] = {
            "model_key": model_key,
            "provider": provider,
            "route_class": route_class,
            "allowed_stages": sorted(allowed_stages),
            "qualification_signal": _normalize_qualification_signal(row.get("qualification_signal"), max_age_seconds=None, provider_max_age_seconds=None),
            "decision": dict(row),
        }
    return index


def _stage_is_allowed(required_stage: str, allowed_stages: set[str]) -> bool:
    if required_stage == "canary":
        return "canary" in allowed_stages or "active" in allowed_stages
    if required_stage == "active":
        return "active" in allowed_stages
    return False


def _check_quota_freshness(repo_root: Path) -> Dict[str, Any]:
    """Check quota matrix freshness and return a gate dict."""
    quota_matrix_path = repo_root / "state" / "codex_quota_matrix_latest.json"
    if not quota_matrix_path.exists():
        return {
            "gate": "quota_freshness",
            "status": "skip",
            "reason": "quota_matrix_missing",
            "details": {"path": str(quota_matrix_path)},
        }
    try:
        with quota_matrix_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "gate": "quota_freshness",
            "status": "warn",
            "reason": "quota_matrix_unreadable",
            "details": {"path": str(quota_matrix_path), "error": str(e)},
        }
    snapshot_str = data.get("snapshotTime")
    if not snapshot_str:
        return {
            "gate": "quota_freshness",
            "status": "warn",
            "reason": "quota_matrix_missing_snapshot",
            "details": {"path": str(quota_matrix_path)},
        }
    try:
        # Parse ISO 8601 with timezone
        snapshot_dt = dt.datetime.fromisoformat(snapshot_str.replace("Z", "+00:00"))
    except Exception as e:
        return {
            "gate": "quota_freshness",
            "status": "warn",
            "reason": "quota_matrix_invalid_snapshot",
            "details": {"path": str(quota_matrix_path), "snapshot": snapshot_str, "error": str(e)},
        }
    now = dt.datetime.now(dt.timezone.utc)
    age = now - snapshot_dt
    max_age = dt.timedelta(hours=24)
    if age > max_age:
        return {
            "gate": "quota_freshness",
            "status": "warn",
            "reason": "quota_matrix_stale",
            "details": {
                "path": str(quota_matrix_path),
                "snapshot": snapshot_str,
                "age_hours": round(age.total_seconds() / 3600, 1),
                "max_age_hours": 24,
            },
        }
    # Fresh
    return {
        "gate": "quota_freshness",
        "status": "pass",
        "details": {
            "path": str(quota_matrix_path),
            "snapshot": snapshot_str,
            "age_hours": round(age.total_seconds() / 3600, 1),
        },
    }

def gate_pool_policy_alignment(
    topology: Mapping[str, Any],
    route_class: str,
    selected_model: Optional[str],
    pool_policy: Mapping[str, Any],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    route_entry = policy_route_entry(pool_policy, route_class)
    if not route_entry:
        return False, "pool_policy_violation", {"error": "route_class_not_defined", "route_class": route_class}

    allowed_models = policy_allowed_models(pool_policy, route_class)

    model_pools = topology.get("model_pools") if isinstance(topology.get("model_pools"), Mapping) else {}
    pool = model_pools.get(route_class)

    if route_class != "NO_LLM":
        if not isinstance(pool, list) or not pool:
            return False, "pool_policy_violation", {"error": "model_pool_missing", "route_class": route_class}

        invalid_pool_models = [
            str(model_key)
            for model_key in pool
            if isinstance(model_key, str) and model_key.strip() and str(model_key) not in allowed_models
        ]
        if invalid_pool_models:
            return (
                False,
                "pool_policy_violation",
                {
                    "error": "topology_pool_contains_unapproved_models",
                    "route_class": route_class,
                    "invalid_models": invalid_pool_models,
                    "allowed_models": sorted(allowed_models),
                },
            )

        if selected_model and selected_model not in allowed_models:
            return (
                False,
                "pool_policy_violation",
                {
                    "error": "selected_model_not_approved",
                    "route_class": route_class,
                    "selected_model": selected_model,
                    "allowed_models": sorted(allowed_models),
                },
            )

    return (
        True,
        None,
        {
            "policy_id": pool_policy.get("policy_id"),
            "route_class": route_class,
            "selected_model": selected_model,
            "route_owner_lane_id": route_entry.get("owner_lane_id"),
            "allowed_model_count": len(allowed_models),
        },
    )


def _resolve_repo_jsonl_path(repo_root: Path, raw_path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    if raw_path is None:
        return None, None
    path = raw_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    if not is_within(repo_root, path):
        return None, "unsafe_path"
    return path, None


def _append_jsonl_payload(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() and not path.is_file():
        raise ValueError("path_not_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stable_json_dumps(dict(payload)) + "\n")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_decision_record(*, decision_log_path: Optional[Path], repo_root: Path, decision_row: Dict[str, Any]) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path, resolve_error = _resolve_repo_jsonl_path(repo_root, decision_log_path)
    if path is None:
        return {"enabled": True, "appended": False, "reason": resolve_error or "unsafe_path"}

    try:
        _append_jsonl_payload(path, decision_row)
        return {"enabled": True, "appended": True, "path": str(path)}
    except Exception as exc:
        reason = str(exc) if str(exc) == "path_not_file" else "append_failed"
        return {
            "enabled": True,
            "appended": False,
            "reason": reason,
            "path": str(path),
            "error": None if reason == "path_not_file" else str(exc),
        }


def _event_backbone_sequence(raw: Any) -> int:
    try:
        value = int(raw)
    except Exception:
        return 1
    return value if value > 0 else 1


def _event_backbone_backoff_ms(*, idempotency_key: str, attempt: int, base_backoff_ms: int) -> int:
    seed = hashlib.sha256(f"{idempotency_key}|{attempt}".encode("utf-8")).hexdigest()
    jitter = int(seed[:4], 16) % max(1, int(base_backoff_ms))
    return (max(1, int(base_backoff_ms)) * (2 ** max(0, int(attempt) - 1))) + jitter


def _event_backbone_normalize_snapshot(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        out: Dict[str, Any] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key)
            if key in {
                "decision_record",
                "event_backbone",
                "token_violation_record",
                "operator_diagnostics",
                "cache",
                "actionable_failure",
                "context_transport",
                "context_compaction",
                "workflow_state_machine",
            }:
                continue
            if key in {"updated_at", "created_at", "created_at_epoch", "expires_at_epoch"}:
                continue
            out[key] = _event_backbone_normalize_snapshot(raw_value)
        return out
    if isinstance(payload, list):
        return [_event_backbone_normalize_snapshot(item) for item in payload]
    return payload


def _event_backbone_payload(
    *,
    request: Mapping[str, Any],
    legacy_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    request_obj = dict(request) if isinstance(request, Mapping) else {}
    event_cfg = request_obj.get("event_backbone") if isinstance(request_obj.get("event_backbone"), Mapping) else {}
    transport_route = request_obj.get("transport_route") if isinstance(request_obj.get("transport_route"), Mapping) else {}
    route_obj = legacy_payload.get("route") if isinstance(legacy_payload.get("route"), Mapping) else {}
    request_view = legacy_payload.get("request") if isinstance(legacy_payload.get("request"), Mapping) else {}
    prompt_guardrail = legacy_payload.get("prompt_guardrail") if isinstance(legacy_payload.get("prompt_guardrail"), Mapping) else {}
    prompt_lint = prompt_guardrail.get("prompt_lint") if isinstance(prompt_guardrail.get("prompt_lint"), Mapping) else {}
    prompt_trimmed = prompt_lint.get("trimmed") if isinstance(prompt_lint.get("trimmed"), Mapping) else {}
    context_transport = legacy_payload.get("context_transport") if isinstance(legacy_payload.get("context_transport"), Mapping) else {}
    context_integrity = context_transport.get("integrity") if isinstance(context_transport.get("integrity"), Mapping) else {}
    knowledge_retrieval = request_obj.get("knowledge_retrieval") if isinstance(request_obj.get("knowledge_retrieval"), Mapping) else {}

    correlation_id = (
        str(event_cfg.get("correlation_id") or "").strip()
        or str(transport_route.get("session_key") or "").strip()
        or str(transport_route.get("transport_key") or "").strip()
        or str((request_view.get("transport_binding") or {}).get("session_key") if isinstance(request_view.get("transport_binding"), Mapping) else "").strip()
        or str((request_view.get("transport_binding") or {}).get("transport_key") if isinstance(request_view.get("transport_binding"), Mapping) else "").strip()
        or str((route_obj.get("selected_rule_id") or "")).strip()
        or "session_topology_router"
    )

    prompt_hash = str(prompt_trimmed.get("sha256") or "").strip() or _prompt_hash(str(request_obj.get("invocation_prompt") or ""))
    idempotency_source = {
        "router": "session_topology_router",
        "session_kind": request_view.get("session_kind"),
        "task_class": request_view.get("task_class"),
        "risk_tier": request_view.get("risk_tier"),
        "selected_rule_id": route_obj.get("selected_rule_id"),
        "route_class": route_obj.get("route_class"),
        "selected_model": route_obj.get("selected_model"),
        "required_rollout_stage": route_obj.get("required_rollout_stage"),
        "correlation_id": correlation_id,
        "prompt_sha256": prompt_hash,
        "context_snapshot_hash": context_integrity.get("reconstructed_snapshot_hash"),
        "knowledge_query": knowledge_retrieval.get("query"),
    }
    explicit_idempotency_key = str(event_cfg.get("idempotency_key") or "").strip()
    idempotency_key = explicit_idempotency_key or ("auto:" + hashlib.sha256(stable_json_dumps(idempotency_source).encode("utf-8")).hexdigest())

    legacy_snapshot = _event_backbone_normalize_snapshot(_json_clone(dict(legacy_payload)))
    legacy_parity_fingerprint = hashlib.sha256(stable_json_dumps(legacy_snapshot).encode("utf-8")).hexdigest()
    event_id = "evt_" + hashlib.sha256(f"{idempotency_key}|{legacy_parity_fingerprint}".encode("utf-8")).hexdigest()[:24]
    sequence = _event_backbone_sequence(event_cfg.get("sequence"))
    parent_event_id = str(event_cfg.get("parent_event_id") or "").strip() or None
    occurred_at = str(legacy_payload.get("evaluated_at") or now_iso())

    envelope = {
        "schema": EVENT_BACKBONE_EVENT_SCHEMA,
        "specversion": "1.0",
        "type": "session.route.decision",
        "source": "openclaw.session_topology_router",
        "id": event_id,
        "time": occurred_at,
        "subject": ":".join(
            [
                str(request_view.get("session_kind") or "unknown"),
                str(request_view.get("task_class") or "unknown"),
                str(request_view.get("risk_tier") or "unknown"),
            ]
        ),
        "correlation_id": correlation_id,
        "idempotency_key": idempotency_key,
        "sequence": sequence,
        "parent_event_id": parent_event_id,
        "legacy_parity_fingerprint": legacy_parity_fingerprint,
        "legacy_schema": legacy_payload.get("schema"),
        "mode": "dual_write",
        "data": {
            "decision": legacy_payload.get("decision"),
            "final_state": legacy_payload.get("final_state"),
            "block_gate": legacy_payload.get("block_gate"),
            "block_reason": legacy_payload.get("block_reason"),
            "request": {
                "session_kind": request_view.get("session_kind"),
                "task_class": request_view.get("task_class"),
                "risk_tier": request_view.get("risk_tier"),
            },
            "route": _json_clone(route_obj),
            "context_transport": _json_clone(context_transport),
            "context_compaction": _json_clone(legacy_payload.get("context_compaction") or {}),
            "hybrid_retrieval": _json_clone(legacy_payload.get("hybrid_retrieval") or {}),
        },
    }

    return {
        "event_envelope": envelope,
        "legacy_snapshot": legacy_snapshot,
        "legacy_parity_fingerprint": legacy_parity_fingerprint,
        "idempotency_key": idempotency_key,
        "event_id": event_id,
    }


def _event_backbone_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=1.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS event_backbone_outbox (
          idempotency_key TEXT PRIMARY KEY,
          event_id TEXT NOT NULL,
          legacy_parity_fingerprint TEXT NOT NULL,
          created_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          published_at TEXT,
          dlq_at TEXT,
          correlation_id TEXT,
          sequence INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL DEFAULT 'pending',
          legacy_written INTEGER NOT NULL DEFAULT 0,
          typed_written INTEGER NOT NULL DEFAULT 0,
          attempt_count INTEGER NOT NULL DEFAULT 0,
          duplicate_count INTEGER NOT NULL DEFAULT 0,
          retry_backoff_json TEXT NOT NULL DEFAULT '[]',
          last_error TEXT,
          legacy_payload_json TEXT NOT NULL,
          typed_payload_json TEXT NOT NULL,
          legacy_log_path TEXT NOT NULL,
          typed_log_path TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con


def _event_backbone_get_row(con: sqlite3.Connection, idempotency_key: str) -> Optional[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM event_backbone_outbox WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()


def _event_backbone_write_metrics(
    *,
    con: sqlite3.Connection,
    metrics_path: Path,
    db_path: Path,
    current_status: str,
    current_event_id: str,
    latency_ms: int,
) -> Dict[str, Any]:
    row = con.execute(
        """
        SELECT
          SUM(CASE WHEN status = 'published' THEN 1 ELSE 0 END) AS published_count,
          SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
          SUM(CASE WHEN status = 'dead_letter' THEN 1 ELSE 0 END) AS dlq_count,
          SUM(duplicate_count) AS duplicate_suppressed_count,
          SUM(CASE WHEN attempt_count > 1 THEN attempt_count - 1 ELSE 0 END) AS retry_attempt_count,
          MAX(published_at) AS last_published_at,
          MAX(dlq_at) AS last_dlq_at
        FROM event_backbone_outbox
        """
    ).fetchone()
    published_count = int((row[0] or 0) if row is not None else 0)
    pending_count = int((row[1] or 0) if row is not None else 0)
    dlq_count = int((row[2] or 0) if row is not None else 0)
    duplicate_suppressed_count = int((row[3] or 0) if row is not None else 0)
    retry_attempt_count = int((row[4] or 0) if row is not None else 0)
    last_published_at = row[5] if row is not None else None
    last_dlq_at = row[6] if row is not None else None

    if dlq_count > 0:
        backpressure_state = "critical"
    elif pending_count > 0 or retry_attempt_count > 0:
        backpressure_state = "elevated"
    else:
        backpressure_state = "normal"

    metrics = {
        "schema": "clawd.session_topology.event_backbone_metrics.v1",
        "generated_at": now_iso(),
        "stream": "session_topology_router",
        "db_path": str(db_path),
        "published_count": published_count,
        "pending_count": pending_count,
        "dlq_count": dlq_count,
        "duplicate_suppressed_count": duplicate_suppressed_count,
        "retry_attempt_count": retry_attempt_count,
        "current_publish_status": current_status,
        "current_event_id": current_event_id,
        "last_publish_latency_ms": int(max(0, latency_ms)),
        "last_published_at": last_published_at,
        "last_dlq_at": last_dlq_at,
        "backpressure_state": backpressure_state,
    }
    _atomic_write_json(metrics_path, metrics)
    return metrics


def publish_event_backbone(
    *,
    repo_root: Path,
    request: Mapping[str, Any],
    legacy_payload: Mapping[str, Any],
    legacy_log_path: Optional[Path],
    typed_log_path: Path,
    db_path: Path,
    dlq_path: Path,
    metrics_path: Path,
    enabled: bool = True,
    max_attempts: int = DEFAULT_EVENT_BACKBONE_MAX_ATTEMPTS,
    base_backoff_ms: int = DEFAULT_EVENT_BACKBONE_BASE_BACKOFF_MS,
) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "disabled", "legacy_record": {"enabled": False, "appended": False, "reason": "disabled"}}

    if legacy_log_path is None:
        return {"enabled": False, "status": "disabled", "legacy_record": {"enabled": False, "appended": False, "reason": "disabled"}}

    started = time.monotonic()
    legacy_path, legacy_err = _resolve_repo_jsonl_path(repo_root, legacy_log_path)
    typed_path, typed_err = _resolve_repo_jsonl_path(repo_root, typed_log_path)
    db_resolved, db_err = _resolve_repo_jsonl_path(repo_root, db_path)
    dlq_resolved, dlq_err = _resolve_repo_jsonl_path(repo_root, dlq_path)
    metrics_resolved, metrics_err = _resolve_repo_jsonl_path(repo_root, metrics_path)
    path_error = legacy_err or typed_err or db_err or dlq_err or metrics_err
    if any(x is None for x in [legacy_path, typed_path, db_resolved, dlq_resolved, metrics_resolved]):
        return {
            "enabled": True,
            "status": "config_error",
            "reason": path_error or "unsafe_path",
            "legacy_record": {"enabled": True, "appended": False, "reason": path_error or "unsafe_path"},
        }

    payload_meta = _event_backbone_payload(request=request, legacy_payload=legacy_payload)
    event_envelope = payload_meta["event_envelope"]
    idempotency_key = str(payload_meta["idempotency_key"])
    event_id = str(payload_meta["event_id"])
    legacy_fingerprint = str(payload_meta["legacy_parity_fingerprint"])
    now = now_iso()

    try:
        con = _event_backbone_connect(db_resolved)
    except Exception as exc:
        return {
            "enabled": True,
            "status": "config_error",
            "reason": "db_open_failed",
            "error": str(exc),
            "legacy_record": {"enabled": True, "appended": False, "reason": "db_open_failed", "error": str(exc)},
        }

    try:
        row = _event_backbone_get_row(con, idempotency_key)
        if row is None:
            con.execute(
                """
                INSERT INTO event_backbone_outbox (
                  idempotency_key, event_id, legacy_parity_fingerprint, created_at, last_seen_at, correlation_id, sequence,
                  status, legacy_written, typed_written, attempt_count, duplicate_count, retry_backoff_json, last_error,
                  legacy_payload_json, typed_payload_json, legacy_log_path, typed_log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, 0, 0, 0, '[]', NULL, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    event_id,
                    legacy_fingerprint,
                    now,
                    now,
                    str(event_envelope.get("correlation_id") or ""),
                    int(event_envelope.get("sequence") or 1),
                    json.dumps(payload_meta["legacy_snapshot"], ensure_ascii=False, sort_keys=True),
                    json.dumps(event_envelope, ensure_ascii=False, sort_keys=True),
                    str(legacy_path),
                    str(typed_path),
                ),
            )
            con.commit()
            row = _event_backbone_get_row(con, idempotency_key)

        if row is None:
            raise RuntimeError("event_backbone_row_missing")

        if str(row["legacy_parity_fingerprint"] or "") != legacy_fingerprint:
            return {
                "enabled": True,
                "status": "idempotency_conflict",
                "event_id": str(row["event_id"] or event_id),
                "idempotency_key": idempotency_key,
                "legacy_parity_fingerprint": legacy_fingerprint,
                "stored_legacy_parity_fingerprint": str(row["legacy_parity_fingerprint"] or ""),
                "legacy_record": {"enabled": True, "appended": False, "reason": "idempotency_conflict", "path": str(legacy_path)},
            }

        if str(row["status"] or "") == "published" and int(row["legacy_written"] or 0) == 1 and int(row["typed_written"] or 0) == 1:
            con.execute(
                "UPDATE event_backbone_outbox SET duplicate_count = duplicate_count + 1, last_seen_at = ? WHERE idempotency_key = ?",
                (now_iso(), idempotency_key),
            )
            con.commit()
            metrics = _event_backbone_write_metrics(
                con=con,
                metrics_path=metrics_resolved,
                db_path=db_resolved,
                current_status="duplicate_suppressed",
                current_event_id=str(row["event_id"] or event_id),
                latency_ms=int(max(0, round((time.monotonic() - started) * 1000))),
            )
            return {
                "enabled": True,
                "status": "duplicate_suppressed",
                "event_id": str(row["event_id"] or event_id),
                "idempotency_key": idempotency_key,
                "attempts": int(row["attempt_count"] or 0),
                "retry_backoff_ms": json.loads(str(row["retry_backoff_json"] or "[]")),
                "legacy_parity_fingerprint": legacy_fingerprint,
                "legacy_record": {"enabled": True, "appended": False, "reason": "duplicate_suppressed", "path": str(legacy_path)},
                "typed_record": {"enabled": True, "appended": False, "reason": "duplicate_suppressed", "path": str(typed_path)},
                "db_path": str(db_resolved),
                "typed_log_path": str(typed_path),
                "dlq_path": str(dlq_resolved),
                "metrics_path": str(metrics_resolved),
                "backpressure_state": metrics.get("backpressure_state"),
            }

        legacy_written = int(row["legacy_written"] or 0) == 1
        typed_written = int(row["typed_written"] or 0) == 1
        attempt_count = int(row["attempt_count"] or 0)
        retry_backoff = json.loads(str(row["retry_backoff_json"] or "[]"))
        legacy_record: Dict[str, Any] = {"enabled": True, "appended": False, "reason": "pending", "path": str(legacy_path)}
        typed_record: Dict[str, Any] = {"enabled": True, "appended": False, "reason": "pending", "path": str(typed_path)}
        last_error = str(row["last_error"] or "")

        max_attempts = max(1, int(max_attempts))
        base_backoff_ms = max(1, int(base_backoff_ms))

        while attempt_count < max_attempts and not (legacy_written and typed_written):
            attempt_count += 1
            try:
                if not legacy_written:
                    _append_jsonl_payload(legacy_path, payload_meta["legacy_snapshot"])
                    legacy_written = True
                    legacy_record = {"enabled": True, "appended": True, "path": str(legacy_path)}
                if not typed_written:
                    _append_jsonl_payload(typed_path, event_envelope)
                    typed_written = True
                    typed_record = {"enabled": True, "appended": True, "path": str(typed_path)}
                con.execute(
                    """
                    UPDATE event_backbone_outbox
                    SET status = 'published', legacy_written = ?, typed_written = ?, attempt_count = ?, retry_backoff_json = ?,
                        published_at = ?, last_seen_at = ?, last_error = NULL
                    WHERE idempotency_key = ?
                    """,
                    (
                        1 if legacy_written else 0,
                        1 if typed_written else 0,
                        attempt_count,
                        json.dumps(retry_backoff, ensure_ascii=False, sort_keys=True),
                        now_iso(),
                        now_iso(),
                        idempotency_key,
                    ),
                )
                con.commit()
                metrics = _event_backbone_write_metrics(
                    con=con,
                    metrics_path=metrics_resolved,
                    db_path=db_resolved,
                    current_status="published",
                    current_event_id=event_id,
                    latency_ms=int(max(0, round((time.monotonic() - started) * 1000))),
                )
                return {
                    "enabled": True,
                    "status": "published",
                    "event_id": event_id,
                    "idempotency_key": idempotency_key,
                    "attempts": attempt_count,
                    "retry_backoff_ms": retry_backoff,
                    "legacy_parity_fingerprint": legacy_fingerprint,
                    "legacy_record": legacy_record,
                    "typed_record": typed_record,
                    "db_path": str(db_resolved),
                    "typed_log_path": str(typed_path),
                    "dlq_path": str(dlq_resolved),
                    "metrics_path": str(metrics_resolved),
                    "backpressure_state": metrics.get("backpressure_state"),
                }
            except Exception as exc:
                last_error = str(exc)
                legacy_reason = str(exc) if (not legacy_written and str(exc) == "path_not_file") else "append_failed"
                typed_reason = str(exc) if (legacy_written and not typed_written and str(exc) == "path_not_file") else "append_failed"
                if not legacy_written:
                    legacy_record = {"enabled": True, "appended": False, "reason": legacy_reason, "path": str(legacy_path), "error": str(exc)}
                if not typed_written:
                    typed_record = {"enabled": True, "appended": False, "reason": typed_reason, "path": str(typed_path), "error": str(exc)}
                retry_backoff.append(_event_backbone_backoff_ms(idempotency_key=idempotency_key, attempt=attempt_count, base_backoff_ms=base_backoff_ms))
                con.execute(
                    """
                    UPDATE event_backbone_outbox
                    SET status = 'pending', legacy_written = ?, typed_written = ?, attempt_count = ?, retry_backoff_json = ?,
                        last_error = ?, last_seen_at = ?
                    WHERE idempotency_key = ?
                    """,
                    (
                        1 if legacy_written else 0,
                        1 if typed_written else 0,
                        attempt_count,
                        json.dumps(retry_backoff, ensure_ascii=False, sort_keys=True),
                        last_error,
                        now_iso(),
                        idempotency_key,
                    ),
                )
                con.commit()

        dlq_entry = {
            "schema": EVENT_BACKBONE_DLQ_SCHEMA,
            "failed_at": now_iso(),
            "event": event_envelope,
            "legacy_log_path": str(legacy_path),
            "typed_log_path": str(typed_path),
            "db_path": str(db_resolved),
            "attempt_count": attempt_count,
            "retry_backoff_ms": retry_backoff,
            "legacy_written": legacy_written,
            "typed_written": typed_written,
            "last_error": last_error or None,
        }
        _append_jsonl_payload(dlq_resolved, dlq_entry)
        con.execute(
            """
            UPDATE event_backbone_outbox
            SET status = 'dead_letter', legacy_written = ?, typed_written = ?, attempt_count = ?, retry_backoff_json = ?,
                last_error = ?, dlq_at = ?, last_seen_at = ?
            WHERE idempotency_key = ?
            """,
            (
                1 if legacy_written else 0,
                1 if typed_written else 0,
                attempt_count,
                json.dumps(retry_backoff, ensure_ascii=False, sort_keys=True),
                last_error or None,
                now_iso(),
                now_iso(),
                idempotency_key,
            ),
        )
        con.commit()
        metrics = _event_backbone_write_metrics(
            con=con,
            metrics_path=metrics_resolved,
            db_path=db_resolved,
            current_status="dlq",
            current_event_id=event_id,
            latency_ms=int(max(0, round((time.monotonic() - started) * 1000))),
        )
        return {
            "enabled": True,
            "status": "dlq",
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "attempts": attempt_count,
            "retry_backoff_ms": retry_backoff,
            "legacy_parity_fingerprint": legacy_fingerprint,
            "legacy_record": legacy_record,
            "typed_record": typed_record,
            "db_path": str(db_resolved),
            "typed_log_path": str(typed_path),
            "dlq_path": str(dlq_resolved),
            "metrics_path": str(metrics_resolved),
            "backpressure_state": metrics.get("backpressure_state"),
            "last_error": last_error or None,
        }
    finally:
        con.close()


def _workflow_state_machine_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("workflow_state_machine") if isinstance(request.get("workflow_state_machine"), Mapping) else {}
    expected_current_state = str(raw.get("expected_current_state") or "").strip().upper() or None
    return {
        "workflow_id": str(raw.get("workflow_id") or "").strip() or None,
        "expected_current_state": expected_current_state,
    }


def _workflow_state_machine_id(
    *,
    request: Mapping[str, Any],
    route_result: Mapping[str, Any],
    event_backbone_record: Mapping[str, Any],
) -> str:
    workflow_request = _workflow_state_machine_request(request)
    if workflow_request.get("workflow_id"):
        return str(workflow_request["workflow_id"])

    transport_binding = (
        (route_result.get("transport_conformance") or {}).get("resolved_binding")
        if isinstance(route_result.get("transport_conformance"), Mapping)
        else {}
    )
    request_transport = (
        (route_result.get("request") or {}).get("transport_binding")
        if isinstance(route_result.get("request"), Mapping)
        else {}
    )
    fallback_tokens = [
        str(event_backbone_record.get("idempotency_key") or "").strip(),
        str(((request.get("event_backbone") or {}).get("correlation_id") if isinstance(request.get("event_backbone"), Mapping) else "") or "").strip(),
        str(transport_binding.get("session_key") or "").strip(),
        str(transport_binding.get("transport_key") or "").strip(),
        str(request_transport.get("session_key") or "").strip(),
        str(request_transport.get("transport_key") or "").strip(),
    ]
    for token in fallback_tokens:
        if token:
            return token

    request_fingerprint = {
        "session_kind": request.get("session_kind"),
        "task_class": request.get("task_class"),
        "risk_tier": request.get("risk_tier"),
        "route_class": ((route_result.get("route") or {}).get("route_class") if isinstance(route_result.get("route"), Mapping) else None),
    }
    return "workflow:auto:" + hashlib.sha256(stable_json_dumps(request_fingerprint).encode("utf-8")).hexdigest()[:24]


def _workflow_state_machine_latest_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema": WORKFLOW_STATE_LATEST_SCHEMA, "generated_at": None, "workflows": {}}
    if not path.is_file():
        raise ValueError("path_not_file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("workflow_state_latest_invalid")
    workflows = raw.get("workflows") if isinstance(raw.get("workflows"), Mapping) else {}
    return {
        "schema": str(raw.get("schema") or WORKFLOW_STATE_LATEST_SCHEMA),
        "generated_at": raw.get("generated_at"),
        "workflows": dict(workflows),
    }


def _workflow_state_machine_load_prior(*, journal_path: Path, latest_path: Path, workflow_id: str) -> Dict[str, Any]:
    latest_payload = _workflow_state_machine_latest_snapshot(latest_path)
    workflows = latest_payload.get("workflows") if isinstance(latest_payload.get("workflows"), Mapping) else {}
    current = workflows.get(workflow_id) if isinstance(workflows, Mapping) else None
    if isinstance(current, Mapping):
        prior_state = str(current.get("next_state") or current.get("state") or "INIT").strip().upper() or "INIT"
        replay_depth = int(current.get("replay_depth") or current.get("transition_count") or 0)
        return {
            "prior_state": prior_state if prior_state in WORKFLOW_ALLOWED_STATES else "INIT",
            "replay_depth": max(0, replay_depth),
            "snapshot": dict(current),
        }

    prior_state = "INIT"
    replay_depth = 0
    if journal_path.exists():
        if not journal_path.is_file():
            raise ValueError("path_not_file")
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token:
                continue
            row = json.loads(token)
            if str(row.get("workflow_id") or "") != workflow_id:
                continue
            replay_depth += 1
            candidate_state = str(row.get("next_state") or row.get("state") or prior_state).strip().upper()
            if candidate_state in WORKFLOW_ALLOWED_STATES:
                prior_state = candidate_state
    return {"prior_state": prior_state, "replay_depth": replay_depth, "snapshot": {}}


def _workflow_state_machine_transition(
    *,
    prior_state: str,
    route_result: Mapping[str, Any],
    event_backbone_record: Mapping[str, Any],
) -> Tuple[str, str, Optional[str]]:
    route_decision = str(route_result.get("decision") or "BLOCK")
    if route_decision == "BLOCK":
        return "ROUTE_BLOCKED", "route_blocked", None

    event_status = str(event_backbone_record.get("status") or "disabled")
    if event_status == "published":
        if prior_state == "RECOVERY_REQUIRED":
            return "ACTIVE", "recovered_after_replay", None
        return "ACTIVE", "route_committed", None
    if event_status == "duplicate_suppressed":
        return "ACTIVE", "idempotent_replay_confirmed", None
    if event_status == "dlq":
        return "RECOVERY_REQUIRED", "publish_failed_requires_recovery", "workflow_event_dlq"
    if event_status == "idempotency_conflict":
        return "RECOVERY_REQUIRED", "idempotency_conflict_requires_recovery", "workflow_idempotency_conflict"
    if event_status in {"disabled", "disabled_by_flag"}:
        return "RECOVERY_REQUIRED", "cutover_disabled_requires_recovery", "workflow_state_tracking_disabled"
    if event_status == "config_error":
        return "RECOVERY_REQUIRED", "event_backbone_config_requires_recovery", "workflow_event_backbone_config_error"
    return "RECOVERY_REQUIRED", "unknown_publish_status_requires_recovery", "workflow_unknown_publish_status"


def evaluate_workflow_state_machine(
    *,
    repo_root: Path,
    request: Mapping[str, Any],
    route_result: Mapping[str, Any],
    event_backbone_record: Mapping[str, Any],
    journal_path: Path,
    latest_path: Path,
) -> Dict[str, Any]:
    request_view = _workflow_state_machine_request(request)
    workflow_id = _workflow_state_machine_id(request=request, route_result=route_result, event_backbone_record=event_backbone_record)
    expected_current_state = request_view.get("expected_current_state")

    journal_resolved, journal_error = _resolve_repo_jsonl_path(repo_root, journal_path)
    latest_resolved, latest_error = _resolve_repo_jsonl_path(repo_root, latest_path)
    if journal_resolved is None or latest_resolved is None:
        return {
            "schema": WORKFLOW_STATE_TRANSITION_SCHEMA,
            "enabled": True,
            "status": "fail",
            "reason": "workflow_state_persistence_failed",
            "workflow_id": workflow_id,
            "prior_state": "INIT",
            "next_state": "RECOVERY_REQUIRED",
            "transition": "persistence_blocked",
            "authoritative_green": False,
            "control_path": "code_driven",
            "journal_record": {"enabled": True, "appended": False, "reason": journal_error or latest_error or "unsafe_path"},
            "latest_record": {"enabled": True, "updated": False, "reason": latest_error or journal_error or "unsafe_path"},
        }

    try:
        prior = _workflow_state_machine_load_prior(journal_path=journal_resolved, latest_path=latest_resolved, workflow_id=workflow_id)
    except Exception as exc:
        return {
            "schema": WORKFLOW_STATE_TRANSITION_SCHEMA,
            "enabled": True,
            "status": "fail",
            "reason": "workflow_state_replay_unreadable",
            "workflow_id": workflow_id,
            "prior_state": "INIT",
            "next_state": "RECOVERY_REQUIRED",
            "transition": "replay_load_failed",
            "authoritative_green": False,
            "control_path": "code_driven",
            "error": str(exc),
            "journal_record": {"enabled": True, "appended": False, "reason": "replay_load_failed", "path": str(journal_resolved)},
            "latest_record": {"enabled": True, "updated": False, "reason": "replay_load_failed", "path": str(latest_resolved)},
        }

    prior_state = str(prior.get("prior_state") or "INIT")
    replay_depth = int(prior.get("replay_depth") or 0)

    if expected_current_state and expected_current_state != prior_state:
        return {
            "schema": WORKFLOW_STATE_TRANSITION_SCHEMA,
            "enabled": True,
            "status": "fail",
            "reason": "workflow_state_stale",
            "workflow_id": workflow_id,
            "expected_current_state": expected_current_state,
            "prior_state": prior_state,
            "next_state": prior_state,
            "transition": "expected_state_mismatch",
            "authoritative_green": prior_state in WORKFLOW_HEALTHY_STATES,
            "control_path": "code_driven",
            "replay_depth": replay_depth,
            "journal_record": {"enabled": True, "appended": False, "reason": "expected_state_mismatch", "path": str(journal_resolved)},
            "latest_record": {"enabled": True, "updated": False, "reason": "expected_state_mismatch", "path": str(latest_resolved)},
        }

    next_state, transition, failure_reason = _workflow_state_machine_transition(
        prior_state=prior_state,
        route_result=route_result,
        event_backbone_record=event_backbone_record,
    )
    allowed_next = WORKFLOW_ALLOWED_TRANSITIONS.get(prior_state, set())
    if next_state not in WORKFLOW_ALLOWED_STATES or next_state not in allowed_next:
        return {
            "schema": WORKFLOW_STATE_TRANSITION_SCHEMA,
            "enabled": True,
            "status": "fail",
            "reason": "workflow_transition_invalid",
            "workflow_id": workflow_id,
            "prior_state": prior_state,
            "next_state": next_state,
            "transition": transition,
            "authoritative_green": False,
            "control_path": "code_driven",
            "replay_depth": replay_depth,
            "journal_record": {"enabled": True, "appended": False, "reason": "workflow_transition_invalid", "path": str(journal_resolved)},
            "latest_record": {"enabled": True, "updated": False, "reason": "workflow_transition_invalid", "path": str(latest_resolved)},
        }

    transition_row = {
        "schema": WORKFLOW_STATE_TRANSITION_SCHEMA,
        "recorded_at": now_iso(),
        "workflow_id": workflow_id,
        "prior_state": prior_state,
        "next_state": next_state,
        "transition": transition,
        "authoritative_green": next_state in WORKFLOW_HEALTHY_STATES,
        "route_decision": route_result.get("decision"),
        "route_final_state": route_result.get("final_state"),
        "route_block_gate": route_result.get("block_gate"),
        "route_block_reason": route_result.get("block_reason"),
        "event_backbone_status": event_backbone_record.get("status"),
        "event_id": event_backbone_record.get("event_id"),
        "idempotency_key": event_backbone_record.get("idempotency_key"),
        "replay_depth": replay_depth + 1,
        "control_path": "code_driven",
        "expected_current_state": expected_current_state,
    }
    if failure_reason:
        transition_row["failure_reason"] = failure_reason

    journal_record = append_decision_record(decision_log_path=journal_resolved, repo_root=repo_root, decision_row=transition_row)
    latest_record: Dict[str, Any]
    if journal_record.get("appended") is True:
        try:
            latest_payload = _workflow_state_machine_latest_snapshot(latest_resolved)
            workflows = latest_payload.get("workflows") if isinstance(latest_payload.get("workflows"), Mapping) else {}
            updated_workflows = dict(workflows)
            updated_workflows[workflow_id] = {
                **transition_row,
                "transition_count": replay_depth + 1,
            }
            _atomic_write_json(
                latest_resolved,
                {
                    "schema": WORKFLOW_STATE_LATEST_SCHEMA,
                    "generated_at": now_iso(),
                    "workflows": updated_workflows,
                },
            )
            latest_record = {"enabled": True, "updated": True, "path": str(latest_resolved)}
        except Exception as exc:
            latest_record = {
                "enabled": True,
                "updated": False,
                "reason": "write_failed",
                "path": str(latest_resolved),
                "error": str(exc),
            }
    else:
        latest_record = {
            "enabled": True,
            "updated": False,
            "reason": str(journal_record.get("reason") or "append_failed"),
            "path": str(latest_resolved),
        }

    if journal_record.get("appended") is not True or latest_record.get("updated") is not True:
        return {
            **transition_row,
            "enabled": True,
            "status": "fail",
            "reason": "workflow_state_persistence_failed",
            "next_state": "RECOVERY_REQUIRED",
            "authoritative_green": False,
            "journal_record": journal_record,
            "latest_record": latest_record,
        }

    route_decision = str(route_result.get("decision") or "BLOCK")
    status = "pass"
    reason = None
    if route_decision == "PASS" and next_state not in WORKFLOW_HEALTHY_STATES:
        status = "fail"
        reason = failure_reason or "workflow_recovery_required"

    return {
        **transition_row,
        "enabled": True,
        "status": status,
        "reason": reason,
        "journal_record": journal_record,
        "latest_record": latest_record,
    }


def build_operator_diagnostics(*, decision: str, block_gate: Optional[str], block_reason: Optional[str]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "router": "route_policy_topology",
        "router_cli": "scripts/session_topology_router.py",
        "docs_ref": "docs/ops/SESSION_TOPOLOGY.md",
        "disambiguation": {
            "route_policy_router": "scripts/session_topology_router.py",
            "transport_router": "scripts/session_topology_transport_router.py",
        },
        "suggested_commands": [
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-transport-route --topology docs/ops/templates/session_topology_transport_contract.template.json --request docs/ops/templates/session_topology_transport_route_request.template.json --json > /tmp/transport_decision.json",
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
        ],
    }

    if decision != "BLOCK":
        return diagnostics

    next_steps: List[str] = []
    reason = str(block_reason or "")
    gate = str(block_gate or "")

    if reason == "routing_request_invalid":
        next_steps.append(
            "Fix request.session_kind/task_class/risk_tier using docs/ops/templates/session_route_request.template.json (risk_tier must be low|medium|high|critical)."
        )
    elif reason == "proposal_packet_required_for_coding":
        next_steps.append(
            "Coding request is in proposal-first strict mode: provide proposal_packet.v1 (task_id/task_class/risk_assessment) before routing."
        )
    elif reason == "proposal_packet_invalid":
        next_steps.append(
            "proposal_packet failed contract checks: fix schema_version, required fields, and ensure task_class/risk_tier align with request envelope."
        )
    elif reason == "proposal_approval_missing":
        next_steps.append(
            "Proposal approval hook not satisfied: include proposal_approval.approved=true and enough approver_ids for the risk tier before dispatch."
        )
    elif reason == "proposal_phase_invalid":
        next_steps.append(
            "proposal_phase must be one of proposal|apply|archive when provided. Fix request.proposal_phase or remove it to use inferred phase mode."
        )
    elif reason == "proposal_apply_missing_delta_spec":
        next_steps.append(
            "Apply/archive phase requires explicit delta_spec instructions. Provide delta_spec.instructions (or prompt) before routing."
        )
    elif reason == "proposal_archive_missing":
        next_steps.append(
            "Archive phase requires proposal_archive packet with schema_version, task_id, and artifacts list before routing."
        )
    elif reason == "proposal_archive_invalid":
        next_steps.append(
            "proposal_archive contract failed: use schema_version=proposal_archive_packet.v1, keep task_id aligned with proposal_packet.task_id, and include all required artifacts."
        )
    elif reason == "delta_spec_invalid":
        next_steps.append(
            "delta_spec must contain a non-empty instructions or prompt field so worker apply instructions are explicit."
        )
    elif reason == "regression_risk_packet_required_for_coding":
        next_steps.append(
            "Coding request is in regression-risk strict mode: provide regression_risk_packet with evidence, risk_assessment, and validation fields before routing."
        )
    elif reason == "regression_risk_packet_invalid":
        next_steps.append(
            "regression_risk_packet failed contract checks: fix packet_id/version/risk scores/evidence/validation.required_approvals alignment; for v2 also fix replay_evidence and blocking_classification parity."
        )
    elif reason == "regression_risk_not_approved":
        next_steps.append(
            "Strict regression-risk mode requires validation.status=approved before coding dispatch. Route review/approval first, then rerun routing."
        )
    elif reason == "regression_risk_approval_missing":
        next_steps.append(
            "Regression-risk approval threshold not met: add approval records so approval_count satisfies required approvals for the overall_tier."
        )
    elif reason == "refactor_risk_packet_required_for_coding":
        next_steps.append(
            "Coding request is in refactor-risk strict mode: provide refactor_risk_packet with risk_assessment, decomposition_plan, and validation fields before routing."
        )
    elif reason == "refactor_risk_packet_invalid":
        next_steps.append(
            "refactor_risk_packet failed contract checks: fix packet_id/version/risk dimensions/decomposition plan/validation.required_approvals alignment."
        )
    elif reason == "refactor_risk_not_approved":
        next_steps.append(
            "Strict refactor-risk mode requires validation.status=approved before coding dispatch. Route review/approval first, then rerun routing."
        )
    elif reason == "refactor_risk_approval_missing":
        next_steps.append(
            "Refactor-risk approval threshold not met: add approvers/approval records so approver_count satisfies required approvals for the overall_tier."
        )
    elif reason == "no_qualified_model_for_route":
        next_steps.append("Regenerate PASS qualification decisions with continuity.sh model-rollout-gate and pass them via --qualification-decision.")
    elif reason == "requested_route_mismatch":
        next_steps.append("Update request.route_lock (route_class/model_key/rule_id/stage) to match expected route, or remove stale lock constraints.")
    elif reason == "pool_policy_violation":
        next_steps.append("Align topology model_pools with docs/ops/model_pool_policy_v1.json allowlist for the resolved route_class.")
    elif reason == "escalation_evidence_missing":
        next_steps.append(
            "Default-down taxonomy blocked escalation: provide request.escalation_evidence with at least one signal and artifact_refs when escalating beyond baseline tier."
        )
    elif reason in {"schema_invalid", "route_policy_invalid"}:
        next_steps.append("Validate topology against docs/ops/schemas/session_topology_contract.schema.json and fix invalid rule/default fields.")
    elif reason in {"transport_route_conformance_failed", "transport_route_mismatch"}:
        next_steps.append("Run session-transport-route first, then pass --transport-decision to continuity.sh session-route (strict transport conformance is default).")
        next_steps.append(
            "Only for bounded legacy tooling windows, use scripts/session_topology_router.py --legacy-allow-missing-transport-decision explicitly."
        )
    elif reason == "prompt_token_budget_exceeded":
        next_steps.append(
            "Prompt guardrail blocked invocation: trim redundancy, lower requested_output_tokens, or split work into bounded slices before rerouting."
        )
    elif reason == "worker_allocation_contract_violation":
        next_steps.append(
            "Worker-allocation contract failed: provide scope_shape, verification_class, worker_topology, and fold_in_target for worker_slice requests and satisfy high-risk/coupled-scope guardrails."
        )
    elif reason == "telegram_direct_heavy_offload_required":
        next_steps.append(
            "Telegram DM is cockpit-only for heavy work: hand off HEAVY or multi-surface/parallel worker_slice requests to codex-worker-plus-* lanes before rerouting."
        )
        next_steps.append(
            "Only for a bounded compatibility window, use scripts/session_topology_router.py --legacy-allow-telegram-direct-heavy-on-dm explicitly."
        )
    elif reason == "telegram_direct_worker_offload_required":
        next_steps.append(
            "Telegram DM is cockpit-only for non-trivial worker slices: route medium/high/critical, validator-required, or coding worker work to codex-worker-plus-* lanes (worker_lane=subagent_default) before rerouting."
        )
        next_steps.append(
            "Use main-session execution only for explicit tiny exceptions (single-surface, low-risk, self-check) with delegation_basis evidence."
        )
    elif reason == "telegram_direct_worker_target_evidence_missing":
        next_steps.append(
            "Telegram DM offload declaration is missing worker-target topology evidence. Provide transport-bound lane_name, agent_id, and session_key for the worker handoff target before rerouting."
        )
    elif reason == "telegram_direct_worker_target_evidence_invalid":
        next_steps.append(
            "Telegram DM offload declaration includes invalid worker-target topology evidence. Use a worker lane/agent target that matches policy patterns (codex-worker-plus-*/codex-worker-pro), keep lane_name and agent_id aligned, and ensure session_key encodes the same agent_id."
        )
    elif reason == "telegram_direct_session_kind_invalid":
        next_steps.append(
            "Telegram direct lane accepts only worker_slice routing requests. Do not run watchdog/internal session kinds on user DM transport; reroute watchdog checks through non-DM/internal transport lanes."
        )
    elif reason == "hybrid_retrieval_abstain":
        next_steps.append(
            "Hybrid retrieval abstained under XE-203 thresholds: tighten the query, provide stronger knowledge candidates, or lower thresholds only with explicit evidence."
        )
    elif reason == "retrieval_search_error":
        next_steps.append("Hybrid retrieval search failed: restore openclaw memory search or inject candidate_results for deterministic replay.")
    elif reason == "gate_unavailable":
        next_steps.append("Install/restore jsonschema validator dependencies and ensure schema paths are readable.")
    elif reason == "workflow_event_dlq":
        next_steps.append("Typed event publish exhausted retries and entered DLQ: inspect event_backbone.dlq_path, fix the sink/path failure, and rerun the same workflow_id to recover the authoritative state.")
    elif reason == "workflow_state_stale":
        next_steps.append("Workflow replay state drifted: reload workflow_state_machine.latest_record, use the reported prior_state as expected_current_state, and replay from that authoritative checkpoint.")
    elif reason == "workflow_state_persistence_failed":
        next_steps.append("Replay journal persistence failed: restore workflow_state_machine journal/latest paths so lifecycle state can be durably recorded before treating the route as successful.")
    elif reason == "workflow_event_backbone_config_error":
        next_steps.append("Event backbone path/config validation failed: fix typed-log/DB/DLQ paths inside the repo root and rerun so the workflow can reach an ACTIVE authoritative state.")
    elif reason == "workflow_idempotency_conflict":
        next_steps.append("Workflow idempotency conflict detected: use a new idempotency_key for the corrected replay or reuse the original payload exactly.")

    if gate == "requested_route_alignment" and not next_steps:
        next_steps.append("Requested route lock mismatch: check request.route_lock values against actual selected route fields.")
    if gate == "transport_route_conformance" and not next_steps:
        next_steps.append("Transport conformance mismatch: verify transport decision binding and remove stale request agent/session overrides.")
    if gate == "tier_escalation_evidence" and not next_steps:
        next_steps.append("Escalation evidence missing: keep default-down tier or provide escalation_evidence with verifiable artifact_refs.")
    if gate == "proposal_first_delta_spec" and not next_steps:
        next_steps.append(
            "Proposal-first contract blocked routing: provide proposal_packet + approval hooks, valid proposal_phase state, delta_spec for apply/archive, and archive artifacts for archive phase."
        )
    if gate == "regression_risk_packet" and not next_steps:
        next_steps.append("Regression-risk contract blocked routing: provide a valid regression_risk_packet and satisfy strict approval posture when enabled.")
    if gate == "refactor_risk_decomposition" and not next_steps:
        next_steps.append(
            "Refactor-risk decomposition contract blocked routing: provide a valid refactor_risk_packet with bounded decomposition chunks and satisfy strict approval posture when enabled."
        )

    if next_steps:
        diagnostics["next_steps"] = next_steps

    return diagnostics


def _actionable_failure(block_gate: Optional[str], block_reason: Optional[str], gates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not block_gate or not block_reason:
        return None

    hint = "Inspect gate details and rerun the same request with corrected contract inputs."
    commands: List[str] = []

    if block_gate == "transport_route_conformance":
        hint = (
            "Recompute transport routing first and pass --transport-decision so agent/session binding cannot drift. "
            "If you must bypass temporarily, use --legacy-allow-missing-transport-decision and treat it as bounded legacy mode."
        )
        commands = [
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-transport-route --topology docs/ops/templates/session_topology_transport_contract.template.json --request docs/ops/templates/session_topology_transport_route_request.template.json --json",
        ]
    elif block_gate == "pool_policy_alignment":
        hint = "Align topology model_pools with docs/ops/model_pool_policy_v1.json before route-policy evaluation."
    elif block_gate == "routing_policy_alignment":
        hint = "Fix docs/ops/session_topology_routing_policy_v1.json and schema conformance before route-policy evaluation."
    elif block_gate == "qualification_model_selection":
        hint = "Run model-rollout-gate and provide PASS qualification decisions for the selected route class and required rollout stage."
        commands = [
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-gate --packet docs/ops/templates/model_qualification_packet.template.json --json",
        ]
    elif block_gate == "requested_route_alignment":
        hint = "Do not override route_lock fields unless they exactly match deterministic policy output."
    elif block_gate == "tier_escalation_evidence":
        hint = (
            "Heavy-tier escalation requires evidence. Provide escalation_evidence with signal booleans "
            "(quality_gate_failed/unresolved_blocker/explicit_criticality/previous_tier_failed) and artifact_refs."
        )
    elif block_gate == "proposal_first_delta_spec":
        hint = (
            "Proposal-first delta-spec contract blocked routing. Provide proposal_packet.v1, satisfy risk-tier approval hooks "
            "(proposal_approval), use valid proposal_phase (proposal|apply|archive), include delta_spec instructions for apply/archive, "
            "and include a valid proposal_archive packet when archiving."
        )
    elif block_gate == "regression_risk_packet":
        hint = (
            "Regression-risk packet contract blocked routing. Provide a valid regression_risk_packet and, in strict mode, ensure validation.status=approved "
            "with required approvals for the packet overall tier."
        )
    elif block_gate == "refactor_risk_decomposition":
        hint = (
            "Refactor-risk decomposition contract blocked routing. Provide a valid refactor_risk_packet (risk_assessment + decomposition_plan + validation), "
            "ensure decomposition chunks are bounded/typed, and in strict mode require validation.status=approved with tier-aligned approvals."
        )
    elif block_gate == "routing_request" and block_reason == "worker_allocation_contract_violation":
        hint = (
            "Worker allocation contract blocked routing. For worker_slice requests, include scope_shape, verification_class, "
            "worker_topology, and fold_in_target; high/critical risk cannot use self_check or parallel_fanout."
        )
    elif block_gate == "telegram_direct_offload":
        if block_reason == "telegram_direct_session_kind_invalid":
            hint = (
                "Telegram direct lane only accepts user-facing worker_slice requests. Route watchdog/internal session kinds via non-DM transport lanes and rerun."
            )
            commands = [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
            ]
        elif block_reason == "telegram_direct_worker_target_evidence_missing":
            hint = (
                "Telegram direct lane received worker_lane=subagent_default but no worker-target topology attestation. "
                "Provide lane_name, agent_id, and session_key in transport binding evidence (or a transport decision) and rerun."
            )
            commands = [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
            ]
        elif block_reason == "telegram_direct_worker_target_evidence_invalid":
            hint = (
                "Telegram direct lane received worker_lane=subagent_default but worker-target attestation failed conformance checks. "
                "Use worker-target lane/agent identities that follow policy patterns, keep lane_name and agent_id aligned, and provide session_key with matching agent prefix."
            )
            commands = [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
            ]
        elif block_reason == "telegram_direct_worker_offload_required":
            hint = (
                "Telegram direct lane blocked non-trivial worker execution. Move medium/high/critical, validator-required, or coding worker slices "
                "to codex-worker-plus-* lanes (worker_lane=subagent_default) and rerun from the worker handoff path."
            )
            commands = [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
            ]
        else:
            hint = (
                "Telegram direct lane blocked heavy engine-room execution. Move HEAVY or multi-surface/parallel worker slices "
                "to codex-worker-plus-* lanes and rerun from the worker handoff path."
            )
            commands = [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --legacy-allow-telegram-direct-heavy-on-dm --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
            ]
    elif block_gate == "prompt_guardrails":
        hint = (
            "Invocation prompt/token guardrail blocked routing. Trim duplicated boilerplate, reduce requested output tokens, "
            "or split into smaller bounded prompts until prompt+output fits budget."
        )
    elif block_gate == "hybrid_retrieval_efficiency":
        hint = (
            "XE-203 hybrid retrieval blocked selective recall. Strengthen the retrieval query, provide deterministic candidate_results, "
            "or only relax abstain thresholds with supporting relevance evidence."
        )
    elif block_gate == "workflow_state_machine":
        hint = (
            "XE-302 workflow state machine did not reach an authoritative ACTIVE state. Inspect workflow_state_machine reason/journal paths, "
            "repair the event backbone or replay checkpoint, then rerun with the same workflow_id."
        )

    matched_gate = next((row for row in gates if row.get("gate") == block_gate), None)
    return {
        "gate": block_gate,
        "reason": block_reason,
        "hint": hint,
        "commands": commands,
        "details": (matched_gate or {}).get("details") if isinstance(matched_gate, Mapping) else None,
    }


def evaluate_routing(
    *,
    topology: Any,
    request: Mapping[str, Any],
    topology_schema_path: Path,
    gate_decisions: List[Mapping[str, Any]],
    pool_policy: Mapping[str, Any],
    pool_policy_meta: Mapping[str, Any],
    routing_policy: Mapping[str, Any],
    routing_policy_meta: Mapping[str, Any],
    transport_decision: Optional[Mapping[str, Any]] = None,
    require_transport_decision: bool = True,
    require_worker_allocation_contract: bool = False,
    require_telegram_direct_heavy_offload: bool = True,
    require_proposal_packet_for_coding: bool = False,
    require_regression_risk_packet_for_coding: bool = False,
    require_refactor_risk_packet_for_coding: bool = False,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> Dict[str, Any]:
    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    selected_rule: Optional[Mapping[str, Any]] = None
    route_class: Optional[str] = None
    required_stage: Optional[str] = None
    selected_model: Optional[str] = None
    selected_qualification_signal: Dict[str, Any] = _normalize_qualification_signal(None, max_age_seconds=None, provider_max_age_seconds=None)
    ranked_candidates_trace: List[Dict[str, Any]] = []
    eligible_candidates_trace: List[Dict[str, Any]] = []
    disqualified_candidates_trace: List[Dict[str, Any]] = []
    routing_request: Dict[str, Any] = dict(request)
    selection_rubric: Dict[str, Any] = _build_selection_rubric(
        request=routing_request,
        routing_policy=routing_policy,
        status="not_evaluated",
    )
    route_lock = _request_route_lock(request)
    request_transport_binding = _request_transport_binding(request)
    request_agent_binding = _request_agent_binding(request)
    resolved_transport_binding: Dict[str, Any] = {}
    prompt_guardrail_details: Dict[str, Any] = {
        "schema": PROMPT_GUARDRAIL_SCHEMA,
        "status": "not_evaluated",
    }
    token_violation_row: Optional[Dict[str, Any]] = None
    workflow_dag_packet: Dict[str, Any] = {
        "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
        "status": "not_requested",
        "enabled": False,
        "requested": False,
    }
    hybrid_retrieval_packet: Dict[str, Any] = {
        "schema": "clawd.session_topology.hybrid_retrieval_efficiency.v1",
        "status": "not_evaluated",
        "enabled": False,
        "required": False,
        "mode": "not_evaluated",
        "context_policy": {"mode": "full_context_unchanged", "stuffing_blocked_by_default": False},
    }
    telegram_direct_offload: Dict[str, Any] = {
        "schema": "clawd.session_topology.telegram_direct_offload_gate.v1",
        "policy_id": "telegram_dm_heavy_offload_v1",
        "enforced": bool(require_telegram_direct_heavy_offload),
        "status": "not_evaluated",
        "reason": None,
        "requires_worker_handoff": False,
        "handoff_recommended": False,
        "target_worker_pattern": TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS[0],
        "target_worker_patterns": list(TELEGRAM_DIRECT_WORKER_TARGET_PATTERNS),
    }
    proposal_flow_packet: Dict[str, Any] = {
        "schema": PROPOSAL_FIRST_DELTA_SPEC_SCHEMA,
        "status": "not_evaluated",
        "coding_task": _is_coding_task_class(str(request.get("task_class") or "").strip()),
        "proposal_packet": {
            "present": bool(isinstance(request.get("proposal_packet"), Mapping)),
            "valid": False,
        },
    }
    regression_risk_packet_flow: Dict[str, Any] = {
        "schema": REGRESSION_RISK_PACKET_SCHEMA,
        "status": "not_evaluated",
        "coding_task": _is_coding_task_class(str(request.get("task_class") or "").strip()),
        "packet": {
            "present": bool(isinstance(request.get("regression_risk_packet"), Mapping)),
            "valid": False,
        },
        "routing": {
            "request_risk_tier": str(request.get("risk_tier") or "").strip() or None,
            "effective_risk_tier": str(request.get("risk_tier") or "").strip() or None,
            "tier_elevated": False,
        },
    }
    refactor_risk_packet_flow: Dict[str, Any] = {
        "schema": REFACTOR_RISK_PACKET_SCHEMA,
        "status": "not_evaluated",
        "coding_task": _is_coding_task_class(str(request.get("task_class") or "").strip()),
        "packet": {
            "present": bool(isinstance(request.get("refactor_risk_packet"), Mapping)),
            "valid": False,
        },
        "routing": {
            "request_risk_tier": str(request.get("risk_tier") or "").strip() or None,
            "effective_risk_tier": str(request.get("risk_tier") or "").strip() or None,
            "tier_elevated": False,
        },
        "decomposition": {
            "present": bool(isinstance((request.get("refactor_risk_packet") if isinstance(request.get("refactor_risk_packet"), Mapping) else {}).get("decomposition_plan"), list)),
            "chunk_count": 0,
            "review_chunk_present": False,
        },
    }

    # 1) Topology schema.
    try:
        schema_ok, reason, details = gate_schema(topology, topology_schema_path)
    except Exception as exc:  # pragma: no cover
        schema_ok, reason, details = False, "gate_unavailable", {"error": "gate_exception", "detail": str(exc)}

    if schema_ok:
        gates.append({"gate": "topology_schema", "status": "pass", "details": details})
    else:
        blocked = True
        block_gate = "topology_schema"
        block_reason = reason or "gate_unavailable"
        gates.append({"gate": "topology_schema", "status": "fail", "reason": block_reason, "details": details})

    # 2) Request contract.
    if blocked:
        gates.append({"gate": "routing_request", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        req_ok, reason, details = _request_contract_gate(
            request,
            routing_policy=routing_policy,
            require_worker_allocation_contract=require_worker_allocation_contract,
        )
        if req_ok:
            gates.append({"gate": "routing_request", "status": "pass", "details": details})
            workflow_dag_packet = dict(details.get("workflow_dag") or {}) if isinstance(details, Mapping) else {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "not_evaluated",
                "enabled": False,
            }
        else:
            blocked = True
            block_gate = "routing_request"
            block_reason = reason or "routing_request_invalid"
            workflow_dag_packet = dict(details.get("workflow_dag") or {
                "schema": WORKFLOW_DAG_ORCHESTRATION_SCHEMA,
                "status": "not_evaluated",
                "enabled": False,
            })
            gates.append({"gate": "routing_request", "status": "fail", "reason": block_reason, "details": details})

    # 3) Proposal-first delta-spec gate.
    if blocked:
        gates.append({"gate": "proposal_first_delta_spec", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        proposal_ok, reason, details = _proposal_first_delta_spec_gate(
            request=request,
            require_proposal_packet_for_coding=require_proposal_packet_for_coding,
        )
        proposal_flow_packet = dict(details or {}) if isinstance(details, Mapping) else {
            "schema": PROPOSAL_FIRST_DELTA_SPEC_SCHEMA,
            "status": "fail",
            "reason": "proposal_gate_internal_error",
        }
        if proposal_ok:
            gates.append({"gate": "proposal_first_delta_spec", "status": "pass", "details": proposal_flow_packet})
        else:
            blocked = True
            block_gate = "proposal_first_delta_spec"
            block_reason = reason or "proposal_first_contract_violation"
            gates.append(
                {
                    "gate": "proposal_first_delta_spec",
                    "status": "fail",
                    "reason": block_reason,
                    "details": proposal_flow_packet,
                }
            )

    # 4) Regression-risk packet gate.
    if blocked:
        gates.append({"gate": "regression_risk_packet", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        regression_ok, reason, details = _regression_risk_packet_gate(
            request=request,
            require_regression_risk_packet_for_coding=require_regression_risk_packet_for_coding,
        )
        regression_risk_packet_flow = dict(details or {}) if isinstance(details, Mapping) else {
            "schema": REGRESSION_RISK_PACKET_SCHEMA,
            "status": "fail",
            "reason": "regression_risk_gate_internal_error",
            "routing": {
                "request_risk_tier": str(request.get("risk_tier") or "").strip() or None,
                "effective_risk_tier": str(request.get("risk_tier") or "").strip() or None,
                "tier_elevated": False,
            },
        }
        effective_risk_tier = str(
            ((regression_risk_packet_flow.get("routing") if isinstance(regression_risk_packet_flow.get("routing"), Mapping) else {}).get("effective_risk_tier"))
            or request.get("risk_tier")
            or ""
        ).strip().lower()
        if effective_risk_tier in ALLOWED_RISK_TIERS:
            routing_request["risk_tier"] = effective_risk_tier
        selection_rubric = _build_selection_rubric(
            request=routing_request,
            routing_policy=routing_policy,
            status="not_evaluated",
        )
        if regression_ok:
            gates.append({"gate": "regression_risk_packet", "status": "pass", "details": regression_risk_packet_flow})
        else:
            blocked = True
            block_gate = "regression_risk_packet"
            block_reason = reason or "regression_risk_contract_violation"
            gates.append(
                {
                    "gate": "regression_risk_packet",
                    "status": "fail",
                    "reason": block_reason,
                    "details": regression_risk_packet_flow,
                }
            )

    # 5) Refactor-risk + decomposition gate.
    if blocked:
        gates.append({"gate": "refactor_risk_decomposition", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        refactor_ok, reason, details = _refactor_risk_packet_gate(
            request=routing_request,
            require_refactor_risk_packet_for_coding=require_refactor_risk_packet_for_coding,
        )
        refactor_risk_packet_flow = dict(details or {}) if isinstance(details, Mapping) else {
            "schema": REFACTOR_RISK_PACKET_SCHEMA,
            "status": "fail",
            "reason": "refactor_risk_gate_internal_error",
            "routing": {
                "request_risk_tier": str(routing_request.get("risk_tier") or "").strip() or None,
                "effective_risk_tier": str(routing_request.get("risk_tier") or "").strip() or None,
                "tier_elevated": False,
            },
        }
        effective_risk_tier = str(
            ((refactor_risk_packet_flow.get("routing") if isinstance(refactor_risk_packet_flow.get("routing"), Mapping) else {}).get("effective_risk_tier"))
            or routing_request.get("risk_tier")
            or ""
        ).strip().lower()
        if effective_risk_tier in ALLOWED_RISK_TIERS:
            routing_request["risk_tier"] = effective_risk_tier
        selection_rubric = _build_selection_rubric(
            request=routing_request,
            routing_policy=routing_policy,
            status="not_evaluated",
        )

        if refactor_ok:
            gates.append({"gate": "refactor_risk_decomposition", "status": "pass", "details": refactor_risk_packet_flow})
        else:
            blocked = True
            block_gate = "refactor_risk_decomposition"
            block_reason = reason or "refactor_risk_contract_violation"
            gates.append(
                {
                    "gate": "refactor_risk_decomposition",
                    "status": "fail",
                    "reason": block_reason,
                    "details": refactor_risk_packet_flow,
                }
            )

    # 6) Transport-route conformance.
    if blocked:
        gates.append({"gate": "transport_route_conformance", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        transport_ok, reason, details = _transport_route_conformance_gate(
            request_transport_binding=request_transport_binding,
            request_agent_binding=request_agent_binding,
            transport_decision=transport_decision,
            require_transport_decision=require_transport_decision,
        )
        if transport_ok:
            resolved_transport_binding = {}
            if isinstance(details, Mapping):
                explicit_binding = details.get("binding")
                if isinstance(explicit_binding, Mapping) and explicit_binding:
                    resolved_transport_binding = dict(explicit_binding)
                else:
                    fallback_request_binding = details.get("request_transport_binding")
                    if isinstance(fallback_request_binding, Mapping) and fallback_request_binding:
                        resolved_transport_binding = dict(fallback_request_binding)
            gates.append({"gate": "transport_route_conformance", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "transport_route_conformance"
            block_reason = reason or "transport_route_conformance_failed"
            gates.append(
                {
                    "gate": "transport_route_conformance",
                    "status": "fail",
                    "reason": block_reason,
                    "details": details,
                }
            )

    # 7) Route selection.
    candidate_rules: List[Mapping[str, Any]] = []
    if blocked:
        gates.append({"gate": "route_selection", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        topology_obj = topology if isinstance(topology, Mapping) else {}
        winner, matches = _resolve_rule(topology_obj, routing_request)
        candidate_rules = matches

        if winner is None:
            route_class = str(topology_obj.get("default_route_class") or "").strip()
            required_stage = str(topology_obj.get("default_required_rollout_stage") or "").strip()
            if route_class not in ALLOWED_ROUTE_CLASSES or required_stage not in ALLOWED_ROLLOUT_STAGES:
                blocked = True
                block_gate = "route_selection"
                block_reason = "route_policy_invalid"
                gates.append(
                    {
                        "gate": "route_selection",
                        "status": "fail",
                        "reason": block_reason,
                        "details": {
                            "error": "default_route_invalid",
                            "default_route_class": route_class or None,
                            "default_required_rollout_stage": required_stage or None,
                        },
                    }
                )
            else:
                gates.append(
                    {
                        "gate": "route_selection",
                        "status": "pass",
                        "details": {
                            "selected_via": "default",
                            "route_class": route_class,
                            "required_rollout_stage": required_stage,
                            "candidate_rule_count": 0,
                        },
                    }
                )
        else:
            selected_rule = winner
            route_class = str(winner.get("route_class") or "").strip()
            required_stage = str(winner.get("required_rollout_stage") or "").strip()
            if route_class not in ALLOWED_ROUTE_CLASSES or required_stage not in ALLOWED_ROLLOUT_STAGES:
                blocked = True
                block_gate = "route_selection"
                block_reason = "route_policy_invalid"
                gates.append(
                    {
                        "gate": "route_selection",
                        "status": "fail",
                        "reason": block_reason,
                        "details": {
                            "error": "selected_rule_invalid",
                            "rule_id": winner.get("rule_id"),
                            "route_class": route_class or None,
                            "required_rollout_stage": required_stage or None,
                        },
                    }
                )
            else:
                gates.append(
                    {
                        "gate": "route_selection",
                        "status": "pass",
                        "details": {
                            "selected_via": "rule",
                            "rule_id": winner.get("rule_id"),
                            "priority": winner.get("priority"),
                            "route_class": route_class,
                            "required_rollout_stage": required_stage,
                            "candidate_rule_count": len(matches),
                        },
                    }
                )

    # 8) Telegram direct-lane heavy offload gate.
    if blocked:
        gates.append({"gate": "telegram_direct_offload", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        offload_ok, reason, details = _telegram_direct_offload_gate(
            request=routing_request,
            route_class=route_class,
            resolved_transport_binding=resolved_transport_binding,
            routing_policy=routing_policy,
            enforce=require_telegram_direct_heavy_offload,
        )
        telegram_direct_offload = dict(details or {}) if isinstance(details, Mapping) else {}
        if offload_ok:
            gates.append({"gate": "telegram_direct_offload", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "telegram_direct_offload"
            block_reason = reason or "telegram_direct_heavy_offload_required"
            gates.append({"gate": "telegram_direct_offload", "status": "fail", "reason": block_reason, "details": details})

    # 9) Escalation evidence (default-down taxonomy gate).
    task_taxonomy = _task_taxonomy_for_request(
        routing_request,
        topology if isinstance(topology, Mapping) else {},
        routing_policy,
    )
    escalation_evidence = _normalize_escalation_evidence(routing_request)
    escalation_gate_details: Dict[str, Any] = {
        "taxonomy": task_taxonomy,
        "evidence": escalation_evidence,
    }

    if blocked:
        gates.append({"gate": "tier_escalation_evidence", "status": "skipped", "reason": "blocked_by_previous_gate", "details": escalation_gate_details})
    else:
        topology_obj = topology if isinstance(topology, Mapping) else {}
        escalation_ok, reason, details = _evaluate_escalation_gate(
            request=routing_request,
            topology=topology_obj,
            routing_policy=routing_policy,
            selected_route_class=route_class,
            selected_required_stage=required_stage,
        )
        escalation_gate_details = details
        if escalation_ok:
            gates.append({"gate": "tier_escalation_evidence", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "tier_escalation_evidence"
            block_reason = reason or "escalation_evidence_missing"
            gates.append({"gate": "tier_escalation_evidence", "status": "fail", "reason": block_reason, "details": details})

    # 10) Pool-policy alignment.
    if blocked:
        gates.append({"gate": "pool_policy_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        topology_obj = topology if isinstance(topology, Mapping) else {}
        pool_ok, reason, details = gate_pool_policy_alignment(
            topology=topology_obj,
            route_class=str(route_class or ""),
            selected_model=None,
            pool_policy=pool_policy,
        )
        if pool_ok:
            gates.append({"gate": "pool_policy_alignment", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "pool_policy_alignment"
            block_reason = reason or "pool_policy_violation"
            gates.append({"gate": "pool_policy_alignment", "status": "fail", "reason": block_reason, "details": details})

    # 11) Qualification / model selection.
    # Quota freshness gate
    if not blocked:
        quota_freshness_gate = _check_quota_freshness(repo_root)
        gates.append(quota_freshness_gate)

    if blocked:
        gates.append({"gate": "qualification_model_selection", "status": "skipped", "reason": "blocked_by_previous_gate"})
    elif route_class == "NO_LLM":
        selected_model = None
        gates.append(
            {
                "gate": "qualification_model_selection",
                "status": "pass",
                "details": {
                    "route_class": route_class,
                    "required_rollout_stage": required_stage,
                    "selected_model": None,
                    "reason": "deterministic_no_llm_route",
                },
            }
        )
    else:
        topology_obj = topology if isinstance(topology, Mapping) else {}
        model_pools = topology_obj.get("model_pools") if isinstance(topology_obj.get("model_pools"), Mapping) else {}
        pool = model_pools.get(route_class)
        if not isinstance(pool, list) or not pool:
            blocked = True
            block_gate = "qualification_model_selection"
            block_reason = "route_policy_invalid"
            gates.append(
                {
                    "gate": "qualification_model_selection",
                    "status": "fail",
                    "reason": block_reason,
                    "details": {"error": "model_pool_missing", "route_class": route_class},
                }
            )
        else:
            gate_index = _collect_model_gate_index(gate_decisions)
            if not gate_index:
                blocked = True
                block_gate = "qualification_model_selection"
                block_reason = "no_qualified_model_for_route"
                gates.append(
                    {
                        "gate": "qualification_model_selection",
                        "status": "fail",
                        "reason": block_reason,
                        "details": {
                            "error": "qualification_decisions_missing_or_not_passed",
                            "route_class": route_class,
                        },
                    }
                )
            else:
                candidates: List[Dict[str, Any]] = []
                for model_key in [str(item) for item in pool if isinstance(item, str) and item.strip()]:
                    meta = gate_index.get(model_key)
                    if not meta:
                        continue
                    if str(meta.get("route_class") or "") != route_class:
                        continue
                    allowed_stages = set(str(x) for x in (meta.get("allowed_stages") or []))
                    if not _stage_is_allowed(str(required_stage or ""), allowed_stages):
                        continue
                    candidates.append(meta)

                if not candidates:
                    blocked = True
                    block_gate = "qualification_model_selection"
                    block_reason = "no_qualified_model_for_route"
                    gates.append(
                        {
                            "gate": "qualification_model_selection",
                            "status": "fail",
                            "reason": block_reason,
                            "details": {
                                "route_class": route_class,
                                "required_rollout_stage": required_stage,
                                "pool": list(pool),
                            },
                        }
                    )
                else:
                    task_taxonomy = _task_taxonomy_for_request(routing_request, topology_obj, routing_policy)
                    prioritized_candidates = _prioritize_candidates_by_model_family(
                        request=routing_request,
                        task_taxonomy=task_taxonomy,
                        candidates=candidates,
                    )
                    ranked_candidates: List[Dict[str, Any]] = []
                    eligible_candidates: List[Dict[str, Any]] = []
                    disqualified_candidates: List[Dict[str, Any]] = []
                    coding_signal_required = bool(
                        _is_coding_task_class(str(routing_request.get("task_class") or "").strip())
                        and (
                            routing_policy_coding_require_qualification_signal(routing_policy)
                            or isinstance(
                                routing_policy_coding_min_score(
                                    routing_policy,
                                    str(routing_request.get("risk_tier") or "").strip(),
                                ),
                                (int, float),
                            )
                        )
                    )

                    # Get max age for qualification signals and provider evidence
                    risk_tier = str(routing_request.get("risk_tier") or "").strip()
                    max_age_seconds = routing_policy_qualification_signal_max_age_seconds_by_risk_tier(routing_policy, risk_tier)
                    provider_max_age_seconds = routing_policy_provider_evidence_max_age_seconds_by_risk_tier(routing_policy, risk_tier)
                    legacy_grace_status = routing_policy_legacy_missing_timestamp_grace_status_by_risk_tier(
                        routing_policy,
                        risk_tier,
                    )
                    legacy_grace_period_seconds = (
                        int(legacy_grace_status.get("grace_period_seconds"))
                        if legacy_grace_status.get("grace_window_active") is True
                        and isinstance(legacy_grace_status.get("grace_period_seconds"), int)
                        else None
                    )
                    
                    for row in prioritized_candidates:
                        signal = _normalize_qualification_signal(
                            row.get("qualification_signal"), 
                            max_age_seconds, 
                            provider_max_age_seconds,
                            legacy_grace_period_seconds,
                            legacy_grace_status,
                        )
                        coding_gate = _coding_signal_gate(
                            request=routing_request,
                            qualification_signal=signal,
                            routing_policy=routing_policy,
                            require_signal_for_coding=coding_signal_required,
                            max_age_seconds=max_age_seconds,
                            provider_max_age_seconds=provider_max_age_seconds,
                            legacy_grace_period_seconds=legacy_grace_period_seconds,
                        )
                        enriched = {
                            **dict(row),
                            "qualification_signal": signal,
                            "coding_signal_gate": coding_gate,
                        }
                        ranked_candidates.append(enriched)
                        if coding_gate.get("allowed_for_selection") is True:
                            eligible_candidates.append(enriched)
                        else:
                            disqualified_candidates.append(enriched)

                    ranked_candidates_trace = list(ranked_candidates)
                    eligible_candidates_trace = list(eligible_candidates)
                    disqualified_candidates_trace = list(disqualified_candidates)

                    if not eligible_candidates:
                        blocked = True
                        block_gate = "qualification_model_selection"
                        block_reason = "no_qualified_model_for_route"
                        selection_rubric = _build_selection_rubric(
                            request=routing_request,
                            routing_policy=routing_policy,
                            status="blocked",
                            coding_signal_required=coding_signal_required,
                            candidate_count=len(ranked_candidates),
                            eligible_candidate_count=0,
                            disqualified_candidate_count=len(disqualified_candidates),
                        )
                        gates.append(
                            {
                                "gate": "qualification_model_selection",
                                "status": "fail",
                                "reason": block_reason,
                                "details": {
                                    "error": "candidate_selection_gate_rejected",
                                    "route_class": route_class,
                                    "required_rollout_stage": required_stage,
                                    "coding_signal_required": coding_signal_required,
                                    "disqualified_candidates": [
                                        {
                                            "model_key": str(row.get("model_key") or ""),
                                            "model_family": _model_family_from_model_key(str(row.get("model_key") or "")),
                                            "coding_signal_gate": row.get("coding_signal_gate"),
                                            "qualification_signal": row.get("qualification_signal"),
                                        }
                                        for row in disqualified_candidates
                                    ],
                                },
                            }
                        )
                    else:
                        selected = eligible_candidates[0]
                        selected_model = str(selected.get("model_key"))
                        selected_family = _model_family_from_model_key(selected_model)
                        selected_signal = selected.get("qualification_signal")
                        if isinstance(selected_signal, Mapping):
                            selected_qualification_signal = _json_clone(dict(selected_signal))
                        else:
                            selected_qualification_signal = _normalize_qualification_signal(
                                selected_signal,
                                max_age_seconds=max_age_seconds,
                                provider_max_age_seconds=provider_max_age_seconds,
                                legacy_grace_period_seconds=legacy_grace_period_seconds,
                                legacy_grace_status=legacy_grace_status,
                            )

                        pool_ok, reason, details = gate_pool_policy_alignment(
                            topology=topology_obj,
                            route_class=str(route_class or ""),
                            selected_model=selected_model,
                            pool_policy=pool_policy,
                        )
                        if not pool_ok:
                            blocked = True
                            block_gate = "pool_policy_alignment"
                            block_reason = reason or "pool_policy_violation"
                            gates.append(
                                {
                                    "gate": "qualification_model_selection",
                                    "status": "fail",
                                    "reason": block_reason,
                                    "details": details,
                                }
                            )
                        else:
                            selection_rubric = _build_selection_rubric(
                                request=routing_request,
                                routing_policy=routing_policy,
                                status="applied",
                                coding_signal_required=coding_signal_required,
                                candidate_count=len(ranked_candidates),
                                eligible_candidate_count=len(eligible_candidates),
                                disqualified_candidate_count=len(disqualified_candidates),
                            )
                            gates.append(
                                {
                                    "gate": "qualification_model_selection",
                                    "status": "pass",
                                    "details": {
                                        "route_class": route_class,
                                        "required_rollout_stage": required_stage,
                                        "selected_model": selected_model,
                                        "selected_model_family": selected_family,
                                        "selected_allowed_stages": selected.get("allowed_stages"),
                                        "selected_qualification_signal": selected_qualification_signal,
                                        "prioritized_candidates": [
                                            {
                                                "model_key": str(row.get("model_key") or ""),
                                                "model_family": _model_family_from_model_key(str(row.get("model_key") or "")),
                                                "allowed_stages": row.get("allowed_stages"),
                                                "qualification_signal": row.get("qualification_signal"),
                                                "coding_signal_gate": row.get("coding_signal_gate"),
                                            }
                                            for row in ranked_candidates
                                        ],
                                        "family_priority": {
                                            "default": task_taxonomy.get("default_model_family"),
                                            "fallback": task_taxonomy.get("fallback_model_families"),
                                        },
                                        "selection_rubric": selection_rubric,
                                    },
                                }
                            )

    if blocked:
        gates.append({"gate": "requested_route_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        mismatches: List[Dict[str, Any]] = []

        expected_route_class = route_lock.get("route_class")
        if expected_route_class and expected_route_class != str(route_class or ""):
            mismatches.append(
                {
                    "field": "route_class",
                    "expected": expected_route_class,
                    "actual": route_class,
                }
            )

        expected_stage = route_lock.get("required_rollout_stage")
        if expected_stage and expected_stage != str(required_stage or ""):
            mismatches.append(
                {
                    "field": "required_rollout_stage",
                    "expected": expected_stage,
                    "actual": required_stage,
                }
            )

        expected_model = route_lock.get("model_key")
        if expected_model and expected_model != str(selected_model or ""):
            mismatches.append(
                {
                    "field": "model_key",
                    "expected": expected_model,
                    "actual": selected_model,
                }
            )

        expected_rule_id = route_lock.get("rule_id")
        actual_rule_id = selected_rule.get("rule_id") if isinstance(selected_rule, Mapping) else None
        if expected_rule_id and expected_rule_id != str(actual_rule_id or ""):
            mismatches.append(
                {
                    "field": "rule_id",
                    "expected": expected_rule_id,
                    "actual": actual_rule_id,
                }
            )

        if mismatches:
            blocked = True
            block_gate = "requested_route_alignment"
            block_reason = "requested_route_mismatch"
            gates.append(
                {
                    "gate": "requested_route_alignment",
                    "status": "fail",
                    "reason": block_reason,
                    "details": {
                        "requested": route_lock,
                        "mismatches": mismatches,
                    },
                }
            )
        else:
            gates.append(
                {
                    "gate": "requested_route_alignment",
                    "status": "pass",
                    "details": {
                        "requested": route_lock,
                    },
                }
            )

    # 12) Prompt lint + token guardrail gate.
    if blocked:
        gates.append({"gate": "prompt_guardrails", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        prompt_ok, reason, details, violation_row = _evaluate_prompt_guardrails(
            request=routing_request,
            route_class=route_class,
            required_rollout_stage=required_stage,
        )
        prompt_guardrail_details = details
        token_violation_row = violation_row
        if prompt_ok:
            gates.append({"gate": "prompt_guardrails", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "prompt_guardrails"
            block_reason = reason or "prompt_token_budget_exceeded"
            gates.append({"gate": "prompt_guardrails", "status": "fail", "reason": block_reason, "details": details})

    # 13) Hybrid retrieval efficiency pack (XE-203).
    if blocked:
        gates.append({"gate": "hybrid_retrieval_efficiency", "status": "skipped", "reason": "blocked_by_previous_gate"})
    else:
        hybrid_retrieval_packet = evaluate_hybrid_retrieval_efficiency(routing_request)
        hybrid_block = bool(hybrid_retrieval_packet.get("block") is True)
        hybrid_status = str(hybrid_retrieval_packet.get("status") or "not_requested")
        if hybrid_block:
            blocked = True
            block_gate = "hybrid_retrieval_efficiency"
            block_reason = str(hybrid_retrieval_packet.get("block_reason") or "hybrid_retrieval_abstain")
            gates.append(
                {
                    "gate": "hybrid_retrieval_efficiency",
                    "status": "fail",
                    "reason": block_reason,
                    "details": hybrid_retrieval_packet,
                }
            )
        else:
            gates.append(
                {
                    "gate": "hybrid_retrieval_efficiency",
                    "status": "pass",
                    "details": hybrid_retrieval_packet,
                }
            )

    decision = "BLOCK" if blocked else "PASS"
    final_state = "BLOCKED" if blocked else "ROUTED"
    provider_route_decision = _build_provider_route_decision(
        request=routing_request,
        route_class=route_class,
        required_stage=required_stage,
        selected_model=selected_model,
        selected_qualification_signal=selected_qualification_signal,
        selection_rubric=selection_rubric,
        ranked_candidates=ranked_candidates_trace,
        eligible_candidates=eligible_candidates_trace,
        disqualified_candidates=disqualified_candidates_trace,
    )
    routing_telemetry = _routing_telemetry_summary(
        request=routing_request,
        routing_policy=routing_policy,
        task_taxonomy=task_taxonomy,
        route_class=route_class,
        selected_model=selected_model,
        escalation_gate_details=escalation_gate_details,
        telegram_direct_offload=telegram_direct_offload,
    )

    return {
        "schema": "clawd.session_topology_routing.decision.v1",
        "router_layer": ROUTER_LAYER,
        "router_module": "scripts/session_topology_router.py",
        "evaluated_at": now_iso(),
        "decision": decision,
        "final_state": final_state,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "request": {
            "session_kind": request.get("session_kind"),
            "task_class": request.get("task_class"),
            "risk_tier": request.get("risk_tier"),
            "effective_risk_tier": routing_request.get("risk_tier"),
            "fold_in_target": _normalize_fold_in_target(request.get("fold_in_target")),
            "support_only_helper": _is_support_only_helper_request(request),
            "route_lock": route_lock,
            "transport_binding": request_transport_binding,
            "agent_binding": request_agent_binding,
            "task_taxonomy": task_taxonomy,
            "escalation_evidence": escalation_evidence,
            "prompt_guardrails": _normalize_prompt_guardrail_request(request),
            "invocation_prompt_present": isinstance(request.get("invocation_prompt"), str),
            "knowledge_retrieval": {
                "present": isinstance(request.get("knowledge_retrieval"), Mapping),
                "enabled": bool((request.get("knowledge_retrieval") if isinstance(request.get("knowledge_retrieval"), Mapping) else {}).get("enabled") is True),
            },
            "proposal_packet_present": bool(isinstance(request.get("proposal_packet"), Mapping)),
            "regression_risk_packet_present": bool(isinstance(request.get("regression_risk_packet"), Mapping)),
            "refactor_risk_packet_present": bool(isinstance(request.get("refactor_risk_packet"), Mapping)),
            "proposal_flow": proposal_flow_packet,
            "regression_risk": regression_risk_packet_flow,
            "refactor_risk": refactor_risk_packet_flow,
            "workflow_dag": workflow_dag_packet,
            "telegram_direct_offload": telegram_direct_offload,
        },
        "transport_conformance": {
            "required": require_transport_decision,
            "resolved_binding": resolved_transport_binding,
            "decision_schema": TRANSPORT_DECISION_SCHEMA if transport_decision is not None else None,
            "telegram_direct_heavy_offload_required": bool(require_telegram_direct_heavy_offload),
        },
        "route": {
            "selected_rule_id": selected_rule.get("rule_id") if isinstance(selected_rule, Mapping) else None,
            "candidate_rule_ids": [str(row.get("rule_id")) for row in candidate_rules if isinstance(row, Mapping)],
            "route_class": route_class,
            "required_rollout_stage": required_stage,
            "task_class": routing_request.get("task_class"),
            "request_risk_tier": request.get("risk_tier"),
            "effective_risk_tier": routing_request.get("risk_tier"),
            "selected_model": selected_model,
            "selected_qualification_signal": selected_qualification_signal,
            "selection_rubric": selection_rubric,
            "rubric_rule_id": selection_rubric.get("rubric_rule_id"),
            "rubric_rule_id_v2": selection_rubric.get("rubric_rule_id_v2"),
            "provider_route_decision": provider_route_decision,
            "proposal_task_id": ((proposal_flow_packet.get("proposal_packet") or {}).get("task_id") if isinstance(proposal_flow_packet.get("proposal_packet"), Mapping) else None),
            "proposal_flow_status": proposal_flow_packet.get("status"),
            "proposal_phase": ((proposal_flow_packet.get("state_flow") or {}).get("effective_phase") if isinstance(proposal_flow_packet.get("state_flow"), Mapping) else None),
            "proposal_flow": proposal_flow_packet,
            "regression_risk": regression_risk_packet_flow,
            "refactor_risk": refactor_risk_packet_flow,
            "selected_model_family": routing_telemetry.get("selected_model_family"),
            "default_model_family": routing_telemetry.get("default_model_family"),
            "fallback_model_families": routing_telemetry.get("fallback_model_families"),
            "task_class_guard": routing_telemetry.get("task_class_guard"),
            "taxonomy_tier": (task_taxonomy or {}).get("taxonomy_tier"),
            "support_only_helper": bool((escalation_gate_details or {}).get("support_only_helper") is True),
            "baseline_route_class": ((escalation_gate_details or {}).get("baseline_route") or {}).get("route_class"),
            "baseline_required_rollout_stage": ((escalation_gate_details or {}).get("baseline_route") or {}).get("required_rollout_stage"),
            "escalation_required": bool((escalation_gate_details or {}).get("escalation_required") is True),
            "escalation_signals": list((escalation_gate_details or {}).get("signals") or []),
            "misrouting_signals": list(routing_telemetry.get("misrouting_signals") or []),
            "telegram_direct_offload": telegram_direct_offload,
            "prompt_guardrails": {
                "status": prompt_guardrail_details.get("status"),
                "effective_prompt_tokens": prompt_guardrail_details.get("effective_prompt_tokens"),
                "effective_total_tokens": prompt_guardrail_details.get("effective_total_tokens"),
                "violations": prompt_guardrail_details.get("violations"),
                "prompt_lint": prompt_guardrail_details.get("prompt_lint"),
            },
            "hybrid_retrieval": {
                "status": hybrid_retrieval_packet.get("status"),
                "confidence_tier": hybrid_retrieval_packet.get("confidence_tier"),
                "selected_top_k": hybrid_retrieval_packet.get("selected_top_k"),
                "saved_tokens": ((hybrid_retrieval_packet.get("context_policy") or {}).get("saved_tokens") if isinstance(hybrid_retrieval_packet.get("context_policy"), Mapping) else None),
                "saved_pct": ((hybrid_retrieval_packet.get("context_policy") or {}).get("saved_pct") if isinstance(hybrid_retrieval_packet.get("context_policy"), Mapping) else None),
                "stuffing_blocked_by_default": ((hybrid_retrieval_packet.get("context_policy") or {}).get("stuffing_blocked_by_default") if isinstance(hybrid_retrieval_packet.get("context_policy"), Mapping) else None),
            },
        },
        "pool_policy": {
            "policy_id": pool_policy.get("policy_id"),
            "policy_path": pool_policy_meta.get("path"),
            "policy_schema_path": pool_policy_meta.get("schema_path"),
        },
        "routing_policy": {
            "policy_id": routing_policy.get("policy_id"),
            "policy_path": routing_policy_meta.get("path"),
            "policy_schema_path": routing_policy_meta.get("schema_path"),
        },
        "routing_telemetry": routing_telemetry,
        "telegram_direct_offload": telegram_direct_offload,
        "proposal_flow": proposal_flow_packet,
        "regression_risk": regression_risk_packet_flow,
        "refactor_risk": refactor_risk_packet_flow,
        "routing_audit": {
            "schema": "clawd.session_topology.routing_audit.v1",
            "task_class": routing_telemetry.get("task_class"),
            "request_risk_tier": request.get("risk_tier"),
            "effective_risk_tier": routing_request.get("risk_tier"),
            "task_class_guard": routing_telemetry.get("task_class_guard"),
            "selected_model_family": routing_telemetry.get("selected_model_family"),
            "default_model_family": routing_telemetry.get("default_model_family"),
            "misrouting_signals": list(routing_telemetry.get("misrouting_signals") or []),
            "misrouting_incident": bool(list(routing_telemetry.get("misrouting_signals") or [])),
            "selected_qualification_signal": selected_qualification_signal,
            "selection_rubric": selection_rubric,
            "rubric_rule_id": selection_rubric.get("rubric_rule_id"),
            "rubric_rule_id_v2": selection_rubric.get("rubric_rule_id_v2"),
            "provider_route_decision": provider_route_decision,
            "proposal_flow": {
                "status": proposal_flow_packet.get("status"),
                "reason": proposal_flow_packet.get("reason"),
                "proposal_task_id": ((proposal_flow_packet.get("proposal_packet") or {}).get("task_id") if isinstance(proposal_flow_packet.get("proposal_packet"), Mapping) else None),
                "phase": ((proposal_flow_packet.get("state_flow") or {}).get("effective_phase") if isinstance(proposal_flow_packet.get("state_flow"), Mapping) else None),
                "approval_checkpoint": ((proposal_flow_packet.get("approval_hooks") or {}).get("proposal_checkpoint") if isinstance(proposal_flow_packet.get("approval_hooks"), Mapping) else None),
            },
            "regression_risk": {
                "status": regression_risk_packet_flow.get("status"),
                "reason": regression_risk_packet_flow.get("reason"),
                "overall_tier": ((regression_risk_packet_flow.get("risk_assessment") or {}).get("overall_tier") if isinstance(regression_risk_packet_flow.get("risk_assessment"), Mapping) else None),
                "effective_risk_tier": ((regression_risk_packet_flow.get("routing") or {}).get("effective_risk_tier") if isinstance(regression_risk_packet_flow.get("routing"), Mapping) else None),
                "approval_satisfied": ((regression_risk_packet_flow.get("validation") or {}).get("approval_satisfied") if isinstance(regression_risk_packet_flow.get("validation"), Mapping) else None),
            },
            "refactor_risk": {
                "status": refactor_risk_packet_flow.get("status"),
                "reason": refactor_risk_packet_flow.get("reason"),
                "overall_tier": ((refactor_risk_packet_flow.get("risk_assessment") or {}).get("overall_tier") if isinstance(refactor_risk_packet_flow.get("risk_assessment"), Mapping) else None),
                "effective_risk_tier": ((refactor_risk_packet_flow.get("routing") or {}).get("effective_risk_tier") if isinstance(refactor_risk_packet_flow.get("routing"), Mapping) else None),
                "decomposition_chunk_count": ((refactor_risk_packet_flow.get("decomposition") or {}).get("chunk_count") if isinstance(refactor_risk_packet_flow.get("decomposition"), Mapping) else None),
                "approval_satisfied": ((refactor_risk_packet_flow.get("validation") or {}).get("approval_satisfied") if isinstance(refactor_risk_packet_flow.get("validation"), Mapping) else None),
            },
        },
        "prompt_guardrail": prompt_guardrail_details,
        "workflow_dag": workflow_dag_packet,
        "hybrid_retrieval": hybrid_retrieval_packet,
        "token_violation": token_violation_row,
        "gates": gates,
        "actionable_failure": _actionable_failure(block_gate, block_reason, gates),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic route-policy session topology router (v1)")
    ap.add_argument("--topology", required=True, help="Path to session topology contract JSON")
    ap.add_argument("--topology-schema", default=str(DEFAULT_TOPOLOGY_SCHEMA), help="Topology schema path")

    ap.add_argument("--request", default=None, help="Routing request JSON path")
    ap.add_argument("--session-kind", default="", help="Inline request session_kind")
    ap.add_argument("--task-class", default="", help="Inline request task_class")
    ap.add_argument("--risk-tier", default="", help="Inline request risk_tier")
    ap.add_argument("--requested-route-class", default="", help="Optional route lock: expected route_class")
    ap.add_argument(
        "--requested-required-rollout-stage",
        default="",
        help="Optional route lock: expected required_rollout_stage",
    )
    ap.add_argument("--requested-model-key", default="", help="Optional route lock: expected selected model key")
    ap.add_argument("--requested-rule-id", default="", help="Optional route lock: expected selected rule_id")

    ap.add_argument(
        "--qualification-decision",
        action="append",
        default=[],
        help="Model rollout gate decision JSON path (repeatable)",
    )
    ap.add_argument(
        "--transport-decision",
        default="",
        help="Transport-topology decision JSON path (from session-transport-route)",
    )
    ap.add_argument(
        "--legacy-allow-missing-transport-decision",
        action="store_true",
        help=(
            "Legacy bounded escape hatch: allow route-policy evaluation without a transport decision. "
            "Strict transport conformance is default."
        ),
    )
    ap.add_argument(
        "--require-worker-allocation-contract",
        action="store_true",
        help=(
            "Require worker-allocation dispatch contract fields for worker_slice requests "
            "(scope_shape, verification_class, worker_topology, fold_in_target)."
        ),
    )
    ap.add_argument(
        "--legacy-allow-missing-worker-allocation-contract",
        action="store_true",
        help=(
            "Legacy bounded escape hatch: allow worker_slice requests without full worker-allocation fields."
        ),
    )
    ap.add_argument(
        "--legacy-allow-telegram-direct-heavy-on-dm",
        action="store_true",
        help=(
            "Legacy bounded escape hatch: allow HEAVY, multi-surface/parallel, or validator-required worker_slice routing while bound to telegram|direct transport."
        ),
    )
    ap.add_argument(
        "--require-proposal-first-coding",
        action="store_true",
        help=(
            "Require proposal_packet.v1 ingress for coding task classes and enforce proposal-first approval hooks before routing."
        ),
    )
    ap.add_argument(
        "--require-regression-risk-packet-for-coding",
        action="store_true",
        help=(
            "Require regression_risk_packet for coding task classes and enforce approved validation status before routing."
        ),
    )
    ap.add_argument(
        "--require-refactor-risk-packet-for-coding",
        action="store_true",
        help=(
            "Require refactor_risk_packet for coding task classes and enforce decomposition governance + approved validation status before routing."
        ),
    )

    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--pool-policy", default=str(DEFAULT_POOL_POLICY_PATH), help="Unified model pool policy JSON path")
    ap.add_argument("--pool-policy-schema", default=str(DEFAULT_POOL_POLICY_SCHEMA), help="Unified model pool policy schema path")
    ap.add_argument("--routing-policy", default=str(DEFAULT_ROUTING_POLICY_PATH), help="Session topology routing policy JSON path")
    ap.add_argument("--routing-policy-schema", default=str(DEFAULT_ROUTING_POLICY_SCHEMA), help="Session topology routing policy schema path")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Append-only routing decision log")
    ap.add_argument("--no-decision-log", action="store_true", help="Disable decision log append")
    ap.add_argument(
        "--token-violation-ledger",
        default=str(DEFAULT_TOKEN_VIOLATION_LEDGER),
        help="Append-only token violation ledger (writes only when prompt guardrail blocks)",
    )
    ap.add_argument("--no-token-violation-ledger", action="store_true", help="Disable token violation ledger append")
    ap.add_argument(
        "--prompt-tool-cache",
        default=str(DEFAULT_PROMPT_TOOL_CACHE_PATH),
        help="Prompt/tool deterministic cache store path",
    )
    ap.add_argument("--no-prompt-tool-cache", action="store_true", help="Disable prompt/tool cache")
    ap.add_argument(
        "--prompt-tool-cache-ttl-sec",
        default=str(DEFAULT_PROMPT_TOOL_CACHE_TTL_SEC),
        help="TTL seconds for prompt/tool cache entries",
    )
    ap.add_argument("--prompt-tool-cache-flush", action="store_true", help="Flush prompt/tool cache file before evaluating")
    ap.add_argument(
        "--context-delta-cache",
        default=str(DEFAULT_CONTEXT_DELTA_CACHE_PATH),
        help="Context-delta transport cache store path",
    )
    ap.add_argument("--no-context-delta-transport", action="store_true", help="Disable context-delta transport")
    ap.add_argument("--context-delta-cache-flush", action="store_true", help="Flush context-delta cache file before evaluating")
    ap.add_argument(
        "--context-compaction-cache",
        default=str(DEFAULT_CONTEXT_COMPACTION_CACHE_PATH),
        help="Anchor-preserving context compaction cache store path",
    )
    ap.add_argument("--no-context-compaction", action="store_true", help="Disable anchor-preserving context compaction")
    ap.add_argument("--context-compaction-cache-flush", action="store_true", help="Flush context-compaction cache file before evaluating")
    ap.add_argument(
        "--event-backbone-typed-log",
        default=str(DEFAULT_EVENT_BACKBONE_TYPED_LOG),
        help="Typed orchestration event stream JSONL path",
    )
    ap.add_argument(
        "--event-backbone-db",
        default=str(DEFAULT_EVENT_BACKBONE_DB),
        help="SQLite state/journal path for event backbone idempotency + retry state",
    )
    ap.add_argument(
        "--event-backbone-dlq",
        default=str(DEFAULT_EVENT_BACKBONE_DLQ),
        help="Dead-letter queue JSONL path for failed dual-write attempts",
    )
    ap.add_argument(
        "--event-backbone-metrics",
        default=str(DEFAULT_EVENT_BACKBONE_METRICS),
        help="Backpressure metrics JSON path for the typed event backbone",
    )
    ap.add_argument(
        "--event-backbone-max-attempts",
        default=str(DEFAULT_EVENT_BACKBONE_MAX_ATTEMPTS),
        help="Bounded retry attempt count before DLQ",
    )
    ap.add_argument(
        "--event-backbone-base-backoff-ms",
        default=str(DEFAULT_EVENT_BACKBONE_BASE_BACKOFF_MS),
        help="Base retry backoff used to emit deterministic retry schedules",
    )
    ap.add_argument("--no-event-backbone", action="store_true", help="Disable typed event dual-write; keep legacy decision log only")
    ap.add_argument(
        "--workflow-state-journal",
        default=str(DEFAULT_WORKFLOW_STATE_JOURNAL),
        help="Replayable workflow state machine transition journal JSONL path",
    )
    ap.add_argument(
        "--workflow-state-latest",
        default=str(DEFAULT_WORKFLOW_STATE_LATEST),
        help="Latest replayable workflow state machine snapshot JSON path",
    )
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def _load_request(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.request:
        req_path = Path(args.request).expanduser().resolve()
        payload = load_json_file(req_path)
        request_obj = dict(payload) if isinstance(payload, Mapping) else {}

        proposal_packet_direct = bool(
            isinstance(payload, Mapping)
            and str(payload.get("schema_version") or "").strip() == PROPOSAL_PACKET_SCHEMA_VERSION
            and "risk_assessment" in payload
            and "task_class" in payload
        )

        if proposal_packet_direct:
            direct_packet = dict(payload)
            risk_assessment = (
                direct_packet.get("risk_assessment") if isinstance(direct_packet.get("risk_assessment"), Mapping) else {}
            )
            request_obj = dict(direct_packet)
            request_obj["session_kind"] = "worker_slice"
            request_obj["task_class"] = str(direct_packet.get("task_class") or "").strip()
            request_obj["risk_tier"] = str(risk_assessment.get("initial_tier") or "").strip()
            request_obj["proposal_packet"] = direct_packet
        elif isinstance(request_obj.get("proposal_packet"), Mapping):
            proposal_packet = dict(request_obj.get("proposal_packet") or {})
            proposal_risk_assessment = (
                proposal_packet.get("risk_assessment") if isinstance(proposal_packet.get("risk_assessment"), Mapping) else {}
            )
            if not str(request_obj.get("session_kind") or "").strip():
                request_obj["session_kind"] = "worker_slice"
            if not str(request_obj.get("task_class") or "").strip():
                request_obj["task_class"] = str(proposal_packet.get("task_class") or "").strip()
            if not str(request_obj.get("risk_tier") or "").strip():
                request_obj["risk_tier"] = str(proposal_risk_assessment.get("initial_tier") or "").strip()
    else:
        request_obj = {
            "session_kind": args.session_kind,
            "task_class": args.task_class,
            "risk_tier": args.risk_tier,
        }

    if str(args.requested_route_class or "").strip():
        request_obj["requested_route_class"] = str(args.requested_route_class).strip()
    if str(args.requested_required_rollout_stage or "").strip():
        request_obj["requested_required_rollout_stage"] = str(args.requested_required_rollout_stage).strip()
    if str(args.requested_model_key or "").strip():
        request_obj["requested_model_key"] = str(args.requested_model_key).strip()
    if str(args.requested_rule_id or "").strip():
        request_obj["requested_rule_id"] = str(args.requested_rule_id).strip()

    return request_obj


def _load_decisions(paths: List[str]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        payload = load_json_file(path)
        if isinstance(payload, Mapping):
            out.append(dict(payload))
    return out


def _load_transport_decision(path_raw: str) -> Optional[Mapping[str, Any]]:
    token = str(path_raw or "").strip()
    if not token:
        return None
    path = Path(token).expanduser().resolve()
    payload = load_json_file(path)
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError("transport_decision_not_object")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    require_transport_decision = not bool(args.legacy_allow_missing_transport_decision)
    require_worker_allocation_contract = bool(args.require_worker_allocation_contract)
    if args.legacy_allow_missing_worker_allocation_contract:
        require_worker_allocation_contract = False
    require_telegram_direct_heavy_offload = not bool(args.legacy_allow_telegram_direct_heavy_on_dm)
    require_proposal_packet_for_coding = bool(args.require_proposal_first_coding)
    require_regression_risk_packet_for_coding = bool(args.require_regression_risk_packet_for_coding)
    require_refactor_risk_packet_for_coding = bool(args.require_refactor_risk_packet_for_coding)

    repo_root = Path(args.repo_root).expanduser().resolve()
    topology_path = Path(args.topology).expanduser().resolve()
    topology_schema_path = Path(args.topology_schema).expanduser().resolve()

    pool_policy_path_raw = Path(args.pool_policy).expanduser()
    pool_policy_path = pool_policy_path_raw if pool_policy_path_raw.is_absolute() else (repo_root / pool_policy_path_raw).resolve()
    pool_policy_schema_raw = Path(args.pool_policy_schema).expanduser()
    pool_policy_schema_path = (
        pool_policy_schema_raw if pool_policy_schema_raw.is_absolute() else (repo_root / pool_policy_schema_raw).resolve()
    )

    routing_policy_path_raw = Path(args.routing_policy).expanduser()
    routing_policy_path = (
        routing_policy_path_raw if routing_policy_path_raw.is_absolute() else (repo_root / routing_policy_path_raw).resolve()
    )
    routing_policy_schema_raw = Path(args.routing_policy_schema).expanduser()
    routing_policy_schema_path = (
        routing_policy_schema_raw
        if routing_policy_schema_raw.is_absolute()
        else (repo_root / routing_policy_schema_raw).resolve()
    )

    cache_enabled = not bool(args.no_prompt_tool_cache)
    cache_ttl_raw = str(args.prompt_tool_cache_ttl_sec or "").strip() or str(
        os.environ.get("OPENCLAW_PROMPT_TOOL_CACHE_TTL_SEC", DEFAULT_PROMPT_TOOL_CACHE_TTL_SEC)
    )
    try:
        cache_ttl_sec = max(1, int(cache_ttl_raw))
    except Exception:
        cache_ttl_sec = int(DEFAULT_PROMPT_TOOL_CACHE_TTL_SEC)
    cache_path_raw = Path(args.prompt_tool_cache).expanduser()
    if cache_path_raw.is_absolute() and str(cache_path_raw) == str(DEFAULT_PROMPT_TOOL_CACHE_PATH):
        cache_path = (repo_root / "state" / "continuity" / "session_topology_router" / "prompt_tool_cache_v1.json").resolve()
    else:
        cache_path = cache_path_raw if cache_path_raw.is_absolute() else (repo_root / cache_path_raw).resolve()

    context_delta_enabled = not bool(args.no_context_delta_transport)
    context_delta_path_raw = Path(args.context_delta_cache).expanduser()
    if context_delta_path_raw.is_absolute() and str(context_delta_path_raw) == str(DEFAULT_CONTEXT_DELTA_CACHE_PATH):
        context_delta_cache_path = (repo_root / "state" / "continuity" / "session_topology_router" / "context_delta_cache_v1.json").resolve()
    else:
        context_delta_cache_path = (
            context_delta_path_raw if context_delta_path_raw.is_absolute() else (repo_root / context_delta_path_raw).resolve()
        )

    context_compaction_enabled = not bool(args.no_context_compaction)
    context_compaction_path_raw = Path(args.context_compaction_cache).expanduser()
    if context_compaction_path_raw.is_absolute() and str(context_compaction_path_raw) == str(DEFAULT_CONTEXT_COMPACTION_CACHE_PATH):
        context_compaction_cache_path = (
            repo_root / "state" / "continuity" / "session_topology_router" / "context_compaction_cache_v1.json"
        ).resolve()
    else:
        context_compaction_cache_path = (
            context_compaction_path_raw
            if context_compaction_path_raw.is_absolute()
            else (repo_root / context_compaction_path_raw).resolve()
        )

    event_backbone_enabled = not bool(args.no_event_backbone)
    event_backbone_typed_log_raw = Path(args.event_backbone_typed_log).expanduser()
    if event_backbone_typed_log_raw.is_absolute() and str(event_backbone_typed_log_raw) == str(DEFAULT_EVENT_BACKBONE_TYPED_LOG):
        event_backbone_typed_log_path = (repo_root / "state" / "continuity" / "session_topology_router" / "typed_events.jsonl").resolve()
    else:
        event_backbone_typed_log_path = (
            event_backbone_typed_log_raw
            if event_backbone_typed_log_raw.is_absolute()
            else (repo_root / event_backbone_typed_log_raw).resolve()
        )

    event_backbone_db_raw = Path(args.event_backbone_db).expanduser()
    if event_backbone_db_raw.is_absolute() and str(event_backbone_db_raw) == str(DEFAULT_EVENT_BACKBONE_DB):
        event_backbone_db_path = (repo_root / "state" / "continuity" / "session_topology_router" / "event_backbone.sqlite").resolve()
    else:
        event_backbone_db_path = event_backbone_db_raw if event_backbone_db_raw.is_absolute() else (repo_root / event_backbone_db_raw).resolve()

    event_backbone_dlq_raw = Path(args.event_backbone_dlq).expanduser()
    if event_backbone_dlq_raw.is_absolute() and str(event_backbone_dlq_raw) == str(DEFAULT_EVENT_BACKBONE_DLQ):
        event_backbone_dlq_path = (repo_root / "state" / "continuity" / "session_topology_router" / "event_backbone_dlq.jsonl").resolve()
    else:
        event_backbone_dlq_path = event_backbone_dlq_raw if event_backbone_dlq_raw.is_absolute() else (repo_root / event_backbone_dlq_raw).resolve()

    event_backbone_metrics_raw = Path(args.event_backbone_metrics).expanduser()
    if event_backbone_metrics_raw.is_absolute() and str(event_backbone_metrics_raw) == str(DEFAULT_EVENT_BACKBONE_METRICS):
        event_backbone_metrics_path = (repo_root / "state" / "continuity" / "session_topology_router" / "event_backbone_metrics.json").resolve()
    else:
        event_backbone_metrics_path = (
            event_backbone_metrics_raw if event_backbone_metrics_raw.is_absolute() else (repo_root / event_backbone_metrics_raw).resolve()
        )

    try:
        event_backbone_max_attempts = max(1, int(str(args.event_backbone_max_attempts or DEFAULT_EVENT_BACKBONE_MAX_ATTEMPTS).strip()))
    except Exception:
        event_backbone_max_attempts = int(DEFAULT_EVENT_BACKBONE_MAX_ATTEMPTS)
    try:
        event_backbone_base_backoff_ms = max(1, int(str(args.event_backbone_base_backoff_ms or DEFAULT_EVENT_BACKBONE_BASE_BACKOFF_MS).strip()))
    except Exception:
        event_backbone_base_backoff_ms = int(DEFAULT_EVENT_BACKBONE_BASE_BACKOFF_MS)

    workflow_state_journal_raw = Path(args.workflow_state_journal).expanduser()
    if workflow_state_journal_raw.is_absolute() and str(workflow_state_journal_raw) == str(DEFAULT_WORKFLOW_STATE_JOURNAL):
        workflow_state_journal_path = (
            repo_root / "state" / "continuity" / "session_topology_router" / "workflow_state_machine_journal.jsonl"
        ).resolve()
    else:
        workflow_state_journal_path = (
            workflow_state_journal_raw
            if workflow_state_journal_raw.is_absolute()
            else (repo_root / workflow_state_journal_raw).resolve()
        )

    workflow_state_latest_raw = Path(args.workflow_state_latest).expanduser()
    if workflow_state_latest_raw.is_absolute() and str(workflow_state_latest_raw) == str(DEFAULT_WORKFLOW_STATE_LATEST):
        workflow_state_latest_path = (
            repo_root / "state" / "continuity" / "session_topology_router" / "workflow_state_machine_latest.json"
        ).resolve()
    else:
        workflow_state_latest_path = (
            workflow_state_latest_raw
            if workflow_state_latest_raw.is_absolute()
            else (repo_root / workflow_state_latest_raw).resolve()
        )

    request_doc: Mapping[str, Any] = {}
    transport_decision_doc: Optional[Mapping[str, Any]] = None

    routing_policy_ok, routing_policy_reason, routing_policy_details, routing_policy_doc = load_routing_policy(
        policy_path=routing_policy_path,
        policy_schema_path=routing_policy_schema_path,
    )

    if not routing_policy_ok:
        result = {
            "schema": "clawd.session_topology_routing.decision.v1",
            "router_layer": ROUTER_LAYER,
            "router_module": "scripts/session_topology_router.py",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "routing_policy_alignment",
            "block_reason": routing_policy_reason,
            "request": {
                "session_kind": None,
                "task_class": None,
                "risk_tier": None,
                "route_lock": {},
                "transport_binding": {},
                "agent_binding": {},
            },
            "transport_conformance": {
                "required": require_transport_decision,
                "resolved_binding": {},
                "decision_schema": None,
                "telegram_direct_heavy_offload_required": bool(require_telegram_direct_heavy_offload),
            },
            "route": {
                "selected_rule_id": None,
                "candidate_rule_ids": [],
                "route_class": None,
                "required_rollout_stage": None,
                "selected_model": None,
            },
            "routing_policy": {
                "policy_path": str(routing_policy_path),
                "policy_schema_path": str(routing_policy_schema_path),
            },
            "pool_policy": {
                "policy_path": str(pool_policy_path),
                "policy_schema_path": str(pool_policy_schema_path),
            },
            "gates": [
                {"gate": "topology_schema", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "routing_request", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "proposal_first_delta_spec", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "regression_risk_packet", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "refactor_risk_decomposition", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "transport_route_conformance", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "route_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "telegram_direct_offload", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "tier_escalation_evidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "routing_policy_alignment", "status": "fail", "reason": routing_policy_reason, "details": routing_policy_details},
                {"gate": "pool_policy_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "qualification_model_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "requested_route_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "prompt_guardrails", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "hybrid_retrieval_efficiency", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
            "actionable_failure": {
                "gate": "routing_policy_alignment",
                "reason": routing_policy_reason,
                "hint": "Fix session topology routing policy contract loading before route-policy evaluation.",
                "commands": [],
                "details": routing_policy_details,
            },
        }
    else:
        pool_policy_ok, pool_policy_reason, pool_policy_details, pool_policy_doc = load_pool_policy(
            policy_path=pool_policy_path,
            policy_schema_path=pool_policy_schema_path,
        )

        if not pool_policy_ok:
            result = {
                "schema": "clawd.session_topology_routing.decision.v1",
                "router_layer": ROUTER_LAYER,
                "router_module": "scripts/session_topology_router.py",
                "evaluated_at": now_iso(),
                "decision": "BLOCK",
                "final_state": "BLOCKED",
                "block_gate": "pool_policy_alignment",
                "block_reason": pool_policy_reason,
                "request": {
                    "session_kind": None,
                    "task_class": None,
                    "risk_tier": None,
                    "route_lock": {},
                    "transport_binding": {},
                    "agent_binding": {},
                },
                "transport_conformance": {
                    "required": require_transport_decision,
                    "resolved_binding": {},
                    "decision_schema": None,
                    "telegram_direct_heavy_offload_required": bool(require_telegram_direct_heavy_offload),
                },
                "route": {
                    "selected_rule_id": None,
                    "candidate_rule_ids": [],
                    "route_class": None,
                    "required_rollout_stage": None,
                    "selected_model": None,
                },
                "pool_policy": {
                    "policy_path": str(pool_policy_path),
                    "policy_schema_path": str(pool_policy_schema_path),
                },
                "routing_policy": {
                    "policy_id": routing_policy_doc.get("policy_id") if isinstance(routing_policy_doc, Mapping) else None,
                    "policy_path": str(routing_policy_path),
                    "policy_schema_path": str(routing_policy_schema_path),
                },
                "gates": [
                    {"gate": "topology_schema", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "routing_request", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "proposal_first_delta_spec", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "regression_risk_packet", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "refactor_risk_decomposition", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "transport_route_conformance", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "route_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "telegram_direct_offload", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "tier_escalation_evidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "routing_policy_alignment", "status": "pass", "details": routing_policy_details},
                    {"gate": "pool_policy_alignment", "status": "fail", "reason": pool_policy_reason, "details": pool_policy_details},
                    {"gate": "qualification_model_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "requested_route_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "prompt_guardrails", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    {"gate": "hybrid_retrieval_efficiency", "status": "skipped", "reason": "blocked_by_previous_gate"},
                ],
                "actionable_failure": {
                    "gate": "pool_policy_alignment",
                    "reason": pool_policy_reason,
                    "hint": "Fix model pool policy contract loading before route-policy evaluation.",
                    "commands": [],
                    "details": pool_policy_details,
                },
            }
        else:
            try:
                topology_doc = load_json_file(topology_path)
                request_doc = _load_request(args)
                gate_decisions = _load_decisions(list(args.qualification_decision or []))
                transport_decision_doc = _load_transport_decision(args.transport_decision)
            except Exception as exc:
                result = {
                    "schema": "clawd.session_topology_routing.decision.v1",
                    "router_layer": ROUTER_LAYER,
                    "router_module": "scripts/session_topology_router.py",
                    "evaluated_at": now_iso(),
                    "decision": "BLOCK",
                    "final_state": "BLOCKED",
                    "block_gate": "topology_schema",
                    "block_reason": "schema_invalid",
                    "request": {
                        "session_kind": None,
                        "task_class": None,
                        "risk_tier": None,
                        "route_lock": {},
                        "transport_binding": {},
                        "agent_binding": {},
                    },
                    "transport_conformance": {
                        "required": require_transport_decision,
                        "resolved_binding": {},
                        "decision_schema": None,
                        "telegram_direct_heavy_offload_required": bool(require_telegram_direct_heavy_offload),
                    },
                    "route": {
                        "selected_rule_id": None,
                        "candidate_rule_ids": [],
                        "route_class": None,
                        "required_rollout_stage": None,
                        "selected_model": None,
                    },
                    "pool_policy": {
                        "policy_id": pool_policy_doc.get("policy_id") if isinstance(pool_policy_doc, dict) else None,
                        "policy_path": str(pool_policy_path),
                        "policy_schema_path": str(pool_policy_schema_path),
                    },
                    "routing_policy": {
                        "policy_id": routing_policy_doc.get("policy_id") if isinstance(routing_policy_doc, Mapping) else None,
                        "policy_path": str(routing_policy_path),
                        "policy_schema_path": str(routing_policy_schema_path),
                    },
                    "gates": [
                        {"gate": "topology_schema", "status": "fail", "reason": "schema_invalid", "details": {"error": "input_load_failed", "detail": str(exc)}},
                        {"gate": "routing_request", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "proposal_first_delta_spec", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "regression_risk_packet", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "refactor_risk_decomposition", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "transport_route_conformance", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "route_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "telegram_direct_offload", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "tier_escalation_evidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "routing_policy_alignment", "status": "pass", "details": routing_policy_details},
                        {"gate": "pool_policy_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "qualification_model_selection", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "requested_route_alignment", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "prompt_guardrails", "status": "skipped", "reason": "blocked_by_previous_gate"},
                        {"gate": "hybrid_retrieval_efficiency", "status": "skipped", "reason": "blocked_by_previous_gate"},
                    ],
                    "actionable_failure": {
                        "gate": "topology_schema",
                        "reason": "schema_invalid",
                        "hint": "Fix input JSON loading/schema paths before routing.",
                        "commands": [],
                        "details": {"error": "input_load_failed", "detail": str(exc)},
                    },
                }
            else:
                cache_meta: Dict[str, Any] = {
                    "enabled": bool(cache_enabled),
                    "store_path": str(cache_path),
                    "ttl_sec": int(cache_ttl_sec),
                    "status": "disabled" if not cache_enabled else "miss",
                    "key": None,
                    "semantic_key": None,
                    "lookup": None,
                    "expires_at_epoch": None,
                    "created_at": None,
                }

                active_pool_policy = pool_policy_doc if isinstance(pool_policy_doc, dict) else {}
                active_routing_policy = routing_policy_doc if isinstance(routing_policy_doc, dict) else {}

                result: Optional[Mapping[str, Any]] = None
                exact_key = ""
                semantic_key = ""
                fingerprint: Dict[str, Any] = {}
                cache_store: Optional[Dict[str, Any]] = None

                if cache_enabled:
                    exact_key, semantic_key, fingerprint = _cache_exact_and_semantic_keys(
                        topology=topology_doc,
                        request=request_doc,
                        gate_decisions=gate_decisions,
                        transport_decision=transport_decision_doc,
                        pool_policy=active_pool_policy,
                        routing_policy=active_routing_policy,
                        require_transport_decision=require_transport_decision,
                        require_worker_allocation_contract=require_worker_allocation_contract,
                        require_telegram_direct_heavy_offload=require_telegram_direct_heavy_offload,
                        require_proposal_packet_for_coding=require_proposal_packet_for_coding,
                        require_regression_risk_packet_for_coding=require_regression_risk_packet_for_coding,
                        require_refactor_risk_packet_for_coding=require_refactor_risk_packet_for_coding,
                    )
                    cache_meta["key"] = exact_key
                    cache_meta["semantic_key"] = semantic_key

                    if args.prompt_tool_cache_flush and cache_path.exists():
                        cache_path.unlink()

                    cache_store = _load_prompt_tool_cache(cache_path)
                    lookup_now = _cache_now_epoch()
                    hit_entry, lookup_meta = _cache_lookup(
                        store=cache_store,
                        exact_key=exact_key,
                        semantic_key=semantic_key,
                        now_epoch=lookup_now,
                    )
                    cache_meta["lookup"] = lookup_meta

                    if hit_entry is not None and isinstance(hit_entry.get("result"), Mapping):
                        result = _json_clone(hit_entry.get("result"))
                        cache_meta["status"] = f"hit_{lookup_meta.get('hit_type') or 'unknown'}"
                        cache_meta["expires_at_epoch"] = hit_entry.get("expires_at_epoch")
                        cache_meta["created_at"] = hit_entry.get("created_at")
                    else:
                        cache_meta["status"] = "miss"
                        result = evaluate_routing(
                            topology=topology_doc,
                            request=request_doc,
                            topology_schema_path=topology_schema_path,
                            gate_decisions=gate_decisions,
                            pool_policy=active_pool_policy,
                            pool_policy_meta=pool_policy_details,
                            routing_policy=active_routing_policy,
                            routing_policy_meta=routing_policy_details,
                            transport_decision=transport_decision_doc,
                            require_transport_decision=require_transport_decision,
                            require_worker_allocation_contract=require_worker_allocation_contract,
                            require_telegram_direct_heavy_offload=require_telegram_direct_heavy_offload,
                            require_proposal_packet_for_coding=require_proposal_packet_for_coding,
                            require_regression_risk_packet_for_coding=require_regression_risk_packet_for_coding,
                            require_refactor_risk_packet_for_coding=require_refactor_risk_packet_for_coding,
                            repo_root=repo_root,
                        )
                        _cache_store_result(
                            store=cache_store,
                            key=exact_key,
                            now_epoch=lookup_now,
                            ttl_sec=cache_ttl_sec,
                            result=result,
                            fingerprint=fingerprint,
                        )
                        cache_meta["expires_at_epoch"] = lookup_now + max(1, int(cache_ttl_sec))
                        cache_meta["created_at"] = now_iso()

                    _save_prompt_tool_cache(cache_path, cache_store)
                else:
                    result = evaluate_routing(
                        topology=topology_doc,
                        request=request_doc,
                        topology_schema_path=topology_schema_path,
                        gate_decisions=gate_decisions,
                        pool_policy=active_pool_policy,
                        pool_policy_meta=pool_policy_details,
                        routing_policy=active_routing_policy,
                        routing_policy_meta=routing_policy_details,
                        transport_decision=transport_decision_doc,
                        require_transport_decision=require_transport_decision,
                        require_worker_allocation_contract=require_worker_allocation_contract,
                        require_telegram_direct_heavy_offload=require_telegram_direct_heavy_offload,
                        require_proposal_packet_for_coding=require_proposal_packet_for_coding,
                        require_regression_risk_packet_for_coding=require_regression_risk_packet_for_coding,
                        require_refactor_risk_packet_for_coding=require_refactor_risk_packet_for_coding,
                        repo_root=repo_root,
                    )

                if isinstance(result, Mapping):
                    result = dict(result)
                    result["cache"] = cache_meta

    context_transport = evaluate_context_delta_transport(
        request=request_doc if isinstance(request_doc, Mapping) else {},
        transport_decision=transport_decision_doc,
        cache_path=context_delta_cache_path,
        enabled=context_delta_enabled,
        flush=bool(args.context_delta_cache_flush),
    )
    result["context_transport"] = context_transport
    result["context_compaction"] = evaluate_anchor_preserving_summary_compaction(
        request=request_doc if isinstance(request_doc, Mapping) else {},
        context_transport=context_transport,
        transport_decision=transport_decision_doc,
        cache_path=context_compaction_cache_path,
        enabled=context_compaction_enabled,
        flush=bool(args.context_compaction_cache_flush),
    )

    decision_log_path: Optional[Path] = None
    if not args.no_decision_log:
        decision_log_raw = Path(args.decision_log).expanduser()
        if decision_log_raw.is_absolute() and str(decision_log_raw) == str(DEFAULT_DECISION_LOG):
            decision_log_path = (repo_root / "state" / "continuity" / "session_topology_router" / "decisions.jsonl").resolve()
        else:
            decision_log_path = decision_log_raw if decision_log_raw.is_absolute() else (repo_root / decision_log_raw).resolve()

    legacy_event_payload = _json_clone(result)
    if decision_log_path is not None and event_backbone_enabled:
        event_backbone_record = publish_event_backbone(
            repo_root=repo_root,
            request=request_doc if isinstance(request_doc, Mapping) else {},
            legacy_payload=legacy_event_payload,
            legacy_log_path=decision_log_path,
            typed_log_path=event_backbone_typed_log_path,
            db_path=event_backbone_db_path,
            dlq_path=event_backbone_dlq_path,
            metrics_path=event_backbone_metrics_path,
            enabled=True,
            max_attempts=event_backbone_max_attempts,
            base_backoff_ms=event_backbone_base_backoff_ms,
        )
        result["decision_record"] = dict(event_backbone_record.get("legacy_record") or {"enabled": True, "appended": False, "reason": "unknown"})
        event_backbone_summary = dict(event_backbone_record)
        event_backbone_summary.pop("legacy_record", None)
        result["event_backbone"] = event_backbone_summary
    else:
        record = append_decision_record(decision_log_path=decision_log_path, repo_root=repo_root, decision_row=result)
        result["decision_record"] = record
        result["event_backbone"] = {
            "enabled": False,
            "status": "disabled" if event_backbone_enabled else "disabled_by_flag",
            "reason": "decision_log_disabled" if decision_log_path is None else "disabled_by_flag",
        }

    route_decision_pre_workflow = str(result.get("decision") or "BLOCK")
    route_final_state_pre_workflow = str(result.get("final_state") or "BLOCKED")
    workflow_state_machine = evaluate_workflow_state_machine(
        repo_root=repo_root,
        request=request_doc if isinstance(request_doc, Mapping) else {},
        route_result=result,
        event_backbone_record=result.get("event_backbone") if isinstance(result.get("event_backbone"), Mapping) else {},
        journal_path=workflow_state_journal_path,
        latest_path=workflow_state_latest_path,
    )
    workflow_state_machine["route_decision_pre_workflow"] = route_decision_pre_workflow
    workflow_state_machine["route_final_state_pre_workflow"] = route_final_state_pre_workflow
    result["workflow_state_machine"] = workflow_state_machine

    gates = result.get("gates") if isinstance(result.get("gates"), list) else []
    workflow_gate_status = "pass" if str(workflow_state_machine.get("status") or "fail") == "pass" else "fail"
    workflow_gate_entry: Dict[str, Any] = {
        "gate": "workflow_state_machine",
        "status": workflow_gate_status,
        "details": workflow_state_machine,
    }
    if workflow_gate_status != "pass":
        workflow_gate_entry["reason"] = workflow_state_machine.get("reason") or "workflow_recovery_required"
    gates.append(workflow_gate_entry)
    result["gates"] = gates

    if route_decision_pre_workflow == "PASS" and workflow_gate_status != "pass":
        result["decision"] = "BLOCK"
        result["final_state"] = "RECOVERY_REQUIRED"
        result["block_gate"] = "workflow_state_machine"
        result["block_reason"] = workflow_state_machine.get("reason") or "workflow_recovery_required"
    elif route_decision_pre_workflow == "PASS" and str(workflow_state_machine.get("next_state") or "") == "ACTIVE":
        result["decision"] = "PASS"
        result["final_state"] = "ROUTED"
        result["block_gate"] = None
        result["block_reason"] = None

    result["actionable_failure"] = _actionable_failure(
        result.get("block_gate"),
        result.get("block_reason"),
        result.get("gates") if isinstance(result.get("gates"), list) else [],
    )

    token_violation_record = {"enabled": False, "appended": False, "reason": "not_triggered"}
    token_violation_row = result.get("token_violation") if isinstance(result.get("token_violation"), Mapping) else None
    if token_violation_row is not None:
        violation_ledger_path: Optional[Path] = None
        if not args.no_token_violation_ledger:
            violation_ledger_raw = Path(args.token_violation_ledger).expanduser()
            if violation_ledger_raw.is_absolute() and str(violation_ledger_raw) == str(DEFAULT_TOKEN_VIOLATION_LEDGER):
                violation_ledger_path = (
                    repo_root / "state" / "continuity" / "session_topology_router" / "token_violations.jsonl"
                ).resolve()
            else:
                violation_ledger_path = (
                    violation_ledger_raw if violation_ledger_raw.is_absolute() else (repo_root / violation_ledger_raw).resolve()
                )
        token_violation_record = append_decision_record(
            decision_log_path=violation_ledger_path,
            repo_root=repo_root,
            decision_row=dict(token_violation_row),
        )
    result["token_violation_record"] = token_violation_record

    result["operator_diagnostics"] = build_operator_diagnostics(
        decision=str(result.get("decision") or "BLOCK"),
        block_gate=result.get("block_gate"),
        block_reason=result.get("block_reason"),
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
